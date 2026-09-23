"""Versioned R2R v3 client and transactional document lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
import streamlit as st

import utils.helpers as func
from utils import logs
from utils.source_state import (
    active_index_matches_settings,
    ensure_active_source,
    make_source_state,
    mark_source_failed,
    mark_source_ready,
    mark_source_stale,
    stable_digest,
)

DEFAULT_R2R_BASE_URL = "http://localhost:7272"
R2R_SERVER_CONTRACT = "3.6.5"
R2R_REFERENCE_SDK_VERSION = "3.6.5"
R2R_API_CONTRACT = "v3"
R2R_CONTRACT = f"r2r-{R2R_SERVER_CONTRACT}-{R2R_API_CONTRACT}"
R2R_AUTH_HEADER = "x-api-key"
R2R_REQUEST_TIMEOUT = 120.0
R2R_POLL_TIMEOUT = 180.0
R2R_POLL_INITIAL_INTERVAL = 0.5
R2R_POLL_MAX_INTERVAL = 5.0
R2R_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
R2R_MAX_ANSWER_CHARS = 200_000
R2R_REGISTRY_SCHEMA_VERSION = 1
R2R_STATE_DIR_ENV = "DOCMIND_R2R_STATE_DIR"
R2R_PROJECT_NAME_ENV = "DOCMIND_R2R_PROJECT_NAME"
R2R_REGISTRY_FILENAME_PATTERN = re.compile(r"^[a-f0-9]{32}-[a-f0-9]{32}\.json$")
R2R_DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
R2R_REGISTRY_FIELDS = {
    "schema_version",
    "contract",
    "workspace_id",
    "endpoint_fingerprint",
    "credential_fingerprint",
    "documents",
    "active_document_ids",
    "active_source_id",
    "active_content_signature",
    "active_settings_signature",
    "active_source_signature",
    "active_index_generation",
    "active_display_name",
    "created_at",
    "updated_at",
}
R2R_DOCUMENT_FIELDS = {
    "document_id",
    "task_id",
    "content_signature",
    "source_id",
    "source_signature",
    "status",
    "created_at",
    "updated_at",
    "last_error",
    "endpoint_fingerprint",
    "credential_fingerprint",
    "file_name",
}
_REGISTRY_LOCKS: dict[str, threading.RLock] = {}
_REGISTRY_LOCKS_GUARD = threading.Lock()


class R2RConnectionError(Exception):
    """Base error for safe user-facing R2R failures."""


class R2RHTTPError(R2RConnectionError):
    def __init__(self, status_code: int, category: str = "http_error"):
        self.status_code = status_code
        self.category = category
        super().__init__(f"R2R request failed ({category}, HTTP {status_code}).")


class R2RAuthenticationError(R2RHTTPError):
    def __init__(self, status_code: int):
        super().__init__(status_code, "authentication")


class R2RNotFoundError(R2RHTTPError):
    def __init__(self):
        super().__init__(404, "not_found")


class R2RRedirectError(R2RHTTPError):
    def __init__(self, status_code: int):
        super().__init__(status_code, "redirect_blocked")


class R2RResponseError(R2RConnectionError):
    """Raised when a remote response violates a safety or schema bound."""


class R2RPollTimeout(R2RConnectionError):
    def __init__(self):
        super().__init__("R2R indexing did not become ready before the deadline.")


class R2RIngestionCancelled(R2RConnectionError):
    def __init__(self):
        super().__init__("R2R indexing was cancelled.")


class R2RLifecycleError(R2RConnectionError):
    """Raised when a document transaction cannot be completed safely."""


class R2RRollbackError(R2RLifecycleError):
    """Raised when one or more rollback deletions fail."""


class R2RRegistryError(R2RLifecycleError):
    """Raised when the local ownership registry cannot be read or written."""


@dataclass(frozen=True)
class R2RIdentity:
    endpoint_fingerprint: str
    credential_fingerprint: str | None

    def as_dict(self) -> dict:
        return asdict(self)

    def matches(self, other: Mapping | None) -> bool:
        return bool(
            other
            and other.get("endpoint_fingerprint") == self.endpoint_fingerprint
            and other.get("credential_fingerprint") == self.credential_fingerprint
        )


@dataclass(frozen=True)
class R2RUploadReceipt:
    document_id: str
    task_id: str | None = None


@dataclass(frozen=True)
class R2RDocumentStatus:
    document_id: str
    status: str
    task_id: str | None = None
    extraction_status: str | None = None


@dataclass(frozen=True)
class R2RDocumentWaitResult:
    document_id: str
    task_id: str | None
    status: str
    polls: int


@dataclass(frozen=True)
class R2RChatResult:
    answer: str
    citations: list[dict]
    search_results: dict


@dataclass(frozen=True)
class R2RDeleteResult:
    deleted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    failed: tuple[tuple[str | None, str], ...] = ()

    @property
    def success(self) -> bool:
        return not self.failed

    def as_dict(self) -> dict:
        return {
            "deleted": list(self.deleted),
            "missing": list(self.missing),
            "failed": [
                {"document_id": document_id, "error": error}
                for document_id, error in self.failed
            ],
        }


@dataclass(frozen=True)
class R2RRemoteResetResult:
    delete_result: R2RDeleteResult
    other_identity_count: int = 0
    attempted: bool = True

    @property
    def success(self) -> bool:
        return self.delete_result.success and self.other_identity_count == 0

    def as_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "success": self.success,
            "other_identity_count": self.other_identity_count,
            **self.delete_result.as_dict(),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def credential_fingerprint(api_key: str | None) -> str | None:
    value = str(api_key or "")
    return _fingerprint(value) if value else None


def normalize_base_url(value: str) -> str:
    raw = str(value or DEFAULT_R2R_BASE_URL).strip()
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("R2R URL must be an absolute HTTP or HTTPS URL.")
    if parsed.username or parsed.password:
        raise ValueError("R2R URL must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("R2R URL must not contain a query string or fragment.")
    try:
        port = parsed.port
    except ValueError as err:
        raise ValueError("R2R URL contains an invalid port.") from err
    host = parsed.hostname.lower().rstrip(".")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = 80 if parsed.scheme.lower() == "http" else 443
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def endpoint_fingerprint(value: str) -> str:
    return _fingerprint(normalize_base_url(value))


def _validate_header_value(value: str, label: str) -> str:
    if "\r" in value or "\n" in value:
        raise ValueError(f"{label} contains an invalid line break.")
    return value


def _safe_identifier(value: str, label: str = "R2R identifier") -> str:
    identifier = str(value or "").strip()
    if not R2R_DOCUMENT_ID_PATTERN.fullmatch(identifier):
        raise R2RLifecycleError(f"{label} is invalid.")
    return identifier


def _safe_error_code(error: BaseException) -> str:
    if isinstance(error, R2RHTTPError):
        return error.category
    if isinstance(error, R2RPollTimeout):
        return "poll_timeout"
    if isinstance(error, R2RIngestionCancelled):
        return "cancelled"
    if isinstance(error, R2RResponseError):
        return "invalid_response"
    if isinstance(error, R2RRegistryError):
        return "registry_error"
    if isinstance(error, R2RConnectionError):
        return "connection_error"
    return type(error).__name__


def safe_error_category(error: BaseException) -> str:
    return _safe_error_code(error)


def safe_user_error(error: BaseException) -> str:
    if isinstance(error, R2RAuthenticationError):
        return "R2R authentication failed. Check the configured API key."
    if isinstance(error, R2RRedirectError):
        return "R2R redirected the request. Enter the final HTTPS endpoint explicitly."
    if isinstance(error, R2RNotFoundError):
        return "The requested R2R resource was not found."
    if isinstance(error, R2RPollTimeout):
        return "R2R indexing timed out before the documents became ready."
    if isinstance(error, R2RIngestionCancelled):
        return "R2R indexing was cancelled."
    if isinstance(error, R2RRollbackError):
        return "R2R indexing failed and remote rollback is incomplete. Reset the project to retry cleanup."
    if isinstance(error, R2RResponseError):
        return "R2R returned an invalid or oversized response."
    if isinstance(error, R2RHTTPError):
        return f"R2R rejected the request ({error.category}, HTTP {error.status_code})."
    if isinstance(error, R2RConnectionError):
        return "Could not complete the R2R request. Check the server and network."
    return "R2R lifecycle operation failed."


class R2RClient:
    """Synchronous client for the explicitly supported R2R 3.6.x v3 contract."""

    def __init__(
        self,
        base_url: str = DEFAULT_R2R_BASE_URL,
        api_key: str = "",
        timeout: float = R2R_REQUEST_TIMEOUT,
        project_name: str = "",
        max_response_bytes: int = R2R_MAX_RESPONSE_BYTES,
    ):
        self.base_url = normalize_base_url(base_url)
        self.timeout = max(1.0, float(timeout))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self._api_key = str(api_key or "")
        self._project_name = _validate_header_value(
            str(project_name or ""), "R2R project name"
        )
        self._headers = {"Accept": "application/json"}
        if self._api_key:
            self._headers[R2R_AUTH_HEADER] = _validate_header_value(
                self._api_key, "R2R API key"
            )
        if self._project_name:
            self._headers["x-project-name"] = self._project_name
        self.identity = R2RIdentity(
            endpoint_fingerprint=_fingerprint(self.base_url),
            credential_fingerprint=credential_fingerprint(self._api_key),
        )

    def _request(self, method: str, path: str, **kwargs):
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"{self.base_url}{path}"
        max_response_bytes = int(
            kwargs.pop("max_response_bytes", self.max_response_bytes)
        )
        try:
            response = httpx.request(
                method,
                url,
                headers=self._headers,
                timeout=self.timeout,
                follow_redirects=False,
                **kwargs,
            )
        except httpx.TimeoutException as err:
            raise R2RConnectionError("R2R request timed out.") from err
        except httpx.HTTPError as err:
            raise R2RConnectionError(
                "Could not reach the configured R2R server."
            ) from err

        status_code = int(response.status_code)
        if 300 <= status_code < 400:
            raise R2RRedirectError(status_code)
        if status_code in {401, 403}:
            raise R2RAuthenticationError(status_code)
        if status_code == 404:
            raise R2RNotFoundError()
        if status_code >= 400:
            category = (
                "rate_limit"
                if status_code == 429
                else "server_error"
                if status_code >= 500
                else "http_error"
            )
            raise R2RHTTPError(status_code, category)

        headers = getattr(response, "headers", {}) or {}
        content_length = headers.get("content-length") or headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > max_response_bytes:
                    raise R2RResponseError(
                        "R2R response exceeded the configured size limit."
                    )
            except (TypeError, ValueError):
                pass
        content = getattr(response, "content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")
        if content and len(content) > max_response_bytes:
            raise R2RResponseError("R2R response exceeded the configured size limit.")
        return response

    def health(self) -> bool:
        try:
            response = self._request("GET", "/v3/health", max_response_bytes=64 * 1024)
            _as_dict(response)
            return True
        except R2RConnectionError:
            return False

    def upload_document(
        self,
        file_path: str,
        document_id: str | None = None,
        run_with_orchestration: bool = True,
    ) -> R2RUploadReceipt:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise R2RLifecycleError("R2R upload file is unavailable.")
        name = path.name
        data = {"run_with_orchestration": str(bool(run_with_orchestration)).lower()}
        if document_id:
            data["id"] = _safe_identifier(document_id, "R2R document ID")
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with path.open("rb") as handle:
            files = [("file", (name, handle, content_type))]
            response = self._request("POST", "/v3/documents", data=data, files=files)
        return _extract_upload_receipt(_as_dict(response))

    def upload_documents(self, file_paths: list[str]) -> list[str]:
        return [self.upload_document(file_path).document_id for file_path in file_paths]

    def get_document(self, document_id: str) -> R2RDocumentStatus:
        identifier = _safe_identifier(document_id)
        response = self._request("GET", f"/v3/documents/{quote(identifier, safe='')}")
        return _extract_document_status(_as_dict(response))

    def delete_document(self, document_id: str) -> str:
        identifier = _safe_identifier(document_id)
        try:
            self._request("DELETE", f"/v3/documents/{quote(identifier, safe='')}")
            return "deleted"
        except R2RNotFoundError:
            return "missing"

    def delete_documents(self, document_ids: list[str]) -> R2RDeleteResult:
        deleted = []
        missing = []
        failed = []
        for document_id in dict.fromkeys(str(item) for item in document_ids):
            try:
                outcome = self.delete_document(document_id)
            except R2RConnectionError as err:
                failed.append((document_id, _safe_error_code(err)))
                continue
            if outcome == "missing":
                missing.append(document_id)
            else:
                deleted.append(document_id)
        return R2RDeleteResult(tuple(deleted), tuple(missing), tuple(failed))

    def rag(
        self,
        query: str,
        document_ids: list,
        top_k: int = 10,
        answer_style: str | None = None,
    ) -> R2RChatResult:
        query_text = str(query or "").strip()
        if not query_text:
            raise R2RLifecycleError("R2R query is empty.")
        payload = {
            "query": query_text,
            "search_mode": "custom",
            "search_settings": {
                "filters": _document_filter(document_ids),
                "limit": max(1, min(100, int(top_k))),
                "include_metadatas": True,
                "include_scores": True,
            },
            "rag_generation_config": {"stream": False},
            "include_web_search": False,
        }
        task_prompt = _style_task_prompt(answer_style)
        if task_prompt:
            payload["task_prompt"] = task_prompt
        response = self._request(
            "POST",
            "/v3/retrieval/rag",
            json=payload,
            max_response_bytes=R2R_MAX_RESPONSE_BYTES,
        )
        return _extract_chat_result(_as_dict(response))

    def wait_for_document(
        self,
        document_id: str,
        task_id: str | None = None,
        timeout: float = R2R_POLL_TIMEOUT,
        initial_interval: float = R2R_POLL_INITIAL_INTERVAL,
        max_interval: float = R2R_POLL_MAX_INTERVAL,
        cancel_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_status: Callable[[R2RDocumentStatus], None] | None = None,
    ) -> R2RDocumentWaitResult:
        identifier = _safe_identifier(document_id)
        deadline = clock() + max(0.0, float(timeout))
        delay = max(0.0, float(initial_interval))
        maximum_delay = max(delay, float(max_interval))
        polls = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise R2RIngestionCancelled()
            try:
                status = self.get_document(identifier)
            except (R2RAuthenticationError, R2RRedirectError):
                raise
            except R2RHTTPError as err:
                if err.status_code not in {404, 429} and err.status_code < 500:
                    raise
                if clock() >= deadline:
                    raise R2RPollTimeout() from None
                if cancel_event is not None:
                    if cancel_event.wait(delay):
                        raise R2RIngestionCancelled()
                else:
                    sleep(delay)
                delay = min(maximum_delay, max(0.1, delay * 2))
                continue
            except R2RConnectionError:
                if clock() >= deadline:
                    raise R2RPollTimeout() from None
                if cancel_event is not None:
                    if cancel_event.wait(delay):
                        raise R2RIngestionCancelled()
                else:
                    sleep(delay)
                delay = min(maximum_delay, max(0.1, delay * 2))
                continue
            polls += 1
            if on_status is not None:
                on_status(status)
            normalized = status.status.strip().lower()
            if normalized in {"success", "completed", "ready", "indexed"}:
                return R2RDocumentWaitResult(identifier, task_id, normalized, polls)
            if normalized in {
                "failed",
                "failure",
                "error",
                "cancelled",
                "canceled",
            }:
                raise R2RLifecycleError("R2R document indexing failed.")
            if clock() >= deadline:
                raise R2RPollTimeout()
            if cancel_event is not None:
                if cancel_event.wait(delay):
                    raise R2RIngestionCancelled()
            else:
                sleep(delay)
            delay = min(maximum_delay, max(0.1, delay * 2))


def _as_dict(response) -> dict:
    try:
        payload = response.json()
    except (TypeError, ValueError) as err:
        raise R2RResponseError("R2R returned a non-JSON response.") from err
    if not isinstance(payload, dict):
        raise R2RResponseError("R2R returned an unexpected response shape.")
    return payload


def _result_entries(payload: dict) -> list[dict]:
    results = payload.get("results", payload)
    if isinstance(results, dict):
        return [results]
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    return []


def _extract_upload_receipt(payload: dict) -> R2RUploadReceipt:
    for entry in _result_entries(payload):
        document_id = entry.get("document_id") or entry.get("id")
        if document_id:
            task_id = entry.get("task_id")
            return R2RUploadReceipt(
                _safe_identifier(str(document_id)),
                str(task_id) if task_id else None,
            )
    raise R2RResponseError("R2R upload response did not include a document ID.")


def _extract_document_id(payload: dict) -> str:
    return _extract_upload_receipt(payload).document_id


def _extract_document_status(payload: dict) -> R2RDocumentStatus:
    entries = _result_entries(payload)
    document = entries[0] if entries else {}
    document_id = document.get("id") or document.get("document_id")
    status = document.get("ingestion_status") or document.get("status")
    if not document_id or not status:
        raise R2RResponseError("R2R document status response was incomplete.")
    return R2RDocumentStatus(
        document_id=_safe_identifier(str(document_id)),
        status=str(status),
        task_id=str(document.get("task_id")) if document.get("task_id") else None,
        extraction_status=(
            str(document.get("extraction_status"))
            if document.get("extraction_status") is not None
            else None
        ),
    )


def _bounded_value(value, depth: int = 0):
    if depth >= 6:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:2000]
    if isinstance(value, list):
        return [_bounded_value(item, depth + 1) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _bounded_value(item, depth + 1)
            for key, item in list(value.items())[:100]
        }
    return str(value)[:2000]


def _extract_chat_result(payload: dict) -> R2RChatResult:
    entries = _result_entries(payload)
    result = entries[0] if entries else {}
    answer = result.get("generated_answer")
    if answer is None:
        answer = result.get("rag_response")
    if answer is None:
        answer = result.get("answer")
    if answer is None:
        raise R2RResponseError("R2R response did not include an answer.")
    answer = str(answer)
    if len(answer) > R2R_MAX_ANSWER_CHARS:
        raise R2RResponseError("R2R answer exceeded the configured size limit.")
    citations = result.get("citations")
    search_results = result.get("search_results")
    return R2RChatResult(
        answer=answer,
        citations=_bounded_value(citations) if isinstance(citations, list) else [],
        search_results=(
            _bounded_value(search_results) if isinstance(search_results, dict) else {}
        ),
    )


def _extract_answer(payload: dict) -> str:
    return _extract_chat_result(payload).answer


def _style_task_prompt(answer_style: str | None) -> str | None:
    style = str(answer_style or "").strip()
    if not style:
        return None
    return f"Follow this answer style: {style[:160]}."


def _document_filter(document_ids: list[str]) -> dict:
    identifiers = [
        _safe_identifier(document_id) for document_id in dict.fromkeys(document_ids)
    ]
    if not identifiers:
        raise R2RLifecycleError("R2R document scope is empty.")
    return {"document_id": {"$in": identifiers}}


def _chat_sources(result: R2RChatResult) -> list[tuple[str, float]]:
    chunks = result.search_results.get("chunk_search_results")
    sources = []
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata")
            title = None
            if isinstance(metadata, dict):
                title = metadata.get("title") or metadata.get("filename")
            title = str(
                title or chunk.get("title") or chunk.get("document_id") or "document"
            )
            score = chunk.get("score")
            sources.append(
                (title, float(score) if isinstance(score, (int, float)) else 0.0)
            )
    return sources


def _registry_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _REGISTRY_LOCKS_GUARD:
        return _REGISTRY_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _locked_registry_file(path: Path):
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class R2RDocumentRegistry:
    """Atomic per-workspace ownership registry for remote R2R documents."""

    def __init__(
        self,
        path: str | Path,
        workspace_id: str,
        identity: R2RIdentity,
        contract: str = R2R_CONTRACT,
    ):
        self.path = Path(path).resolve()
        self.workspace_id = str(workspace_id)
        self.identity = identity
        self.contract = str(contract)
        self._thread_lock = _registry_lock(self.path)

    @classmethod
    def for_workspace(
        cls,
        workspace_id: str,
        identity: R2RIdentity,
        state_root: str | Path | None = None,
        project_path: str | Path | None = None,
    ) -> R2RDocumentRegistry:
        root_value = (
            state_root
            or os.getenv(R2R_STATE_DIR_ENV)
            or (Path.home() / ".docmind" / "r2r")
        )
        root = Path(root_value).expanduser().resolve()
        project = Path(project_path or Path.cwd()).expanduser().resolve()
        workspace_token = _fingerprint(str(workspace_id))[:32]
        project_token = _fingerprint(str(project))[:32]
        path = root / f"{project_token}-{workspace_token}.json"
        if not R2R_REGISTRY_FILENAME_PATTERN.fullmatch(path.name):
            raise R2RRegistryError("R2R registry path could not be generated safely.")
        return cls(path, str(workspace_id), identity)

    def _empty(self) -> dict:
        now = _utc_now()
        return {
            "schema_version": R2R_REGISTRY_SCHEMA_VERSION,
            "contract": self.contract,
            "workspace_id": self.workspace_id,
            "endpoint_fingerprint": self.identity.endpoint_fingerprint,
            "credential_fingerprint": self.identity.credential_fingerprint,
            "documents": [],
            "active_document_ids": [],
            "active_source_id": None,
            "active_content_signature": None,
            "active_settings_signature": None,
            "active_source_signature": None,
            "active_index_generation": 0,
            "active_display_name": None,
            "created_at": now,
            "updated_at": now,
        }

    def _load_unlocked(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, TypeError, ValueError) as err:
            raise R2RRegistryError("R2R registry could not be read.") from err
        if not isinstance(data, dict):
            raise R2RRegistryError("R2R registry has an invalid shape.")
        if data.get("schema_version") != R2R_REGISTRY_SCHEMA_VERSION:
            raise R2RRegistryError("R2R registry schema is unsupported.")
        if data.get("contract") != self.contract:
            raise R2RRegistryError("R2R registry contract does not match this client.")
        if data.get("workspace_id") != self.workspace_id:
            raise R2RRegistryError("R2R registry workspace does not match.")
        if not isinstance(data.get("documents"), list):
            raise R2RRegistryError("R2R registry documents are invalid.")
        self._validate(data)
        return data

    def _validate(self, data: Mapping) -> None:
        forbidden = {
            "api_key",
            "authorization",
            "access_token",
            "refresh_token",
            "password",
            "secret",
        }
        for document in data.get("documents", []):
            if not isinstance(document, dict):
                raise R2RRegistryError("R2R registry contains an invalid document.")
            if set(document) - R2R_DOCUMENT_FIELDS:
                raise R2RRegistryError(
                    "R2R registry document contains unsupported fields."
                )
            if forbidden.intersection(str(key).lower() for key in document):
                raise R2RRegistryError(
                    "R2R registry attempted to persist a credential."
                )
            _safe_identifier(str(document.get("document_id") or ""))

    def _write_unlocked(self, data: dict) -> None:
        self._validate(data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(temporary_name, 0o600)
            except OSError:
                pass
            os.replace(temporary_name, self.path)
        except OSError as err:
            raise R2RRegistryError("R2R registry could not be written.") from err
        finally:
            if os.path.exists(temporary_name):
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass

    def _mutate(self, callback: Callable[[dict], None]) -> dict:
        with self._thread_lock, _locked_registry_file(self.path):
            data = self._load_unlocked()
            callback(data)
            data["endpoint_fingerprint"] = self.identity.endpoint_fingerprint
            data["credential_fingerprint"] = self.identity.credential_fingerprint
            data["updated_at"] = _utc_now()
            self._write_unlocked(data)
            return data

    def snapshot(self) -> dict:
        with self._thread_lock, _locked_registry_file(self.path):
            return copy.deepcopy(self._load_unlocked())

    def upsert_document(self, record: Mapping) -> dict:
        clean = {key: record.get(key) for key in R2R_DOCUMENT_FIELDS}
        if not clean.get("created_at"):
            clean["created_at"] = _utc_now()
        clean["updated_at"] = _utc_now()
        clean["endpoint_fingerprint"] = self.identity.endpoint_fingerprint
        clean["credential_fingerprint"] = self.identity.credential_fingerprint

        def update(data: dict):
            documents = data["documents"]
            for index, existing in enumerate(documents):
                if existing.get("document_id") == clean.get("document_id"):
                    clean["created_at"] = (
                        existing.get("created_at") or clean["created_at"]
                    )
                    documents[index] = clean
                    break
            else:
                documents.append(clean)

        return copy.deepcopy(self._mutate(update))

    def update_document(self, document_id: str, **updates) -> dict:
        identifier = _safe_identifier(document_id)
        allowed = R2R_DOCUMENT_FIELDS - {"document_id", "created_at"}
        if set(updates) - allowed:
            raise R2RRegistryError("R2R registry update contains unsupported fields.")
        updated_document = None

        def update(data: dict):
            nonlocal updated_document
            for document in data["documents"]:
                if document.get("document_id") == identifier:
                    document.update(updates)
                    document["updated_at"] = _utc_now()
                    updated_document = copy.deepcopy(document)
                    return
            raise R2RRegistryError("R2R document is not owned by this registry.")

        self._mutate(update)
        if updated_document is None:
            raise R2RRegistryError("R2R document is not owned by this registry.")
        return updated_document

    def get_document(self, document_id: str) -> dict | None:
        identifier = _safe_identifier(document_id)
        for document in self.snapshot()["documents"]:
            if document.get("document_id") == identifier:
                return copy.deepcopy(document)
        return None

    def documents_for_identity(
        self, identity: R2RIdentity | None = None, include_deleted: bool = False
    ) -> list[dict]:
        selected_identity = identity or self.identity
        result = []
        for document in self.snapshot()["documents"]:
            if not selected_identity.matches(document):
                continue
            if not include_deleted and document.get("status") in {
                "deleted",
                "rolled_back",
            }:
                continue
            result.append(copy.deepcopy(document))
        return result

    def active_documents(self, identity: R2RIdentity | None = None) -> list[dict]:
        snapshot = self.snapshot()
        selected_identity = identity or self.identity
        active_ids = set(snapshot.get("active_document_ids") or [])
        return [
            copy.deepcopy(document)
            for document in snapshot["documents"]
            if document.get("document_id") in active_ids
            and document.get("status") == "ready"
            and selected_identity.matches(document)
        ]

    def commit_active(self, records: list[Mapping], metadata: Mapping) -> dict:
        identifiers = [
            _safe_identifier(str(record["document_id"])) for record in records
        ]
        if len(identifiers) != len(records):
            raise R2RRegistryError("R2R active document IDs are invalid.")

        def update(data: dict):
            data["active_document_ids"] = identifiers
            data["active_source_id"] = metadata.get("source_id")
            data["active_content_signature"] = metadata.get("content_signature")
            data["active_settings_signature"] = metadata.get("settings_signature")
            data["active_source_signature"] = metadata.get("source_signature")
            try:
                data["active_index_generation"] = max(
                    0, int(metadata.get("index_generation") or 0)
                )
            except (TypeError, ValueError) as err:
                raise R2RRegistryError("R2R source generation is invalid.") from err
            data["active_display_name"] = metadata.get("display_name")

        return copy.deepcopy(self._mutate(update))

    def update_active_metadata(self, metadata: Mapping) -> dict:
        def update(data: dict):
            data["active_source_id"] = metadata.get("source_id")
            data["active_content_signature"] = metadata.get("content_signature")
            data["active_settings_signature"] = metadata.get("settings_signature")
            data["active_source_signature"] = metadata.get("source_signature")
            try:
                data["active_index_generation"] = max(
                    0, int(metadata.get("index_generation") or 0)
                )
            except (TypeError, ValueError) as err:
                raise R2RRegistryError("R2R source generation is invalid.") from err
            data["active_display_name"] = metadata.get("display_name")

        return copy.deepcopy(self._mutate(update))

    def clear_active(self) -> dict:
        def update(data: dict):
            data["active_document_ids"] = []
            data["active_source_id"] = None
            data["active_content_signature"] = None
            data["active_settings_signature"] = None
            data["active_source_signature"] = None
            data["active_index_generation"] = 0
            data["active_display_name"] = None

        return copy.deepcopy(self._mutate(update))

    def invalidate_active(self) -> dict:
        return self.clear_active()


def _state_root(state: Mapping | None = None) -> Path | None:
    if state is not None:
        configured = state.get("r2r_registry_root")
        if configured:
            return Path(str(configured)).expanduser().resolve()
    configured = os.getenv(R2R_STATE_DIR_ENV)
    return Path(configured).expanduser().resolve() if configured else None


def get_client(state: Mapping | None = None) -> R2RClient:
    source = state if state is not None else st.session_state
    return R2RClient(
        base_url=source.get("r2r_base_url") or DEFAULT_R2R_BASE_URL,
        api_key=source.get("r2r_api_key") or "",
        project_name=os.getenv(R2R_PROJECT_NAME_ENV, ""),
    )


def registry_for_state(
    state: Mapping, client: R2RClient | None = None, create: bool = True
) -> R2RDocumentRegistry:
    workspace_id = str(state.get("r2r_workspace_id") or uuid.uuid4())
    if create:
        state["r2r_workspace_id"] = workspace_id
    selected_client = client or get_client(state)
    registry = R2RDocumentRegistry.for_workspace(
        workspace_id,
        selected_client.identity,
        state_root=_state_root(state),
        project_path=state.get("r2r_project_path") or Path.cwd(),
    )
    state["r2r_registry_path"] = str(registry.path)
    return registry


def _bound_identity(state: Mapping, identity: R2RIdentity) -> dict:
    return {"enabled": bool(state.get("r2r_enabled")), **identity.as_dict()}


def _invalidate_session_r2r(state, reason: str) -> None:
    state["r2r_document_ids"] = []
    state["r2r_document_ids_signature"] = None
    state["r2r_current_source_signature"] = None
    state["r2r_connection_ok"] = None
    source = ensure_active_source(state)
    if source.get("kind") == "r2r" and source.get("status") in {
        "ready",
        "stale",
        "failed",
    }:
        mark_source_stale(state, reason)


def invalidate_source_selection(state) -> None:
    state["r2r_document_ids"] = []
    state["r2r_document_ids_signature"] = None
    state["r2r_current_source_signature"] = None
    source = ensure_active_source(state)
    if source.get("kind") == "r2r":
        mark_source_stale(state, "The selected R2R source changed.")


def _restore_r2r_source(state, registry: R2RDocumentRegistry) -> bool:
    snapshot = registry.snapshot()
    records = registry.active_documents()
    source_signature = snapshot.get("active_source_signature")
    source_id = snapshot.get("active_source_id")
    content_signature = snapshot.get("active_content_signature")
    settings_signature = snapshot.get("active_settings_signature")
    if not records or not source_signature or not source_id or not settings_signature:
        return False
    current = ensure_active_source(state)
    if current.get("kind") not in {None, "r2r"}:
        return False
    generation = max(
        int(state.get("index_generation") or 0),
        int(snapshot.get("active_index_generation") or 0),
    )
    source = make_source_state(
        "r2r",
        source_id=str(source_id),
        display_name=snapshot.get("active_display_name") or "Restored R2R documents",
        content_signature=content_signature,
        settings_signature=str(settings_signature),
        index_generation=generation,
    )
    state["index_generation"] = generation
    mark_source_ready(state, source)
    state["r2r_document_ids"] = [record["document_id"] for record in records]
    state["r2r_document_ids_signature"] = str(source_signature)
    state["r2r_current_source_signature"] = str(source_signature)
    return True


def initialize_r2r_state(state, client: R2RClient | None = None) -> None:
    state.setdefault("r2r_enabled", False)
    state.setdefault("r2r_base_url", DEFAULT_R2R_BASE_URL)
    state.setdefault("r2r_api_key", "")
    workspace_id = str(state.get("r2r_workspace_id") or "")
    if (
        not workspace_id
        or len(workspace_id) > 128
        or any(character in workspace_id for character in "\r\n\x00")
    ):
        workspace_id = str(uuid.uuid4())
        state["r2r_workspace_id"] = workspace_id
    state.setdefault("r2r_registry_path", None)
    state.setdefault("r2r_document_ids", [])
    state.setdefault("r2r_document_ids_signature", None)
    state.setdefault("r2r_current_source_signature", None)
    state.setdefault("r2r_registry_initialized", False)
    state["r2r_server_contract"] = R2R_SERVER_CONTRACT
    state["r2r_reference_sdk_version"] = R2R_REFERENCE_SDK_VERSION
    state["r2r_api_contract"] = R2R_API_CONTRACT
    state["r2r_contract"] = R2R_CONTRACT

    try:
        selected_client = client or get_client(state)
    except (TypeError, ValueError):
        _invalidate_session_r2r(state, "Invalid R2R endpoint configuration.")
        state["r2r_connection_ok"] = False
        return
    registry = registry_for_state(state, selected_client)
    current_bound = _bound_identity(state, selected_client.identity)
    previous_bound = state.get("r2r_bound_identity")
    identity_changed = bool(
        previous_bound
        and (
            previous_bound.get("endpoint_fingerprint")
            != current_bound.get("endpoint_fingerprint")
            or previous_bound.get("credential_fingerprint")
            != current_bound.get("credential_fingerprint")
        )
    )
    toggle_changed = bool(
        previous_bound and previous_bound.get("enabled") != current_bound.get("enabled")
    )
    legacy_ids = bool(
        state.get("r2r_document_ids")
        and not state.get("r2r_registry_initialized")
        and not previous_bound
    )
    if identity_changed or toggle_changed or legacy_ids:
        _invalidate_session_r2r(
            state, "R2R endpoint, credential, or enablement changed."
        )
        if toggle_changed:
            registry.invalidate_active()
    elif not state.get("r2r_registry_initialized") and state.get("r2r_enabled"):
        _restore_r2r_source(state, registry)
    state["r2r_bound_identity"] = current_bound
    state["r2r_registry_initialized"] = True


def r2r_is_ready(state) -> bool:
    if not state.get("r2r_enabled") or not state.get("r2r_document_ids"):
        return False
    if state.get("r2r_contract") not in {None, R2R_CONTRACT}:
        return False
    source_signature = state.get("r2r_current_source_signature")
    document_signature = state.get("r2r_document_ids_signature")
    if not source_signature or source_signature != document_signature:
        return False
    try:
        client = get_client(state)
        bound = state.get("r2r_bound_identity")
        expected = _bound_identity(state, client.identity)
        if not bound or bound != expected:
            return False
    except (TypeError, ValueError):
        return False
    active_source = state.get("active_source")
    if not isinstance(active_source, dict):
        return False
    return (
        active_source.get("kind") == "r2r"
        and active_source.get("status") == "ready"
        and active_index_matches_settings(state)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deterministic_document_id(
    workspace_id: str,
    source_signature: str,
    file_name: str,
    file_hash: str,
) -> str:
    label = (
        f"docmind-r2r:{workspace_id}:{source_signature}:"
        f"{file_name.casefold()}:{file_hash}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, label))


def _rollback_documents(
    client: R2RClient,
    registry: R2RDocumentRegistry,
    document_ids: list[str],
    reason: str,
) -> None:
    failed = []
    for document_id in dict.fromkeys(document_ids):
        try:
            outcome = client.delete_document(document_id)
        except R2RConnectionError as err:
            failed.append(document_id)
            try:
                registry.update_document(
                    document_id,
                    status="rollback_failed",
                    last_error=f"{reason}:{_safe_error_code(err)}",
                )
            except R2RRegistryError:
                pass
            continue
        try:
            registry.update_document(
                document_id,
                status="rolled_back"
                if outcome in {"deleted", "missing"}
                else "rollback_failed",
                last_error=reason,
            )
        except R2RRegistryError:
            if outcome in {"deleted", "missing"}:
                failed.append(document_id)
    if failed:
        raise R2RRollbackError()


def ingest_paths_transaction(
    client: R2RClient,
    file_paths: list[str],
    registry: R2RDocumentRegistry,
    workspace_id: str,
    content_signature: str,
    source_id: str,
    source_signature: str,
    old_document_ids: list[str] | None = None,
    poll_timeout: float = R2R_POLL_TIMEOUT,
    cancel_event: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    operation_ids: list[str] = []
    records: list[dict] = []
    active_ids_before = {
        str(record["document_id"])
        for record in registry.active_documents(client.identity)
    }
    old_ids = {
        _safe_identifier(str(document_id))
        for document_id in (
            old_document_ids if old_document_ids is not None else active_ids_before
        )
    }
    try:
        for file_path in file_paths:
            if cancel_event is not None and cancel_event.is_set():
                raise R2RIngestionCancelled()
            path = Path(file_path).resolve()
            if not path.is_file():
                raise R2RLifecycleError("R2R upload file is unavailable.")
            file_hash = _file_sha256(path)
            requested_id = _deterministic_document_id(
                str(workspace_id), source_signature, path.name, file_hash
            )
            existing = registry.get_document(requested_id)
            if (
                existing
                and existing.get("status") == "ready"
                and existing.get("content_signature") == content_signature
                and client.identity.matches(existing)
            ):
                record = existing
                if requested_id not in old_ids:
                    operation_ids.append(requested_id)
            else:
                record = {
                    "document_id": requested_id,
                    "task_id": None,
                    "content_signature": content_signature,
                    "source_id": source_id,
                    "source_signature": source_signature,
                    "status": "planned",
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "last_error": None,
                    "endpoint_fingerprint": client.identity.endpoint_fingerprint,
                    "credential_fingerprint": client.identity.credential_fingerprint,
                    "file_name": path.name,
                }
                registry.upsert_document(record)
                operation_ids.append(requested_id)
                receipt = client.upload_document(path, document_id=requested_id)
                operation_ids.append(receipt.document_id)
                if receipt.document_id != requested_id:
                    registry.update_document(
                        requested_id,
                        status="failed",
                        last_error="document_id_mismatch",
                    )
                    registry.upsert_document(
                        {
                            **record,
                            "document_id": receipt.document_id,
                            "status": "uploaded",
                            "task_id": receipt.task_id,
                        }
                    )
                    raise R2RLifecycleError("R2R returned an unexpected document ID.")
                record = registry.update_document(
                    requested_id,
                    status="uploaded",
                    task_id=receipt.task_id,
                    last_error=None,
                )
                wait_result = client.wait_for_document(
                    requested_id,
                    task_id=receipt.task_id,
                    timeout=poll_timeout,
                    cancel_event=cancel_event,
                    clock=clock,
                    sleep=sleep,
                    on_status=lambda status, identifier=requested_id, task_id=receipt.task_id: (
                        registry.update_document(
                            identifier,
                            status=(
                                "ready"
                                if status.status.strip().lower()
                                in {"success", "completed", "ready", "indexed"}
                                else status.status.strip().lower()
                            ),
                            task_id=task_id,
                            last_error=None,
                        )
                    ),
                )
                record = registry.update_document(
                    requested_id,
                    status="ready",
                    task_id=wait_result.task_id or receipt.task_id,
                    last_error=None,
                )
            records.append(record)

        selected_old_ids = set(old_ids)
        selected_old_ids.difference_update(
            str(record["document_id"]) for record in records
        )
        old_failures = []
        for document_id in sorted(selected_old_ids):
            try:
                outcome = client.delete_document(document_id)
            except R2RConnectionError as err:
                old_failures.append(document_id)
                try:
                    registry.update_document(
                        document_id,
                        status="delete_failed",
                        last_error=_safe_error_code(err),
                    )
                except R2RRegistryError:
                    pass
                continue
            registry.update_document(
                document_id,
                status="deleted",
                last_error=None,
            )
            if outcome not in {"deleted", "missing"}:
                old_failures.append(document_id)
        if old_failures:
            raise R2RLifecycleError(
                "R2R replacement cleanup did not delete every old document."
            )

        registry.commit_active(
            records,
            {
                "source_id": source_id,
                "content_signature": content_signature,
                "source_signature": source_signature,
                "settings_signature": stable_digest(
                    {
                        "contract": R2R_CONTRACT,
                        **client.identity.as_dict(),
                        "source_signature": source_signature,
                    }
                ),
                "index_generation": 0,
                "display_name": ", ".join(
                    sorted((Path(path).name for path in file_paths), key=str.casefold)
                ),
            },
        )
        return copy.deepcopy(records)
    except BaseException as err:
        rollback_ids = [
            document_id for document_id in operation_ids if document_id not in old_ids
        ]
        _rollback_documents(
            client,
            registry,
            rollback_ids,
            _safe_error_code(err),
        )
        raise


def _upload_content_signature(uploaded_files: list) -> str:
    entries = []
    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        entries.append(
            {
                "name": uploaded_file.name,
                "size": getattr(uploaded_file, "size", len(data)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return stable_digest(sorted(entries, key=lambda item: str(item["name"]).casefold()))


def _source_signature(client: R2RClient, content_signature: str) -> str:
    return stable_digest(
        {
            "contract": R2R_CONTRACT,
            **client.identity.as_dict(),
            "content_signature": content_signature,
        }
    )


def _owned_work_dir(path: str | Path | None) -> bool:
    if not path:
        return False
    work_root = (Path.cwd() / "data" / "work").resolve()
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(work_root)
    except (OSError, ValueError):
        return False
    return resolved != work_root and resolved.name.startswith("r2r-upload-")


def _render_status(status_container, completed_stages, active_stage=None):
    if status_container is None:
        return
    status_container.empty()
    with status_container.container():
        for stage in completed_stages:
            st.caption(f"✔ {stage}")
        if active_stage is not None:
            st.caption(f"Working: {active_stage}")


def _render_completed(status_container, completed_stages):
    if status_container is None:
        return
    status_container.empty()
    with status_container.container():
        for stage in completed_stages:
            st.caption(f"✔ {stage}")
        st.empty()
        st.empty()


def r2r_ingest_files(
    uploaded_files: list,
    status_container=None,
    status_state_key: str = "r2r_ingestion_stages",
    content_signature: str | None = None,
    source_id: str | None = None,
    source_signature: str | None = None,
    client: R2RClient | None = None,
    registry: R2RDocumentRegistry | None = None,
) -> list[str]:
    completed_stages: list[str] = []
    previous = {
        key: copy.deepcopy(st.session_state.get(key))
        for key in (
            "r2r_document_ids",
            "r2r_document_ids_signature",
            "query_engine",
            "retriever",
            "documents",
            status_state_key,
        )
    }
    work_dir = None
    client = client or get_client(st.session_state)
    registry = registry or registry_for_state(st.session_state, client)
    content_signature = content_signature or _upload_content_signature(uploaded_files)
    source_id = source_id or stable_digest(
        {"kind": "r2r", "identity": content_signature}
    )
    source_signature = source_signature or _source_signature(client, content_signature)
    old_ids = list(st.session_state.get("r2r_document_ids") or [])
    old_ids.extend(
        record["document_id"] for record in registry.active_documents(client.identity)
    )
    old_ids = list(dict.fromkeys(old_ids))

    def record_stages():
        st.session_state[status_state_key] = list(completed_stages)

    try:
        func.validate_uploaded_files(uploaded_files)
        st.session_state["r2r_document_ids"] = []
        st.session_state["r2r_document_ids_signature"] = None
        st.session_state["r2r_current_source_signature"] = source_signature
        if not client.health():
            raise R2RConnectionError("R2R health check failed.")
        completed_stages.append("R2R connection OK")
        record_stages()
        _render_status(status_container, completed_stages)

        work_dir = func.create_ingestion_work_dir("r2r-upload")
        session_dirs = st.session_state.setdefault("_session_work_dirs", [])
        if isinstance(session_dirs, list):
            session_dirs.append(work_dir)
        uploaded_paths = []
        for uploaded_file in uploaded_files:
            func.save_uploaded_file(uploaded_file, work_dir)
            uploaded_paths.append(str(Path(work_dir) / uploaded_file.name))
        completed_stages.append("Files prepared")
        record_stages()
        _render_status(status_container, completed_stages, "Uploading and indexing")

        records = ingest_paths_transaction(
            client,
            uploaded_paths,
            registry,
            str(st.session_state.get("r2r_workspace_id") or uuid.uuid4()),
            content_signature,
            source_id,
            source_signature,
            old_document_ids=old_ids,
            cancel_event=st.session_state.get("r2r_cancel_event"),
        )
        document_ids = [record["document_id"] for record in records]
        st.session_state["r2r_document_ids"] = document_ids
        st.session_state["r2r_document_ids_signature"] = source_signature
        st.session_state["query_engine"] = None
        st.session_state["retriever"] = None
        st.session_state["documents"] = None
        completed_stages.append("Documents indexed by R2R")
        record_stages()
        _render_completed(status_container, completed_stages)
        return document_ids
    except (R2RConnectionError, OSError, ValueError) as err:
        for key, value in previous.items():
            if value is None:
                st.session_state.pop(key, None)
            else:
                st.session_state[key] = value
        st.session_state["r2r_current_source_signature"] = source_signature
        source = ensure_active_source(st.session_state)
        if source.get("kind") == "r2r" or old_ids:
            mark_source_failed(st.session_state, _safe_error_code(err))
        logs.log.error(
            "R2R ingestion failed: category=%s status=%s",
            _safe_error_code(err),
            getattr(err, "status_code", None),
        )
        st.error(safe_user_error(err))
        st.stop()
    finally:
        if work_dir and _owned_work_dir(work_dir):
            if func.cleanup_ingestion_work_dir(work_dir):
                session_dirs = st.session_state.get("_session_work_dirs", [])
                if isinstance(session_dirs, list):
                    st.session_state["_session_work_dirs"] = [
                        item for item in session_dirs if item != work_dir
                    ]
            else:
                logs.log.warning("Unable to delete the owned R2R work directory.")


def delete_owned_remote_documents(
    state,
    client: R2RClient | None = None,
    registry: R2RDocumentRegistry | None = None,
) -> R2RRemoteResetResult:
    has_owned_state = bool(state.get("r2r_document_ids"))
    if registry is None and not state.get("r2r_registry_path") and not has_owned_state:
        return R2RRemoteResetResult(R2RDeleteResult(), attempted=False)
    client = client or get_client(state)
    registry = registry or registry_for_state(state, client)
    documents = registry.documents_for_identity(client.identity)
    deleted = []
    missing = []
    failed = []
    for document in documents:
        document_id = str(document["document_id"])
        try:
            outcome = client.delete_document(document_id)
        except R2RConnectionError as err:
            failed.append((document_id, _safe_error_code(err)))
            try:
                registry.update_document(
                    document_id,
                    status="delete_failed",
                    last_error=_safe_error_code(err),
                )
            except R2RRegistryError:
                pass
            continue
        if outcome == "deleted":
            deleted.append(document_id)
        else:
            missing.append(document_id)
        registry.update_document(document_id, status="deleted", last_error=None)
    other_identity_count = len(
        [
            document
            for document in registry.snapshot()["documents"]
            if document.get("status") not in {"deleted", "rolled_back"}
            and not client.identity.matches(document)
        ]
    )
    delete_result = R2RDeleteResult(tuple(deleted), tuple(missing), tuple(failed))
    result = R2RRemoteResetResult(delete_result, other_identity_count)
    if result.success:
        registry.clear_active()
    return result


def r2r_chat(prompt: str):
    try:
        client = get_client(st.session_state)
        document_ids = list(st.session_state.get("r2r_document_ids") or [])
        if not r2r_is_ready(st.session_state):
            raise R2RLifecycleError("R2R documents are not ready.")
        result = client.rag(
            prompt,
            document_ids=document_ids,
            top_k=int(st.session_state.get("top_k") or 3),
            answer_style=st.session_state.get("answer_style"),
        )
        st.session_state["last_doc_sources"] = _chat_sources(result)
        st.session_state["last_r2r_metadata"] = {
            "contract": R2R_CONTRACT,
            "streaming": False,
            "history_sent": False,
            "citations": result.citations,
            "search_results": result.search_results,
        }
        yield result.answer
    except (R2RConnectionError, ValueError) as err:
        metadata = {
            "contract": R2R_CONTRACT,
            "streaming": False,
            "history_sent": False,
            "error_category": _safe_error_code(err),
        }
        st.session_state["last_r2r_metadata"] = metadata
        st.session_state["last_doc_sources"] = []
        logs.log.error(
            "R2R chat failed: category=%s status=%s",
            _safe_error_code(err),
            getattr(err, "status_code", None),
        )
        yield f"**R2R request failed.** {safe_user_error(err)}"
