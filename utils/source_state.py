import copy
import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from utils.provider_config import normalize_provider_kind, provider_slug

SOURCE_KINDS = frozenset({"local", "github", "website", "r2r", None})
SOURCE_STATUSES = frozenset({"idle", "pending", "stale", "ready", "failed"})
PARSER_CACHE_VERSION = "format-ingestion-v1"
INDEX_SETTINGS_VERSION = 1
_UNSET = object()


def initialize_active_source(state) -> dict[str, Any]:
    return ensure_active_source(state)


def reset_source_state(state) -> dict[str, Any]:
    return reset_active_source(state)


def mark_pending(state, source=None, **updates) -> dict[str, Any]:
    return mark_source_pending(state, source, **updates)


def mark_stale(state, error: str | None = None) -> dict[str, Any]:
    return mark_source_stale(state, error)


def mark_ready(state, source=None, **updates) -> dict[str, Any]:
    return mark_source_ready(state, source, **updates)


def mark_failed(state, error: Any, source=None) -> dict[str, Any]:
    return mark_source_failed(state, error, source)


def mark_active_source_pending(state, source=None, **updates) -> dict[str, Any]:
    return mark_source_pending(state, source, **updates)


def mark_active_source_stale(state, error: str | None = None) -> dict[str, Any]:
    return mark_source_stale(state, error)


def mark_active_source_ready(state, source=None, **updates) -> dict[str, Any]:
    return mark_source_ready(state, source, **updates)


def mark_active_source_failed(state, error: Any, source=None) -> dict[str, Any]:
    return mark_source_failed(state, error, source)


def normalize_effective_chunk_settings(state=None, **kwargs) -> dict[str, int]:
    return normalize_chunk_settings(state, **kwargs)


def current_indexing_settings(state, *, strict: bool = True) -> dict[str, Any]:
    return effective_indexing_settings(state, strict=strict)


def is_active_index_current(state, current_settings=None) -> bool:
    return active_index_matches_settings(state, current_settings)


def index_matches_settings(state, current_settings=None) -> bool:
    return active_index_matches_settings(state, current_settings)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_digest(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def credential_fingerprint(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def initial_source_state() -> dict[str, Any]:
    return {
        "id": None,
        "source_id": None,
        "kind": None,
        "display_name": None,
        "source_uri": None,
        "content_signature": None,
        "settings_signature": None,
        "index_generation": 0,
        "status": "idle",
        "cache_key": None,
        "created_at": None,
        "error": None,
    }


def source_state_copy(source: Mapping[str, Any] | None) -> dict[str, Any]:
    result = initial_source_state()
    if not isinstance(source, Mapping):
        return result
    for key in result:
        if key in source:
            result[key] = copy.deepcopy(source[key])
    kind = result.get("kind")
    if kind not in SOURCE_KINDS:
        result["kind"] = None
    status = result.get("status")
    if status not in SOURCE_STATUSES:
        result["status"] = "idle"
    try:
        generation = int(result.get("index_generation") or 0)
    except (TypeError, ValueError):
        generation = 0
    result["index_generation"] = max(0, generation)
    result["id"] = result["id"] or result.get("source_id")
    if result["id"] is None and result.get("kind") is not None:
        result["id"] = str(uuid.uuid4())
    result["source_id"] = result["id"]
    return result


def ensure_active_source(state) -> dict[str, Any]:
    current = state.get("active_source")
    normalized = source_state_copy(current)
    if not isinstance(current, Mapping) or normalized != source_state_copy(current):
        state["active_source"] = normalized
    else:
        state["active_source"] = normalized
    return copy.deepcopy(normalized)


def reset_active_source(state) -> dict[str, Any]:
    state["active_source"] = initial_source_state()
    state["index_generation"] = 0
    return copy.deepcopy(state["active_source"])


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{label} must be a finite number.") from err
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if number < minimum or number > maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
    return number


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    number = _number(value, label, minimum, maximum)
    if not number.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    return int(number)


def normalize_chunk_settings(
    state=None,
    chunk_size=_UNSET,
    chunk_overlap=_UNSET,
    chunk_overlap_pct=_UNSET,
    *,
    strict: bool = True,
) -> dict[str, int]:
    if isinstance(state, Mapping):
        raw_size = state.get("chunk_size", 256)
        raw_overlap = state.get("chunk_overlap")
        raw_pct = state.get("chunk_overlap_pct")
    elif state is not None and chunk_size is _UNSET:
        raw_size = state
        raw_overlap = None if chunk_overlap is _UNSET else chunk_overlap
        raw_pct = None if chunk_overlap_pct is _UNSET else chunk_overlap_pct
        state = None
    else:
        raw_size = 256
        raw_overlap = None
        raw_pct = None
    if chunk_size is not _UNSET:
        raw_size = chunk_size
    if chunk_overlap is not _UNSET:
        raw_overlap = chunk_overlap
    if chunk_overlap_pct is not _UNSET:
        raw_pct = chunk_overlap_pct

    try:
        size = _integer(raw_size if raw_size not in (None, "") else 256, "Chunk Size", 1, 1_000_000)
        if raw_pct not in (None, ""):
            pct = _number(raw_pct, "Chunk Overlap Percentage", 0, 50)
            overlap = int(size * pct // 100)
        elif raw_overlap not in (None, ""):
            overlap = _integer(raw_overlap, "Chunk Overlap", 0, size - 1)
            pct = overlap / size * 100
        else:
            pct = 12.0
            overlap = int(size * pct // 100)
        if overlap < 0 or overlap >= size:
            raise ValueError("Chunk Overlap must be less than Chunk Size.")
        pct = _number(pct, "Chunk Overlap Percentage", 0, 50)
    except (TypeError, ValueError):
        if strict:
            raise
        size = 256
        pct = 12.0
        overlap = 30

    normalized = {
        "chunk_size": size,
        "chunk_overlap": overlap,
        "chunk_overlap_pct": int(round(pct)),
    }
    if isinstance(state, Mapping):
        for key, value in normalized.items():
            if state.get(key) != value:
                state[key] = value
    return normalized


def effective_indexing_settings(state, *, strict: bool = True) -> dict[str, Any]:
    chunks = normalize_chunk_settings(state, strict=strict)
    backend = str(state.get("llm_backend", "Ollama") or "Ollama")
    if bool(state.get("r2r_enabled")):
        provider = "r2r"
        embedding_backend = "r2r"
        model = None
        endpoint = str(state.get("r2r_base_url", "http://localhost:7272") or "").rstrip("/")
        api_key = None
    else:
        embedding_backend = str(
            state.get("embedding_backend") or backend or "Ollama"
        )
        provider = normalize_provider_kind(embedding_backend)
        if provider == "ollama":
            model = state.get("embedding_model") or state.get("ollama_embedding_model")
            endpoint = str(
                state.get("embedding_base_url")
                or state.get("ollama_endpoint", "http://localhost:11434")
                or ""
            ).rstrip("/")
            api_key = None
        else:
            slug = provider_slug(embedding_backend)
            profile_model = {
                "openai": state.get("openai_embedding_model"),
                "lm_studio": state.get("lm_studio_embedding_model"),
                "tabbyapi": state.get("tabby_embedding_model"),
                "openai_compatible": state.get("openai_compatible_embedding_model"),
            }.get(slug)
            model = state.get("embedding_model") or profile_model or state.get(
                "openai_embedding_model"
            )
            profile_endpoint = {
                "openai": state.get("openai_base_url"),
                "lm_studio": state.get("lm_studio_base_url"),
                "tabbyapi": state.get("tabby_base_url"),
                "openai_compatible": state.get("openai_compatible_base_url"),
            }.get(slug)
            endpoint = str(
                state.get("embedding_base_url")
                or profile_endpoint
                or (
                    "https://api.openai.com/v1"
                    if provider == "openai_official"
                    else "http://localhost:1234/v1"
                )
            ).rstrip("/")
            api_key = state.get("embedding_api_key")
            if (
                not state.get("embedding_profile_migrated")
                and "embedding_api_key" not in state
                and provider == "openai_compatible"
            ):
                api_key = ""
        settings = {
            "version": INDEX_SETTINGS_VERSION,
            "parser_cache_version": PARSER_CACHE_VERSION,
            "chunk_size": chunks["chunk_size"],
            "chunk_overlap": chunks["chunk_overlap"],
            "chunk_overlap_pct": chunks["chunk_overlap_pct"],
            "embedding_backend": embedding_backend,
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_endpoint": endpoint,
            "embedding_base_url": endpoint,
            "embedding_credential_fingerprint": credential_fingerprint(api_key),
            "embedding_api_key_fingerprint": credential_fingerprint(api_key),
        }
        if bool(state.get("r2r_enabled")):
            settings["r2r_base_url"] = str(
                state.get("r2r_base_url", "http://localhost:7272") or ""
            ).rstrip("/")
            settings["r2r_credential_fingerprint"] = credential_fingerprint(
                state.get("r2r_api_key")
            )
        return settings
    settings = {
        "version": INDEX_SETTINGS_VERSION,
        "parser_cache_version": PARSER_CACHE_VERSION,
        "chunk_size": chunks["chunk_size"],
        "chunk_overlap": chunks["chunk_overlap"],
        "chunk_overlap_pct": chunks["chunk_overlap_pct"],
        "embedding_backend": embedding_backend,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_endpoint": endpoint,
        "embedding_base_url": endpoint,
        "embedding_credential_fingerprint": None,
        "embedding_api_key_fingerprint": None,
    }
    settings["r2r_base_url"] = str(
        state.get("r2r_base_url", "http://localhost:7272") or ""
    ).rstrip("/")
    settings["r2r_credential_fingerprint"] = credential_fingerprint(
        state.get("r2r_api_key")
    )
    return settings


def indexing_settings_signature(state, *, strict: bool = True) -> str:
    return stable_digest(effective_indexing_settings(state, strict=strict))


indexing_settings_signature.__doc__ = "Return a stable signature of active indexing settings."


def settings_signature(state, *, strict: bool = True) -> str:
    return indexing_settings_signature(state, strict=strict)


def source_identity(kind: str, identity: Any) -> str:
    if kind not in SOURCE_KINDS or kind is None:
        raise ValueError("A concrete source kind is required for source identity.")
    return stable_digest({"kind": kind, "identity": identity})


def make_source_state(
    kind: str | None,
    *,
    source_id: str | None = None,
    display_name: str | None = None,
    source_uri: str | None = None,
    content_signature: str | None = None,
    settings_signature: str | None = None,
    index_generation: int = 0,
    status: str = "pending",
    cache_key: str | None = None,
    created_at: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if kind not in SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind: {kind!r}.")
    if status not in SOURCE_STATUSES:
        raise ValueError(f"Unsupported source status: {status!r}.")
    source_id = str(source_id) if source_id else str(uuid.uuid4())
    return {
        "id": source_id,
        "source_id": source_id,
        "kind": kind,
        "display_name": display_name,
        "source_uri": source_uri,
        "content_signature": content_signature,
        "settings_signature": settings_signature,
        "index_generation": max(0, int(index_generation)),
        "status": status,
        "cache_key": cache_key,
        "created_at": created_at or utc_now(),
        "error": error,
    }


def _record_from_source(state, source=None, **updates) -> dict[str, Any]:
    current = ensure_active_source(state)
    if isinstance(source, Mapping):
        record = source_state_copy(source)
    elif source is not None:
        record = make_source_state(source, status="pending")
    else:
        record = current
    if "source_id" in updates and "id" not in updates:
        updates["id"] = updates["source_id"]
    if "id" in updates and updates["id"]:
        record["id"] = str(updates.pop("id"))
        record["source_id"] = record["id"]
    if "source_id" in updates and updates["source_id"]:
        record["source_id"] = str(updates.pop("source_id"))
        record["id"] = record["source_id"]
    for key, value in updates.items():
        if key in record or key in {"source_id"}:
            record[key] = value
    if record.get("kind") not in SOURCE_KINDS:
        record["kind"] = None
    if record.get("status") not in SOURCE_STATUSES:
        record["status"] = "pending"
    if not record.get("id"):
        record["id"] = str(uuid.uuid4())
        record["source_id"] = record["id"]
    if not record.get("created_at"):
        record["created_at"] = utc_now()
    record["source_id"] = record["id"]
    try:
        record["index_generation"] = max(0, int(record.get("index_generation") or 0))
    except (TypeError, ValueError):
        record["index_generation"] = current["index_generation"] + 1
    state["active_source"] = copy.deepcopy(record)
    return copy.deepcopy(record)


def mark_source_pending(state, source=None, **updates) -> dict[str, Any]:
    current = ensure_active_source(state)
    source_has_generation = isinstance(source, Mapping) and "index_generation" in source
    if "index_generation" not in updates and not source_has_generation:
        updates["index_generation"] = current["index_generation"] + 1
    updates.setdefault("status", "pending")
    updates["error"] = None
    return _record_from_source(state, source, **updates)


def mark_source_stale(state, error: str | None = None) -> dict[str, Any]:
    return _record_from_source(state, status="stale", error=error)


def mark_source_ready(state, source=None, **updates) -> dict[str, Any]:
    updates["status"] = "ready"
    updates["error"] = None
    return _record_from_source(state, source, **updates)


def mark_source_failed(state, error: Any, source=None) -> dict[str, Any]:
    updates = {"status": "failed", "error": str(error)}
    return _record_from_source(state, source, **updates)


def restore_active_source(state, source: Mapping[str, Any] | None) -> dict[str, Any]:
    state["active_source"] = source_state_copy(source)
    state["index_generation"] = state["active_source"]["index_generation"]
    return copy.deepcopy(state["active_source"])


def active_source(state) -> dict[str, Any]:
    return ensure_active_source(state)


def active_index_matches_settings(state, current_settings: Mapping[str, Any] | None = None) -> bool:
    source = ensure_active_source(state)
    if source["kind"] is None or source["status"] != "ready":
        return False
    if not source.get("settings_signature"):
        return False
    if current_settings is None:
        try:
            current_settings = effective_indexing_settings(state)
        except ValueError:
            return False
    current_signature = (
        str(current_settings)
        if isinstance(current_settings, str)
        else stable_digest(dict(current_settings))
    )
    if current_signature != source["settings_signature"]:
        return False
    return source.get("index_generation", 0) == state.get(
        "index_generation", source.get("index_generation", 0)
    )


def active_index_is_current(state, current_settings: Mapping[str, Any] | None = None) -> bool:
    return active_index_matches_settings(state, current_settings)


def mark_active_index_stale_if_needed(state) -> bool:
    source = ensure_active_source(state)
    if source["kind"] is None or source["status"] not in {"ready", "stale"}:
        return False
    if active_index_matches_settings(state):
        return source["status"] == "ready"
    mark_source_stale(state)
    return False


def source_matches(
    source: Mapping[str, Any] | None,
    *,
    kind: str | None = None,
    source_id: str | None = None,
    content_signature: str | None = None,
    index_generation: int | None = None,
) -> bool:
    if not isinstance(source, Mapping):
        return False
    if kind is not None and source.get("kind") != kind:
        return False
    if source_id is not None and source.get("id") != source_id:
        return False
    if content_signature is not None and source.get("content_signature") != content_signature:
        return False
    if index_generation is not None and source.get("index_generation") != index_generation:
        return False
    return True


def tag_report(
    report: list[Mapping[str, Any]],
    source_id: str | None,
    index_generation: int | None,
) -> list[dict[str, Any]]:
    tagged = []
    for entry in report:
        item = dict(entry)
        if source_id is not None:
            item["source_id"] = source_id
        if index_generation is not None:
            item["index_generation"] = index_generation
        tagged.append(item)
    return tagged


def report_matches_active_source(
    report: list[Mapping[str, Any]],
    state,
    source_id: str | None = None,
    index_generation: int | None = None,
) -> bool:
    if not report:
        return False
    tagged = [entry for entry in report if "source_id" in entry or "index_generation" in entry]
    active = ensure_active_source(state)
    if not tagged:
        return "active_source" not in state and active.get("kind") is None
    if len(tagged) != len(report):
        return False
    expected_id = source_id if source_id is not None else active.get("id")
    expected_generation = (
        index_generation
        if index_generation is not None
        else active.get("index_generation")
    )
    return all(
        (expected_id is None or entry.get("source_id", expected_id) == expected_id)
        and (
            expected_generation is None
            or entry.get("index_generation", expected_generation) == expected_generation
        )
        for entry in tagged
    )


__all__ = [
    "INDEX_SETTINGS_VERSION",
    "PARSER_CACHE_VERSION",
    "SOURCE_KINDS",
    "SOURCE_STATUSES",
    "active_index_is_current",
    "active_index_matches_settings",
    "active_source",
    "credential_fingerprint",
    "effective_indexing_settings",
    "ensure_active_source",
    "indexing_settings_signature",
    "initial_source_state",
    "initialize_active_source",
    "is_active_index_current",
    "index_matches_settings",
    "current_indexing_settings",
    "make_source_state",
    "mark_failed",
    "mark_pending",
    "mark_ready",
    "mark_stale",
    "normalize_effective_chunk_settings",
    "mark_active_index_stale_if_needed",
    "mark_source_failed",
    "mark_source_pending",
    "mark_source_ready",
    "mark_source_stale",
    "mark_active_source_failed",
    "mark_active_source_pending",
    "mark_active_source_ready",
    "mark_active_source_stale",
    "normalize_chunk_settings",
    "report_matches_active_source",
    "reset_active_source",
    "reset_source_state",
    "restore_active_source",
    "settings_signature",
    "source_identity",
    "source_matches",
    "source_state_copy",
    "stable_digest",
    "tag_report",
    "utc_now",
]
