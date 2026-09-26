from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

os.environ["DOCMIND_LOG_FILE"] = str(
    Path(tempfile.gettempdir()) / f"docmind-e2e-import-{os.getpid()}.log"
)

from components.page_state import WELCOME_MESSAGE, perform_project_reset
from utils import ollama as ollama_module
from utils import r2r
from utils.source_state import make_source_state

ROOT = Path(__file__).resolve().parents[1]
MAX_FAKE_REQUEST_BYTES = 2 * 1024 * 1024
MAX_FAKE_RESPONSE_BYTES = 256 * 1024
SERVER_TIMEOUT_SECONDS = 5
TEST_CREDENTIAL = "integration-credential"
STREAMED_RESPONSE = "Fake streamed response from local provider."

NAVIGATION_TIMEOUT_MS = 30_000
ACTION_TIMEOUT_MS = 15_000
FIRST_TOKEN_TIMEOUT_MS = 120_000
RESTORE_TIMEOUT_MS = 60_000


class _BoundedHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = SERVER_TIMEOUT_SECONDS

    def setup(self):
        super().setup()
        self.connection.settimeout(self.timeout)

    def log_message(self, _format_string, *_args):
        return

    def _body(self):
        value = self.headers.get("Content-Length", "0")
        try:
            length = int(value)
        except (TypeError, ValueError):
            self._send(400, b"invalid request length", "text/plain")
            return None
        if length < 0 or length > self.server.max_body_bytes:
            self._send(413, b"request body limit exceeded", "text/plain")
            return None
        return self.rfile.read(length)

    def _send(self, status, body=b"", content_type="application/json", headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        body = bytes(body)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode("utf-8"))

    def _authorized(self):
        accepted = getattr(self.server.state, "accepted_keys", set())
        return not accepted or self.headers.get("x-api-key") in accepted

    def _record(self, method, path, body=None):
        self.server.state.requests.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "content_type": self.headers.get("Content-Type", ""),
                "authorization": self.headers.get("Authorization", ""),
                "api_key": self.headers.get("x-api-key", ""),
            }
        )


class _LoopbackServer:
    def __init__(self, handler, state):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = False
        self.server.block_on_close = True
        self.server.max_body_bytes = MAX_FAKE_REQUEST_BYTES
        self.server.state = state
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _OpenAIState:
    def __init__(self):
        self.lock = threading.RLock()
        self.mode = "normal"
        self.requests = []
        self.accepted_keys = set()
        self.models = ["fake-model", "fake-embedding"]
        self.error_secret = "api_key=provider-secret"


class _OpenAIHandler(_BoundedHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        self._record("GET", path)
        if not self._authorized():
            self._json(401, {"error": {"message": self.server.state.error_secret}})
            return
        if path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model", "type": "model"}
                        for model in self.server.state.models
                    ],
                },
            )
            return
        if path == "/api/tags":
            self._json(
                200,
                {
                    "models": [
                        {"name": "fake-model", "model": "fake-model"},
                        {"name": "fake-embedding", "model": "fake-embedding"},
                    ]
                },
            )
            return
        if path == "/v1/embeddings":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                    "model": "fake-embedding",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        path = urlsplit(self.path).path
        body = self._body()
        if body is None:
            return
        self._record("POST", path, body.decode("utf-8", errors="replace"))
        if not self._authorized():
            self._json(401, {"error": {"message": self.server.state.error_secret}})
            return
        if path == "/api/show":
            try:
                payload = json.loads(body)
            except (TypeError, ValueError):
                self._json(400, {"error": "invalid json"})
                return
            model = str(payload.get("name") or payload.get("model") or "")
            capability = "embedding" if "embedding" in model else "completion"
            self._json(
                200,
                {
                    "capabilities": [capability],
                    "model_info": {model: {"general.architecture": "test"}},
                },
            )
            return
        if path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            self._json(400, {"error": {"message": "invalid request"}})
            return
        model = str(payload.get("model") or "")
        mode = self.server.state.mode
        if mode == "http_failure":
            self._json(503, {"error": {"message": self.server.state.error_secret}})
            return
        if mode == "malformed":
            self._send(502, b"api_key=provider-secret not-json", "application/json")
            return
        if mode == "oversized":
            self._send(
                502,
                b"x" * (MAX_FAKE_RESPONSE_BYTES + 1),
                "text/plain",
            )
            return
        if model not in self.server.state.models or model == "fake-embedding":
            self._json(404, {"error": {"message": self.server.state.error_secret}})
            return
        if payload.get("stream"):
            chunks = [
                {"role": "assistant", "content": "Fake "},
                {"content": "streamed response "},
                {"content": "from local provider."},
            ]
            data = bytearray()
            for index, delta in enumerate(chunks):
                item = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": (
                                "stop" if index == len(chunks) - 1 else None
                            ),
                        }
                    ],
                }
                data.extend(f"data: {json.dumps(item)}\n\n".encode())
            data.extend(b"data: [DONE]\n\n")
            self._send(200, bytes(data), "text/event-stream")
            return
        self._json(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": STREAMED_RESPONSE},
                        "finish_reason": "stop",
                    }
                ],
            },
        )


class _R2RState:
    def __init__(self):
        self.lock = threading.RLock()
        self.requests = []
        self.documents = {}
        self.poll_counts = {}
        self.uploads = []
        self.accepted_keys = {TEST_CREDENTIAL, "second-credential"}
        self.mode = "normal"
        self.delay_seconds = 0.0
        self.fail_next_indexing = False
        self.fail_delete_ids = set()
        self.stay_pending = False
        self.error_secret = "api_key=r2r-secret"
        self.sequence = 0

    def next_id(self):
        self.sequence += 1
        return f"integration-doc-{self.sequence}"


class _R2RHandler(_BoundedHandler):
    def _document(self, document_id):
        return self.server.state.documents.get(document_id)

    def _auth_failure(self):
        self._json(401, {"detail": self.server.state.error_secret})

    def do_GET(self):
        path = urlsplit(self.path).path
        self._record("GET", path)
        if not self._authorized():
            self._auth_failure()
            return
        if path == "/v3/health":
            if self.server.state.delay_seconds:
                time.sleep(min(self.server.state.delay_seconds, SERVER_TIMEOUT_SECONDS))
            self._json(200, {"results": {"status": "ok"}})
            return
        if path == "/v3/documents":
            with self.server.state.lock:
                documents = [
                    {
                        "id": document_id,
                        "status": record["status"],
                        "task_id": record["task_id"],
                        "metadata": {"filename": record["file_name"]},
                    }
                    for document_id, record in self.server.state.documents.items()
                ]
            self._json(200, {"results": documents})
            return
        match = re.fullmatch(r"/v3/documents/([^/]+)", path)
        if not match:
            self._json(404, {"detail": "not found"})
            return
        document_id = unquote(match.group(1))
        with self.server.state.lock:
            record = self._document(document_id)
            if record is None:
                self._json(404, {"detail": "not found"})
                return
            if self.server.state.mode == "document_5xx":
                self._json(503, {"detail": self.server.state.error_secret})
                return
            if self.server.state.delay_seconds:
                time.sleep(min(self.server.state.delay_seconds, SERVER_TIMEOUT_SECONDS))
            count = self.server.state.poll_counts.get(document_id, 0) + 1
            self.server.state.poll_counts[document_id] = count
            if record.get("fail_indexing"):
                record["status"] = "failed"
            elif self.server.state.stay_pending or count < 2:
                record["status"] = "pending"
            else:
                record["status"] = "success"
            self._json(
                200,
                {
                    "results": {
                        "id": document_id,
                        "task_id": record["task_id"],
                        "ingestion_status": record["status"],
                    }
                },
            )

    def do_POST(self):
        path = urlsplit(self.path).path
        body = self._body()
        if body is None:
            return
        self._record("POST", path, body.decode("utf-8", errors="replace"))
        if not self._authorized():
            self._auth_failure()
            return
        if path == "/v3/documents":
            content_type = self.headers.get("Content-Type", "")
            fields, files = self._multipart(body, content_type)
            if self.server.state.mode == "upload_5xx":
                self._json(503, {"detail": self.server.state.error_secret})
                return
            with self.server.state.lock:
                document_id = str(fields.get("id") or self.server.state.next_id())
                if document_id in self.server.state.documents:
                    self._json(409, {"detail": "document already exists"})
                    return
                task_id = f"task-{document_id}"
                self.server.state.documents[document_id] = {
                    "status": "pending",
                    "task_id": task_id,
                    "file_name": next(iter(files), "upload.bin"),
                    "content": files.get(next(iter(files), ""), b""),
                    "fail_indexing": bool(self.server.state.fail_next_indexing),
                }
                if self.server.state.fail_next_indexing:
                    self.server.state.fail_next_indexing = False
                self.server.state.uploads.append(
                    {"document_id": document_id, "fields": fields, "files": files}
                )
            self._json(
                202,
                {
                    "results": {
                        "document_id": document_id,
                        "task_id": task_id,
                    }
                },
            )
            return
        if path == "/v3/retrieval/rag":
            if self.server.state.mode == "rag_5xx":
                self._json(503, {"detail": self.server.state.error_secret})
                return
            try:
                payload = json.loads(body)
            except (TypeError, ValueError):
                self._json(400, {"detail": "invalid json"})
                return
            query = str(payload.get("query") or "")
            if "no-match" in query:
                self._json(
                    200,
                    {
                        "results": {
                            "generated_answer": "No matching passage was found.",
                            "citations": [],
                            "search_results": {},
                        }
                    },
                )
                return
            document_ids = (
                payload.get("search_settings", {})
                .get("filters", {})
                .get("document_id", {})
                .get("$in", [])
            )
            document_id = str(document_ids[0]) if document_ids else "document"
            self._json(
                200,
                {
                    "results": {
                        "generated_answer": "R2R answer with a citation.",
                        "citations": [
                            {
                                "id": "citation-1",
                                "document_id": document_id,
                                "metadata": {"title": "notes.txt"},
                            }
                        ],
                        "search_results": {
                            "chunk_search_results": [
                                {
                                    "document_id": document_id,
                                    "score": 0.91,
                                    "metadata": {"title": "notes.txt"},
                                }
                            ]
                        },
                    }
                },
            )
            return
        self._json(404, {"detail": "not found"})

    def do_DELETE(self):
        path = urlsplit(self.path).path
        self._record("DELETE", path)
        if not self._authorized():
            self._auth_failure()
            return
        match = re.fullmatch(r"/v3/documents/([^/]+)", path)
        if not match:
            self._json(404, {"detail": "not found"})
            return
        document_id = unquote(match.group(1))
        with self.server.state.lock:
            if (
                document_id in self.server.state.fail_delete_ids
                or self.server.state.mode == "delete_5xx"
            ):
                self._json(503, {"detail": self.server.state.error_secret})
                return
            if document_id not in self.server.state.documents:
                self._json(404, {"detail": "not found"})
                return
            del self.server.state.documents[document_id]
        self._json(200, {"results": {"id": document_id, "status": "deleted"}})

    def _multipart(self, body, content_type):
        if "multipart/form-data" not in content_type.lower():
            return {}, {}
        envelope = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
        )
        message = BytesParser(policy=policy.default).parsebytes(envelope)
        fields = {}
        files = {}
        if not message.is_multipart():
            return fields, files
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            value = part.get_payload(decode=True) or b""
            if part.get_filename():
                files[str(name)] = value
            else:
                fields[str(name)] = value.decode("utf-8", errors="replace")
        return fields, files


class StreamlitAppProcess:
    def __init__(self, work_dir, port, extra_environment=None):
        self.work_dir = Path(work_dir)
        self.reservation = port if isinstance(port, _PortReservation) else None
        self.port = int(getattr(port, "port", port))
        self.extra_environment = dict(extra_environment or {})
        self.process = None
        self.log_path = self.work_dir / "streamlit.log"
        self.log_handle = None
        self.log_text = ""

    @property
    def app_url(self):
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        home = self.work_dir / "home"
        home.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(ROOT)
                + (
                    os.pathsep + environment["PYTHONPATH"]
                    if environment.get("PYTHONPATH")
                    else ""
                ),
                "HOME": str(home),
                "USERPROFILE": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "STREAMLIT_CONFIG_DIR": str(home / ".streamlit"),
                "DOCMIND_LOG_FILE": str(self.work_dir / "docmind.log"),
                "DOCMIND_R2R_STATE_DIR": str(self.work_dir / "r2r-state"),
                "DOCMIND_INGESTION_LOCK_PATH": str(self.work_dir / "ingestion.lock"),
                "STREAMLIT_SERVER_HEADLESS": "true",
                "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
                "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
            }
        )
        environment.update(self.extra_environment)
        self.log_handle = self.log_path.open("w+b")
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "main.py"),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={self.port}",
            "--server.fileWatcherType=none",
        ]
        process_options = {
            "cwd": str(self.work_dir),
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": self.log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        if self.reservation is not None:
            self.reservation.release()
            self.reservation = None
        self.process = subprocess.Popen(command, **process_options)
        try:
            self._wait_for_health()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_for_health(self):
        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 35
        url = f"http://127.0.0.1:{self.port}/_stcore/health"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(self._log_tail())
            try:
                with opener.open(
                    Request(url, headers={"Cache-Control": "no-cache"}), timeout=1
                ) as response:
                    if response.status == 200 and response.read(64).strip():
                        self._confirm_own_instance()
                        return
            except (OSError, URLError, TimeoutError):
                time.sleep(0.1)
        raise AssertionError(self._log_tail())

    def _confirm_own_instance(self):
        time.sleep(0.5)
        if self.process.poll() is not None:
            raise AssertionError(
                f"Streamlit exited after reporting healthy on port {self.port}; "
                f"another process likely owns that port.\n{self._log_tail()}"
            )

    def _log_tail(self):
        if self.log_handle:
            self.log_handle.flush()
        try:
            self.log_text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            self.log_text = ""
        return self.log_text[-4000:]

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.reservation is not None:
            self.reservation.release()
            self.reservation = None
        process = self.process
        if process is not None:
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    with contextlib.suppress(OSError):
                        process.terminate()
            else:
                with contextlib.suppress(OSError, ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    with contextlib.suppress(OSError, ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                with contextlib.suppress(OSError):
                    process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        self._log_tail()
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        return False


class _FakeStreamlit:
    def __init__(self, state):
        self.session_state = state
        self.errors = []
        self.captions = []

    def chat_message(self, *_args, **_kwargs):
        return contextlib.nullcontext()

    def spinner(self, *_args, **_kwargs):
        return contextlib.nullcontext()

    def caption(self, value, *_args, **_kwargs):
        self.captions.append(str(value))

    def markdown(self, *_args, **_kwargs):
        return None

    def write_stream(self, stream):
        return "".join(str(item) for item in stream)

    def button(self, *_args, **_kwargs):
        return False

    def error(self, value, *_args, **_kwargs):
        self.errors.append(str(value))


class _NoMatchRetriever:
    def retrieve(self, _query):
        return []


class _PortReservation:
    def __init__(self, port, sock):
        self.port = int(port)
        self.sock = sock

    def release(self):
        sock, self.sock = self.sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()


def _reserve_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        with contextlib.suppress(OSError, AttributeError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(("127.0.0.1", 0))
        return _PortReservation(sock.getsockname()[1], sock)
    except BaseException:
        sock.close()
        raise


def _storage_script(settings):
    value = json.dumps(settings, separators=(",", ":"))
    return (
        "window.localStorage.clear();"
        f"window.localStorage.setItem('docmind:settings', {json.dumps(value)});"
    )


def _browser_executable():
    try:
        from playwright.sync_api import sync_playwright
    except Exception as error:
        return (
            None,
            f"Playwright Python package is unavailable: {type(error).__name__}.",
        )
    configured = os.getenv("DOCMIND_E2E_BROWSER", "").strip()
    if configured and Path(configured).is_file():
        return configured, None
    try:
        with sync_playwright() as playwright:
            bundled = Path(playwright.chromium.executable_path)
            if bundled.is_file():
                return None, None
    except Exception as error:
        return None, f"Playwright browser discovery failed: {type(error).__name__}."
    for command in ("msedge.exe", "chrome.exe", "chromium.exe", "chromium-browser.exe"):
        path = shutil.which(command)
        if path and Path(path).is_file():
            return path, None
    for path in (
        Path(os.getenv("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.getenv("PROGRAMFILES(X86)", ""))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.getenv("PROGRAMFILES(X86)", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.getenv("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
    ):
        if path.is_file():
            return str(path), None
    return None, "No Chromium, Edge, or Chrome executable is installed."


_BROWSER_PATH, _BROWSER_SKIP_REASON = _browser_executable()


def _launch_browser(playwright):
    options = {"headless": True}
    if _BROWSER_PATH:
        options["executable_path"] = _BROWSER_PATH
    return playwright.chromium.launch(**options)


@contextlib.contextmanager
def _browser_page(playwright, app, settings=None):
    browser = _launch_browser(playwright)
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    context.add_init_script(script=_storage_script(settings or {}))
    page = context.new_page()
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    try:
        page.goto(
            app.app_url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS
        )
        yield page, page_errors
    finally:
        context.close()
        browser.close()


def _assert_no_streamlit_exception(page, page_errors, app=None):
    exception_nodes = page.locator('[data-testid="stException"]')
    if exception_nodes.count():
        raise AssertionError(
            "Streamlit exception: " + exception_nodes.first.inner_text()
        )
    if page_errors:
        raise AssertionError("Browser page error: " + " | ".join(page_errors))
    if app is not None:
        app._log_tail()
    if app is not None and re.search(
        r"Traceback \(most recent call last\)|StreamlitAPIException",
        app.log_text,
    ):
        raise AssertionError("Streamlit process reported an exception.")


def _wait_for_chat(page):
    chat_input = page.locator('[data-testid="stChatInput"]')
    chat_input.wait_for(state="visible", timeout=NAVIGATION_TIMEOUT_MS)
    page.get_by_text(re.compile(r"How to use", re.IGNORECASE)).first.wait_for(
        state="visible", timeout=NAVIGATION_TIMEOUT_MS
    )


def _wait_for_restored_provider(page, key="llm_backend", timeout=RESTORE_TIMEOUT_MS):
    """Wait until the app has re-persisted the restored setting to localStorage.

    Persisting only happens after a successful restore, so this is an app-level
    signal that the injected settings were applied rather than overwritten.
    """
    deadline = time.monotonic() + timeout / 1000.0
    last = None
    while time.monotonic() < deadline:
        last = page.evaluate(
            "key => { try { return JSON.parse("
            "localStorage.getItem('docmind:settings') || '{}')[key] || null; }"
            " catch (error) { return null; } }",
            key,
        )
        if last:
            return last
        time.sleep(0.2)
    raise AssertionError(f"the app never re-persisted the restored {key}; saw {last!r}")


def _visible_locator(locator):
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            return candidate
    raise AssertionError("Expected a visible browser control.")


def _wait_for_visible(locator, timeout=ACTION_TIMEOUT_MS):
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        for index in range(locator.count()):
            candidate = locator.nth(index)
            if candidate.is_visible():
                return candidate
        time.sleep(0.1)
    raise AssertionError("Expected a visible browser control.")


def _wait_for_text_visibility(page, value, visible, timeout=ACTION_TIMEOUT_MS):
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        locator = page.get_by_text(value, exact=False)
        states = [locator.nth(index).is_visible() for index in range(locator.count())]
        if visible and any(states):
            return
        if not visible and not any(states):
            return
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for browser text visibility.")


def _click_text(page, value):
    locator = page.get_by_text(value, exact=True)
    if locator.count():
        _visible_locator(locator).click()
        return
    _visible_locator(page.get_by_role("radio", name=value, exact=True)).click()


_PLAYWRIGHT_ERRORS = None


def _playwright_errors():
    """Resolve Playwright exception types lazily, like the browser tests do."""
    global _PLAYWRIGHT_ERRORS
    if _PLAYWRIGHT_ERRORS is None:
        from playwright.sync_api import Error

        _PLAYWRIGHT_ERRORS = (Error,)
    return _PLAYWRIGHT_ERRORS


def _chat_textarea(page):
    return page.locator('[data-testid="stChatInput"] textarea').first


def _user_message_count(page):
    return page.locator('[data-testid="stChatMessage"]').count()


def _submit_button(page):
    return page.locator('[data-testid="stChatInputSubmitButton"]').first


def _send_chat_prompt(page, prompt, app=None, attempts=3, attempt_timeout_ms=45000):
    """Submit a chat prompt and confirm the user turn actually landed.

    Streamlit can drop a submission that arrives while the app is still running
    a script, and it exposes no element to poll for that state, so the user
    message bubble is the only trustworthy confirmation. The text fill and the
    Enter press must stay back to back, otherwise Streamlit commits the text as
    its own script run and discards the key press.
    """
    observed = []
    for attempt in range(1, attempts + 1):
        before = _user_message_count(page)
        area = _chat_textarea(page)
        area.wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)
        if attempt > 1:
            # A dropped submit leaves the text behind, and re-filling an
            # unchanged value is invisible to the widget's own state, so the
            # retry would never submit. Clear it first.
            area.fill("")
        area.fill(prompt)
        area.press("Enter")
        deadline = time.monotonic() + attempt_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if _user_message_count(page) > before:
                return
            time.sleep(0.25)
        observed.append(f"attempt {attempt}: no user turn after {attempt_timeout_ms}ms")
    raise AssertionError(
        f"the chat input dropped {prompt!r} on all {attempts} attempts\n"
        + "\n".join(observed)
        + _submission_diagnostics(page, app)
    )


def _submission_diagnostics(page, app=None):
    """Describe the chat widget state so a dropped submission is explainable."""
    try:
        area = _chat_textarea(page)
        value = area.input_value()
        disabled = area.is_disabled()
    except _playwright_errors() as error:
        return f"\nchat input unreachable: {error}"
    button = _submit_button(page)
    try:
        button_state = (
            "missing" if button.count() == 0 else f"disabled={button.is_disabled()}"
        )
    except _playwright_errors() as error:
        button_state = f"unreadable: {error}"
    detail = (
        f"\nchat input: value={value!r} disabled={disabled}"
        f"\nsubmit button: {button_state}"
        f"\nmessages={_user_message_count(page)}"
        f"\nexceptions={page.locator('[data-testid=\"stException\"]').count()}"
    )
    if app is not None:
        detail += f"\napp log tail:\n{app.log_text[-1500:]}"
    return detail


def _open_reset_expander(page):
    expander = _visible_locator(page.get_by_text(re.compile(r"Reset Project")))
    details = expander.locator("xpath=ancestor::details[1]")
    if not details.evaluate("(element) => element.open"):
        expander.click()
    button = page.get_by_role("button", name=re.compile(r"Clear Chat"))
    try:
        _visible_locator(button).wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)
    except AssertionError:
        expander.click()
        _visible_locator(button).wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)


@unittest.skipUnless(
    _BROWSER_PATH is not None or _BROWSER_SKIP_REASON is None,
    _BROWSER_SKIP_REASON or "",
)
class BrowserSmokeTests(unittest.TestCase):
    def test_real_browser_smoke_clears_chat_and_local_resets(self):
        with tempfile.TemporaryDirectory(prefix="docmind-browser-") as temp_dir:
            state = _OpenAIState()
            with _LoopbackServer(_OpenAIHandler, state) as provider:
                settings = {
                    "llm_backend": "LM Studio (Local AI)",
                    "lm_studio_base_url": provider.base_url + "/v1",
                    "lm_studio_model": "fake-model",
                    "ollama_endpoint": provider.base_url,
                    "ollama_embedding_model": "fake-embedding",
                    "embedding_backend": "LM Studio (Local AI)",
                    "embedding_base_url": provider.base_url + "/v1",
                    "embedding_model": "fake-embedding",
                }
                with StreamlitAppProcess(
                    temp_dir,
                    _reserve_port(),
                    {
                        "DOCMIND_E2E_TEST": "1",
                    },
                ) as app:
                    from playwright.sync_api import sync_playwright

                    with (
                        sync_playwright() as playwright,
                        _browser_page(playwright, app, settings) as (page, errors),
                    ):
                        _wait_for_chat(page)
                        page.get_by_role("tab", name="Data Sources").click()
                        page.get_by_text(
                            "Directly import your data", exact=True
                        ).wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)
                        page.get_by_role("tab", name="Settings").click()
                        page.get_by_text("Settings", exact=True).first.wait_for(
                            state="visible", timeout=ACTION_TIMEOUT_MS
                        )
                        page.get_by_role("tab", name="Data Sources").click()
                        _send_chat_prompt(page, "smoke prompt", app)
                        page.get_by_text(STREAMED_RESPONSE, exact=False).last.wait_for(
                            state="visible", timeout=FIRST_TOKEN_TIMEOUT_MS
                        )
                        _open_reset_expander(page)
                        _visible_locator(
                            page.get_by_role("button", name=re.compile(r"Clear Chat"))
                        ).click()
                        _wait_for_text_visibility(
                            page,
                            STREAMED_RESPONSE,
                            False,
                        )
                        page.get_by_text(
                            re.compile(r"How to use", re.IGNORECASE)
                        ).first.wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)
                        _open_reset_expander(page)
                        _click_text(page, "Keep R2R documents (local-only reset)")
                        _visible_locator(
                            page.get_by_role(
                                "checkbox",
                                name="I understand this reset cannot be undone",
                            )
                        ).check(force=True)
                        _wait_for_visible(
                            page.get_by_role(
                                "button",
                                name="Reset local project only",
                                exact=True,
                            )
                        ).click()
                        page.get_by_text(
                            re.compile("Local project state was reset", re.IGNORECASE)
                        ).wait_for(state="visible", timeout=ACTION_TIMEOUT_MS)
                        _assert_no_streamlit_exception(page, errors, app)
                        self.assertTrue(app.process.poll() is None)
                        self.assertTrue(
                            any(
                                request["path"] == "/v1/chat/completions"
                                and request["body"]
                                and "smoke prompt" in request["body"]
                                for request in state.requests
                            )
                        )


@unittest.skipUnless(
    _BROWSER_PATH is not None or _BROWSER_SKIP_REASON is None,
    _BROWSER_SKIP_REASON or "",
)
class BrowserProviderTests(unittest.TestCase):
    def test_real_browser_streams_from_threaded_fake_openai_provider(self):
        with tempfile.TemporaryDirectory(
            prefix="docmind-provider-browser-"
        ) as temp_dir:
            state = _OpenAIState()
            with _LoopbackServer(_OpenAIHandler, state) as provider:
                settings = {
                    "llm_backend": "LM Studio (Local AI)",
                    "lm_studio_base_url": provider.base_url + "/v1",
                    "lm_studio_model": "fake-model",
                    "ollama_endpoint": provider.base_url,
                    "ollama_embedding_model": "fake-embedding",
                    "embedding_backend": "LM Studio (Local AI)",
                    "embedding_base_url": provider.base_url + "/v1",
                    "embedding_model": "fake-embedding",
                }
                with StreamlitAppProcess(temp_dir, _reserve_port()) as app:
                    from playwright.sync_api import expect, sync_playwright

                    with (
                        sync_playwright() as playwright,
                        _browser_page(playwright, app, settings) as (page, errors),
                    ):
                        _wait_for_chat(page)
                        page.get_by_role("tab", name="Settings").click()
                        provider_input = page.get_by_role("combobox").first
                        provider_input.wait_for(
                            state="visible", timeout=ACTION_TIMEOUT_MS
                        )
                        _wait_for_restored_provider(page)
                        page.get_by_role("tab", name="Data Sources").click()
                        page.get_by_role("tab", name="Settings").click()
                        provider_input = page.get_by_role("combobox").first
                        try:
                            expect(provider_input).to_have_value(
                                "LM Studio (Local AI)", timeout=RESTORE_TIMEOUT_MS
                            )
                        except AssertionError as error:
                            stored = page.evaluate(
                                "() => localStorage.getItem('docmind:settings')"
                            )
                            raise AssertionError(
                                "the restored provider never settled: "
                                f"value={provider_input.input_value()!r} "
                                f"stored={stored!r}\napp log:\n{app.log_text[-2000:]}"
                            ) from error
                        page.get_by_role("tab", name="Data Sources").click()
                        _send_chat_prompt(page, "browser integration prompt", app)
                        page.get_by_text(STREAMED_RESPONSE, exact=False).last.wait_for(
                            state="visible", timeout=FIRST_TOKEN_TIMEOUT_MS
                        )
                        _assert_no_streamlit_exception(page, errors, app)
                        self.assertTrue(
                            any(
                                request["path"] == "/v1/chat/completions"
                                and request["body"]
                                and "browser integration prompt" in request["body"]
                                for request in state.requests
                            )
                        )
                        self.assertTrue(
                            all(
                                request["authorization"] in {"", "Bearer local"}
                                for request in state.requests
                            )
                        )
                        storage = page.evaluate(
                            "Object.fromEntries(Object.entries(localStorage))"
                        )
                        self.assertNotIn(
                            "provider-secret",
                            json.dumps(storage),
                        )
                        persisted = json.loads(storage.get("docmind:settings", "{}"))
                        self.assertTrue(
                            all(
                                "api_key" not in key.casefold()
                                and "secret" not in key.casefold()
                                and "password" not in key.casefold()
                                for key in persisted
                            )
                        )
                        self.assertNotIn("lm_studio_api_key", persisted)


class R2RServerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="docmind-r2r-")
        self.root = Path(self.temp_dir.name)
        self.state = _R2RState()
        self.server_context = _LoopbackServer(_R2RHandler, self.state)
        self.server = self.server_context.__enter__()
        self.client = r2r.R2RClient(
            base_url=self.server_context.base_url,
            api_key=TEST_CREDENTIAL,
            timeout=1.0,
        )

    def tearDown(self):
        self.server_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def _file(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def _registry(
        self, workspace="integration-workspace", identity=None, project="project"
    ):
        return r2r.R2RDocumentRegistry.for_workspace(
            workspace,
            identity or self.client.identity,
            state_root=self.root / "registry",
            project_path=self.root / project,
        )

    def _ingest_one(self, registry, name, content, old_document_ids=None):
        path = self._file(name, content)
        arguments = {
            "old_document_ids": old_document_ids,
        }
        return r2r.ingest_paths_transaction(
            self.client,
            [str(path)],
            registry,
            "integration-workspace",
            f"content-{name}",
            f"source-{name}",
            f"signature-{name}",
            **arguments,
        )[0]

    def test_health_multipart_upload_listing_polling_and_rag(self):
        self.assertTrue(self.client.health())
        path = self._file("notes.txt", "local integration content")
        receipt = self.client.upload_document(str(path), document_id="doc-health")
        self.assertEqual(receipt.document_id, "doc-health")
        self.assertTrue(receipt.task_id)
        status = self.client.wait_for_document(
            receipt.document_id,
            task_id=receipt.task_id,
            timeout=2,
            initial_interval=0.01,
            max_interval=0.02,
        )
        self.assertEqual(status.status, "success")
        self.assertGreaterEqual(status.polls, 2)
        listing = self.client._request("GET", "/v3/documents")
        self.assertIn("doc-health", str(listing.json()))
        result = self.client.rag(
            "What is in the note?",
            document_ids=[receipt.document_id],
            top_k=2,
            answer_style="Concise",
        )
        self.assertEqual(result.answer, "R2R answer with a citation.")
        self.assertEqual(result.citations[0]["document_id"], "doc-health")
        self.assertEqual(
            result.search_results["chunk_search_results"][0]["metadata"]["title"],
            "notes.txt",
        )
        upload_request = next(
            request
            for request in self.state.requests
            if request["method"] == "POST" and request["path"] == "/v3/documents"
        )
        self.assertIn("multipart/form-data", upload_request["content_type"])
        self.assertIn("local integration content", upload_request["body"])
        self.assertEqual(
            self.state.uploads[0]["fields"]["run_with_orchestration"], "true"
        )
        rag_request = next(
            request
            for request in self.state.requests
            if request["method"] == "POST" and request["path"] == "/v3/retrieval/rag"
        )
        self.assertIn('"filters"', rag_request["body"])

    def test_r2r_chat_returns_citations_and_metadata_through_the_app_adapter(self):
        registry = self._registry()
        record = self._ingest_one(registry, "chat.txt", "chat content")
        state = {
            "r2r_document_ids": [record["document_id"]],
            "top_k": 3,
            "answer_style": "Concise",
            "last_doc_sources": [],
        }
        fake_streamlit = _FakeStreamlit(state)
        with (
            patch.object(r2r, "st", fake_streamlit),
            patch.object(r2r, "get_client", return_value=self.client),
            patch.object(r2r, "r2r_is_ready", return_value=True),
        ):
            answer = "".join(r2r.r2r_chat("What does the note say?"))
        self.assertEqual(answer, "R2R answer with a citation.")
        self.assertEqual(
            state["last_r2r_metadata"]["citations"][0]["document_id"],
            record["document_id"],
        )
        self.assertEqual(state["last_doc_sources"], [("notes.txt", 0.91)])

    def test_replacement_is_atomic_and_registry_survives_reload(self):
        registry = self._registry()
        first = self._file("first.txt", "first")
        first_records = r2r.ingest_paths_transaction(
            self.client,
            [str(first)],
            registry,
            "integration-workspace",
            "content-first",
            "source-first",
            "signature-first",
        )
        first_id = first_records[0]["document_id"]
        second = self._file("second.txt", "second")
        second_records = r2r.ingest_paths_transaction(
            self.client,
            [str(second)],
            registry,
            "integration-workspace",
            "content-second",
            "source-second",
            "signature-second",
            old_document_ids=[first_id],
        )
        second_id = second_records[0]["document_id"]
        snapshot = registry.snapshot()
        self.assertEqual(snapshot["active_document_ids"], [second_id])
        self.assertEqual(registry.get_document(first_id)["status"], "deleted")
        reloaded = self._registry()
        self.assertEqual(reloaded.snapshot()["active_document_ids"], [second_id])
        self.assertNotIn(TEST_CREDENTIAL, registry.path.read_text(encoding="utf-8"))

    def test_failed_replacement_rolls_back_new_document_and_preserves_old_active(self):
        registry = self._registry()
        first = self._file("first.txt", "first")
        first_records = r2r.ingest_paths_transaction(
            self.client,
            [str(first)],
            registry,
            "integration-workspace",
            "content-first",
            "source-first",
            "signature-first",
        )
        first_id = first_records[0]["document_id"]
        self.state.fail_next_indexing = True
        second = self._file("second.txt", "second")
        with self.assertRaises(r2r.R2RLifecycleError):
            r2r.ingest_paths_transaction(
                self.client,
                [str(second)],
                registry,
                "integration-workspace",
                "content-second",
                "source-second",
                "signature-second",
                old_document_ids=[first_id],
            )
        self.assertEqual(registry.snapshot()["active_document_ids"], [first_id])
        new_ids = set(self.state.documents) - {first_id}
        self.assertEqual(new_ids, set())
        self.assertTrue(
            any(
                record["status"] == "rolled_back"
                for record in registry.snapshot()["documents"]
            )
        )

    def test_remote_reset_deletes_owned_documents_and_clears_local_state(self):
        registry = self._registry()
        record = self._ingest_one(registry, "reset.txt", "reset")
        state = {
            "r2r_enabled": True,
            "r2r_base_url": self.server_context.base_url,
            "r2r_api_key": TEST_CREDENTIAL,
            "r2r_workspace_id": "integration-workspace",
            "r2r_registry_root": str(self.root / "registry"),
            "r2r_project_path": str(self.root / "project"),
            "r2r_registry_path": str(registry.path),
            "r2r_document_ids": [record["document_id"]],
            "r2r_document_ids_signature": "signature-reset",
            "r2r_current_source_signature": "signature-reset",
            "messages": [{"role": "user", "content": "remove"}],
            "last_doc_sources": [("notes.txt", 0.9)],
            "last_rag_evidence": [{"source": "notes.txt"}],
        }
        with patch.dict(
            os.environ,
            {
                r2r.R2R_STATE_DIR_ENV: str(self.root / "reset-state"),
                "DOCMIND_INGESTION_LOCK_PATH": str(self.root / "ingestion.lock"),
            },
        ):
            result = perform_project_reset(state)
        self.assertTrue(result["remote_delete_success"])
        self.assertEqual(state["r2r_document_ids"], [])
        self.assertEqual(state["last_doc_sources"], [])
        self.assertEqual(state["last_rag_evidence"], [])
        self.assertEqual(state["messages"], [dict(WELCOME_MESSAGE)])
        self.assertTrue(registry.path.exists())
        self.assertEqual(
            registry.get_document(record["document_id"])["status"], "deleted"
        )
        self.assertEqual(registry.snapshot()["active_document_ids"], [])

    def test_remote_delete_failure_clears_local_state_but_retains_failed_registry_record(
        self,
    ):
        registry = self._registry()
        record = self._ingest_one(registry, "failure.txt", "failure")
        self.state.fail_delete_ids.add(record["document_id"])
        state = {
            "r2r_enabled": True,
            "r2r_base_url": self.server_context.base_url,
            "r2r_api_key": TEST_CREDENTIAL,
            "r2r_workspace_id": "integration-workspace",
            "r2r_registry_root": str(self.root / "registry"),
            "r2r_project_path": str(self.root / "project"),
            "r2r_registry_path": str(registry.path),
            "r2r_document_ids": [record["document_id"]],
            "r2r_document_ids_signature": "signature-failure",
            "messages": [{"role": "user", "content": "remove"}],
        }
        with patch.dict(
            os.environ,
            {
                r2r.R2R_STATE_DIR_ENV: str(self.root / "failure-state"),
                "DOCMIND_INGESTION_LOCK_PATH": str(self.root / "ingestion.lock"),
            },
        ):
            result = perform_project_reset(state)
        self.assertFalse(result["remote_delete_success"])
        self.assertEqual(state["r2r_document_ids"], [])
        self.assertEqual(state["messages"], [dict(WELCOME_MESSAGE)])
        self.assertTrue(registry.path.exists())
        self.assertEqual(
            registry.get_document(record["document_id"])["status"], "delete_failed"
        )
        self.assertEqual(
            registry.snapshot()["active_document_ids"], [record["document_id"]]
        )

    def test_wrong_endpoint_and_credential_ownership_are_retained(self):
        registry = self._registry(workspace="ownership-workspace")
        first = self._ingest_one(registry, "owned.txt", "owned")
        other_client = r2r.R2RClient(
            base_url=self.server_context.base_url,
            api_key="second-credential",
            timeout=1.0,
        )
        other_registry = r2r.R2RDocumentRegistry.for_workspace(
            "ownership-workspace",
            other_client.identity,
            state_root=self.root / "registry",
            project_path=self.root / "project",
        )
        second = self._ingest_one(
            other_registry,
            "other.txt",
            "other",
            old_document_ids=[],
        )
        wrong_client = r2r.R2RClient(
            base_url=self.server_context.base_url,
            api_key="wrong-credential",
            timeout=1.0,
        )
        self.assertFalse(wrong_client.health())
        state = {
            "r2r_enabled": True,
            "r2r_base_url": self.server_context.base_url.replace(
                "127.0.0.1", "localhost"
            ),
            "r2r_api_key": "wrong-credential",
            "r2r_workspace_id": "ownership-workspace",
            "r2r_registry_root": str(self.root / "registry"),
            "r2r_project_path": str(self.root / "project"),
            "r2r_registry_path": str(registry.path),
            "r2r_document_ids": [first["document_id"]],
            "messages": [{"role": "user", "content": "remove"}],
        }
        with patch.dict(
            os.environ,
            {"DOCMIND_INGESTION_LOCK_PATH": str(self.root / "ingestion.lock")},
        ):
            result = perform_project_reset(state)
        self.assertFalse(result["remote_delete_success"])
        self.assertEqual(result["remote"]["other_identity_count"], 2)
        self.assertEqual(state["r2r_document_ids"], [])
        self.assertIn(first["document_id"], self.state.documents)
        self.assertIn(second["document_id"], self.state.documents)
        self.assertTrue(registry.path.exists())
        self.assertNotIn("wrong-credential", registry.path.read_text(encoding="utf-8"))


class ProviderFailureIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="docmind-provider-")
        self.root = Path(self.temp_dir.name)
        self.state = _OpenAIState()
        self.server_context = _LoopbackServer(_OpenAIHandler, self.state)
        self.server_context.__enter__()
        self.base_url = self.server_context.base_url + "/v1"

    def tearDown(self):
        self.server_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def _state(self, model="fake-model"):
        return {
            "llm_backend": "LM Studio (Local AI)",
            "lm_studio_base_url": self.base_url,
            "lm_studio_model": model,
            "lm_studio_api_key": "",
            "openai_model": model,
            "messages": [],
            "system_prompt": "system",
            "temperature": 0.1,
            "eco_mode": False,
            "last_doc_sources": [],
            "last_rag_evidence": [],
            "last_rag_no_result": False,
            "last_rag_question": None,
        }

    def _run_chat(self, mode, model="fake-model"):
        self.state.mode = mode
        state = self._state(model)
        fake_streamlit = _FakeStreamlit(state)
        ollama_module._create_openai_llm_cached.clear()
        with (
            patch.object(ollama_module, "st", fake_streamlit),
            patch.object(ollama_module.logs.log, "error") as error_log,
        ):
            answer = "".join(ollama_module.chat("provider failure prompt"))
        rendered_log = " ".join(str(call) for call in error_log.call_args_list)
        return answer, rendered_log

    def test_provider_http_failure_is_generic_and_redacted(self):
        answer, rendered_log = self._run_chat("http_failure")
        self.assertIn("configured provider could not complete", answer)
        self.assertNotIn("provider-secret", answer)
        self.assertNotIn("provider-secret", rendered_log)
        self.assertIn("[REDACTED]", rendered_log)

    def test_malformed_and_oversized_provider_responses_are_generic(self):
        for mode in ("malformed", "oversized"):
            with self.subTest(mode=mode):
                answer, rendered_log = self._run_chat(mode)
                self.assertIn("configured provider could not complete", answer)
                self.assertNotIn("provider-secret", answer)
                self.assertNotIn("provider-secret", rendered_log)

    def test_unavailable_model_is_generic(self):
        answer, rendered_log = self._run_chat("normal", "missing-model")
        self.assertIn("configured provider could not complete", answer)
        self.assertNotIn("provider-secret", answer)
        self.assertNotIn("provider-secret", rendered_log)

    def test_ollama_model_discovery_reports_unavailable_model(self):
        state = {"ollama_endpoint": self.server_context.base_url}
        with patch.object(ollama_module, "st", SimpleNamespace(session_state=state)):
            self.assertFalse(
                ollama_module.verify_chat_model(
                    "missing-model", self.server_context.base_url
                )
            )
            self.assertTrue(
                ollama_module.verify_chat_model(
                    "fake-model", self.server_context.base_url
                )
            )


class R2RFailureIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="docmind-r2r-failure-")
        self.root = Path(self.temp_dir.name)
        self.state = _R2RState()
        self.server_context = _LoopbackServer(_R2RHandler, self.state)
        self.server_context.__enter__()
        self.client = r2r.R2RClient(
            base_url=self.server_context.base_url,
            api_key=TEST_CREDENTIAL,
            timeout=1.0,
        )

    def tearDown(self):
        self.server_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_r2r_timeout_and_5xx_are_generic_and_redacted(self):
        self.state.delay_seconds = 2
        self.assertFalse(self.client.health())
        self.state.delay_seconds = 0
        self.state.mode = "rag_5xx"
        with self.assertRaises(r2r.R2RHTTPError) as raised:
            self.client.rag("question", ["doc-1"])
        self.assertNotIn("r2r-secret", str(raised.exception))
        state = {
            "r2r_document_ids": ["doc-1"],
            "top_k": 3,
            "answer_style": "Concise",
            "last_doc_sources": [("stale.txt", 0.8)],
            "active_source": make_source_state("r2r", status="ready"),
            "r2r_enabled": True,
            "r2r_current_source_signature": "signature",
            "r2r_document_ids_signature": "signature",
            "r2r_bound_identity": self.client.identity.as_dict() | {"enabled": True},
            "r2r_contract": r2r.R2R_CONTRACT,
            "r2r_base_url": self.server_context.base_url,
            "r2r_api_key": TEST_CREDENTIAL,
        }
        fake_streamlit = _FakeStreamlit(state)
        with (
            patch.object(r2r, "st", fake_streamlit),
            patch.object(r2r, "get_client", return_value=self.client),
            patch.object(r2r, "r2r_is_ready", return_value=True),
            patch.object(r2r.logs.log, "error") as error_log,
        ):
            answer = "".join(r2r.r2r_chat("question"))
        self.assertNotIn("r2r-secret", answer)
        self.assertIn("R2R request failed", answer)
        self.assertEqual(state["last_doc_sources"], [])
        self.assertEqual(state["last_r2r_metadata"]["error_category"], "server_error")
        self.assertNotIn(
            "r2r-secret", " ".join(str(call) for call in error_log.call_args_list)
        )

    def test_r2r_poll_timeout_is_bounded(self):
        path = self.root / "pending.txt"
        path.write_text("pending", encoding="utf-8")
        self.state.stay_pending = True
        self.client.upload_document(str(path), document_id="pending-doc")
        with self.assertRaises(r2r.R2RPollTimeout):
            self.client.wait_for_document(
                "pending-doc",
                timeout=0.05,
                initial_interval=0.01,
                max_interval=0.01,
            )

    def test_no_match_rag_clears_stale_evidence(self):
        state = {
            "retriever": None,
            "query_engine": SimpleNamespace(_retriever=_NoMatchRetriever()),
            "messages": [],
            "system_prompt": "",
            "last_doc_sources": [("stale.txt", 0.99)],
            "last_rag_evidence": [{"source": "stale.txt", "score": 0.99}],
            "last_rag_no_result": False,
            "last_rag_question": None,
        }
        fake_streamlit = _FakeStreamlit(state)
        with (
            patch.object(ollama_module, "st", fake_streamlit),
            patch.object(ollama_module, "_session_llm", return_value=object()),
        ):
            answer = "".join(
                ollama_module.context_chat(
                    "unknown question",
                    state["query_engine"],
                )
            )
        self.assertIn("could not find", answer.casefold())
        self.assertEqual(state["last_doc_sources"], [])
        self.assertEqual(state["last_rag_evidence"], [])
        self.assertTrue(state["last_rag_no_result"])
        self.assertEqual(state["last_rag_question"], "unknown question")

    def test_reset_cleanup_failure_clears_local_state_without_leaking_paths(self):
        owned = self.root / "owned"
        owned.mkdir()
        state = {
            "active_source": {"kind": "local", "status": "ready"},
            "query_engine": object(),
            "retriever": object(),
            "documents": [object()],
            "file_list": [object()],
            "messages": [{"role": "user", "content": "remove"}],
            "last_doc_sources": [("stale.txt", 0.8)],
            "last_rag_evidence": [{"source": "stale.txt"}],
            "_session_work_dirs": [str(owned)],
        }
        with (
            patch.dict(
                os.environ,
                {
                    r2r.R2R_STATE_DIR_ENV: str(self.root / "cleanup-state"),
                    "DOCMIND_INGESTION_LOCK_PATH": str(self.root / "ingestion.lock"),
                },
            ),
            patch("components.page_state._remove_dir_retry", return_value=False),
        ):
            result = perform_project_reset(
                state, owned_dirs=[owned], delete_remote=False
            )
        self.assertEqual(result["failed"], [str(owned)])
        self.assertTrue(owned.exists())
        self.assertEqual(state["messages"], [dict(WELCOME_MESSAGE)])
        self.assertEqual(state["last_doc_sources"], [])
        self.assertEqual(state["last_rag_evidence"], [])
        self.assertEqual(state["_session_work_dirs"], [str(owned)])


def _component_chat_app():
    import streamlit as st

    from components.chatbox import chatbox
    from components.page_state import WELCOME_MESSAGE

    st.session_state.update(
        {
            "messages": [dict(WELCOME_MESSAGE)],
            "last_doc_sources": [],
            "last_rag_evidence": [],
            "last_rag_no_result": False,
            "last_rag_question": None,
            "system_prompt": "system",
            "answer_style": "Balanced (default)",
            "quick_answer_style": "Balanced (default)",
            "llm_backend": "LM Studio (Local AI)",
            "lm_studio_base_url": "http://127.0.0.1:1/v1",
            "lm_studio_model": "fake-model",
            "openai_model": "fake-model",
        }
    )
    chatbox()


class StreamlitComponentTests(unittest.TestCase):
    def test_apptest_renders_real_chat_component_without_a_browser(self):
        script = (
            "from streamlit.testing.v1 import AppTest\n"
            "from tests.test_e2e_integration import _component_chat_app\n"
            "app = AppTest.from_function(_component_chat_app).run(timeout=20)\n"
            "assert not app.exception\n"
            "assert app.chat_input\n"
            "assert app.pills\n"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
