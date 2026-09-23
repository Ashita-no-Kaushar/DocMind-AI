import json
import os
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from components import chatbox
from components.page_state import perform_project_reset
from utils import r2r
from utils.browser_settings import serialize_persisted_settings
from utils.source_state import (
    effective_indexing_settings,
    make_source_state,
    mark_source_ready,
    stable_digest,
)


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text="ok", headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"content-type": "application/json"}
        if payload is None:
            self.content = text.encode("utf-8")
        else:
            self.content = json.dumps(payload).encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeLifecycleClient:
    def __init__(self, fail_upload_number=None, fail_delete_ids=None):
        self.identity = r2r.R2RIdentity("endpoint-fp", "credential-fp")
        self.fail_upload_number = fail_upload_number
        self.fail_delete_ids = set(fail_delete_ids or [])
        self.upload_count = 0
        self.deleted = []
        self.events = []
        self.waited = []

    def upload_document(self, file_path, document_id=None, run_with_orchestration=True):
        self.upload_count += 1
        self.events.append(("upload", document_id))
        if self.upload_count == self.fail_upload_number:
            raise r2r.R2RLifecycleError("upload failed")
        return r2r.R2RUploadReceipt(document_id, f"task-{document_id}")

    def wait_for_document(
        self,
        document_id,
        task_id=None,
        timeout=None,
        cancel_event=None,
        clock=None,
        sleep=None,
        on_status=None,
    ):
        self.events.append(("ready", document_id))
        self.waited.append((document_id, task_id))
        status = r2r.R2RDocumentStatus(document_id, "success", task_id)
        if on_status:
            on_status(status)
        return r2r.R2RDocumentWaitResult(document_id, task_id, "success", 1)

    def delete_document(self, document_id):
        self.events.append(("delete", document_id))
        if document_id in self.fail_delete_ids:
            raise r2r.R2RConnectionError("delete failed")
        self.deleted.append(document_id)
        return "deleted"


class R2RClientTests(unittest.TestCase):
    def test_health_uses_v3_endpoint_and_api_key_header(self):
        client = r2r.R2RClient(base_url="http://localhost:7272/", api_key="secret")
        with patch(
            "utils.r2r.httpx.request", return_value=_FakeResponse({"results": {}})
        ) as request:
            self.assertTrue(client.health())
        self.assertEqual(
            request.call_args.args, ("GET", "http://localhost:7272/v3/health")
        )
        headers = request.call_args.kwargs["headers"]
        self.assertEqual(headers["x-api-key"], "secret")
        self.assertNotIn("Authorization", headers)
        self.assertFalse(request.call_args.kwargs["follow_redirects"])

    def test_health_false_when_server_unreachable(self):
        client = r2r.R2RClient()
        with patch(
            "utils.r2r.httpx.request",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            self.assertFalse(client.health())

    def test_upload_uses_official_multipart_field_and_idempotency_id(self):
        client = r2r.R2RClient(api_key="secret")
        response = _FakeResponse(
            {"results": {"document_id": "doc-1", "task_id": "task-1"}}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("content", encoding="utf-8")
            with patch("utils.r2r.httpx.request", return_value=response) as request:
                receipt = client.upload_document(str(path), document_id="doc-1")
        self.assertEqual(receipt, r2r.R2RUploadReceipt("doc-1", "task-1"))
        self.assertEqual(
            request.call_args.kwargs["data"],
            {"id": "doc-1", "run_with_orchestration": "true"},
        )
        self.assertEqual(request.call_args.kwargs["files"][0][0], "file")
        self.assertEqual(request.call_args.kwargs["files"][0][1][0], "notes.txt")
        self.assertEqual(request.call_args.kwargs["headers"]["x-api-key"], "secret")

    def test_rag_exact_filter_shape_and_no_top_level_document_ids(self):
        client = r2r.R2RClient(api_key="secret")
        response = _FakeResponse(
            {
                "results": {
                    "generated_answer": "Answer",
                    "citations": [{"id": "c1", "payload": {"document_id": "doc-1"}}],
                    "search_results": {
                        "chunk_search_results": [
                            {
                                "document_id": "doc-1",
                                "score": 0.8,
                                "metadata": {"title": "notes.txt"},
                            }
                        ]
                    },
                }
            }
        )
        with patch("utils.r2r.httpx.request", return_value=response) as request:
            result = client.rag(
                "Question?",
                document_ids=["doc-1", "doc-2", "doc-1"],
                top_k=4,
                answer_style="Concise",
            )
        self.assertEqual(result.answer, "Answer")
        self.assertEqual(result.citations[0]["id"], "c1")
        payload = request.call_args.kwargs["json"]
        self.assertEqual(
            payload,
            {
                "query": "Question?",
                "search_mode": "custom",
                "search_settings": {
                    "filters": {"document_id": {"$in": ["doc-1", "doc-2"]}},
                    "limit": 4,
                    "include_metadatas": True,
                    "include_scores": True,
                },
                "rag_generation_config": {"stream": False},
                "include_web_search": False,
                "task_prompt": "Follow this answer style: Concise.",
            },
        )
        self.assertNotIn("document_ids", payload)
        self.assertNotIn("messages", payload)
        self.assertNotIn("history", payload)

    def test_redirect_auth_and_server_errors_are_classified(self):
        client = r2r.R2RClient()
        cases = [
            (302, r2r.R2RRedirectError),
            (401, r2r.R2RAuthenticationError),
            (500, r2r.R2RHTTPError),
        ]
        for status_code, error_type in cases:
            with (
                self.subTest(status_code=status_code),
                patch(
                    "utils.r2r.httpx.request",
                    return_value=_FakeResponse(
                        {"detail": "raw-secret-response"},
                        status_code=status_code,
                        text="raw-secret-response",
                    ),
                ),
                self.assertRaises(error_type) as raised,
            ):
                client.rag("question", ["doc-1"])
                self.assertNotIn("raw-secret-response", str(raised.exception))

    def test_oversized_and_non_json_responses_are_bounded_and_redacted(self):
        client = r2r.R2RClient(max_response_bytes=1024)
        oversized = _FakeResponse(
            None,
            text="S" * 2048,
            headers={"content-length": "2048", "content-type": "text/plain"},
        )
        with (
            patch("utils.r2r.httpx.request", return_value=oversized),
            self.assertRaises(r2r.R2RResponseError) as raised,
        ):
            client._request("GET", "/v3/health", max_response_bytes=1024)
        self.assertNotIn("S" * 20, str(raised.exception))
        non_json = _FakeResponse(
            None, text="credential=do-not-show", headers={"content-type": "text/plain"}
        )
        with (
            patch("utils.r2r.httpx.request", return_value=non_json),
            self.assertRaises(r2r.R2RResponseError) as raised,
        ):
            response = client._request("GET", "/v3/health")
            r2r._as_dict(response)
        self.assertNotIn("do-not-show", str(raised.exception))

    def test_document_identifier_rejects_path_traversal(self):
        client = r2r.R2RClient()
        with self.assertRaises(r2r.R2RLifecycleError):
            client.get_document("../../secret")


class _FakeStreamlit:
    def __init__(self, state):
        self.session_state = state
        self.captions = []
        self.warnings = []

    def chat_message(self, *args, **kwargs):
        return nullcontext()

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def markdown(self, *args, **kwargs):
        pass

    def caption(self, value, *args, **kwargs):
        self.captions.append(str(value))

    def write_stream(self, stream):
        return "".join(str(item) for item in stream)

    def warning(self, value, *args, **kwargs):
        self.warnings.append(str(value))

    def button(self, *args, **kwargs):
        return False

    def error(self, *args, **kwargs):
        pass


class R2RChatTests(unittest.TestCase):
    def test_r2r_routes_before_local_and_openai_model_validation(self):
        state = {
            "messages": [],
            "last_doc_sources": [],
            "top_k": 3,
            "answer_style": "Balanced (default)",
            "r2r_enabled": True,
        }

        def fake_r2r_chat(prompt):
            state["last_r2r_metadata"] = {
                "contract": r2r.R2R_CONTRACT,
                "streaming": False,
                "citations": [{"id": "c1"}],
                "search_results": {},
            }
            yield "Remote answer"

        streamlit = _FakeStreamlit(state)
        with (
            patch.object(chatbox, "st", streamlit),
            patch.object(chatbox.r2r, "r2r_is_ready", return_value=True),
            patch.object(chatbox.r2r, "r2r_chat", side_effect=fake_r2r_chat),
            patch.object(chatbox, "is_openai_compatible_backend") as openai_backend,
            patch.object(chatbox, "get_models") as get_models,
            patch.object(chatbox, "get_embedding_models") as get_embeddings,
        ):
            chatbox._process_prompt("Question?")

        openai_backend.assert_not_called()
        get_models.assert_not_called()
        get_embeddings.assert_not_called()
        self.assertEqual(streamlit.warnings, [])
        self.assertTrue(
            any("non-streaming" in caption for caption in streamlit.captions)
        )
        assistant = state["messages"][-1]
        self.assertEqual(assistant["content"], "Remote answer")
        self.assertEqual(assistant["r2r"]["citations"], [{"id": "c1"}])

    def test_r2r_chat_stores_structured_results_without_history(self):
        state = {
            "r2r_document_ids": ["doc-1"],
            "top_k": 3,
            "answer_style": "Technical",
            "last_doc_sources": [],
        }
        client = SimpleNamespace(
            rag=lambda *args, **kwargs: r2r.R2RChatResult(
                "Answer",
                [{"id": "citation-1"}],
                {
                    "chunk_search_results": [
                        {
                            "document_id": "doc-1",
                            "score": 0.7,
                            "metadata": {"title": "notes.txt"},
                        }
                    ]
                },
            )
        )
        with (
            patch.object(r2r, "st", SimpleNamespace(session_state=state)),
            patch.object(r2r, "get_client", return_value=client),
            patch.object(r2r, "r2r_is_ready", return_value=True),
        ):
            answer = "".join(r2r.r2r_chat("Question?"))
        self.assertEqual(answer, "Answer")
        self.assertEqual(
            state["last_r2r_metadata"]["citations"], [{"id": "citation-1"}]
        )
        self.assertFalse(state["last_r2r_metadata"]["history_sent"])
        self.assertEqual(state["last_doc_sources"], [("notes.txt", 0.7)])

    def test_r2r_error_text_and_log_do_not_expose_credentials(self):
        state = {
            "r2r_document_ids": ["doc-1"],
            "top_k": 3,
            "answer_style": "Concise",
            "last_doc_sources": [],
        }
        client = SimpleNamespace(
            rag=lambda *args, **kwargs: (_ for _ in ()).throw(
                r2r.R2RConnectionError("api-key=super-secret")
            )
        )
        fake_log = SimpleNamespace(error=lambda *args, **kwargs: None)
        with (
            patch.object(r2r, "st", SimpleNamespace(session_state=state)),
            patch.object(r2r, "get_client", return_value=client),
            patch.object(r2r, "r2r_is_ready", return_value=True),
            patch.object(r2r.logs, "log", fake_log),
        ):
            answer = "".join(r2r.r2r_chat("Question?"))
        self.assertNotIn("super-secret", answer)
        self.assertEqual(
            state["last_r2r_metadata"]["error_category"], "connection_error"
        )


class R2RPollAndLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.identity = r2r.R2RIdentity("endpoint-fp", "credential-fp")
        self.registry = r2r.R2RDocumentRegistry.for_workspace(
            "workspace", self.identity, state_root=self.root
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _file(self, name, content="content"):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def _seed_ready(self, document_id="old-doc", source_signature="old-source"):
        record = {
            "document_id": document_id,
            "task_id": "old-task",
            "content_signature": "old-content",
            "source_id": "old-source-id",
            "source_signature": source_signature,
            "status": "ready",
            "created_at": r2r._utc_now(),
            "updated_at": r2r._utc_now(),
            "last_error": None,
            "endpoint_fingerprint": self.identity.endpoint_fingerprint,
            "credential_fingerprint": self.identity.credential_fingerprint,
            "file_name": "old.txt",
        }
        self.registry.upsert_document(record)
        self.registry.commit_active(
            [record],
            {
                "source_id": "old-source-id",
                "content_signature": "old-content",
                "settings_signature": "old-settings",
                "source_signature": source_signature,
                "index_generation": 1,
                "display_name": "old.txt",
            },
        )
        return record

    def test_task_id_is_tracked_and_document_status_must_be_successful(self):
        path = self._file("notes.txt")
        document_ids = []
        status_calls = 0

        def request_side_effect(method, url, **kwargs):
            nonlocal status_calls
            if method == "POST" and url.endswith("/v3/documents"):
                document_id = kwargs["data"]["id"]
                document_ids.append(document_id)
                return _FakeResponse(
                    {
                        "results": {
                            "document_id": document_id,
                            "task_id": "task-123",
                        }
                    },
                    status_code=202,
                )
            status_calls += 1
            document_id = url.rsplit("/", 1)[-1]
            status = "pending" if status_calls == 1 else "success"
            return _FakeResponse(
                {"results": {"id": document_id, "ingestion_status": status}}
            )

        client = r2r.R2RClient()
        with patch("utils.r2r.httpx.request", side_effect=request_side_effect):
            records = r2r.ingest_paths_transaction(
                client,
                [str(path)],
                self.registry,
                "workspace",
                "content-signature",
                "source-id",
                "source-signature",
            )
        self.assertEqual(records[0]["task_id"], "task-123")
        self.assertEqual(records[0]["status"], "ready")
        self.assertEqual(records[0]["source_id"], "source-id")

    def test_indexing_failure_rolls_back_new_document(self):
        path = self._file("notes.txt")
        requested_id = {}

        def request_side_effect(method, url, **kwargs):
            if method == "POST":
                requested_id["value"] = kwargs["data"]["id"]
                return _FakeResponse(
                    {
                        "results": {
                            "document_id": requested_id["value"],
                            "task_id": "task-1",
                        }
                    },
                    status_code=202,
                )
            if method == "GET":
                return _FakeResponse(
                    {
                        "results": {
                            "id": requested_id["value"],
                            "ingestion_status": "failed",
                        }
                    }
                )
            if method == "DELETE":
                return _FakeResponse({"results": {"success": True}})
            raise AssertionError(url)

        client = r2r.R2RClient()
        with (
            patch("utils.r2r.httpx.request", side_effect=request_side_effect),
            self.assertRaises(r2r.R2RLifecycleError),
        ):
            r2r.ingest_paths_transaction(
                client,
                [str(path)],
                self.registry,
                "workspace",
                "content-signature",
                "source-id",
                "source-signature",
            )
        stored = self.registry.get_document(requested_id["value"])
        self.assertEqual(stored["status"], "rolled_back")

    def test_poll_timeout_rolls_back_and_stays_bounded(self):
        path = self._file("notes.txt")
        requested_id = {}
        status_calls = 0

        def request_side_effect(method, url, **kwargs):
            nonlocal status_calls
            if method == "POST":
                requested_id["value"] = kwargs["data"]["id"]
                return _FakeResponse(
                    {
                        "results": {
                            "document_id": requested_id["value"],
                            "task_id": "task-1",
                        }
                    },
                    status_code=202,
                )
            if method == "GET":
                status_calls += 1
                return _FakeResponse(
                    {
                        "results": {
                            "id": requested_id["value"],
                            "ingestion_status": "pending",
                        }
                    }
                )
            return _FakeResponse({"results": {"success": True}})

        now = [0.0]

        def sleep(delay):
            now[0] += delay

        client = r2r.R2RClient()
        with (
            patch("utils.r2r.httpx.request", side_effect=request_side_effect),
            self.assertRaises(r2r.R2RPollTimeout),
        ):
            r2r.ingest_paths_transaction(
                client,
                [str(path)],
                self.registry,
                "workspace",
                "content-signature",
                "source-id",
                "source-signature",
                poll_timeout=1,
                clock=lambda: now[0],
                sleep=sleep,
            )
        self.assertGreaterEqual(status_calls, 2)
        self.assertEqual(
            self.registry.get_document(requested_id["value"])["status"],
            "rolled_back",
        )

    def test_partial_upload_failure_rolls_back_every_new_id(self):
        first = self._file("first.txt", "first")
        second = self._file("second.txt", "second")
        client = _FakeLifecycleClient(fail_upload_number=2)
        with self.assertRaises(r2r.R2RLifecycleError):
            r2r.ingest_paths_transaction(
                client,
                [str(first), str(second)],
                self.registry,
                "workspace",
                "content-signature",
                "source-id",
                "source-signature",
            )
        self.assertEqual(len(client.deleted), 2)
        self.assertTrue(
            all(
                record["status"] == "rolled_back"
                for record in self.registry.snapshot()["documents"]
            )
        )

    def test_cancellation_rolls_back_documents_already_uploaded(self):
        first = self._file("first.txt", "first")
        second = self._file("second.txt", "second")
        client = _FakeLifecycleClient()
        cancel_event = threading.Event()
        original_wait = client.wait_for_document

        def wait_and_cancel(*args, **kwargs):
            result = original_wait(*args, **kwargs)
            cancel_event.set()
            return result

        client.wait_for_document = wait_and_cancel
        with self.assertRaises(r2r.R2RIngestionCancelled):
            r2r.ingest_paths_transaction(
                client,
                [str(first), str(second)],
                self.registry,
                "workspace",
                "content-signature",
                "source-id",
                "source-signature",
                cancel_event=cancel_event,
            )
        self.assertEqual(client.upload_count, 1)
        self.assertEqual(len(client.deleted), 1)
        self.assertEqual(
            self.registry.get_document(client.deleted[0])["status"],
            "rolled_back",
        )

    def test_registry_commit_failure_rolls_back_new_documents(self):
        path = self._file("new.txt", "new")
        client = _FakeLifecycleClient()
        with (
            patch.object(
                self.registry,
                "commit_active",
                side_effect=r2r.R2RRegistryError("commit failed"),
            ),
            self.assertRaises(r2r.R2RRegistryError),
        ):
            r2r.ingest_paths_transaction(
                client,
                [str(path)],
                self.registry,
                "workspace",
                "new-content",
                "new-source-id",
                "new-source-signature",
            )
        self.assertEqual(len(client.deleted), 1)
        self.assertEqual(
            self.registry.get_document(client.deleted[0])["status"],
            "rolled_back",
        )

    def test_idempotent_retry_reuses_ready_document_without_deleting_it(self):
        path = self._file("same.txt", "same-content")
        source_signature = "same-source-signature"
        document_id = r2r._deterministic_document_id(
            "workspace",
            source_signature,
            path.name,
            r2r._file_sha256(path),
        )
        record = {
            "document_id": document_id,
            "task_id": "task-existing",
            "content_signature": "same-content-signature",
            "source_id": "same-source-id",
            "source_signature": source_signature,
            "status": "ready",
            "created_at": r2r._utc_now(),
            "updated_at": r2r._utc_now(),
            "last_error": None,
            "endpoint_fingerprint": self.identity.endpoint_fingerprint,
            "credential_fingerprint": self.identity.credential_fingerprint,
            "file_name": path.name,
        }
        self.registry.upsert_document(record)
        self.registry.commit_active(
            [record],
            {
                "source_id": "same-source-id",
                "content_signature": "same-content-signature",
                "settings_signature": "settings",
                "source_signature": source_signature,
                "index_generation": 1,
                "display_name": path.name,
            },
        )
        client = _FakeLifecycleClient()
        records = r2r.ingest_paths_transaction(
            client,
            [str(path)],
            self.registry,
            "workspace",
            "same-content-signature",
            "same-source-id",
            source_signature,
            old_document_ids=[document_id],
        )
        self.assertEqual(records[0]["document_id"], document_id)
        self.assertEqual(client.upload_count, 0)
        self.assertEqual(client.deleted, [])
        self.assertEqual(self.registry.snapshot()["active_document_ids"], [document_id])

    def test_replacement_deletes_old_only_after_new_is_ready(self):
        old = self._seed_ready()
        new_file = self._file("new.txt", "new")
        client = _FakeLifecycleClient()
        records = r2r.ingest_paths_transaction(
            client,
            [str(new_file)],
            self.registry,
            "workspace",
            "new-content",
            "new-source-id",
            "new-source-signature",
            old_document_ids=[old["document_id"]],
        )
        self.assertEqual(
            client.events[-2:],
            [("ready", records[0]["document_id"]), ("delete", old["document_id"])],
        )
        snapshot = self.registry.snapshot()
        self.assertEqual(snapshot["active_document_ids"], [records[0]["document_id"]])
        self.assertEqual(
            self.registry.get_document(old["document_id"])["status"], "deleted"
        )

    def test_replacement_cleanup_failure_rolls_back_new_ids(self):
        old = self._seed_ready()
        new_file = self._file("new.txt", "new")
        client = _FakeLifecycleClient(fail_delete_ids={old["document_id"]})
        with self.assertRaises(r2r.R2RLifecycleError):
            r2r.ingest_paths_transaction(
                client,
                [str(new_file)],
                self.registry,
                "workspace",
                "new-content",
                "new-source-id",
                "new-source-signature",
                old_document_ids=[old["document_id"]],
            )
        self.assertEqual(
            self.registry.snapshot()["active_document_ids"], [old["document_id"]]
        )
        new_id = next(
            document_id
            for document_id in client.deleted
            if document_id != old["document_id"]
        )
        self.assertEqual(self.registry.get_document(new_id)["status"], "rolled_back")
        self.assertEqual(
            self.registry.get_document(old["document_id"])["status"],
            "delete_failed",
        )

    def test_reset_treats_404_as_success_and_preserves_registry(self):
        record = self._seed_ready()
        client = _FakeLifecycleClient()
        client.delete_document = lambda document_id: "missing"
        state = {"r2r_registry_path": str(self.registry.path)}
        result = r2r.delete_owned_remote_documents(
            state, client=client, registry=self.registry
        )
        self.assertTrue(result.success)
        self.assertEqual(result.delete_result.missing, (record["document_id"],))
        self.assertTrue(self.registry.path.exists())
        self.assertEqual(
            self.registry.get_document(record["document_id"])["status"],
            "deleted",
        )

    def test_reset_partial_failure_is_not_reported_as_success(self):
        first = self._seed_ready("old-1")
        second_record = dict(first)
        second_record["document_id"] = "old-2"
        self.registry.upsert_document(second_record)
        client = _FakeLifecycleClient(fail_delete_ids={second_record["document_id"]})
        result = r2r.delete_owned_remote_documents(
            {"r2r_registry_path": str(self.registry.path)},
            client=client,
            registry=self.registry,
        )
        self.assertFalse(result.success)
        self.assertEqual(
            self.registry.get_document(second_record["document_id"])["status"],
            "delete_failed",
        )
        self.assertEqual(
            self.registry.snapshot()["active_document_ids"],
            [first["document_id"]],
        )


class R2RStateAndRegistryTests(unittest.TestCase):
    def _ready_state(self, state_root, key="secret-key"):
        state = {
            "r2r_enabled": True,
            "r2r_base_url": "http://localhost:7272",
            "r2r_api_key": key,
            "r2r_registry_root": str(state_root),
            "llm_backend": "Ollama",
            "ollama_endpoint": "http://localhost:11434",
            "ollama_embedding_model": "embedding",
            "chunk_size": 256,
            "chunk_overlap_pct": 12,
            "index_generation": 1,
        }
        r2r.initialize_r2r_state(state)
        settings_signature = stable_digest(effective_indexing_settings(state))
        source = make_source_state(
            "r2r",
            source_id="source-1",
            content_signature="content-1",
            settings_signature=settings_signature,
            index_generation=1,
        )
        state["index_generation"] = 1
        mark_source_ready(state, source)
        state["r2r_document_ids"] = ["doc-1"]
        state["r2r_document_ids_signature"] = "source-signature"
        state["r2r_current_source_signature"] = "source-signature"
        registry = r2r.registry_for_state(state)
        record = {
            "document_id": "doc-1",
            "task_id": "task-1",
            "content_signature": "content-1",
            "source_id": "source-1",
            "source_signature": "source-signature",
            "status": "ready",
            "created_at": r2r._utc_now(),
            "updated_at": r2r._utc_now(),
            "last_error": None,
            "endpoint_fingerprint": r2r.get_client(state).identity.endpoint_fingerprint,
            "credential_fingerprint": r2r.get_client(
                state
            ).identity.credential_fingerprint,
            "file_name": "notes.txt",
        }
        registry.upsert_document(record)
        registry.commit_active(
            [record],
            {
                "source_id": "source-1",
                "content_signature": "content-1",
                "settings_signature": settings_signature,
                "source_signature": "source-signature",
                "index_generation": 1,
                "display_name": "notes.txt",
            },
        )
        return state, registry

    def test_state_pins_explicit_server_and_api_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = {"r2r_registry_root": temp_dir}
            r2r.initialize_r2r_state(state)
            self.assertEqual(state["r2r_server_contract"], "3.6.5")
            self.assertEqual(state["r2r_api_contract"], "v3")
            self.assertEqual(state["r2r_reference_sdk_version"], "3.6.5")
            self.assertEqual(state["r2r_contract"], r2r.R2R_CONTRACT)

    def test_endpoint_and_credential_changes_invalidate_ids_without_losing_registry(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            state, registry = self._ready_state(temp_dir)
            self.assertTrue(r2r.r2r_is_ready(state))
            state["r2r_api_key"] = "new-secret-key"
            r2r.initialize_r2r_state(state)
            self.assertFalse(r2r.r2r_is_ready(state))
            self.assertEqual(state["r2r_document_ids"], [])
            self.assertIn("doc-1", registry.snapshot()["documents"][0]["document_id"])
            state["r2r_base_url"] = "http://other-r2r:7272"
            r2r.initialize_r2r_state(state)
            self.assertEqual(state["r2r_document_ids"], [])
            serialized = registry.path.read_text(encoding="utf-8")
            self.assertNotIn("secret-key", serialized)
            self.assertNotIn("new-secret-key", serialized)

    def test_reload_restores_ready_registry_for_same_workspace_and_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original, _ = self._ready_state(temp_dir)
            reloaded = {
                "r2r_enabled": True,
                "r2r_base_url": original["r2r_base_url"],
                "r2r_api_key": original["r2r_api_key"],
                "r2r_workspace_id": original["r2r_workspace_id"],
                "r2r_registry_root": temp_dir,
                "llm_backend": "Ollama",
                "ollama_endpoint": original["ollama_endpoint"],
                "ollama_embedding_model": original["ollama_embedding_model"],
                "chunk_size": original["chunk_size"],
                "chunk_overlap_pct": original["chunk_overlap_pct"],
                "index_generation": 0,
            }
            r2r.initialize_r2r_state(reloaded)
            self.assertEqual(reloaded["r2r_document_ids"], ["doc-1"])
            self.assertTrue(r2r.r2r_is_ready(reloaded))

    def test_disable_and_reenable_invalidate_readiness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state, registry = self._ready_state(temp_dir)
            state["r2r_enabled"] = False
            r2r.initialize_r2r_state(state)
            self.assertEqual(state["r2r_document_ids"], [])
            self.assertEqual(registry.snapshot()["active_document_ids"], [])
            state["r2r_enabled"] = True
            r2r.initialize_r2r_state(state)
            self.assertFalse(r2r.r2r_is_ready(state))

    def test_source_signature_change_invalidates_without_sending_old_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state, _ = self._ready_state(temp_dir)
            state["r2r_current_source_signature"] = "new-source-signature"
            self.assertFalse(r2r.r2r_is_ready(state))

    def test_workspace_id_reloads_without_persisting_credentials(self):
        state = {
            "r2r_workspace_id": "workspace-123",
            "r2r_api_key": "never-persist-this",
        }
        persisted = serialize_persisted_settings(state)
        self.assertEqual(persisted["r2r_workspace_id"], "workspace-123")
        self.assertNotIn("r2r_api_key", persisted)

    def test_workspace_path_is_traversal_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = r2r.R2RDocumentRegistry.for_workspace(
                "../../escape",
                r2r.R2RIdentity("endpoint", None),
                state_root=root,
            )
            self.assertEqual(registry.path.parent, root.resolve())
            self.assertNotIn("..", registry.path.name)
            self.assertTrue(
                r2r.R2R_REGISTRY_FILENAME_PATTERN.fullmatch(registry.path.name)
            )

    def test_concurrent_registry_writes_preserve_every_owned_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = r2r.R2RDocumentRegistry.for_workspace(
                "workspace",
                r2r.R2RIdentity("endpoint", "credential"),
                state_root=temp_dir,
            )
            errors = []

            def write(index):
                try:
                    registry.upsert_document(
                        {
                            "document_id": f"doc-{index}",
                            "task_id": None,
                            "content_signature": f"content-{index}",
                            "source_id": f"source-{index}",
                            "source_signature": f"signature-{index}",
                            "status": "uploaded",
                            "created_at": r2r._utc_now(),
                            "updated_at": r2r._utc_now(),
                            "last_error": None,
                            "endpoint_fingerprint": "endpoint",
                            "credential_fingerprint": "credential",
                            "file_name": f"{index}.txt",
                        }
                    )
                except (r2r.R2RRegistryError, OSError) as err:
                    errors.append(err)

            threads = [
                threading.Thread(target=write, args=(index,)) for index in range(12)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(
                {
                    document["document_id"]
                    for document in registry.snapshot()["documents"]
                },
                {f"doc-{index}" for index in range(12)},
            )

    def test_project_reset_defaults_to_remote_deletion(self):
        remote_result = r2r.R2RRemoteResetResult(r2r.R2RDeleteResult())
        with patch(
            "components.page_state.r2r.delete_owned_remote_documents",
            return_value=remote_result,
        ) as delete_remote:
            result = perform_project_reset({})
        delete_remote.assert_called_once()
        self.assertTrue(result["remote_delete_success"])

    def test_project_reset_offers_explicit_local_only_mode(self):
        with patch(
            "components.page_state.r2r.delete_owned_remote_documents"
        ) as delete_remote:
            result = perform_project_reset({}, delete_remote=False)
        delete_remote.assert_not_called()
        self.assertIsNone(result["remote_delete_success"])
        self.assertFalse(result["remote"]["attempted"])

    def test_only_owned_work_directory_is_cleanup_eligible(self):
        self.assertTrue(
            r2r._owned_work_dir(Path.cwd() / "data" / "work" / "r2r-upload-example")
        )
        self.assertFalse(r2r._owned_work_dir(Path.cwd() / "data"))
        self.assertFalse(r2r._owned_work_dir(Path.cwd() / "docmind.log"))


@unittest.skipUnless(
    os.getenv(r2r.R2R_STATE_DIR_ENV) == "live",
    "set DOCMIND_R2R_STATE_DIR=live for the opt-in R2R integration test",
)
class R2RLiveIntegrationTests(unittest.TestCase):
    def test_live_v3_upload_chat_and_cleanup(self):
        base_url = os.getenv("DOCMIND_R2R_BASE_URL", r2r.DEFAULT_R2R_BASE_URL)
        api_key = os.getenv("DOCMIND_R2R_API_KEY", "")
        client = r2r.R2RClient(base_url=base_url, api_key=api_key)
        self.assertTrue(client.health())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "live.txt"
            path.write_text("DocMind live contract test: 42.", encoding="utf-8")
            registry = r2r.R2RDocumentRegistry.for_workspace(
                str(uuid_value := uuid_test_value()),
                client.identity,
                state_root=root / "registry",
            )
            records = r2r.ingest_paths_transaction(
                client,
                [str(path)],
                registry,
                str(uuid_value),
                "live-content",
                "live-source",
                "live-signature",
                poll_timeout=300,
            )
            try:
                result = client.rag(
                    "What number is mentioned?",
                    [records[0]["document_id"]],
                    top_k=3,
                )
                self.assertIn("42", result.answer)
            finally:
                client.delete_document(records[0]["document_id"])


def uuid_test_value():
    import uuid

    return uuid.uuid4()


if __name__ == "__main__":
    unittest.main()
