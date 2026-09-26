import math
import os
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from utils.source_state import (
    active_index_matches_settings,
    effective_indexing_settings,
    ensure_active_source,
    initial_source_state,
    make_source_state,
    mark_active_index_stale_if_needed,
    mark_source_failed,
    mark_source_pending,
    mark_source_ready,
    normalize_chunk_settings,
    report_matches_active_source,
    stable_digest,
    tag_report,
)


class SourceStateRecordTests(unittest.TestCase):
    def test_initial_record_contains_authoritative_contract_fields(self):
        state = {}
        record = ensure_active_source(state)

        self.assertEqual(state["active_source"], record)
        for field in (
            "id",
            "kind",
            "display_name",
            "source_uri",
            "content_signature",
            "settings_signature",
            "index_generation",
            "status",
            "cache_key",
            "created_at",
            "error",
        ):
            self.assertIn(field, record)

    def test_state_transitions_are_explicit_and_generation_is_preserved(self):
        state = {"index_generation": 4}
        pending = make_source_state(
            "github",
            source_id="repo-1",
            display_name="owner/repo",
            content_signature="content",
            settings_signature="settings",
            index_generation=5,
        )
        mark_source_pending(state, pending)
        self.assertEqual(state["active_source"]["status"], "pending")
        ready = mark_source_ready(state, pending)
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["index_generation"], 5)
        mark_source_failed(state, "embedding failed")
        self.assertEqual(state["active_source"]["status"], "failed")
        self.assertIn("embedding failed", state["active_source"]["error"])

    def test_reset_returns_a_fresh_record(self):
        state = {"active_source": make_source_state("local"), "index_generation": 3}
        state["active_source"]["id"] = "old"
        from utils.source_state import reset_active_source

        reset_active_source(state)
        self.assertEqual(state["active_source"], initial_source_state())
        self.assertEqual(state["index_generation"], 0)


class ChunkSettingsTests(unittest.TestCase):
    def test_overlap_and_percentage_are_normalized_together(self):
        state = {"chunk_size": "256", "chunk_overlap": 32, "chunk_overlap_pct": 12}
        self.assertEqual(
            normalize_chunk_settings(state),
            {"chunk_size": 256, "chunk_overlap": 30, "chunk_overlap_pct": 12},
        )

    def test_non_finite_and_out_of_bounds_values_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_chunk_settings({"chunk_size": value})
        with self.assertRaises(ValueError):
            normalize_chunk_settings({"chunk_size": 10, "chunk_overlap_pct": 50.1})
        with self.assertRaises(ValueError):
            normalize_chunk_settings({"chunk_size": 10, "chunk_overlap": 10})


class SettingsAndStalenessTests(unittest.TestCase):
    def _state(self, kind):
        state = {
            "llm_backend": "Ollama",
            "ollama_endpoint": "http://one",
            "ollama_embedding_model": "embedding-a",
            "chunk_size": 256,
            "chunk_overlap_pct": 12,
            "index_generation": 1,
        }
        source = make_source_state(
            kind,
            source_id=f"{kind}-1",
            content_signature="content",
            settings_signature=stable_digest(effective_indexing_settings(state)),
            index_generation=1,
        )
        state["active_source"] = source
        state["index_generation"] = 1
        mark_source_ready(state, source)
        return state

    def test_chat_only_changes_do_not_change_index_settings(self):
        state = self._state("local")
        before = effective_indexing_settings(state)
        state["selected_model"] = "another-chat-model"
        state["answer_style"] = "Detailed"
        self.assertEqual(effective_indexing_settings(state), before)
        self.assertTrue(active_index_matches_settings(state))

    def test_embedding_and_chunk_changes_stale_each_source_kind(self):
        for kind in ("local", "github", "website"):
            with self.subTest(kind=kind):
                state = self._state(kind)
                state["ollama_embedding_model"] = "embedding-b"
                self.assertFalse(active_index_matches_settings(state))
                self.assertFalse(mark_active_index_stale_if_needed(state))
                self.assertEqual(state["active_source"]["status"], "stale")

    def test_endpoint_change_stales_source(self):
        state = self._state("github")
        state["ollama_endpoint"] = "http://two"
        mark_active_index_stale_if_needed(state)
        self.assertEqual(state["active_source"]["status"], "stale")


class ReportAssociationTests(unittest.TestCase):
    def test_report_is_renderable_only_for_matching_source_generation(self):
        state = {"active_source": make_source_state("local", index_generation=2)}
        report = tag_report([{"filename": "a.txt", "status": "loaded"}], "local-1", 2)
        self.assertTrue(report_matches_active_source(report, state, "local-1", 2))
        self.assertFalse(report_matches_active_source(report, state, "local-1", 1))
        self.assertFalse(report_matches_active_source(report, state, "other", 2))


class TransactionalPipelineTests(unittest.TestCase):
    def test_failed_replacement_restores_committed_generation_and_cleans_work_dir(self):
        import utils.rag_pipeline as rag_pipeline

        class FakeUpload:
            name = "notes.txt"
            size = 1
            type = "text/plain"

            def getvalue(self):
                return b"x"

            def getbuffer(self):
                return b"x"

        state = {
            "llm_backend": "Ollama",
            "ollama_endpoint": "http://localhost:11434",
            "ollama_embedding_model": "embedding-a",
            "chunk_size": 256,
            "chunk_overlap_pct": 12,
            "selected_model": "chat-a",
            "system_prompt": "",
            "index_generation": 1,
            "query_engine": "old-engine",
            "retriever": "old-retriever",
            "documents": ["old-document"],
            "r2r_document_ids": ["old-r2r"],
            "file_extraction_report": [],
            "extraction_report": [],
            "_session_work_dirs": [],
        }
        old_source = make_source_state(
            "local",
            source_id="old-source",
            content_signature="old-content",
            settings_signature=stable_digest(effective_indexing_settings(state)),
            index_generation=1,
        )
        mark_source_ready(state, old_source)
        state["active_source"]["id"] = "old-source"
        state["active_source"]["source_id"] = "old-source"

        class FakeStreamlit:
            def __init__(self):
                self.session_state = state

            def spinner(self, *args, **kwargs):
                return nullcontext()

            def exception(self, error):
                pass

            def stop(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, "operation")
            os.mkdir(work_dir)
            with (
                patch.object(rag_pipeline, "st", FakeStreamlit()),
                patch.object(rag_pipeline, "render_pipeline_status"),
                patch.object(rag_pipeline, "render_embedding_progress"),
                patch.object(rag_pipeline, "render_completed_ingestion_status"),
                patch.object(
                    rag_pipeline.llama_index,
                    "load_documents",
                    return_value=(["new-document"], []),
                ),
                patch.object(rag_pipeline.llama_index, "setup_embedding_model"),
                patch.object(
                    rag_pipeline.llama_index,
                    "create_query_engine",
                    side_effect=RuntimeError("embedding failed"),
                ),
                patch.object(
                    rag_pipeline.ollama, "verify_chat_model", return_value=True
                ),
                patch.object(rag_pipeline.ollama, "create_llm", return_value="new-llm"),
                patch.object(
                    rag_pipeline.func,
                    "create_ingestion_work_dir",
                    return_value=work_dir,
                ),
            ):
                error = rag_pipeline.rag_pipeline(
                    [FakeUpload()],
                    source_kind="local",
                    content_signature="new-content",
                )

            self.assertIsInstance(error, RuntimeError)
            self.assertFalse(os.path.exists(work_dir))
            self.assertEqual(state["active_source"]["id"], "old-source")
            self.assertEqual(state["active_source"]["status"], "ready")
            self.assertEqual(state["index_generation"], 1)
            self.assertEqual(state["query_engine"], "old-engine")
            self.assertEqual(state["retriever"], "old-retriever")
            self.assertEqual(state["documents"], ["old-document"])
            self.assertEqual(state["r2r_document_ids"], ["old-r2r"])


class IngestionLockTests(unittest.TestCase):
    def test_process_wide_lock_serializes_two_ingestion_threads(self):
        from utils.llama_index import ingestion_lock

        active = 0
        maximum = 0
        state_lock = threading.Lock()
        entered = threading.Event()
        release = threading.Event()

        def worker():
            nonlocal active, maximum
            with ingestion_lock():
                with state_lock:
                    active += 1
                    maximum = max(maximum, active)
                entered.set()
                release.wait(2)
                with state_lock:
                    active -= 1

        first = threading.Thread(target=worker)
        second = threading.Thread(target=worker)
        first.start()
        self.assertTrue(entered.wait(2))
        second.start()
        time.sleep(0.05)
        with state_lock:
            observed = active
        release.set()
        first.join(2)
        second.join(2)
        self.assertEqual(observed, 1)
        self.assertEqual(maximum, 1)


if __name__ == "__main__":
    unittest.main()
