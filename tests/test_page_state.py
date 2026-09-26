import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from components.page_state import (
    default_chat_model,
    ensure_valid_model_selections,
    set_initial_state,
)


class PageStateTests(unittest.TestCase):
    def test_default_chat_model_prefers_gemma4_latest(self):
        self.assertEqual(
            default_chat_model(["llama3:8b", "gemma4:latest"]),
            "gemma4:latest",
        )

    def test_ensure_valid_model_selections_repairs_inconsistent_chat_model(self):
        state = {
            "selected_model": "missing:latest",
            "ollama_models": ["gemma4:latest"],
        }
        ensure_valid_model_selections(state)
        self.assertEqual(state["selected_model"], "gemma4:latest")

    def test_ensure_valid_model_selections_clears_missing_ollama_embedding_model(self):
        state = {
            "selected_model": "gemma4:latest",
            "ollama_models": ["gemma4:latest"],
            "ollama_embedding_model": "embeddinggemma",
            "ollama_embedding_models": [],
        }
        ensure_valid_model_selections(state)
        self.assertIsNone(state["ollama_embedding_model"])

    def test_ensure_valid_model_selections_prefers_embeddinggemma_latest(self):
        state = {
            "selected_model": "gemma4:latest",
            "ollama_models": ["gemma4:latest"],
            "ollama_embedding_model": "missing:latest",
            "ollama_embedding_models": [
                "nomic-embed-text:latest",
                "embeddinggemma:latest",
            ],
        }
        ensure_valid_model_selections(state)
        self.assertEqual(state["ollama_embedding_model"], "embeddinggemma:latest")

    def test_initial_state_uses_persisted_endpoint_before_model_discovery(self):
        state = {}

        def restore_from_local_storage():
            state.update(
                {
                    "browser_settings_restored": True,
                    "ollama_endpoint": "https://192.168.4.2:11434",
                    "selected_model": "gemma4:latest",
                    "ollama_embedding_model": "embeddinggemma",
                }
            )

        with (
            patch("components.page_state.st.session_state", state),
            patch(
                "components.page_state.restore_settings_from_browser_storage",
                side_effect=restore_from_local_storage,
            ),
            patch(
                "components.page_state.get_models", return_value=["gemma4:latest"]
            ) as get_models,
            patch(
                "components.page_state.get_embedding_models",
                return_value=["embeddinggemma"],
            ) as get_embedding_models,
        ):
            set_initial_state()
        get_models.assert_called_once()
        get_embedding_models.assert_called_once()
        self.assertEqual(state["ollama_endpoint"], "https://192.168.4.2:11434")
        self.assertEqual(state["ollama_models_endpoint"], "https://192.168.4.2:11434")
        self.assertEqual(
            state["ollama_embedding_models_endpoint"],
            "https://192.168.4.2:11434",
        )
        self.assertEqual(state["selected_model"], "gemma4:latest")
        self.assertEqual(state["ollama_embedding_model"], "embeddinggemma")

    def test_project_reset_removes_only_owned_work_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            owned = root / "owned"
            failed = root / "failed"
            shared = root / "data" / "shared.txt"
            owned.mkdir()
            failed.mkdir()
            shared.parent.mkdir()
            shared.write_text("keep", encoding="utf-8")
            state = {
                "active_source": {"kind": "local", "status": "ready"},
                "query_engine": object(),
                "retriever": object(),
                "documents": [object()],
                "file_list": [object()],
                "messages": [],
                "_session_work_dirs": [],
            }
            from components.page_state import _remove_dir_retry as real_remove

            def remove_owned(path):
                if Path(path) == failed:
                    return False
                return real_remove(path)

            with patch(
                "components.page_state._remove_dir_retry",
                side_effect=remove_owned,
            ):
                result = __import__(
                    "components.page_state", fromlist=["perform_project_reset"]
                ).perform_project_reset(
                    state, owned_dirs=[owned, failed, root / "missing"]
                )

            self.assertFalse(owned.exists())
            self.assertTrue(failed.exists())
            self.assertTrue(shared.exists())
            self.assertEqual(result["failed"], [str(failed)])
            self.assertEqual(state["active_source"]["kind"], None)
            self.assertIsNone(state["query_engine"])

    def test_project_reset_does_not_call_removal_for_absent_owned_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {}
            with patch("components.page_state._remove_dir_retry") as remove:
                result = __import__(
                    "components.page_state", fromlist=["perform_project_reset"]
                ).perform_project_reset(state, owned_dirs=[Path(tmpdir) / "missing"])
        remove.assert_not_called()
        self.assertEqual(result["failed"], [])

        state = {
            "browser_settings_restored": True,
            "ollama_endpoint": "",
            "selected_model": "gemma4:latest",
            "ollama_embedding_model": "embeddinggemma",
        }
        with (
            patch("components.page_state.st.session_state", state),
            patch("components.page_state.restore_settings_from_browser_storage"),
            patch(
                "components.page_state.get_models", return_value=["gemma4:latest"]
            ) as get_models,
            patch(
                "components.page_state.get_embedding_models",
                return_value=["embeddinggemma"],
            ) as get_embedding_models,
        ):
            set_initial_state()
        get_models.assert_called_once()
        get_embedding_models.assert_called_once()
        self.assertEqual(state["ollama_endpoint"], "http://localhost:11434")
        self.assertEqual(state["ollama_models_endpoint"], "http://localhost:11434")
        self.assertEqual(
            state["ollama_embedding_models_endpoint"],
            "http://localhost:11434",
        )


if __name__ == "__main__":
    unittest.main()
