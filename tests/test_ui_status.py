import unittest
from types import SimpleNamespace
from unittest.mock import patch

from components import chatbox, status
from utils.retrieval_map import RetrievalMapConfig, build_document_map


class _RecordingStreamlit:
    def __init__(self, state=None):
        self.session_state = {} if state is None else state
        self.calls = {
            name: []
            for name in ("markdown", "caption", "success", "info", "error", "warning")
        }

    def _record(self, name):
        def inner(value, *args, **kwargs):
            self.calls[name].append(value)

        return inner

    def markdown(self, value, *args, **kwargs):
        self._record("markdown")(value)

    def caption(self, value, *args, **kwargs):
        self._record("caption")(value)

    def success(self, value, *args, **kwargs):
        self._record("success")(value)

    def info(self, value, *args, **kwargs):
        self._record("info")(value)

    def error(self, value, *args, **kwargs):
        self._record("error")(value)

    def warning(self, value, *args, **kwargs):
        self._record("warning")(value)

    def text(self, name):
        return " ".join(str(value) for value in self.calls[name])


def _state(**overrides):
    state = {
        "llm_backend": "Ollama",
        "ollama_endpoint": "http://localhost:11434",
        "ollama_models": [],
        "r2r_enabled": False,
        "r2r_document_ids": [],
        "retrieval_map_enabled": True,
        "query_engine": None,
    }
    state.update(overrides)
    return state


class StatusResolutionTests(unittest.TestCase):
    def test_direct_mode_without_an_index(self):
        resolved = status.resolve_status(_state())
        self.assertEqual(resolved["mode"], status.MODE_DIRECT)
        self.assertEqual(resolved["mode_label"], "Direct chat")
        self.assertFalse(resolved["index_ready"])
        self.assertFalse(resolved["map"]["built"])

    def test_model_label_names_the_backend_and_model(self):
        resolved = status.resolve_status(
            _state(ollama_models=["llama3:8b"], selected_model="llama3:8b")
        )
        self.assertIn("Ollama", resolved["model"])
        self.assertIn("llama3:8b", resolved["model"])

    def test_r2r_mode_requires_ready_documents(self):
        pending = status.resolve_status(
            _state(r2r_enabled=True, r2r_document_ids=["a"])
        )
        self.assertEqual(pending["mode"], status.MODE_DIRECT)
        self.assertIn("not ready", pending["note"])

    def test_map_summary_reads_a_built_map(self):
        from research.fixtures import build_retriever

        document_map = build_document_map(
            build_retriever(),
            config=RetrievalMapConfig(chunks_per_section=2),
        )
        resolved = status.resolve_status(_state(retrieval_map=document_map))
        self.assertTrue(resolved["map"]["built"])
        self.assertEqual(resolved["map"]["sections"], len(document_map.sections))
        self.assertEqual(resolved["map"]["nodes"], int(document_map.input_node_count))

    def test_map_summary_tolerates_a_missing_or_broken_map(self):
        self.assertFalse(status.resolve_status(_state())["map"]["built"])
        self.assertFalse(
            status.resolve_status(_state(retrieval_map=object()))["map"]["built"]
        )

    def test_long_model_and_source_values_are_clipped(self):
        resolved = status.resolve_status(
            _state(
                selected_model="m" * 200,
                active_source={
                    "kind": "local",
                    "display_name": "d" * 200,
                    "status": "ready",
                },
            )
        )
        self.assertLessEqual(len(resolved["model"]), 90)
        self.assertLessEqual(len(resolved["source_display"]), 48)
        self.assertTrue(resolved["source_display"].endswith(chr(0x2026)))


class StatusRenderingTests(unittest.TestCase):
    def test_strip_marks_the_mode_and_reports_the_map(self):
        streamlit = _RecordingStreamlit()
        document_map = SimpleNamespace(
            sections=(1, 2, 3), input_node_count=9, truncated=True
        )
        with patch("components.status.st", streamlit):
            status.render_status_strip(
                status.resolve_status(_state(retrieval_map=document_map))
            )
        markup = streamlit.text("markdown")
        self.assertIn("dm-strip", markup)
        self.assertIn("Direct chat", markup)
        self.assertIn("Map 3 sections+", markup)
        self.assertNotIn("dark", markup)

    def test_strip_warns_when_the_map_is_missing_for_a_ready_index(self):
        streamlit = _RecordingStreamlit()
        resolved = status.resolve_status(
            _state(
                query_engine=object(),
                active_source={"kind": "local", "status": "ready"},
            )
        )
        resolved["index_ready"] = True
        with patch("components.status.st", streamlit):
            status.render_status_strip(resolved)
        markup = streamlit.text("markdown")
        self.assertIn("Map not built", markup)
        self.assertIn("dm-warn", markup)

    def test_badge_escapes_untrusted_model_names(self):
        markup = status._badge('<img src=x onerror="alert(1)">', "idle")
        self.assertNotIn("<img", markup)
        self.assertIn("&lt;img", markup)
        self.assertNotIn('onerror="', markup)

    def test_strip_escapes_a_hostile_model_name(self):
        streamlit = _RecordingStreamlit()
        resolved = status.resolve_status(
            _state(selected_model='<script>alert("x")</script>')
        )
        with patch("components.status.st", streamlit):
            status.render_status_strip(resolved)
        markup = streamlit.text("markdown")
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)

    def test_mode_card_keeps_the_existing_wording(self):
        streamlit = _RecordingStreamlit()
        with patch("components.status.st", streamlit):
            status.render_mode_card(status.resolve_status(_state()))
        self.assertEqual(len(streamlit.calls["info"]), 1)
        self.assertIn("Chat Mode", streamlit.calls["info"][0])


class RouteSummaryTests(unittest.TestCase):
    def test_empty_route_has_no_summary(self):
        self.assertEqual(chatbox._route_summary({}), "")
        self.assertEqual(chatbox._route_summary(None), "")

    def test_summary_reports_sections_steps_and_tokens(self):
        summary = chatbox._route_summary(
            {
                "selected_section_ids": ["a", "b", "c"],
                "step_count": 2,
                "metrics": {"selected_evidence_tokens": 86},
                "reflection_flag": True,
            }
        )
        self.assertIn("3 sections routed", summary)
        self.assertIn("2 steps", summary)
        self.assertIn("86 context tokens", summary)
        self.assertIn("reflected", summary)

    def test_summary_uses_singular_wording_for_one_section(self):
        summary = chatbox._route_summary(
            {"selected_section_ids": ["a"], "step_count": 1}
        )
        self.assertIn("1 section routed", summary)
        self.assertIn("1 step", summary)
        self.assertNotIn("reflected", summary)

    def test_summary_ignores_a_route_that_selected_nothing(self):
        self.assertEqual(
            chatbox._route_summary({"selected_section_ids": [], "step_count": 1}), ""
        )

    def test_summary_survives_a_hostile_route_shape(self):
        for route in (
            {"selected_section_ids": "not-a-list"},
            {"selected_section_ids": [None, 1], "step_count": "many"},
            {"selected_section_ids": ["a"], "metrics": "broken"},
        ):
            self.assertIsInstance(chatbox._route_summary(route), str)


if __name__ == "__main__":
    unittest.main()
