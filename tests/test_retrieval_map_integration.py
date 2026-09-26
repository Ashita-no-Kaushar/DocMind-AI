import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core.schema import NodeWithScore, TextNode

from components import chatbox, page_state
from utils import browser_settings
from utils import ollama as ollama_module
from utils.llama_index import HybridRetriever
from utils.retrieval_map import (
    MAP_SETTING_DEFAULTS,
    MAP_SETTING_RANGES,
    DocumentMap,
    build_document_map,
    normalize_map_settings,
    retriever_supports_map,
)


class _Docstore:
    def __init__(self, nodes):
        self.docs = {node.node_id: node for node in nodes}

    def get_node(self, node_id):
        return self.docs[node_id]


class _VectorRetriever:
    def __init__(self, results):
        self.results = list(results)

    def retrieve(self, query):
        return list(self.results)


class _SpyRetriever(HybridRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs.get("allowed_node_ids")))
        return super().retrieve(query, **kwargs)


class _EmptyRetriever(_SpyRetriever):
    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs.get("allowed_node_ids")))
        return []


class _FakeLlm:
    def __init__(self, completion=None, completion_error=None, fail_stream=False):
        self._tokenizer = SimpleNamespace(encode=lambda value: value.split())
        self.complete_calls = []
        self.stream_calls = []
        self.completion = completion
        self.completion_error = completion_error
        self.fail_stream = fail_stream

    def complete(self, prompt, **kwargs):
        self.complete_calls.append(prompt)
        if self.completion_error is not None:
            raise self.completion_error
        return SimpleNamespace(
            text=self.completion(prompt) if self.completion else "[]"
        )

    def stream_chat(self, messages):
        self.stream_calls.append(messages)
        if self.fail_stream:
            raise RuntimeError("stream failed")
        yield SimpleNamespace(delta="grounded answer")


class _FakeStreamlit:
    def __init__(self, state):
        self.session_state = state
        self.captions = []

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

    def warning(self, *args, **kwargs):
        pass

    def button(self, *args, **kwargs):
        return False


def _node(file_name, index, body=None):
    text = body or (
        f"{file_name} chunk {index} topic {index}alpha keyword {index * 7} "
        "contains operational details and unique terminology"
    )
    return TextNode(
        text=text,
        id_=f"{file_name}-{index}",
        metadata={"file_name": file_name, "start_char_idx": index * 100},
    )


def _retriever(files=(("alpha.txt", 3), ("beta.txt", 3)), **kwargs):
    nodes = []
    for file_name, count in files:
        nodes.extend(_node(file_name, index) for index in range(count))
    docstore = _Docstore(nodes)
    corpus = [node.node_id for node in nodes]
    vector = _VectorRetriever([NodeWithScore(node=node, score=0.85) for node in nodes])
    return _SpyRetriever(
        vector,
        docstore,
        corpus,
        top_k=kwargs.pop("top_k", 3),
        similarity_cutoff=kwargs.pop("similarity_cutoff", 0.3),
        **kwargs,
    )


def _empty_retriever():
    nodes = [_node("alpha.txt", index) for index in range(3)]
    return _EmptyRetriever(
        _VectorRetriever([]),
        _Docstore(nodes),
        [node.node_id for node in nodes],
        top_k=3,
        similarity_cutoff=0.3,
    )


def _state(retriever, **overrides):
    state = {
        "messages": [],
        "system_prompt": "system",
        "retriever": retriever,
        "query_engine": SimpleNamespace(_retriever=retriever),
        "selected_model": "model",
        "ollama_endpoint": "http://localhost:11434",
        "llm_backend": "Ollama",
        "top_k": 3,
        "similarity_cutoff": 0.3,
        "eco_mode": False,
    }
    state.update(overrides)
    return state


def _ask(prompt, state, llm=None, route_sink=None):
    llm = llm or _FakeLlm()
    streamlit = _FakeStreamlit(state)
    with (
        patch.object(ollama_module, "st", streamlit),
        patch.object(ollama_module, "_session_llm", return_value=llm),
    ):
        answer = "".join(
            ollama_module.context_chat(
                prompt, state["query_engine"], route_sink=route_sink
            )
        )
    return answer, llm


def _payload_ids(prompt):
    request = prompt.strip().splitlines()[-1]
    return [section["id"] for section in json.loads(request)["candidate_sections"]]


class MapSessionSettingsTests(unittest.TestCase):
    def test_defaults_enable_deterministic_map_routing_without_secrets(self):
        settings = normalize_map_settings({})

        self.assertEqual(settings, dict(MAP_SETTING_DEFAULTS))
        self.assertTrue(settings["retrieval_map_enabled"])
        self.assertEqual(settings["retrieval_map_planner_mode"], "deterministic")
        self.assertFalse(settings["retrieval_map_measure_baseline"])
        self.assertNotIn("api_key", settings)
        self.assertFalse(any("key" in key for key in settings))

    def test_hostile_values_are_bounded_and_normalized(self):
        settings = normalize_map_settings(
            {
                "retrieval_map_enabled": "off",
                "retrieval_map_planner_mode": "chain-of-thought",
                "retrieval_map_measure_baseline": "yes",
                "retrieval_map_chunks_per_section": 10_000,
                "retrieval_map_max_selected_sections": -4,
                "retrieval_map_neighbor_sections": 99,
                "retrieval_map_max_results": "eight",
            }
        )

        low, high = MAP_SETTING_RANGES["retrieval_map_chunks_per_section"]
        self.assertEqual(settings["retrieval_map_chunks_per_section"], high)
        self.assertEqual(settings["retrieval_map_max_selected_sections"], 1)
        self.assertEqual(settings["retrieval_map_neighbor_sections"], 4)
        self.assertEqual(settings["retrieval_map_max_results"], 8)
        self.assertFalse(settings["retrieval_map_enabled"])
        self.assertTrue(settings["retrieval_map_measure_baseline"])
        self.assertEqual(settings["retrieval_map_planner_mode"], "deterministic")
        self.assertEqual(low, 1)

        floors = normalize_map_settings(
            {
                "retrieval_map_neighbor_sections": -1,
                "retrieval_map_max_selected_sections": "not-a-number",
                "retrieval_map_measure_baseline": 1,
            }
        )
        self.assertEqual(floors["retrieval_map_neighbor_sections"], 0)
        self.assertEqual(floors["retrieval_map_max_selected_sections"], 4)
        self.assertFalse(floors["retrieval_map_measure_baseline"])

    def test_page_state_initializes_bounded_map_defaults(self):
        state = {"retrieval_map_max_results": 999, "retrieval_map_enabled": "no"}

        with (
            patch.object(page_state.st, "session_state", state),
            patch.object(page_state, "restore_settings_from_browser_storage"),
            patch.object(page_state, "get_models", return_value=[]),
            patch.object(page_state, "get_embedding_models", return_value=[]),
        ):
            page_state.set_initial_state()

        self.assertEqual(state["retrieval_map_max_results"], 50)
        self.assertFalse(state["retrieval_map_enabled"])
        self.assertEqual(state["retrieval_map_planner_mode"], "deterministic")
        self.assertIsNone(state["retrieval_map"])
        self.assertEqual(state["last_retrieval_route"], {})


class MapRoutedRetrievalTests(unittest.TestCase):
    def test_local_hybrid_defaults_to_deterministic_map_routing(self):
        retriever = _retriever()
        state = _state(retriever)
        route = {}

        answer, _llm = _ask("alpha chunk 1", state, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertTrue(route["selected_section_ids"])
        self.assertTrue(route["planned_section_ids"])
        self.assertEqual(route["planner_mode"], "deterministic")
        self.assertEqual(route["planner_card_token_estimate"], 0)
        self.assertEqual(
            route["token_accounting"]["token_basis"],
            (ollama_module.TOKEN_ACCOUNTING_BASIS),
        )
        self.assertEqual(state["last_retrieval_route"], route)
        self.assertTrue(
            all(call[1] is not None for call in retriever.calls),
            retriever.calls,
        )
        json.dumps(route, allow_nan=False)

    def test_deterministic_mode_makes_no_extra_llm_calls(self):
        retriever = _retriever()
        state = _state(retriever, retrieval_map_planner_mode="deterministic")
        llm = _FakeLlm()

        _answer, _llm = _ask("alpha chunk 1", state, llm=llm)

        self.assertEqual(llm.complete_calls, [])
        self.assertEqual(len(llm.stream_calls), 1)

    def test_disabled_agent_keeps_the_existing_raw_retrieval_path(self):
        retriever = _retriever()
        state = _state(retriever, retrieval_map_enabled=False)
        route = {}

        answer, llm = _ask("alpha chunk 1", state, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(route, {})
        self.assertEqual(state["last_retrieval_route"], {})
        self.assertEqual([call[1] for call in retriever.calls], [None])
        self.assertEqual(llm.complete_calls, [])

    def test_two_query_candidates_are_preserved_on_the_raw_path(self):
        node = TextNode(
            text="Refunds are available for 30 days after purchase.",
            metadata={"file_name": "policy.txt"},
        )

        class FollowupRetriever:
            def __init__(self):
                self.queries = []

            def retrieve(self, query):
                self.queries.append(query)
                if len(self.queries) == 1:
                    return []
                return [NodeWithScore(node=node, score=0.8)]

        retriever = FollowupRetriever()
        state = _state(
            retriever,
            messages=[
                {"role": "user", "content": "What is the refund policy?"},
                {"role": "user", "content": "What about electronics?"},
            ],
        )
        retriever_supports_map(retriever)
        self.assertFalse(retriever_supports_map(retriever))
        route = {}

        answer, _llm = _ask("What about electronics?", state, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(len(retriever.queries), 2)
        self.assertEqual(route, {})

    def test_non_map_retriever_still_answers(self):
        node = TextNode(
            text="The warranty covers 24 months of parts.",
            metadata={"file_name": "warranty.txt"},
        )
        retriever = SimpleNamespace(
            retrieve=lambda query: [NodeWithScore(node=node, score=0.9)]
        )
        state = _state(retriever)
        route = {}

        answer, _llm = _ask("warranty terms", state, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(route, {})
        self.assertEqual(state["last_rag_evidence"][0]["source"], "warranty.txt")

    def test_map_failures_fall_back_to_the_raw_path(self):
        retriever = _retriever()
        state = _state(retriever)
        route = {}

        with patch.object(
            ollama_module, "_build_map_agent", side_effect=RuntimeError("no map")
        ):
            answer, _llm = _ask("alpha chunk 1", state, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(route, {})
        self.assertEqual([call[1] for call in retriever.calls], [None])

    def test_route_metric_failures_never_fail_the_answer(self):
        state = _state(_retriever())
        route = {}

        with patch.object(
            ollama_module,
            "measure_rag_token_accounting",
            side_effect=RuntimeError("metrics failed"),
        ):
            answer, _llm = _ask("alpha chunk 1", state, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(route, {})
        self.assertEqual(state["last_retrieval_route"], {})
        self.assertTrue(state["last_rag_evidence"])


class OptionalLlmPlannerTests(unittest.TestCase):
    def test_llm_planner_routes_the_sections_it_selected(self):
        retriever = _retriever()
        state = _state(retriever, retrieval_map_planner_mode="llm")
        route = {}
        llm = _FakeLlm(completion=lambda prompt: json.dumps([_payload_ids(prompt)[-1]]))

        answer, _llm = _ask("alpha chunk 1", state, llm=llm, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(len(llm.complete_calls), 1)
        self.assertEqual(route["planner_mode"], "llm")
        self.assertEqual(route["planner_output_token_estimate"], 1)
        self.assertGreater(route["planner_card_token_estimate"], 0)
        self.assertIsNone(route["planner_failure_reason"])
        self.assertEqual(len(route["planned_section_ids"]), 1)
        self.assertEqual(
            route["metrics"]["planner_mode"],
            "llm",
        )
        request = llm.complete_calls[0]
        self.assertIn("JSON array", request)
        self.assertIn("alpha chunk 1", request)
        self.assertEqual(
            set(json.loads(request.strip().splitlines()[-1])["candidate_sections"][0]),
            {"id", "title", "keywords", "summary", "score"},
        )

    def test_malformed_llm_planner_output_falls_back_deterministically(self):
        query = "alpha chunk 1"
        deterministic_state = _state(
            _retriever(), retrieval_map_planner_mode="deterministic"
        )
        _answer, _llm = _ask(query, deterministic_state)
        expected = deterministic_state["last_retrieval_route"]["planned_section_ids"]

        state = _state(_retriever(), retrieval_map_planner_mode="llm")
        route = {}
        llm = _FakeLlm(completion=lambda prompt: "The best section is number 2.")

        answer, _llm = _ask(query, state, llm=llm, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(route["planner_mode"], "llm_fallback_deterministic")
        self.assertEqual(route["planner_failure_reason"], "planner_output_not_json")
        self.assertEqual(route["planned_section_ids"], expected)
        self.assertTrue(route["selected_section_ids"])

    def test_failing_llm_planner_falls_back_with_a_redacted_reason(self):
        query = "alpha chunk 1"
        state = _state(_retriever(), retrieval_map_planner_mode="llm")
        route = {}
        llm = _FakeLlm(completion_error=RuntimeError("api_key=SECRET-PLANNER-VALUE"))

        answer, _llm = _ask(query, state, llm=llm, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(route["planner_mode"], "llm_fallback_deterministic")
        self.assertTrue(
            route["planner_failure_reason"].startswith("planner_call_failed")
        )
        self.assertNotIn("SECRET", json.dumps(route))
        self.assertTrue(route["selected_section_ids"])

    def test_llm_planner_output_must_be_a_json_list_of_known_ids(self):
        state = _state(_retriever(), retrieval_map_planner_mode="llm")
        route = {}
        llm = _FakeLlm(
            completion=lambda prompt: json.dumps({"id": _payload_ids(prompt)[0]})
        )

        _answer, _llm = _ask("alpha chunk 1", state, llm=llm, route_sink=route)

        self.assertEqual(route["planner_mode"], "llm_fallback_deterministic")
        self.assertEqual(route["planner_failure_reason"], "planner_output_not_list")

        unknown_state = _state(_retriever(), retrieval_map_planner_mode="llm")
        unknown_route = {}
        unknown_llm = _FakeLlm(completion=lambda prompt: '["map_sec_not_a_section"]')

        _answer, _llm = _ask(
            "alpha chunk 1", unknown_state, llm=unknown_llm, route_sink=unknown_route
        )

        self.assertEqual(unknown_route["planner_mode"], "llm_fallback_deterministic")
        self.assertEqual(
            unknown_route["planner_failure_reason"], "planner_rejected_output"
        )


class RouteLifecycleTests(unittest.TestCase):
    def test_no_match_clears_the_route_and_the_route_sink(self):
        retriever = _empty_retriever()
        state = _state(
            retriever,
            last_retrieval_route={"selected_section_ids": ["stale"]},
            last_rag_evidence=[{"stale": True}],
        )
        route = {"selected_section_ids": ["stale"]}

        answer, _llm = _ask("unmatched quasar question", state, route_sink=route)

        self.assertIn("could not find", answer.lower())
        self.assertEqual(route, {})
        self.assertEqual(state["last_retrieval_route"], {})
        self.assertEqual(state["last_rag_evidence"], [])
        self.assertTrue(state["last_rag_no_result"])

    def test_stream_errors_clear_the_route_without_stale_evidence(self):
        state = _state(_retriever())
        route = {}
        llm = _FakeLlm(fail_stream=True)

        answer, _llm = _ask("alpha chunk 1", state, llm=llm, route_sink=route)

        self.assertIn("could not complete", answer.lower())
        self.assertEqual(route, {})
        self.assertEqual(state["last_retrieval_route"], {})
        self.assertEqual(state["last_rag_evidence"], [])

    def test_project_reset_clears_map_and_route_state(self):
        state = {
            "retrieval_map": object(),
            "last_retrieval_route": {"selected_section_ids": ["stale"]},
            "messages": [{"role": "user", "content": "reset"}],
            "_session_work_dirs": [],
        }

        page_state.perform_project_reset(state, delete_remote=False)

        self.assertIsNone(state["retrieval_map"])
        self.assertEqual(state["last_retrieval_route"], {})
        self.assertEqual(len(state["messages"]), 1)

    def test_local_assistant_message_carries_a_deep_copied_route(self):
        retriever = _retriever()
        state = _state(
            retriever,
            messages=[],
            retrieval_map_chunks_per_section=1,
            retrieval_map_max_selected_sections=1,
            retrieval_map_initial_sections=1,
            retrieval_map_neighbor_sections=0,
        )
        streamlit = _FakeStreamlit(state)
        llm = _FakeLlm()

        with (
            patch.object(chatbox, "st", streamlit),
            patch.object(chatbox.r2r, "r2r_is_ready", return_value=False),
            patch.object(chatbox, "is_openai_compatible_backend", return_value=False),
            patch.object(chatbox, "_local_index_ready", return_value=True),
            patch.object(ollama_module, "st", streamlit),
            patch.object(ollama_module, "_session_llm", return_value=llm),
        ):
            chatbox._process_prompt("alpha chunk 1")

        assistant = state["messages"][-1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertTrue(assistant["retrieval_route"]["selected_section_ids"])
        self.assertEqual(assistant["retrieval_route"], state["last_retrieval_route"])
        assistant["retrieval_route"]["selected_section_ids"].append("mutated")
        self.assertNotIn(
            "mutated", state["last_retrieval_route"]["selected_section_ids"]
        )

    def test_r2r_assistant_messages_do_not_carry_a_local_map_route(self):
        state = _state(
            _retriever(),
            messages=[],
            last_r2r_metadata={"citations": ["a"]},
        )
        streamlit = _FakeStreamlit(state)

        with (
            patch.object(chatbox, "st", streamlit),
            patch.object(chatbox.r2r, "r2r_is_ready", return_value=True),
            patch.object(chatbox.r2r, "r2r_chat", return_value=iter(["r2r answer"])),
        ):
            chatbox._process_prompt("question")

        assistant = state["messages"][-1]
        self.assertNotIn("retrieval_route", assistant)
        self.assertNotIn("evidence", assistant)


class TokenAccountingTests(unittest.TestCase):
    def _accounted_state(self, **overrides):
        settings = {
            "retrieval_map_chunks_per_section": 1,
            "retrieval_map_max_selected_sections": 1,
            "retrieval_map_initial_sections": 1,
            "retrieval_map_neighbor_sections": 0,
            "retrieval_map_max_results": 6,
        }
        settings.update(overrides)
        return _state(_retriever(), **settings)

    def test_actual_and_baseline_tokens_are_recorded(self):
        state = self._accounted_state(retrieval_map_measure_baseline=True)
        route = {}
        llm = _FakeLlm()

        answer, _llm = _ask("alpha chunk 1", state, llm=llm, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        accounting = route["token_accounting"]
        self.assertTrue(accounting["baseline_measured"])
        self.assertGreater(accounting["actual_rag_input_prompt_tokens"], 0)
        self.assertGreater(accounting["actual_selected_evidence_tokens"], 0)
        self.assertGreater(accounting["baseline_input_prompt_tokens"], 0)
        self.assertGreater(accounting["baseline_selected_evidence_tokens"], 0)
        self.assertEqual(
            accounting["selected_minus_baseline_input_tokens"],
            accounting["actual_rag_input_prompt_tokens"]
            - accounting["baseline_input_prompt_tokens"],
        )
        self.assertLess(accounting["selected_minus_baseline_input_tokens"], 0)
        self.assertLess(accounting["selected_minus_baseline_input_pct"], 0)
        self.assertIn("not total compute", accounting["notes"])
        self.assertEqual(len(llm.stream_calls), 1)
        self.assertEqual(
            accounting["actual_rag_input_prompt_tokens"],
            ollama_module._messages_token_count(
                llm.stream_calls[0], ollama_module._resolve_tokenizer(llm)
            ),
        )

    def test_no_baseline_work_happens_when_measurement_is_disabled(self):
        retriever = _retriever()
        state = self._accounted_state(retrieval_map_measure_baseline=False)
        route = {}

        _answer, _llm = _ask("alpha chunk 1", state, route_sink=route)

        accounting = route["token_accounting"]
        self.assertFalse(accounting["baseline_measured"])
        self.assertIsNone(accounting["baseline_input_prompt_tokens"])
        self.assertIsNone(accounting["selected_minus_baseline_input_tokens"])
        self.assertIsNone(accounting["selected_minus_baseline_input_pct"])
        self.assertEqual([call[1] for call in retriever.calls if call[1] is None], [])

    def test_accounting_omits_deltas_when_the_baseline_is_empty(self):
        state = self._accounted_state()
        route = {}

        _answer, _llm = _ask("alpha chunk 1", state, route_sink=route)

        accounting = route["token_accounting"]
        self.assertFalse(accounting["baseline_measured"])
        self.assertIsNone(accounting["selected_minus_baseline_input_pct"])


class _PipelineStreamlit:
    def __init__(self, state):
        self.session_state = state

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def error(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass


class MapIngestionLifecycleTests(unittest.TestCase):
    def _run_pipeline(self, state, retriever, work_dir):
        import utils.rag_pipeline as rag_pipeline

        bundle = {
            "query_engine": "new-engine",
            "retriever": retriever,
            "index": "new-index",
            "cache_key": "new-cache",
        }
        with (
            patch.object(rag_pipeline, "st", _PipelineStreamlit(state)),
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
                rag_pipeline.llama_index, "create_query_engine", return_value=bundle
            ),
            patch.object(rag_pipeline.ollama, "verify_chat_model", return_value=True),
            patch.object(rag_pipeline.ollama, "create_llm", return_value="new-llm"),
            patch.object(
                rag_pipeline.func, "create_ingestion_work_dir", return_value=work_dir
            ),
        ):
            return rag_pipeline.rag_pipeline(
                data_dir=work_dir,
                source_kind="local",
                content_signature="new-content",
            )

    def _state(self):
        return {
            "llm_backend": "Ollama",
            "ollama_endpoint": "http://localhost:11434",
            "ollama_embedding_model": "embedding-a",
            "chunk_size": 256,
            "chunk_overlap_pct": 12,
            "selected_model": "chat-a",
            "system_prompt": "",
            "index_generation": 0,
            "retrieval_map": None,
            "last_retrieval_route": {"selected_section_ids": ["stale"]},
            "_session_work_dirs": [],
        }

    def test_map_is_built_after_a_successful_local_commit(self):
        state = self._state()
        retriever = _retriever()

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, "operation")
            os.mkdir(work_dir)
            error = self._run_pipeline(state, retriever, work_dir)

        self.assertIsNone(error)
        self.assertEqual(state["active_source"]["status"], "ready")
        self.assertEqual(state["query_engine"], "new-engine")
        document_map = state["retrieval_map"]
        self.assertIsInstance(document_map, DocumentMap)
        self.assertTrue(document_map.sections)
        self.assertEqual(
            document_map.source_identity,
            f"{state['active_source']['id']}:{state['index_generation']}",
        )
        self.assertEqual(state["last_retrieval_route"], {})

    def test_ingestion_keeps_the_map_out_of_the_query_bundle_contract(self):
        state = self._state()
        retriever = _retriever()

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, "operation")
            os.mkdir(work_dir)
            self._run_pipeline(state, retriever, work_dir)

        self.assertIsNone(retriever.__dict__.get("retrieval_map"))
        self.assertEqual(state["active_source"]["cache_key"], "new-cache")
        self.assertIsInstance(state["retrieval_map"], DocumentMap)

    def test_map_build_failure_does_not_roll_back_a_valid_index(self):
        import utils.rag_pipeline as rag_pipeline

        state = self._state()
        retriever = _retriever()

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, "operation")
            os.mkdir(work_dir)
            with patch.object(
                rag_pipeline, "build_document_map", side_effect=RuntimeError("boom")
            ):
                error = self._run_pipeline(state, retriever, work_dir)

        self.assertIsNone(error)
        self.assertEqual(state["active_source"]["status"], "ready")
        self.assertEqual(state["query_engine"], "new-engine")
        self.assertIs(state["retriever"], retriever)
        self.assertIsNone(state["retrieval_map"])
        self.assertEqual(state["last_retrieval_route"], {})
        self.assertIsNone(state["last_ingestion_error"])

    def test_failed_replacement_restores_the_previous_map_and_route(self):
        import utils.rag_pipeline as rag_pipeline

        previous_map = build_document_map(_retriever(), source_identity="old:1")
        state = self._state()
        state["retrieval_map"] = previous_map
        state["last_retrieval_route"] = {"selected_section_ids": ["old"]}
        state["query_engine"] = "old-engine"
        state["index_generation"] = 1
        state["active_source"] = {
            "id": "old",
            "source_id": "old",
            "kind": "local",
            "status": "ready",
            "index_generation": 1,
        }
        bundle = {
            "query_engine": "new-engine",
            "retriever": _retriever(),
            "index": "new-index",
            "cache_key": "new-cache",
        }
        state["retriever"] = bundle["retriever"]

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, "operation")
            os.mkdir(work_dir)
            with (
                patch.object(rag_pipeline, "st", _PipelineStreamlit(state)),
                patch.object(rag_pipeline, "render_pipeline_status"),
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
                    data_dir=work_dir,
                    source_kind="local",
                    content_signature="new-content",
                )

        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(state["query_engine"], "old-engine")
        self.assertEqual(state["retrieval_map"].source_identity, "old:1")
        self.assertEqual(
            state["last_retrieval_route"], {"selected_section_ids": ["old"]}
        )
        self.assertEqual(state["index_generation"], 1)

    def test_committed_map_is_reused_by_the_routing_path(self):
        state = self._state()
        retriever = _retriever()

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, "operation")
            os.mkdir(work_dir)
            self._run_pipeline(state, retriever, work_dir)

        document_map = state["retrieval_map"]
        route = {}
        answer, _llm = _ask("alpha chunk 1", state, route_sink=route)

        self.assertEqual(answer, "grounded answer")
        self.assertIs(state["retrieval_map"], document_map)
        self.assertTrue(
            set(route["selected_section_ids"]).issubset(
                {section.section_id for section in document_map.sections}
            )
        )


class BrowserPersistenceTests(unittest.TestCase):
    def test_map_settings_round_trip_with_strict_types(self):
        state = {}
        browser_settings.apply_persisted_settings(
            state,
            {
                "retrieval_map_enabled": "true",
                "retrieval_map_planner_mode": "LLM",
                "retrieval_map_measure_baseline": "false",
                "retrieval_map_chunks_per_section": "4",
                "retrieval_map_max_selected_sections": "3",
                "retrieval_map_initial_sections": "2",
                "retrieval_map_neighbor_sections": "0",
                "retrieval_map_max_results": "10",
            },
        )

        self.assertEqual(
            browser_settings.serialize_persisted_settings(state),
            {
                "retrieval_map_enabled": True,
                "retrieval_map_planner_mode": "llm",
                "retrieval_map_measure_baseline": False,
                "retrieval_map_chunks_per_section": 4,
                "retrieval_map_max_selected_sections": 3,
                "retrieval_map_initial_sections": 2,
                "retrieval_map_neighbor_sections": 0,
                "retrieval_map_max_results": 10,
            },
        )

    def test_out_of_range_and_unsupported_map_values_are_dropped(self):
        state = {}
        browser_settings.apply_persisted_settings(
            state,
            {
                "retrieval_map_chunks_per_section": 9999,
                "retrieval_map_max_results": 0,
                "retrieval_map_neighbor_sections": -1,
                "retrieval_map_planner_mode": "chain-of-thought",
                "retrieval_map_enabled": "maybe",
            },
        )

        self.assertEqual(state, {})

    def test_persisted_bounds_match_the_map_settings_ranges(self):
        for key, bounds in MAP_SETTING_RANGES.items():
            with self.subTest(key=key):
                self.assertEqual(
                    browser_settings.RANGED_SETTING_BOUNDS[key],
                    tuple(bounds),
                )

    def test_secrets_and_document_text_never_enter_local_storage(self):
        payload = browser_settings.serialize_persisted_settings(
            {
                "retrieval_map_enabled": True,
                "retrieval_map_planner_mode": "deterministic",
                "openai_api_key": "sk-secret",
                "embedding_api_key": "sk-secret",
                "r2r_api_key": "r2r-secret",
                "documents": ["confidential body text"],
                "retrieval_map": {"sections": [{"summary": "confidential body text"}]},
                "last_retrieval_route": {"result_metadata": ["confidential"]},
                "messages": [{"role": "user", "content": "confidential body text"}],
            }
        )

        self.assertEqual(
            payload,
            {
                "retrieval_map_enabled": True,
                "retrieval_map_planner_mode": "deterministic",
            },
        )
        self.assertNotIn("confidential", json.dumps(payload))
        self.assertNotIn("secret", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
