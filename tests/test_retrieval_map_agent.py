import inspect
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core.schema import NodeWithScore, TextNode

from utils.llama_index import HybridRetriever
from utils.retrieval_map import (
    HARD_MAX_KEYWORDS,
    HARD_MAX_NODES,
    HARD_MAX_SECTIONS,
    HARD_MAX_STEPS,
    HARD_MAX_SUMMARY_CHARS,
    MapRetrievalAgent,
    RetrievalMapConfig,
    build_document_map,
    build_map_retrieval_agent,
    count_map_tokens,
    json_safe,
)


class _Docstore:
    def __init__(self, nodes):
        self.docs = {node.node_id: node for node in nodes}

    def get_node(self, node_id):
        return self.docs[node_id]


class _VectorRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.results)


class _FakeMapRetriever:
    def __init__(self, nodes, allowed=None, global_results=None):
        self.docstore = _Docstore(nodes)
        self.corpus = [node.node_id for node in nodes]
        self.allowed = [] if allowed is None else list(allowed)
        self.global_results = [] if global_results is None else list(global_results)
        self.calls = []

    def retrieve(self, query, allowed_node_ids=None):
        self.calls.append((query, allowed_node_ids))
        if allowed_node_ids is None:
            return list(self.global_results)
        return list(self.allowed)


class _DocstoreMapRetriever:
    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.docstore = SimpleNamespace(
            get_node=lambda node_id: (
                self.corpus[node_id] if isinstance(node_id, int) else None
            )
        )

    def retrieve(self, query, allowed_node_ids=None):
        return []


def _whitespace_tokenizer(value):
    return value.split()


def _node(file_name, index, body=None, source_id=None):
    text = body or (
        f"{file_name} chunk {index} topic {index}alpha keyword {index * 7} "
        "contains operational details and unique terminology"
    )
    metadata = {"file_name": file_name, "start_char_idx": index * 100}
    if source_id:
        metadata["source_id"] = source_id
    return TextNode(text=text, id_=f"{file_name}-{index}", metadata=metadata)


def _hybrid(files, vector_results=None, **kwargs):
    nodes = []
    for file_name, count in files:
        nodes.extend(_node(file_name, index) for index in range(count))
    docstore = _Docstore(nodes)
    corpus = [node.node_id for node in nodes]
    if vector_results is None:
        vector_results = [NodeWithScore(node=node, score=0.85) for node in nodes]
    vector = _VectorRetriever(vector_results)
    return HybridRetriever(
        vector,
        docstore,
        corpus,
        top_k=kwargs.pop("top_k", 3),
        similarity_cutoff=kwargs.pop("similarity_cutoff", 0.3),
        **kwargs,
    )


class RetrievalMapBuildTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("utils.llama_index.st", SimpleNamespace(session_state={}))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_map_is_deterministic_sections_are_bounded_and_ids_are_stable(self):
        retriever = _hybrid([("alpha.txt", 5)])
        config = RetrievalMapConfig(
            chunks_per_section=2,
            max_sections=2,
            max_keywords=3,
            max_summary_chars=40,
        )

        first = build_document_map(retriever, config=config, source_identity="src-1")
        second = build_document_map(retriever, config=config, source_identity="src-1")

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            [section.section_id for section in first.sections],
            [section.section_id for section in second.sections],
        )
        self.assertEqual(
            [section.chunk_count for section in first.sections],
            [2, 2],
        )
        self.assertTrue(first.truncated)
        self.assertTrue(all(len(section.keywords) <= 3 for section in first.sections))
        self.assertTrue(all(len(section.summary) <= 40 for section in first.sections))
        self.assertTrue(all(section.token_count > 0 for section in first.sections))
        self.assertEqual(first.source_identity, "src-1")
        self.assertTrue(any(edge.kind == "contains" for edge in first.edges))
        self.assertTrue(any(edge.kind == "next" for edge in first.edges))
        self.assertEqual(first.map_id, second.map_id)

    def test_hard_caps_clamp_configuration(self):
        config = RetrievalMapConfig(
            chunks_per_section=1000,
            max_sections=100000,
            max_nodes=100000,
            max_keywords=1000,
            max_summary_chars=100000,
        )

        self.assertLessEqual(config.chunks_per_section, 32)
        self.assertLessEqual(config.max_sections, HARD_MAX_SECTIONS)
        self.assertLessEqual(config.max_nodes, HARD_MAX_NODES)
        self.assertLessEqual(config.max_keywords, HARD_MAX_KEYWORDS)
        self.assertLessEqual(config.max_summary_chars, HARD_MAX_SUMMARY_CHARS)

    def test_map_does_not_duplicate_full_document_text(self):
        body = "SENTINEL-BODY " + ("long unique passage " * 60)
        nodes = [_node("long.txt", index, body=body + str(index)) for index in range(4)]
        retriever = SimpleNamespace(
            docstore=_Docstore(nodes),
            corpus=[node.node_id for node in nodes],
            retrieve=lambda query, allowed_node_ids=None: [],
        )

        document_map = build_document_map(
            retriever,
            chunks_per_section=2,
            max_summary_chars=60,
        )
        serialized = json.dumps(document_map.to_dict(), allow_nan=False)

        self.assertNotIn(body, serialized)
        self.assertLess(len(serialized), len(body) * 3)
        self.assertEqual(
            list(document_map.member_node_ids()),
            [node.node_id for node in nodes],
        )
        self.assertTrue(
            all("get_content" not in json_safe(item) for item in document_map.sections)
        )

    def test_missing_metadata_and_missing_node_ids_are_tolerated(self):
        valid = TextNode(
            text="A usable chunk with enough content to map.",
            id_="valid-0",
            metadata={},
        )
        missing_id = SimpleNamespace(
            text="A chunk without a node id or metadata.",
            metadata={},
        )
        retriever = _DocstoreMapRetriever([missing_id, valid])

        document_map = build_document_map(retriever, source_identity="src-2")

        self.assertEqual(len(document_map.sections), 1)
        self.assertEqual(document_map.mapped_node_count, 1)
        self.assertEqual(document_map.skipped_node_count, 1)
        self.assertEqual(document_map.input_node_count, 2)
        self.assertEqual(document_map.sections[0].member_node_ids, ("valid-0",))


class MapRetrievalAgentTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("utils.llama_index.st", SimpleNamespace(session_state={}))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tokenizer = _whitespace_tokenizer

    def _agent(self, retriever, **kwargs):
        return MapRetrievalAgent(
            retriever,
            tokenizer=self.tokenizer,
            chunks_per_section=kwargs.pop("chunks_per_section", 1),
            **kwargs,
        )

    def test_default_planner_is_deterministic_and_sends_no_planner_tokens(self):
        retriever = _hybrid([("alpha.txt", 3), ("beta.txt", 3)])

        first = MapRetrievalAgent(
            retriever, tokenizer=self.tokenizer, chunks_per_section=1
        ).retrieve("alpha chunk 1")
        second = MapRetrievalAgent(
            retriever, tokenizer=self.tokenizer, chunks_per_section=1
        ).retrieve("alpha chunk 1")

        self.assertEqual(first[1], second[1])
        self.assertEqual(first[1]["planner_mode"], "deterministic")
        self.assertEqual(first[1]["planner_card_token_estimate"], 0)
        self.assertTrue(first[0])
        self.assertLessEqual(
            len(first[1]["selected_section_ids"]),
            first[1]["metrics"]["result_count"] + 16,
        )

    def test_neighbor_expansion_adds_bounded_adjacent_sections(self):
        retriever = _hybrid([("alpha.txt", 5)])

        results, trace = self._agent(
            retriever,
            initial_sections=1,
            max_selected_sections=4,
            neighbor_sections=1,
        ).retrieve("topic 2alpha")

        self.assertTrue(results)
        self.assertEqual(len(trace["selected_section_ids"]), 3)
        self.assertEqual(
            trace["planned_section_ids"], [trace["selected_section_ids"][0]]
        )
        self.assertEqual(trace["step_count"], 1)
        self.assertFalse(trace["reflection_flag"])
        self.assertLessEqual(len(trace["selected_section_ids"]), 4)

    def test_weak_routes_perform_one_global_fallback(self):
        nodes = [_node("alpha.txt", index) for index in range(2)]
        fallback = NodeWithScore(node=nodes[1], score=0.9)
        retriever = _FakeMapRetriever(nodes, global_results=[fallback])

        results, trace = self._agent(
            retriever,
            initial_sections=2,
            max_selected_sections=2,
        ).retrieve("quasar unknown")

        self.assertEqual(
            [result.node.node_id for result in results], [nodes[1].node_id]
        )
        self.assertEqual(trace["step_count"], 2)
        self.assertTrue(trace["reflection_flag"])
        self.assertEqual(
            [action["action"] for action in trace["actions"]],
            ["observe", "plan", "retrieve_sections", "reflect", "global_fallback"],
        )
        self.assertEqual(
            len([call for call in retriever.calls if call[1] is None]),
            1,
        )
        self.assertLessEqual(trace["step_count"], HARD_MAX_STEPS)

    def test_weak_global_evidence_is_rejected(self):
        nodes = [_node("alpha.txt", index) for index in range(2)]
        weak = NodeWithScore(node=nodes[0], score=0.1)
        retriever = _FakeMapRetriever(nodes, allowed=[weak], global_results=[weak])

        results, trace = self._agent(
            retriever,
            initial_sections=2,
            max_selected_sections=2,
            weak_score=0.5,
        ).retrieve("quasar unknown")

        self.assertEqual(results, [])
        self.assertTrue(trace["reflection_flag"])
        self.assertEqual(trace["metrics"]["result_count"], 0)
        self.assertEqual(trace["metrics"]["selected_evidence_tokens"], 0)

    def test_injected_planner_receives_only_bounded_cards(self):
        retriever = _hybrid([("alpha.txt", 3), ("beta.txt", 3)])
        seen = {}

        def planner(query, cards):
            seen["query"] = query
            seen["cards"] = cards
            return [cards[1]["id"]]

        results, trace = self._agent(retriever, planner=planner).retrieve(
            "alpha chunk 1"
        )

        self.assertTrue(results)
        self.assertEqual(seen["query"], "alpha chunk 1")
        self.assertEqual(
            set(seen["cards"][0]),
            {"id", "title", "keywords", "summary", "score"},
        )
        self.assertEqual(trace["planner_mode"], "injected")
        self.assertEqual(len(trace["planned_section_ids"]), 1)
        self.assertGreater(trace["planner_card_token_estimate"], 0)

    def test_malformed_unknown_and_failed_planners_fall_back_deterministically(self):
        retriever = _hybrid([("alpha.txt", 4)])
        expected = MapRetrievalAgent(
            retriever,
            tokenizer=self.tokenizer,
            chunks_per_section=1,
            initial_sections=1,
        ).retrieve("alpha chunk 1")[1]

        cases = [
            (lambda query, cards: "not-a-list", "planner_rejected_output"),
            (lambda query, cards: [1, 2, 3], "planner_rejected_output"),
            (lambda query, cards: ["map_sec_unknown"], "planner_rejected_output"),
            (
                lambda query, cards: (_ for _ in ()).throw(
                    RuntimeError("secret-planner-detail")
                ),
                "planner_failed",
            ),
        ]

        for planner, reason in cases:
            with self.subTest(reason=reason, planner=planner):
                _results, trace = self._agent(
                    retriever, planner=planner, initial_sections=1
                ).retrieve("alpha chunk 1")
                self.assertEqual(
                    trace["planner_mode"], "injected_fallback_deterministic"
                )
                plan_action = next(
                    action for action in trace["actions"] if action["action"] == "plan"
                )
                self.assertEqual(plan_action["reason"], reason)
                self.assertEqual(
                    trace["planned_section_ids"],
                    expected["planned_section_ids"],
                )
                self.assertNotIn("secret-planner-detail", json.dumps(trace))

    def test_token_metrics_and_global_context_comparison_use_injected_tokenizer(self):
        retriever = _hybrid(
            [("alpha.txt", 2), ("beta.txt", 2), ("gamma.txt", 2)], top_k=3
        )
        agent = self._agent(retriever, max_results=3)

        results, trace = agent.retrieve("alpha topic 0alpha")

        self.assertEqual(trace["planner_card_token_estimate"], 0)
        expected_tokens = sum(
            len(result.node.get_content().split()) for result in results
        )
        self.assertEqual(
            trace["metrics"]["selected_evidence_tokens"],
            expected_tokens,
        )
        self.assertEqual(trace["metrics"]["token_basis"], "selected_evidence_text")

        comparison = agent.compare_global_context(
            "alpha topic 0alpha",
            results,
            baseline_limit=3,
        ).to_dict()

        self.assertEqual(
            comparison["selected_evidence_tokens"],
            trace["metrics"]["selected_evidence_tokens"],
        )
        self.assertGreater(comparison["global_baseline_tokens"], 0)
        self.assertEqual(
            comparison["selected_minus_baseline_tokens"],
            comparison["selected_evidence_tokens"]
            - comparison["global_baseline_tokens"],
        )
        self.assertEqual(comparison["global_baseline_result_count"], 3)

    def test_conversation_history_uses_recent_user_turns_without_assistant_text(self):
        nodes = [
            _node("alpha.txt", 0, body="boomerang parts are listed in the catalog"),
            _node("alpha.txt", 1, body="different operational details"),
        ]
        docstore = _Docstore(nodes)
        retriever = HybridRetriever(
            _VectorRetriever([NodeWithScore(node=node, score=0.4) for node in nodes]),
            docstore,
            [node.node_id for node in nodes],
            top_k=2,
            similarity_cutoff=0.3,
        )
        history = [
            {"role": "user", "content": "boomerang handling"},
            {"role": "assistant", "content": "ASSISTANT-SECRET"},
        ]

        results, trace = self._agent(
            retriever,
            initial_sections=1,
            max_selected_sections=1,
            neighbor_sections=0,
        ).retrieve("what about the warranty?", history=history)

        self.assertTrue(results)
        self.assertEqual(len(retriever.vector_retriever.queries), 2)
        self.assertIn("boomerang", retriever.vector_retriever.queries[-1])
        self.assertIn("warranti", retriever.vector_retriever.queries[-1])
        self.assertNotIn("ASSISTANT-SECRET", retriever.vector_retriever.queries[-1])
        self.assertNotIn("ASSISTANT-SECRET", json.dumps(trace))
        self.assertEqual(trace["step_count"], 1)

    def test_map_cache_is_attached_per_retriever_and_does_not_mutate_bm25(self):
        retriever = _hybrid([("alpha.txt", 3)])
        corpus_before = list(retriever.corpus)
        scores_before = list(retriever._bm25.get_scores(["alpha"]))

        first = build_map_retrieval_agent(
            retriever,
            tokenizer=self.tokenizer,
            source_identity="src-1",
            chunks_per_section=1,
        )
        second = build_map_retrieval_agent(
            retriever,
            tokenizer=self.tokenizer,
            source_identity="src-1",
            chunks_per_section=1,
        )

        self.assertIs(first.document_map, second.document_map)
        self.assertIs(retriever.retrieval_map, first.document_map)
        self.assertIs(retriever.retrieval_map_agent, second)

        third = build_map_retrieval_agent(
            retriever,
            tokenizer=self.tokenizer,
            source_identity="src-2",
            chunks_per_section=1,
        )

        self.assertIs(retriever.retrieval_map_agent, third)
        self.assertIsNot(third.document_map, first.document_map)
        self.assertEqual(retriever.corpus, corpus_before)
        self.assertEqual(list(retriever._bm25.get_scores(["alpha"])), scores_before)

    def test_trace_and_map_serialization_is_strict_json_safe(self):
        retriever = _hybrid([("alpha.txt", 3)])

        _results, trace = self._agent(retriever).retrieve("alpha chunk 1")
        payload = json.dumps(trace, allow_nan=False, sort_keys=True)
        document_payload = json.dumps(
            build_document_map(retriever, source_identity="src-1").to_dict(),
            allow_nan=False,
        )

        self.assertTrue(payload)
        self.assertTrue(document_payload)
        self.assertTrue(all(math.isfinite(value) for value in [trace["step_count"]]))
        self.assertEqual(count_map_tokens("one two three", self.tokenizer), 3)
        self.assertEqual(count_map_tokens("one two three", lambda value: 99), 99)
        self.assertEqual(count_map_tokens("", self.tokenizer), 0)
        self.assertEqual(
            count_map_tokens(
                "one two", lambda value: (_ for _ in ()).throw(ValueError)
            ),
            2,
        )


class HybridRoutingTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("utils.llama_index.st", SimpleNamespace(session_state={}))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_allowed_node_ids_filter_vector_and_bm25_pools(self):
        nodes = [
            _node("allowed.txt", 0, body="alpha widget ships with every unit"),
            _node("blocked.txt", 0, body="alpha widget covers the retired chassis"),
        ]
        docstore = _Docstore(nodes)
        corpus = [node.node_id for node in nodes]
        vector = _VectorRetriever([NodeWithScore(node=nodes[1], score=0.95)])
        retriever = HybridRetriever(
            vector,
            docstore,
            corpus,
            top_k=3,
            similarity_cutoff=0.3,
            vector_candidate_depth=1,
            bm25_candidate_depth=2,
        )

        unrestricted = retriever.retrieve("alpha widget")
        routed = retriever.retrieve("alpha widget", allowed_node_ids={nodes[0].node_id})

        self.assertEqual(
            {result.node.node_id for result in unrestricted},
            {nodes[0].node_id, nodes[1].node_id},
        )
        self.assertEqual(
            [result.node.node_id for result in routed],
            [nodes[0].node_id],
        )
        self.assertEqual(retriever.corpus, corpus)

    def test_bm25_only_pool_is_restricted_without_mutating_bm25_state(self):
        nodes = [
            _node("allowed.txt", 0, body="alpha widget ships with every unit"),
            _node("blocked.txt", 0, body="alpha widget covers the retired chassis"),
        ]
        docstore = _Docstore(nodes)
        corpus = [node.node_id for node in nodes]
        retriever = HybridRetriever(
            _VectorRetriever([]),
            docstore,
            corpus,
            top_k=3,
            similarity_cutoff=0.3,
        )
        before = list(retriever._bm25.get_scores(["alpha"]))

        routed = retriever.retrieve("alpha widget", allowed_node_ids=[nodes[0].node_id])

        self.assertEqual([result.node.node_id for result in routed], [nodes[0].node_id])
        self.assertEqual(retriever.corpus, corpus)
        self.assertEqual(list(retriever._bm25.get_scores(["alpha"])), before)

    def test_default_signature_and_allow_global_fallback_are_backward_compatible(self):
        retriever = _hybrid([("alpha.txt", 3), ("beta.txt", 3)], top_k=3)
        first = retriever.retrieve("alpha topic 0alpha")
        second = retriever.retrieve("alpha topic 0alpha", allowed_node_ids=None)

        self.assertEqual(
            [result.node.node_id for result in first],
            [result.node.node_id for result in second],
        )
        signature = inspect.signature(HybridRetriever.retrieve)
        self.assertIn("allowed_node_ids", signature.parameters)
        self.assertIs(
            signature.parameters["allowed_node_ids"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

        fallback = retriever.retrieve(
            "summarize this document",
            allowed_node_ids={"not-in-corpus"},
            allow_global_fallback=True,
        )

        self.assertTrue(fallback)
        self.assertLessEqual(len(fallback), 5)

    def test_map_agent_filters_retrievers_without_routing_keyword(self):
        nodes = [_node("alpha.txt", 0), _node("beta.txt", 0)]
        allowed = NodeWithScore(node=nodes[0], score=0.9)
        blocked = NodeWithScore(node=nodes[1], score=0.8)

        class LegacyRetriever:
            def __init__(self):
                self.docstore = _Docstore(nodes)
                self.corpus = [node.node_id for node in nodes]

            def retrieve(self, query):
                return [blocked, allowed]

        agent = MapRetrievalAgent(
            LegacyRetriever(),
            tokenizer=_whitespace_tokenizer,
            chunks_per_section=1,
            initial_sections=1,
            max_selected_sections=1,
            neighbor_sections=0,
        )
        results, _trace = agent.retrieve("alpha topic 0alpha")

        self.assertEqual(
            [result.node.node_id for result in results], [nodes[0].node_id]
        )

    def test_empty_allowed_set_returns_no_results(self):
        retriever = _hybrid([("alpha.txt", 2)])

        self.assertEqual(
            retriever.retrieve("alpha topic 0alpha", allowed_node_ids=[]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
