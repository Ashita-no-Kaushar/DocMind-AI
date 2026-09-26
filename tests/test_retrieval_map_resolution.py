import unittest

from llama_index.core.schema import NodeWithScore, TextNode

from utils.retrieval_map import (
    DEFAULT_CHUNKS_PER_SECTION,
    HARD_MAX_CHUNKS_PER_SECTION,
    MapAgentConfig,
    MapRetrievalAgent,
    RetrievalMapConfig,
    _bounded_excerpt,
    _section_step,
    build_document_map,
    map_document_config,
    normalize_map_settings,
)

CHUNKS = (
    "Refund Policy Overview. Refunds are available within 30 days of purchase. "
    "Eligibility is checked against the order record.",
    "Staging Promotion. Promotion to staging requires approval. The staging soak "
    "time is 30 minutes before production promotion is allowed.",
    "Rollback Plan. Rollback names the previous version and the trigger conditions "
    "for a rollback.",
    "Incident Cadence. A severity 1 incident posts an internal update every 30 "
    "minutes and a public update every 60 minutes.",
)


class _Docstore:
    def __init__(self, nodes):
        self.docs = {node.node_id: node for node in nodes}

    def get_node(self, node_id):
        return self.docs.get(node_id)


class _Retriever:
    def __init__(self, nodes, top_k=4):
        self.docstore = _Docstore(nodes)
        self.corpus = [node.node_id for node in nodes]
        self.top_k = top_k
        self.calls = []

    def retrieve(self, query, allowed_node_ids=None, top_k=None):
        self.calls.append((query, allowed_node_ids))
        allowed = (
            set(allowed_node_ids) if allowed_node_ids is not None else set(self.corpus)
        )
        scored = []
        for node_id in self.corpus:
            if node_id not in allowed:
                continue
            node = self.docstore.get_node(node_id)
            overlap = len(
                set(query.lower().split()) & set(node.get_content().lower().split())
            )
            if overlap:
                scored.append((node_id, float(overlap)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            NodeWithScore(node=self.docstore.get_node(node_id), score=score)
            for node_id, score in scored[: (top_k or self.top_k)]
        ]


def _corpus(docs=1, chunks=CHUNKS):
    nodes = []
    for doc in range(docs):
        for index, chunk in enumerate(chunks):
            nodes.append(
                TextNode(
                    text=chunk,
                    id_=f"doc{doc}#{index}",
                    metadata={"file_name": f"doc{doc}.txt", "source_id": f"doc{doc}"},
                )
            )
    return _Retriever(nodes)


class SectionStepTests(unittest.TestCase):
    def test_fixed_resolution_is_used_when_adaptive_is_disabled(self):
        config = RetrievalMapConfig(chunks_per_section=3)
        self.assertEqual(_section_step(20, config), 3)
        self.assertEqual(_section_step(2, config), 3)

    def test_adaptive_resolution_divides_each_document_evenly(self):
        config = RetrievalMapConfig(target_sections_per_document=3)
        self.assertEqual(_section_step(3, config), 1)
        self.assertEqual(_section_step(6, config), 2)
        self.assertEqual(_section_step(7, config), 3)

    def test_adaptive_resolution_never_produces_an_empty_section(self):
        config = RetrievalMapConfig(target_sections_per_document=12)
        for run_length in range(1, 40):
            self.assertGreaterEqual(_section_step(run_length, config), 1)
            self.assertLessEqual(
                _section_step(run_length, config), HARD_MAX_CHUNKS_PER_SECTION
            )

    def test_adaptive_resolution_collapses_a_short_document_into_one_section(self):
        fixed = RetrievalMapConfig(chunks_per_section=DEFAULT_CHUNKS_PER_SECTION)
        document_map = build_document_map(_corpus(), config=fixed)
        self.assertEqual(len(document_map.sections), 1)


class AdaptiveMapBuildTests(unittest.TestCase):
    def test_adaptive_build_splits_short_documents_into_several_sections(self):
        document_map = build_document_map(
            _corpus(), config=RetrievalMapConfig(target_sections_per_document=2)
        )
        self.assertEqual(len(document_map.sections), 2)
        self.assertTrue(
            all(section.chunk_count == 2 for section in document_map.sections)
        )

    def test_every_section_keeps_its_member_nodes_and_ordering(self):
        document_map = build_document_map(
            _corpus(), config=RetrievalMapConfig(target_sections_per_document=3)
        )
        members = [
            node_id
            for section in document_map.sections
            for node_id in section.member_node_ids
        ]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(sorted(members), sorted(_corpus().corpus))

    def test_map_identity_changes_with_the_resolution(self):
        fixed = build_document_map(
            _corpus(), config=RetrievalMapConfig(chunks_per_section=4)
        )
        adaptive = build_document_map(
            _corpus(), config=RetrievalMapConfig(target_sections_per_document=2)
        )
        self.assertNotEqual(fixed.map_id, adaptive.map_id)


class BoundedExcerptTests(unittest.TestCase):
    def test_excerpt_keeps_terms_that_appear_late_in_the_chunk(self):
        text = (
            "Staging Promotion. Promotion requires approval. The staging soak time "
            "is 30 minutes."
        )
        excerpt = _bounded_excerpt(text, 240, "Release Checklist")
        self.assertIn("soak", excerpt.lower())

    def test_excerpt_respects_its_budget(self):
        text = " ".join(
            f"Sentence number {index} carries some words." for index in range(40)
        )
        for limit in (24, 40, 80, 240):
            self.assertLessEqual(len(_bounded_excerpt(text, limit)), limit)

    def test_excerpt_skips_a_leading_title_sentence(self):
        excerpt = _bounded_excerpt(
            "Release Checklist. The soak time is 30 minutes.", 240, "Release Checklist"
        )
        self.assertNotIn("Release Checklist", excerpt)
        self.assertIn("soak", excerpt)

    def test_card_summary_exposes_a_late_discriminative_term(self):
        document_map = build_document_map(
            _corpus(), config=RetrievalMapConfig(chunks_per_section=1)
        )
        summaries = " ".join(section.summary for section in document_map.sections)
        self.assertIn("soak", summaries.lower())


class MapSettingTests(unittest.TestCase):
    def test_adaptive_target_is_a_bounded_persisted_setting(self):
        self.assertIn("retrieval_map_target_sections", normalize_map_settings({}))
        self.assertEqual(normalize_map_settings({})["retrieval_map_target_sections"], 0)
        self.assertEqual(
            normalize_map_settings({"retrieval_map_target_sections": 3})[
                "retrieval_map_target_sections"
            ],
            3,
        )
        self.assertEqual(
            normalize_map_settings({"retrieval_map_target_sections": 999})[
                "retrieval_map_target_sections"
            ],
            12,
        )
        self.assertEqual(
            normalize_map_settings({"retrieval_map_target_sections": -5})[
                "retrieval_map_target_sections"
            ],
            0,
        )

    def test_settings_flow_into_the_document_config(self):
        config = map_document_config({"retrieval_map_target_sections": 4})
        self.assertEqual(config.target_sections_per_document, 4)
        self.assertEqual(map_document_config({}).target_sections_per_document, 0)


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.retriever = _corpus()
        self.document_map = build_document_map(
            self.retriever, config=RetrievalMapConfig(chunks_per_section=1)
        )

    def _agent(self, **kwargs):
        return MapRetrievalAgent(
            self.retriever,
            self.document_map,
            tokenizer=str.split,
            agent_config=MapAgentConfig(
                max_selected_sections=4,
                initial_sections=2,
                neighbor_sections=0,
                max_steps=1,
                max_results=4,
                **kwargs,
            ),
        )

    def test_refinement_is_off_by_default(self):
        self.assertFalse(MapAgentConfig().refine_candidates)

    def test_refinement_reports_itself_in_the_plan_reason(self):
        _results, trace = self._agent().retrieve("staging soak time")
        reason = next(
            action["reason"]
            for action in trace["actions"]
            if action["action"] == "plan"
        )
        self.assertEqual(reason, "deterministic_keyword_scores")

    def test_refinement_can_be_enabled_and_is_reported(self):
        _results, trace = self._agent(refine_candidates=True).retrieve(
            "staging soak time"
        )
        reason = next(
            action["reason"]
            for action in trace["actions"]
            if action["action"] == "plan"
        )
        self.assertEqual(reason, "deterministic_keyword_scores_refined")

    def test_refinement_never_selects_a_section_outside_the_shortlist(self):
        agent = self._agent(refine_candidates=True, refine_pool=2)
        cards = agent._cards("staging soak time")
        pool = {str(card["id"]) for card in cards[:2]}
        selected = agent._refine_candidates(["staging soak time"], cards, 2)
        self.assertTrue(set(selected) <= pool)

    def test_failed_planner_falls_back_to_the_same_plan_as_the_deterministic_path(self):
        def broken(_query, _cards):
            raise RuntimeError("planner exploded")

        _results, fallback = MapRetrievalAgent(
            self.retriever,
            self.document_map,
            tokenizer=str.split,
            planner=broken,
            agent_config=MapAgentConfig(
                max_selected_sections=4, initial_sections=2, max_steps=1
            ),
        ).retrieve("staging soak time")
        _results, direct = self._agent().retrieve("staging soak time")
        self.assertEqual(fallback["planned_section_ids"], direct["planned_section_ids"])

    def test_trace_reports_whether_routing_actually_engaged(self):
        _results, routed = self._agent().retrieve("staging soak time")
        self.assertTrue(routed["routed"])
        self.assertTrue(routed["selected_section_ids"])

    def test_a_routed_miss_is_still_counted_as_a_routed_attempt(self):
        results, trace = self._agent().retrieve("zzzz nonexistent topic")
        self.assertEqual(results, [])
        self.assertTrue(
            trace["routed"],
            "a routed attempt that found nothing must not look like a silent "
            "global fallback",
        )
        self.assertFalse(
            any(
                action["action"] in ("retrieve_global", "global_fallback")
                for action in trace["actions"]
            )
        )

    def test_empty_query_never_claims_a_route(self):
        _results, trace = self._agent().retrieve("   ")
        self.assertFalse(trace["routed"])
        self.assertEqual(trace["selected_section_ids"], [])


class ScorerTests(unittest.TestCase):
    def test_a_rare_body_term_outranks_a_common_title_term(self):
        document_map = build_document_map(
            _corpus(), config=RetrievalMapConfig(chunks_per_section=1)
        )
        agent = MapRetrievalAgent(
            _corpus(), document_map, tokenizer=str.split, agent_config=MapAgentConfig()
        )
        cards = agent._cards("staging soak time")
        top = max(cards, key=lambda card: card["score"])
        self.assertIn("soak", top["summary"].lower() + " " + " ".join(top["keywords"]))


if __name__ == "__main__":
    unittest.main()
