import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from components import retrieval_map_view as view
from research import fixtures, map_ablation
from utils.retrieval_map import RetrievalMapConfig, build_document_map

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 8
_TEMP_ROOT = Path(tempfile.mkdtemp(prefix="docmind_ablation_shared_"))


def setUpModule():
    map_ablation.run_study(top_k=TOP_K, out_dir=_TEMP_ROOT)


def tearDownModule():
    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)


def _build_map(top_k=TOP_K):
    retriever = fixtures.build_retriever(top_k=top_k)
    return retriever, build_document_map(
        retriever,
        config=RetrievalMapConfig(chunks_per_section=fixtures.CHUNKS_PER_SECTION),
    )


class FixtureCorpusTests(unittest.TestCase):
    def test_corpus_is_grouped_and_fully_mapped(self):
        retriever, document_map = _build_map()
        self.assertEqual(len(retriever.corpus), 48)
        self.assertEqual(len(document_map.sections), 24)
        self.assertFalse(document_map.truncated)
        self.assertEqual(int(document_map.input_node_count), 48)
        self.assertEqual(
            len({section.source_id for section in document_map.sections}), 8
        )

    def test_every_answerable_query_names_known_chunks(self):
        retriever, document_map = _build_map()
        known = set(retriever.corpus)
        sections = {section.section_id for section in document_map.sections}
        self.assertTrue(sections)
        answers = map_ablation.answer_nodes()
        self.assertEqual(len(answers), 25)
        for nodes in answers.values():
            self.assertTrue(nodes)
            for node_id in nodes:
                self.assertIn(node_id, known)
        no_match = [
            query["id"] for query in fixtures.QUERIES if query["id"] not in answers
        ]
        self.assertEqual(len(no_match), 6)
        self.assertTrue(
            all(
                not query.get("doc")
                for query in fixtures.QUERIES
                if query["id"] in no_match
            )
        )

    def test_retrieval_is_deterministic_and_respects_the_allowed_set(self):
        retriever, _document_map = _build_map()
        first = retriever.retrieve("How long is the warranty?")
        second = retriever.retrieve("How long is the warranty?")
        self.assertEqual([n.node_id for n in first], [n.node_id for n in second])
        allowed = [retriever.corpus[0], retriever.corpus[1]]
        restricted = retriever.retrieve(
            "refund window for standard merchandise", allowed_node_ids=allowed
        )
        self.assertTrue(restricted)
        self.assertTrue({node.node_id for node in restricted} <= set(allowed))
        self.assertEqual(
            retriever.retrieve("How long is the warranty?", allowed_node_ids=allowed),
            [],
        )

    def test_unrelated_questions_do_not_match(self):
        retriever, _document_map = _build_map()
        self.assertEqual(
            retriever.retrieve("What is the capital city of Portugal?"), []
        )
        self.assertEqual(retriever.retrieve("   "), [])


class AblationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads(
            (_TEMP_ROOT / "results" / "ablation.json").read_text(encoding="utf-8")
        )
        cls.result = {
            "rows": payload["routes"],
            "aggregates": payload["aggregates"],
            "resolution": payload["resolution"],
            "scaling": payload["scaling"],
        }
        cls.aggregates = {row["variant"]: row for row in payload["aggregates"]}
        cls.resolution = payload["resolution"]
        cls.scaling = payload["scaling"]

    def test_routing_engagement_is_reported_so_fallbacks_cannot_inflate_results(self):
        naive = self.aggregates["naive_vector_rag"]
        self.assertEqual(naive["routing_engaged_rate"], 0.0)
        high = self.aggregates["map_agent_high_resolution"]
        self.assertGreater(high["routing_engaged_rate"], 0.5)
        for row in self.result["rows"]:
            self.assertIn("routed", row)

    def test_scaling_sweep_covers_every_size_with_every_variant(self):
        self.assertTrue(self.scaling)
        for size in map_ablation.SCALING_SIZES:
            at_size = [row for row in self.scaling if row["documents"] == size]
            self.assertEqual(
                len(at_size), len(map_ablation._scaling_variants()), f"size {size}"
            )
        sizes = [row["documents"] for row in self.scaling]
        self.assertEqual(sorted(set(sizes)), sorted(map_ablation.SCALING_SIZES))
        chunks = {row["documents"]: row["chunks"] for row in self.scaling}
        self.assertEqual(chunks[8], 48)
        self.assertGreater(chunks[64], chunks[8])

    def test_map_index_cost_grows_while_naive_prompt_cost_stays_flat(self):
        def at(size, variant):
            return next(
                row
                for row in self.scaling
                if row["documents"] == size and row["variant"] == variant
            )

        naive_small = at(8, "global_topk")["mean_evidence_tokens"]
        naive_large = at(64, "global_topk")["mean_evidence_tokens"]
        self.assertLess(abs(naive_large - naive_small) / naive_small, 0.25)
        self.assertGreater(
            at(64, "map_agent_adaptive")["map_index_tokens"],
            at(8, "map_agent_adaptive")["map_index_tokens"],
        )

    def test_the_best_map_variant_beats_naive_on_context_at_equal_accuracy(self):
        for size in map_ablation.SCALING_SIZES:
            at_size = {
                row["variant"]: row for row in self.scaling if row["documents"] == size
            }
            naive = at_size["global_topk"]
            best = max(
                (
                    row
                    for row in at_size.values()
                    if row["variant"] in map_ablation.MAP_SCALING_KEYS
                ),
                key=lambda row: (row["hit_at_k"], -row["mean_evidence_tokens"]),
            )
            self.assertGreaterEqual(best["hit_at_k"], naive["hit_at_k"])
            self.assertLess(best["mean_evidence_tokens"], naive["mean_evidence_tokens"])

    def test_report_states_that_no_accuracy_crossover_was_observed(self):
        report = (_TEMP_ROOT / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("## Scaling study", report)
        self.assertIn("No accuracy crossover was observed", report)
        self.assertIn("## Cost model", report)
        self.assertIn("Break-even condition", report)

    def test_every_variant_scores_every_query(self):
        self.assertEqual(
            len(self.result["rows"]), len(map_ablation.VARIANTS) * len(fixtures.QUERIES)
        )
        for variant in map_ablation.VARIANTS:
            rows = [
                row for row in self.result["rows"] if row["variant"] == variant["key"]
            ]
            self.assertEqual(len(rows), len(fixtures.QUERIES))
            self.assertTrue(all("latency_ms" in row for row in rows))

    def test_the_study_covers_naive_agentic_and_map_families(self):
        keys = set(self.aggregates)
        self.assertIn("naive_vector_rag", keys)
        self.assertIn("global_topk", keys)
        self.assertIn("agentic_no_map", keys)
        self.assertIn("map_agent_low_resolution", keys)
        self.assertIn("map_agent_high_resolution", keys)
        self.assertIn("map_agent_full", keys)

    def test_naive_vector_baseline_is_a_real_dense_path(self):
        dense = fixtures.build_dense_retriever()
        results = dense.retrieve("What is the default API rate limit quota?")
        self.assertTrue(results)
        self.assertTrue(all(0.0 <= result.score <= 1.0 for result in results))
        first = fixtures.build_dense_retriever().retrieve(
            "What is the default API rate limit quota?"
        )
        self.assertEqual(
            [r.node.node_id for r in results], [r.node.node_id for r in first]
        )

    def test_flat_agent_has_no_document_structure_to_navigate(self):
        agent = fixtures.FlatTextAgent(fixtures.build_retriever())
        results, trace = agent.retrieve("How long is the warranty?")
        self.assertTrue(results)
        self.assertEqual(trace["planner_mode"], "flat_deterministic")
        self.assertEqual(len(trace["planned_section_ids"]), 4)

    def test_map_routes_use_fewer_tokens_than_global_topk(self):
        full = self.aggregates["map_agent_full"]
        baseline = self.aggregates["global_topk"]
        self.assertLess(full["mean_evidence_tokens"], baseline["mean_evidence_tokens"])
        self.assertLess(full["mean_token_delta_pct"], 0.0)
        self.assertLess(full["mean_nodes_scored"], baseline["mean_nodes_scored"])

    def test_precision_and_recall_are_chunk_level_and_bounded(self):
        for row in self.result["rows"]:
            self.assertGreaterEqual(row["precision"], 0.0)
            self.assertLessEqual(row["precision"], 1.0)
            self.assertIn(row["recall"], (0.0, 1.0))
            if row["query_kind"] == "answerable":
                self.assertEqual(row["hit"], bool(row["recall"]))
                if not row["hit"]:
                    self.assertEqual(row["precision"], 0.0)

    def test_finer_map_returns_less_context_than_a_coarser_map(self):
        high = self.aggregates["map_agent_high_resolution"]
        low = self.aggregates["map_agent_low_resolution"]
        self.assertLess(high["mean_evidence_tokens"], low["mean_evidence_tokens"])
        self.assertGreater(high["map_index_tokens"], low["map_index_tokens"])

    def test_the_production_default_resolution_is_the_one_under_test(self):
        from utils.retrieval_map import DEFAULT_CHUNKS_PER_SECTION

        self.assertEqual(map_ablation.RESOLUTION_DEFAULT, DEFAULT_CHUNKS_PER_SECTION)
        self.assertIn(
            str(DEFAULT_CHUNKS_PER_SECTION), self.aggregates["map_agent_full"]["label"]
        )
        self.assertEqual(
            [row["chunks_per_section"] for row in self.resolution][-1],
            DEFAULT_CHUNKS_PER_SECTION,
        )

    def test_resolution_sweep_reports_a_monotonic_token_tradeoff(self):
        rows = [
            row
            for row in self.result["resolution"]
            if not row.get("target_sections_per_document")
        ]
        rows.sort(key=lambda row: row["chunks_per_section"])
        index_tokens = [row["map_index_tokens"] for row in rows]
        evidence = [row["mean_evidence_tokens"] for row in rows]
        sections = [row["sections"] for row in rows]
        self.assertEqual(index_tokens, sorted(index_tokens, reverse=True))
        self.assertEqual(sections, sorted(sections, reverse=True))
        self.assertLess(evidence[0], evidence[-1])
        for row in rows:
            self.assertGreater(row["sections"], 0)
            self.assertEqual(row["nodes"], 48)

    def test_adaptive_resolution_divides_each_document_into_several_sections(self):
        adaptive = next(
            row
            for row in self.result["resolution"]
            if row.get("target_sections_per_document")
        )
        fixed = next(
            row
            for row in self.result["resolution"]
            if not row.get("target_sections_per_document")
            and row["chunks_per_section"] == adaptive["chunks_per_section"]
        )
        self.assertGreater(adaptive["sections"], fixed["sections"])
        self.assertGreaterEqual(adaptive["hit_at_k"], fixed["hit_at_k"])

    def test_report_states_the_negative_result_and_the_resolution_caveat(self):
        report = (_TEMP_ROOT / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("## Map resolution study", report)
        self.assertIn("not pixel or image resolution", report)
        self.assertIn("Precision", report)
        no_map = self.aggregates["agentic_no_map"]
        full = self.aggregates["map_agent_full"]
        if no_map["hit_at_k"] >= full["hit_at_k"]:
            self.assertIn("Negative result worth reporting", report)

    def test_routing_shrinks_the_prompt_without_inventing_hits(self):
        full = self.aggregates["map_agent_full"]
        single = self.aggregates["map_agent_single_section"]
        self.assertLess(single["mean_evidence_tokens"], full["mean_evidence_tokens"])
        self.assertLessEqual(
            full["hit_at_k"], self.aggregates["global_topk"]["hit_at_k"] + 0.05
        )

    def test_oracle_planner_is_an_upper_bound_not_a_win(self):
        oracle = self.aggregates["map_agent_oracle_planner"]
        full = self.aggregates["map_agent_full"]
        self.assertGreaterEqual(oracle["hit_at_k"], full["hit_at_k"])
        self.assertGreaterEqual(oracle["mean_precision"], full["mean_precision"])

    def test_no_match_questions_are_reported_separately_from_hard_negatives(self):
        baseline = self.aggregates["global_topk"]
        self.assertGreater(baseline["no_match_queries"], 0)
        self.assertGreater(baseline["hard_negative_queries"], 0)
        self.assertLess(baseline["no_match_rejection_rate"], 1.0)

    def test_artifacts_are_written_and_consistent(self):
        out = _TEMP_ROOT
        payload = json.loads(
            (out / "results" / "ablation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(payload["routes"]), len(self.result["rows"]))
        self.assertEqual(payload["run"]["top_k"], TOP_K)
        production = next(
            row
            for row in self.result["resolution"]
            if row["chunks_per_section"] == map_ablation.RESOLUTION_DEFAULT
        )
        self.assertEqual(payload["run"]["section_count"], production["sections"])
        self.assertEqual(
            payload["run"]["chunks_per_section"], production["chunks_per_section"]
        )
        with (out / "results" / "ablation.csv").open(encoding="utf-8") as handle:
            self.assertEqual(
                len(list(csv.DictReader(handle))), len(self.result["aggregates"])
            )
        with (out / "results" / "resolution.csv").open(encoding="utf-8") as handle:
            self.assertEqual(
                len(list(csv.DictReader(handle))), len(self.result["resolution"])
            )
        with (out / "results" / "scaling.csv").open(encoding="utf-8") as handle:
            self.assertEqual(
                len(list(csv.DictReader(handle))), len(self.result["scaling"])
            )
        for name in map_ablation.FIGURE_FILES:
            markup = (out / "figures" / name).read_text(encoding="utf-8")
            self.assertTrue(markup.startswith("<svg"))
            self.assertTrue(markup.rstrip().endswith("</svg>"))
        report = (out / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("## Limitations", report)
        self.assertIn("is not an LLM result", report)
        self.assertIn("## Map resolution study", report)
        self.assertIn(payload["run"]["corpus_fingerprint"], report)

    def test_route_figure_matches_the_app_view_model(self):
        out = _TEMP_ROOT
        payload = json.loads(
            (out / "results" / "ablation.json").read_text(encoding="utf-8")
        )
        trace = payload["route_example"]["trace"]
        retriever = fixtures.build_retriever(top_k=TOP_K)
        document_map = build_document_map(
            retriever,
            config=RetrievalMapConfig(
                chunks_per_section=map_ablation.RESOLUTION_DEFAULT
            ),
        )
        model = view.build_map_view_model(document_map, trace)
        self.assertFalse(model["map_mismatch"])
        self.assertEqual(
            [
                node["section_id"]
                for node in model["sections"]
                if node["state"] == view.STATE_SELECTED
            ],
            list(trace["selected_section_ids"]),
        )
        self.assertTrue(model["has_route"])


class MapAblationCliTests(unittest.TestCase):
    def test_cli_writes_outputs_into_the_requested_directory(self):
        target = Path(tempfile.mkdtemp(prefix="docmind_ablation_"))
        self.addCleanup(shutil.rmtree, target, True)
        code = map_ablation.main(["--top-k", "4", "--out-dir", str(target), "--quiet"])
        self.assertEqual(code, 0)
        self.assertTrue((target / "results" / "ablation.json").exists())
        self.assertTrue((target / "REPORT.md").exists())

    def test_check_passes_for_the_committed_research_artifacts(self):
        self.assertEqual(map_ablation.main(["--check", "--quiet"]), 0)
        self.assertEqual(
            map_ablation.check_artifacts(out_dir="research", fresh_dir=_TEMP_ROOT),
            [],
        )

    def test_check_detects_a_stale_committed_artifact(self):
        fresh = Path(tempfile.mkdtemp(prefix="docmind_ablation_fresh_"))
        self.addCleanup(shutil.rmtree, fresh, True)
        map_ablation.run_study(top_k=TOP_K, out_dir=fresh)
        committed = fresh.with_name(fresh.name + "_committed")
        shutil.copytree(fresh, committed)
        self.addCleanup(shutil.rmtree, committed, True)
        self.assertEqual(
            map_ablation.check_artifacts(out_dir=committed, fresh_dir=fresh), []
        )
        aggregates = (committed / "results" / "ablation.csv").read_text(
            encoding="utf-8"
        )
        (committed / "results" / "ablation.csv").write_text(
            aggregates.replace("1.0", "0.9", 1), encoding="utf-8"
        )
        self.assertIn(
            "ablation_csv",
            map_ablation.check_artifacts(out_dir=committed, fresh_dir=fresh),
        )

    def test_check_reports_a_missing_study(self):
        target = Path(tempfile.mkdtemp(prefix="docmind_ablation_absent_"))
        self.addCleanup(shutil.rmtree, target, True)
        self.assertTrue(map_ablation.check_artifacts(out_dir=target))

    def test_top_k_is_clamped_to_the_hard_limit(self):
        target = Path(tempfile.mkdtemp(prefix="docmind_ablation_k_"))
        self.addCleanup(shutil.rmtree, target, True)
        result = map_ablation.run_study(top_k=10_000, out_dir=target)
        self.assertEqual(result["run"]["top_k"], 50)
        for row in result["rows"]:
            self.assertLessEqual(row["result_count"], 50)


class GroundTruthRegressionTests(unittest.TestCase):
    def test_answerable_queries_are_retrievable_by_global_topk(self):
        retriever = fixtures.build_retriever(top_k=TOP_K)
        answers = map_ablation.answer_nodes()
        misses = []
        for query in fixtures.QUERIES:
            expected = answers.get(query["id"], ())
            if not expected:
                continue
            results = retriever.retrieve(query["question"], top_k=TOP_K)
            if not map_ablation._rank(results, expected):
                misses.append(query["id"])
        self.assertEqual(misses, [], f"fixture queries became unanswerable: {misses}")

    def test_answer_node_ids_exist_in_the_corpus(self):
        retriever = fixtures.build_retriever(top_k=TOP_K)
        self.assertEqual(len(map_ablation.answer_nodes()), 25)
        for nodes in map_ablation.answer_nodes().values():
            for node_id in nodes:
                self.assertIn(node_id, set(retriever.corpus))

    def test_multi_hop_questions_name_several_answer_chunks(self):
        hard_queries = fixtures.QUERIES + fixtures.MULTIHOP_QUERIES
        answers = map_ablation.answer_nodes(hard_queries)
        self.assertEqual(len(fixtures.MULTIHOP_QUERIES), 6)
        for query in fixtures.MULTIHOP_QUERIES:
            self.assertIn(query["id"], answers)
            self.assertEqual(len(answers[query["id"]]), len(query["docs"]))
            self.assertGreater(len(answers[query["id"]]), 1)

    def test_multi_hop_recall_requires_every_relevant_chunk(self):
        expected = ("doc#0", "doc#1")

        class _Node:
            def __init__(self, node_id):
                self.node_id = node_id

        class _Result:
            def __init__(self, node_id):
                self.node = _Node(node_id)

        partial = [_Result("doc#0")]
        precision, recall = map_ablation._precision_recall(partial, expected)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 0.5)
        _precision, full_recall = map_ablation._precision_recall(
            [_Result("doc#0"), _Result("doc#1")], expected
        )
        self.assertAlmostEqual(full_recall, 1.0)

    def test_distractor_documents_exist_and_reuse_corpus_vocabulary(self):
        self.assertGreaterEqual(len(fixtures.DISTRACTOR_DOCUMENTS), 4)
        combined = " ".join(
            " ".join(chunks) for _name, chunks in fixtures.DISTRACTOR_DOCUMENTS
        ).lower()
        for shared in ("refund", "incident", "rollback", "token"):
            self.assertIn(shared, combined)

    def test_filler_documents_are_deterministic_and_scale(self):
        first = fixtures.build_filler_documents(5)
        second = fixtures.build_filler_documents(5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({name for name, _ in first}), 5)
        self.assertEqual(len(fixtures.build_filler_documents(20)), 20)


class DocumentationContractTests(unittest.TestCase):
    def test_agents_md_does_not_claim_the_map_study_was_never_run(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("research/map_ablation.py", agents)


if __name__ == "__main__":
    unittest.main()
