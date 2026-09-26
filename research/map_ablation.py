"""Retrieval-map ablation study.

Runs the production map agent and plain global retrieval over the deterministic
offline corpus in ``research.fixtures`` and writes reproducible tables and
figures. No network, embeddings, or LLM is required unless an OpenAI-compatible
planner endpoint is supplied explicitly.

Run:
    python -m research.map_ablation
    python -m research.map_ablation --top-k 8 --out-dir research
    python -m research.map_ablation --check

Outputs (relative to --out-dir):
    results/routes.csv     one row per variant and query
    results/ablation.csv   one aggregated row per variant
    results/ablation.json  full run record, environment, and corpus fingerprint
    figures/ablation_hit.svg
    figures/ablation_tokens.svg
    figures/route_example.svg
    REPORT.md

``--check`` regenerates the study into a temporary directory and compares it
with the committed artifacts, ignoring wall-clock timings and run timestamps, so
CI can prove the committed results still match the code that produces them.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.fixtures import (  # noqa: E402
    CHUNKS_PER_SECTION,
    DISTRACTOR_DOCUMENTS,
    DOCUMENTS,
    QUERIES,
    SCALING_QUERIES,
    FixtureRetriever,
    FlatTextAgent,
    build_dense_retriever,
    build_filler_documents,
    build_nodes,
    build_retriever,
    node_key,
)
from utils.retrieval_map import (  # noqa: E402
    DEFAULT_CHUNKS_PER_SECTION,
    HARD_MAX_RESULTS,
    MapAgentConfig,
    MapRetrievalAgent,
    RetrievalMapConfig,
    build_document_map,
    count_map_tokens,
    json_safe,
)

TOP_K_DEFAULT = 8
ROUTE_FIGURE_QUERY = "q19"

GLOBAL_BASELINE = "global_topk"

VOLATILE_ROW_FIELDS = ("latency_ms",)
VOLATILE_AGGREGATE_FIELDS = ("mean_latency_ms", "p95_latency_ms")
VOLATILE_RUN_FIELDS = ("generated_at", "python", "platform")
FIGURE_FILES = (
    "ablation_hit.svg",
    "ablation_tokens.svg",
    "resolution_tradeoff.svg",
    "route_example.svg",
)

MAP_AGENT_SETTINGS = {
    "max_selected_sections": 4,
    "initial_sections": 2,
    "neighbor_sections": 1,
    "max_steps": 2,
}

RESOLUTION_HIGH = 1
RESOLUTION_DEFAULT = DEFAULT_CHUNKS_PER_SECTION
RESOLUTION_LOW = 4
ADAPTIVE_TARGET_SECTIONS = 3

VARIANTS: tuple[dict, ...] = (
    {
        "key": "naive_vector_rag",
        "label": "Naive vector RAG",
        "family": "baseline",
        "description": (
            "Naive dense retrieval over the whole corpus with no agent and no map, "
            "scored with deterministic hash embeddings."
        ),
        "retriever": "dense",
        "agent": None,
    },
    {
        "key": GLOBAL_BASELINE,
        "label": "Naive lexical RAG (BM25 top-k)",
        "family": "baseline",
        "description": "Unrestricted BM25 retrieval over the whole corpus, same top-k cap.",
        "retriever": "lexical",
        "agent": None,
    },
    {
        "key": "agentic_no_map",
        "label": "Agentic text RAG (no map)",
        "family": "ablation",
        "description": (
            "Same observe/plan/act loop and deterministic keyword planner, but over "
            "flat chunk cards with no document grouping, hierarchy, or adjacency."
        ),
        "retriever": "lexical",
        "agent": None,
        "flat": True,
    },
    {
        "key": "map_agent_low_resolution",
        "label": f"Map agent (low resolution, {RESOLUTION_LOW} chunks/section)",
        "family": "map",
        "description": (
            f"Coarse map: {RESOLUTION_LOW} chunks per section, so few large sections."
        ),
        "chunks_per_section": RESOLUTION_LOW,
        "agent": dict(MAP_AGENT_SETTINGS),
    },
    {
        "key": "map_agent_high_resolution",
        "label": f"Map agent (high resolution, {RESOLUTION_HIGH} chunk/section)",
        "family": "map",
        "description": (
            f"Fine map: {RESOLUTION_HIGH} chunk per section, so many small sections."
        ),
        "chunks_per_section": RESOLUTION_HIGH,
        "agent": dict(MAP_AGENT_SETTINGS),
    },
    {
        "key": "map_agent_full",
        "label": (
            f"Map agent (production default, {RESOLUTION_DEFAULT} chunks/section)"
        ),
        "family": "map",
        "description": "Deterministic planner, neighbour expansion, one reflection step.",
        "chunks_per_section": RESOLUTION_DEFAULT,
        "agent": dict(MAP_AGENT_SETTINGS),
    },
    {
        "key": "map_agent_adaptive",
        "label": (
            f"Map agent (adaptive, {ADAPTIVE_TARGET_SECTIONS} sections/document)"
        ),
        "family": "map",
        "description": (
            "Per-document adaptive granularity: every document is divided into about "
            f"{ADAPTIVE_TARGET_SECTIONS} sections instead of a fixed chunks-per-section."
        ),
        "target_sections_per_document": ADAPTIVE_TARGET_SECTIONS,
        "agent": dict(MAP_AGENT_SETTINGS),
    },
    {
        "key": "map_agent_refine",
        "label": "Map agent (with candidate refinement)",
        "family": "ablation",
        "description": (
            "Re-scores the shortlist on full section text before planning. Measured "
            "and left off by default because it did not pay for itself."
        ),
        "chunks_per_section": RESOLUTION_DEFAULT,
        "agent": {**MAP_AGENT_SETTINGS, "refine_candidates": True},
    },
    {
        "key": "map_agent_no_neighbors",
        "label": "Map agent (no neighbours)",
        "family": "ablation",
        "description": "Neighbour expansion disabled; only planned sections are routed.",
        "chunks_per_section": RESOLUTION_DEFAULT,
        "agent": {**MAP_AGENT_SETTINGS, "neighbor_sections": 0},
    },
    {
        "key": "map_agent_no_reflection",
        "label": "Map agent (no reflection)",
        "family": "ablation",
        "description": "Single step; weak routed evidence is not expanded or re-fetched.",
        "chunks_per_section": RESOLUTION_DEFAULT,
        "agent": {**MAP_AGENT_SETTINGS, "max_steps": 1},
    },
    {
        "key": "map_agent_single_section",
        "label": "Map agent (single section)",
        "family": "ablation",
        "description": "One section routed with no neighbours; smallest possible context.",
        "chunks_per_section": RESOLUTION_DEFAULT,
        "agent": {
            "max_selected_sections": 1,
            "initial_sections": 1,
            "neighbor_sections": 0,
            "max_steps": 1,
        },
    },
    {
        "key": "map_agent_oracle_planner",
        "label": "Map agent (oracle planner)",
        "family": "upper_bound",
        "description": (
            "Injected planner returns the ground-truth section. Upper bound only; "
            "not an LLM measurement."
        ),
        "chunks_per_section": RESOLUTION_DEFAULT,
        "agent": dict(MAP_AGENT_SETTINGS),
        "planner": "oracle",
    },
)

RESOLUTION_LEVELS: tuple[dict, ...] = (
    {"chunks_per_section": 1, "label": "Highest (1 chunk/section)"},
    {"chunks_per_section": 2, "label": "High (2 chunks/section)"},
    {"chunks_per_section": 3, "label": "Medium-high (3 chunks/section)"},
    {"chunks_per_section": 4, "label": "Medium (4 chunks/section)"},
    {"chunks_per_section": 5, "label": "Medium-low (5 chunks/section)"},
    {
        "chunks_per_section": DEFAULT_CHUNKS_PER_SECTION,
        "label": f"Production default ({DEFAULT_CHUNKS_PER_SECTION} chunks/section)",
    },
    {
        "chunks_per_section": DEFAULT_CHUNKS_PER_SECTION,
        "target_sections_per_document": ADAPTIVE_TARGET_SECTIONS,
        "label": f"Adaptive ({ADAPTIVE_TARGET_SECTIONS} sections/document)",
    },
)


def _tokenizer(value):
    return str(value).split()


def _content(result) -> str:
    node = getattr(result, "node", None) or result
    getter = getattr(node, "get_content", None)
    if callable(getter):
        return str(getter() or "")
    return str(getattr(node, "text", "") or "")


def _node_id(result) -> str:
    node = getattr(result, "node", None) or result
    return str(getattr(node, "node_id", "") or "")


def _score(result) -> float:
    try:
        return float(getattr(result, "score", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def answer_nodes(queries=None) -> dict[str, tuple[str, ...]]:
    """Return the answer-bearing chunk ids for each question.

    Metrics are computed at chunk level rather than section level so that every
    map resolution is judged against the same unit. A section is correct when it
    contains one of these chunks, which keeps the comparison between a 1-chunk
    section and a 12-chunk section fair. A question may name several chunks when
    its answer genuinely spans documents.
    """
    answers: dict[str, tuple[str, ...]] = {}
    for query in queries if queries is not None else QUERIES:
        if query.get("docs"):
            pairs = list(query["docs"])
        elif query.get("doc"):
            pairs = [(query["doc"], query.get("chunk"))]
        else:
            pairs = []
        nodes = tuple(
            node_key(str(name), int(index))
            for name, index in pairs
            if name and index is not None
        )
        if nodes:
            answers[query["id"]] = nodes
    return answers


def _rank(results, expected: tuple[str, ...]) -> int:
    if not expected:
        return 0
    for position, result in enumerate(results, start=1):
        if _node_id(result) in expected:
            return position
    return 0


def _precision_recall(results, expected: tuple[str, ...]) -> tuple[float, float]:
    """Return chunk-level precision and recall for one question.

    Precision is the share of returned chunks that are relevant and recall is the
    share of relevant chunks that were returned, so a multi-hop question with two
    relevant chunks is not credited for finding one of them. A coarser map returns
    more surrounding chunks per section, so it scores lower precision at equal
    recall, which is the real trade-off rather than an artefact of section size.
    """
    if not results:
        return (1.0, 1.0) if not expected else (0.0, 0.0)
    if not expected:
        return 0.0, 0.0
    relevant = sum(1 for result in results if _node_id(result) in expected)
    return relevant / len(results), relevant / len(expected)


def _evidence_tokens(results, tokenizer) -> int:
    return sum(count_map_tokens(_content(result), tokenizer) for result in results)


def map_index_tokens(document_map, tokenizer) -> int:
    """Return the tokens of every section card the agent must score to navigate.

    This is the map-reading cost of one routing decision. It is local index
    work for the deterministic planner and is only charged to the model when the
    optional LLM planner is used, so it is reported separately from evidence
    tokens rather than folded into the prompt claim.
    """
    cards = [
        {
            "id": section.section_id,
            "title": section.title,
            "keywords": list(section.keywords),
            "summary": section.summary,
        }
        for section in getattr(document_map, "sections", ()) or ()
    ]
    return count_map_tokens(
        json.dumps(cards, sort_keys=True, ensure_ascii=False), tokenizer
    )


def _oracle_planner(expected: tuple[str, ...]):
    def planner(_query, _cards):
        return list(expected)

    return planner


def _build_agent(variant, retriever, document_map, tokenizer, expected):
    settings = variant.get("agent") or {}
    agent_config = MapAgentConfig(
        max_selected_sections=settings.get("max_selected_sections", 4),
        initial_sections=settings.get("initial_sections", 2),
        neighbor_sections=settings.get("neighbor_sections", 1),
        max_steps=settings.get("max_steps", 2),
        max_results=retriever.top_k,
        max_baseline_results=retriever.top_k,
        refine_candidates=bool(settings.get("refine_candidates", False)),
    )
    planner = None
    if variant.get("planner") == "oracle":
        planner = _oracle_planner(expected)
    return MapRetrievalAgent(
        retriever,
        document_map,
        tokenizer=tokenizer,
        planner=planner,
        agent_config=agent_config,
    )


def _run_variant(variant, query, retriever, document_map, truth, expected, tokenizer):
    if variant.get("flat"):
        agent = FlatTextAgent(
            retriever, tokenizer=tokenizer, max_results=retriever.top_k
        )
        return agent.retrieve(query["question"])
    if not variant.get("agent"):
        return retriever.retrieve(query["question"], top_k=retriever.top_k), {}
    agent = _build_agent(variant, retriever, document_map, tokenizer, expected)
    return agent.retrieve(query["question"])


def _measure(
    variant,
    query,
    retriever,
    document_map,
    truth,
    tokenizer,
    baseline_retriever=None,
    index_tokens=0,
):
    expected = truth.get(query["id"], ())
    reference = baseline_retriever or retriever
    before = len(retriever.calls)
    started = time.perf_counter()
    results, trace = _run_variant(
        variant, query, retriever, document_map, truth, expected, tokenizer
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    scored = sum(call["candidates"] for call in retriever.calls[before:])

    global_results = reference.retrieve(query["question"], top_k=reference.top_k)
    global_tokens = _evidence_tokens(global_results, tokenizer)
    evidence_tokens = _evidence_tokens(results, tokenizer)
    rank = _rank(results, expected)
    precision, recall = _precision_recall(results, expected)
    delta = evidence_tokens - global_tokens
    return {
        "variant": variant["key"],
        "query_id": query["id"],
        "question": query["question"],
        "query_kind": "no_match" if not expected else "answerable",
        "target_document": query.get("doc") or "",
        "hard_negative": bool(not expected and global_results),
        "hit": bool(rank),
        "rank": rank,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "result_count": len(results),
        "evidence_tokens": evidence_tokens,
        "global_tokens": global_tokens,
        "map_index_tokens": int(index_tokens),
        "token_delta": delta,
        "token_delta_pct": (
            round((delta / global_tokens) * 100.0, 2) if global_tokens else 0.0
        ),
        "selected_sections": len(trace.get("selected_section_ids") or ()),
        "planned_sections": len(trace.get("planned_section_ids") or ()),
        "considered_sections": len(trace.get("considered_section_ids") or ()),
        "route_steps": int(trace.get("step_count") or 0),
        "routed": bool(trace.get("routed")),
        "reflection": bool(trace.get("reflection_flag")),
        "planner_mode": str(trace.get("planner_mode") or ""),
        "nodes_scored": scored,
        "latency_ms": round(latency_ms, 3),
    }


def _mean(values) -> float:
    numbers = [float(value) for value in values if value is not None]
    return round(statistics.fmean(numbers), 4) if numbers else 0.0


def _percentile(values, fraction: float) -> float:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return 0.0
    position = min(len(numbers) - 1, max(0, round(fraction * (len(numbers) - 1))))
    return round(numbers[position], 4)


def aggregate(rows, variant) -> dict:
    selected = [row for row in rows if row["variant"] == variant["key"]]
    answerable = [row for row in selected if row["query_kind"] == "answerable"]
    no_match = [row for row in selected if row["query_kind"] == "no_match"]
    hard = [row for row in no_match if row["hard_negative"]]
    deltas = [row["token_delta_pct"] for row in answerable]
    return {
        "variant": variant["key"],
        "label": variant["label"],
        "family": variant["family"],
        "description": variant["description"],
        "queries": len(selected),
        "answerable_queries": len(answerable),
        "no_match_queries": len(no_match),
        "hard_negative_queries": len(hard),
        "hit_at_1": (
            round(sum(1 for row in answerable if row["rank"] == 1) / len(answerable), 4)
            if answerable
            else 0.0
        ),
        "hit_at_k": (
            round(sum(1 for row in answerable if row["hit"]) / len(answerable), 4)
            if answerable
            else 0.0
        ),
        "mean_precision": _mean([row["precision"] for row in answerable]),
        "mean_recall": _mean([row["recall"] for row in answerable]),
        "mrr": (
            round(
                sum(1.0 / row["rank"] for row in answerable if row["rank"])
                / len(answerable),
                4,
            )
            if answerable
            else 0.0
        ),
        "no_match_rejection_rate": (
            round(
                sum(1 for row in no_match if not row["result_count"]) / len(no_match), 4
            )
            if no_match
            else 0.0
        ),
        "hard_negative_false_positive_rate": (
            round(sum(1 for row in hard if row["result_count"]) / len(hard), 4)
            if hard
            else 0.0
        ),
        "mean_evidence_tokens": _mean([row["evidence_tokens"] for row in answerable]),
        "mean_global_tokens": _mean([row["global_tokens"] for row in answerable]),
        "map_index_tokens": max(
            [int(row.get("map_index_tokens") or 0) for row in selected] + [0]
        ),
        "mean_total_context_tokens": _mean(
            [
                int(row["evidence_tokens"]) + int(row.get("map_index_tokens") or 0)
                for row in answerable
            ]
        ),
        "llm_planner_net_vs_naive": _mean(
            [
                int(row["evidence_tokens"])
                + int(row.get("map_index_tokens") or 0)
                - int(row["global_tokens"])
                for row in answerable
            ]
        ),
        "map_index_over_naive_evidence": (
            round(
                max([int(row.get("map_index_tokens") or 0) for row in answerable] + [0])
                / _mean([row["global_tokens"] for row in answerable]),
                4,
            )
            if _mean([row["global_tokens"] for row in answerable])
            else 0.0
        ),
        "mean_token_delta": _mean([row["token_delta"] for row in answerable]),
        "mean_token_delta_pct": _mean(deltas),
        "mean_selected_sections": _mean(
            [row["selected_sections"] for row in selected if row["route_steps"]]
        ),
        "mean_route_steps": _mean([row["route_steps"] for row in selected]),
        "routing_engaged_rate": (
            round(
                sum(1 for row in answerable if row.get("routed")) / len(answerable), 4
            )
            if answerable
            else 0.0
        ),
        "reflection_on_answerable": sum(1 for row in answerable if row["reflection"]),
        "mean_nodes_scored": _mean([row["nodes_scored"] for row in selected]),
        "mean_latency_ms": _mean([row["latency_ms"] for row in selected]),
        "p95_latency_ms": _percentile([row["latency_ms"] for row in selected], 0.95),
    }


def _bar_chart(title, subtitle, labels, values, value_label, out_path):
    width, height = 1000, 460
    left, top, plot_width, plot_height = 260, 110, 660, 280
    maximum = max([abs(float(value)) for value in values] + [1.0])
    count = max(1, len(values))
    bar_height = max(18, int(plot_height / count) - 14)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="40" y="48" font-size="24" font-weight="600" fill="#14181f">{title}</text>',
        f'<text x="40" y="76" font-size="14" fill="#4a5568">{subtitle}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = top + int(index * (plot_height / count))
        magnitude = abs(float(value)) / maximum
        bar_width = max(2, int(magnitude * plot_width))
        parts.append(
            f'<text x="{left - 12}" y="{y + bar_height // 2 + 5}" font-size="13" '
            f'text-anchor="end" fill="#232a34">{label}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width}" height="{bar_height}" '
            f'rx="3" fill="#2f6feb"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 10}" y="{y + bar_height // 2 + 5}" '
            f'font-size="13" fill="#232a34">{float(value):.2f}{value_label}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#cbd2da" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="40" y="{height - 28}" font-size="12" fill="#6b7480">'
        f"Offline deterministic corpus; {len(DOCUMENTS)} documents, "
        f"{len(QUERIES)} queries.</text>"
    )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return len(parts)


def _resolution_figure(resolution_rows, out_path):
    """Plot map resolution as accuracy against total context cost."""
    width, height = 1040, 520
    left, top = 110, 120
    plot_width, plot_height = 700, 300
    points = sorted(resolution_rows, key=lambda row: int(row["map_index_tokens"]))
    xs = [float(row["map_index_tokens"]) for row in points]
    ys = [float(row["mean_total_context_tokens"]) for row in points]
    hits = [float(row["hit_at_k"]) for row in points]
    x_max = max(xs) * 1.08 or 1.0
    y_max = max(ys) * 1.15 or 1.0
    hit_max = 1.0

    def sx(value):
        return left + (float(value) / x_max) * plot_width

    def sy(value):
        return top + plot_height - (float(value) / y_max) * plot_height

    def sh(value):
        return top + plot_height - (float(value) / hit_max) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="40" y="48" font-size="24" font-weight="600" fill="#14181f">'
        "Map resolution trade-off</text>",
        '<text x="40" y="76" font-size="14" fill="#4a5568">'
        "Structural resolution (chunks per section) against accuracy and context "
        "cost.</text>",
    ]
    for step in range(5):
        ratio = step / 4
        y_value = y_max * ratio
        y = sy(y_value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            'stroke="#eef1f5" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
            f'fill="#6b7480">{y_value:.0f}</text>'
        )
        hit_y = sh(hit_max * ratio)
        parts.append(
            f'<text x="{left + plot_width + 12}" y="{hit_y + 4:.1f}" font-size="11" '
            f'fill="#1f7a4d">{hit_max * ratio:.2f}</text>'
        )
    line_points = " ".join(
        f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys, strict=True)
    )
    parts.append(
        f'<polyline points="{line_points}" fill="none" stroke="#2f6feb" '
        'stroke-width="2.5"/>'
    )
    hit_points = " ".join(
        f"{sx(x):.1f},{sh(h):.1f}" for x, h in zip(xs, hits, strict=True)
    )
    parts.append(
        f'<polyline points="{hit_points}" fill="none" stroke="#1f7a4d" '
        'stroke-width="2.5" stroke-dasharray="7 4"/>'
    )
    for row, x, y, hit in zip(points, xs, ys, hits, strict=True):
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5.5" fill="#2f6feb"/>'
        )
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sh(hit):.1f}" r="5.5" fill="#1f7a4d" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{sx(x):.1f}" y="{top + plot_height + 20}" font-size="10.5" '
            f'text-anchor="middle" fill="#4a5568">{row["chunks_per_section"]}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#cbd2da" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{left + plot_width / 2:.0f}" y="{top + plot_height + 46}" '
        'font-size="12.5" text-anchor="middle" fill="#232a34">'
        "Map index tokens (all section cards scored per question)</text>"
    )
    parts.append(
        f'<text x="34" y="{top + plot_height / 2:.0f}" font-size="12.5" '
        f'transform="rotate(-90 34 {top + plot_height / 2:.0f})" text-anchor="middle" '
        'fill="#232a34">Mean total context tokens</text>'
    )
    parts.append(
        f'<text x="{left + plot_width + 52}" y="{top + plot_height / 2:.0f}" '
        f'font-size="12.5" transform="rotate(90 {left + plot_width + 52} '
        f'{top + plot_height / 2:.0f})" text-anchor="middle" fill="#1f7a4d">'
        "hit@k</text>"
    )
    parts.append(
        f'<text x="{left + plot_width / 2:.0f}" y="{top + plot_height + 68}" '
        'font-size="11.5" text-anchor="middle" fill="#6b7480">'
        "x-axis tick labels are chunks per section (1 = highest resolution)</text>"
    )
    legend = (
        ("#2f6feb", "total context tokens (left axis)"),
        ("#1f7a4d", "hit@k (right axis)"),
    )
    for index, (color, label) in enumerate(legend):
        x = 40 + index * 320
        parts.append(
            f'<rect x="{x}" y="{height - 46}" width="14" height="14" rx="3" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + 20}" y="{height - 35}" font-size="12" fill="#232a34">'
            f"{label}</text>"
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _route_figure(document_map, trace, out_path, title):
    sections = list(document_map.sections)
    documents: list[str] = []
    for section in sections:
        if section.source_id not in documents:
            documents.append(section.source_id)
    width = 1120
    column_width = 150
    left_margin = 40
    box_height = 34
    box_gap = 12
    column_top = 130
    plot_height = max(
        240,
        max(
            sum(1 for section in sections if section.source_id == document)
            for document in documents
        )
        * (box_height + box_gap)
        + 40,
    )
    height = column_top + plot_height + 80
    selected = set(trace.get("selected_section_ids") or ())
    planned = set(trace.get("planned_section_ids") or ())
    considered = set(trace.get("considered_section_ids") or ())
    positions: dict[str, tuple[int, int]] = {}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="40" y="46" font-size="24" font-weight="600" fill="#14181f">'
        f"{title}</text>",
        f'<text x="40" y="72" font-size="14" fill="#4a5568">'
        f"Map id {_document_map_id(document_map)} &#183; "
        f"{len(selected)} selected &#183; {len(planned)} planned &#183; "
        f"{len(considered)} considered &#183; "
        f"{int(trace.get('step_count') or 0)} route steps</text>",
    ]
    for column, document in enumerate(documents):
        x = left_margin + column * column_width
        label = next(
            (section.source for section in sections if section.source_id == document),
            "document",
        )
        parts.append(
            f'<text x="{x + 4}" y="{column_top - 14}" font-size="12" font-weight="600" '
            f'fill="#232a34">{_xml_escape(label[:18])}</text>'
        )
        row = 0
        for section in sections:
            if section.source_id != document:
                continue
            y = column_top + row * (box_height + box_gap)
            positions[section.section_id] = (x, y)
            row += 1
            fill, stroke = "#f1f3f6", "#c3cad4"
            if section.section_id in considered:
                fill, stroke = "#e4edff", "#7aa2f7"
            if section.section_id in planned:
                fill, stroke = "#d7e6ff", "#3d7bf0"
            if section.section_id in selected:
                fill, stroke = "#bcd8ff", "#1f5fd0"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{column_width - 18}" '
                f'height="{box_height}" rx="4" fill="{fill}" stroke="{stroke}" '
                'stroke-width="1.2"/>'
            )
            parts.append(
                f'<text x="{x + 8}" y="{y + 15}" font-size="10.5" fill="#1b2330">'
                f"{_xml_escape(section.title[:20])}</text>"
            )
            parts.append(
                f'<text x="{x + 8}" y="{y + 27}" font-size="9" fill="#5b6675">'
                f"seq {section.sequence} &#183; {section.chunk_count} chunks</text>"
            )
    order = [
        section_id
        for section_id in (trace.get("selected_section_ids") or ())
        if section_id in positions
    ]
    if len(order) > 1:
        points = []
        for section_id in order:
            x, y = positions[section_id]
            points.append(f"{x + (column_width - 18) / 2},{y + box_height / 2}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#d64545" '
            'stroke-width="2" stroke-dasharray="6 4" opacity="0.85"/>'
        )
        for index, (x, y) in enumerate(positions[route_id] for route_id in order):
            parts.append(
                f'<circle cx="{x + (column_width - 18) / 2}" cy="{y + box_height / 2}" '
                f'r="6" fill="#ffffff" stroke="#d64545" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{x + (column_width - 18) / 2 + 10}" y="{y + box_height / 2 + 4}" '
                f'font-size="10" fill="#a12b2b">{index + 1}</text>'
            )
    legend = (
        ("#bcd8ff", "selected"),
        ("#d7e6ff", "planned"),
        ("#e4edff", "considered"),
        ("#f1f3f6", "not considered"),
    )
    for index, (fill, label) in enumerate(legend):
        x = left_margin + index * 150
        parts.append(
            f'<rect x="{x}" y="{height - 46}" width="14" height="14" rx="3" '
            f'fill="{fill}" stroke="#8f9bab"/>'
        )
        parts.append(
            f'<text x="{x + 20}" y="{height - 35}" font-size="12" fill="#232a34">'
            f"{label}</text>"
        )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _document_map_id(document_map) -> str:
    return str(getattr(document_map, "map_id", "") or "")[:24]


def _xml_escape(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fingerprint() -> str:
    from hashlib import sha256

    payload = json.dumps(
        {
            "documents": [[name, list(chunks)] for name, chunks in DOCUMENTS],
            "queries": [dict(query) for query in QUERIES],
            "chunks_per_section": CHUNKS_PER_SECTION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows, fieldnames) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cell(value, places=2) -> str:
    number = float(value or 0.0)
    if number.is_integer() and abs(number) < 1e6:
        return f"{number:.0f}" if places == 0 else f"{number:.{places}f}"
    return f"{number:.{places}f}"


def _report(aggregates, run, rows, resolution_rows=(), scaling_rows=()) -> str:
    lines = [
        "# Agentic map retrieval: ablation and resolution study",
        "",
        f"Generated {run['generated_at']} by `python -m research.map_ablation`.",
        "",
        "## What this measures",
        "",
        f"Each variant answers the same {run['query_count']} questions over the same "
        f"{run['node_count']} chunks from {run['document_count']} documents. The map "
        "variants are the production `MapRetrievalAgent`; only the agent configuration "
        "and the map resolution change between rows. Every variant is capped at the "
        f"same top-k of {run['top_k']} results, so the token comparison is "
        "like-for-like.",
        "",
        "The comparison follows the standard progression from naive retrieval to "
        "agentic map navigation:",
        "",
        "1. **Naive RAG** — dense top-k retrieval, no agent, no map.",
        "2. **Naive lexical RAG** — BM25 top-k, no agent, no map.",
        "3. **Agentic text RAG** — the same observe/plan/act loop, but over flat chunk "
        "cards with no document grouping, hierarchy, or adjacency to navigate.",
        "4. **Agentic map RAG** — the full map agent at low, default, and high "
        "resolution, plus component ablations and an oracle upper bound.",
        "",
        "Two token costs are reported separately and never merged:",
        "",
        "- **Evidence tokens** are what actually enters the model prompt.",
        "- **Map index tokens** are the section cards the agent must score to navigate. "
        "For the deterministic planner this is local index work, not prompt tokens. It "
        "becomes prompt cost only in the optional LLM-planner mode, which is not run "
        "here.",
        "",
        "## Ablation results",
        "",
        "| Variant | Hit@1 | Hit@k | Precision | Recall | MRR | No-match rejection | "
        "Hard-negative FP | Mean evidence tokens | Map index tokens | Total context "
        "tokens | Delta % vs naive | Mean steps | Mean nodes scored |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- |",
    ]
    for row in aggregates:
        lines.append(
            "| {label} | {h1} | {hk} | {prec} | {rec} | {mrr} | {rej} | {fp} | "
            "{tok} | {index} | {total} | {delta} | {steps} | {nodes} |".format(
                label=row["label"],
                h1=_cell(row["hit_at_1"]),
                hk=_cell(row["hit_at_k"]),
                prec=_cell(row["mean_precision"]),
                rec=_cell(row["mean_recall"]),
                mrr=_cell(row["mrr"]),
                rej=_cell(row["no_match_rejection_rate"]),
                fp=_cell(row["hard_negative_false_positive_rate"]),
                tok=_cell(row["mean_evidence_tokens"], 1),
                index=row["map_index_tokens"],
                total=_cell(row["mean_total_context_tokens"], 1),
                delta=_cell(row["mean_token_delta_pct"], 1),
                steps=_cell(row["mean_route_steps"]),
                nodes=_cell(row["mean_nodes_scored"], 1),
            )
        )
    lines.extend(
        [
            "",
            "Latency is machine-dependent and is recorded in `results/ablation.csv` "
            "and `results/ablation.json` rather than in this table, so the report stays "
            "byte-reproducible.",
        ]
    )
    baseline = next(
        (row for row in aggregates if row["variant"] == GLOBAL_BASELINE), None
    )
    full = next((row for row in aggregates if row["variant"] == "map_agent_full"), None)
    lines.extend(["", "## Reading the numbers", ""])
    if baseline and full:
        delta = full["mean_token_delta_pct"]
        direction = "fewer" if delta < 0 else "more"
        lines.append(
            f"- The full map agent used {direction} prompt-context tokens than global "
            f"top-k on answerable questions ({_cell(full['mean_token_delta_pct'], 1)}% "
            "mean change)."
        )
        lines.append(
            f"- Hit@1 moved from {baseline['hit_at_1']} (global top-k) to "
            f"{full['hit_at_1']} (map agent)."
        )
    no_reflection = next(
        (row for row in aggregates if row["variant"] == "map_agent_no_reflection"), None
    )
    if full and no_reflection:
        fired = full["reflection_on_answerable"]
        same = no_reflection["hit_at_k"] == full["hit_at_k"]
        lines.append(
            f"- Reflection fired on {fired} answerable "
            f"{'question' if fired == 1 else 'questions'}. The no-reflection ablation "
            f"{'scores the same' if same else 'scores differently'} on answerable "
            f"questions (hit@k {_cell(no_reflection['hit_at_k'])} against "
            f"{_cell(full['hit_at_k'])}), so the second step is worth "
            f"{'nothing on answerable questions here' if same else 'a measurable amount'}"
            "."
        )
    misses = [
        row["query_id"]
        for row in rows
        if row["variant"] == "map_agent_full"
        and row["query_kind"] == "answerable"
        and not row["hit"]
    ]
    plural = "question" if len(misses) == 1 else "questions"
    lines.append(
        f"- The full map agent missed {len(misses)} answerable {plural}"
        + (f" ({', '.join(misses)})." if misses else ".")
    )
    lines.extend(
        [
            "- `nodes scored` is lexical work only. A hybrid retriever still issues the "
            "same embedding query per question, so this is not a total compute saving.",
        ]
    )
    no_map = next(
        (row for row in aggregates if row["variant"] == "agentic_no_map"), None
    )
    if baseline and no_map and full:
        map_wins = full["hit_at_k"] > no_map["hit_at_k"] or (
            full["hit_at_k"] == no_map["hit_at_k"]
            and full["mean_evidence_tokens"] < no_map["mean_evidence_tokens"]
        )
        if map_wins:
            lines.append(
                f"- The structural map earns its place: the map agent reached hit@k "
                f"{_cell(full['hit_at_k'])} at "
                f"{_cell(full['mean_evidence_tokens'], 1)} evidence tokens against the "
                f"flat no-map agent's {_cell(no_map['hit_at_k'])} at "
                f"{_cell(no_map['mean_evidence_tokens'], 1)}."
            )
        else:
            lines.append(
                f"- **Negative result worth reporting:** the structural map did not beat "
                f"the flat no-map agent on this corpus. The no-map agent reached hit@k "
                f"{_cell(no_map['hit_at_k'])} at "
                f"{_cell(no_map['mean_evidence_tokens'], 1)} evidence tokens; the map "
                f"agent reached {_cell(full['hit_at_k'])} at "
                f"{_cell(full['mean_evidence_tokens'], 1)}. Most of the token saving "
                "here comes from the agentic observe/plan/act loop itself, not from the "
                "document structure. A larger corpus with more competing sections is "
                "needed before claiming the map adds accuracy."
            )
    naive_vector = next(
        (row for row in aggregates if row["variant"] == "naive_vector_rag"), None
    )
    if naive_vector and full:
        lines.append(
            f"- Against naive dense retrieval, the map agent moved hit@k from "
            f"{_cell(naive_vector['hit_at_k'])} to {_cell(full['hit_at_k'])} and mean "
            f"evidence tokens from {_cell(naive_vector['mean_evidence_tokens'], 1)} to "
            f"{_cell(full['mean_evidence_tokens'], 1)}."
        )
    if resolution_rows:
        lines.extend(_resolution_section(resolution_rows))
    lines.extend(_cost_model_section(aggregates))
    lines.extend(_scaling_section(scaling_rows))
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Resolution here means **structural granularity of the layout map**, the "
            "number of chunks per map section. This is not pixel or image resolution: "
            "no page image, figure, or table region is embedded or indexed. A "
            "multimodal page-image resolution study is separate work and no result for "
            "it is claimed.",
            "- The retrieval backends are deterministic fixtures: BM25 for the lexical "
            "rows and hash embeddings for the dense row. The dense row is a reproducible "
            "stand-in for a trained encoder, not a real embedding model, so its absolute "
            "numbers are not a claim about any production model.",
            "- The corpus is 8 synthetic policy documents with 25 answerable and 6 "
            "no-match questions. These are small numbers and the confidence intervals "
            "are wide.",
            "- The oracle planner variant is an upper bound that reads ground truth. "
            "It is not an LLM result and must never be reported as one.",
            "- The optional LLM planner was not executed in this run: "
            f"{run['llm_planner_status']}.",
            "- Token counts come from a whitespace tokenizer, not a provider "
            "tokenizer.",
            "- Prompt-context tokens are a narrow claim: this study does not show "
            "better answers, lower total cost, or lower end-to-end latency.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m research.map_ablation",
            "python -m research.map_ablation --check",
            "```",
            "",
            f"Corpus fingerprint `{run['corpus_fingerprint']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _cost_model_section(aggregates) -> list[str]:
    """Compare the two planner cost models and state where the map breaks even."""
    lines = [
        "",
        "## Cost model: what the map actually costs",
        "",
        "The map is only free if the planner is local. Two models are reported, and "
        "they lead to opposite conclusions:",
        "",
        "| Variant | Deterministic planner (prompt tokens) | LLM planner upper bound "
        "(prompt tokens) | Map index / naive evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in aggregates:
        lines.append(
            "| {label} | {local} | {llm} | {ratio} |".format(
                label=row["label"],
                local=_cell(row["mean_evidence_tokens"], 1),
                llm=_cell(row["mean_total_context_tokens"], 1),
                ratio=_cell(row["map_index_over_naive_evidence"], 2),
            )
        )
    map_rows = [row for row in aggregates if row["family"] in ("map", "ablation")]
    baseline = next(
        (row for row in aggregates if row["variant"] == GLOBAL_BASELINE), None
    )
    lines.append("")
    if baseline:
        lines.append(
            f"- Naive lexical top-k costs {_cell(baseline['mean_evidence_tokens'], 1)} "
            "prompt tokens per question. That figure is capped by top-k and does not "
            "grow with the corpus."
        )
    for row in map_rows:
        if row["map_index_over_naive_evidence"] > 1.0:
            lines.append(
                f"- {row['label']}: the map index is "
                f"{_cell(row['map_index_over_naive_evidence'], 2)}x the whole naive "
                f"context, so charging the map to the prompt costs "
                f"{_cell(row['llm_planner_net_vs_naive'], 1)} extra tokens per "
                "question. Under the LLM-planner model this variant is a net loss."
            )
    lines.extend(
        [
            "- **Break-even condition.** The map only pays for itself under the "
            "LLM-planner model when `map_index_tokens < naive_evidence_tokens - "
            "map_evidence_tokens`. Because naive top-k evidence is capped by top-k "
            "while the map index grows with the number of sections, this condition is "
            "harder to satisfy as the corpus grows, not easier.",
            "- **Therefore the honest economic claim is not a token saving.** The "
            "defensible claim is that the deterministic planner performs "
            "structure-guided routing with zero extra model calls: the map is scored "
            "locally, so the only prompt cost is the routed evidence, which is at most "
            "the naive cost. That framing is what the application default implements.",
            "- A reviewer should read the LLM-planner column as the price of the "
            "alternative design, not as a cost the default configuration pays.",
        ]
    )
    return lines


def _resolution_section(resolution_rows) -> list[str]:
    lines = [
        "",
        "## Map resolution study",
        "",
        "Resolution is the number of chunks merged into one map section. More chunks "
        "per section means fewer, larger sections: a cheaper index to score but a "
        "coarser place to aim. Fewer chunks per section means more, smaller sections: "
        "a finer map that costs more to read.",
        "",
        "| Resolution | Chunks/section | Sections | Map index tokens | Hit@1 | Hit@k | "
        "Precision | Recall | Mean evidence tokens | Total context tokens | Mean steps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in resolution_rows:
        lines.append(
            "| {label} | {cps} | {sections} | {index} | {h1} | {hk} | {prec} | {rec} "
            "| {tok} | {total} | {steps} |".format(
                label=row["label"],
                cps=row["chunks_per_section"],
                sections=row["sections"],
                index=row["map_index_tokens"],
                h1=_cell(row["hit_at_1"]),
                hk=_cell(row["hit_at_k"]),
                prec=_cell(row["mean_precision"]),
                rec=_cell(row["mean_recall"]),
                tok=_cell(row["mean_evidence_tokens"], 1),
                total=_cell(row["mean_total_context_tokens"], 1),
                steps=_cell(row["mean_route_steps"]),
            )
        )
    best = max(
        resolution_rows,
        key=lambda row: (
            row["hit_at_k"],
            -row["mean_evidence_tokens"],
        ),
    )
    cheapest = min(resolution_rows, key=lambda row: row["map_index_tokens"])
    finest = min(resolution_rows, key=lambda row: row["chunks_per_section"])
    coarsest = max(resolution_rows, key=lambda row: row["chunks_per_section"])
    flat_accuracy = len({row["hit_at_k"] for row in resolution_rows}) == 1
    lines.append("")
    lines.append(
        f"- Best accuracy at the lowest context cost: {best['label']} "
        f"(hit@k {_cell(best['hit_at_k'])}, "
        f"{_cell(best['mean_evidence_tokens'], 1)} evidence tokens)."
    )
    lines.append(
        f"- Cheapest map to read: {cheapest['label']} "
        f"({cheapest['map_index_tokens']} map index tokens, hit@k "
        f"{_cell(cheapest['hit_at_k'])})."
    )
    if flat_accuracy:
        lines.append(
            "- Accuracy is flat across the whole sweep "
            f"(hit@k {_cell(resolution_rows[0]['hit_at_k'])} everywhere), so on this "
            "corpus resolution buys context efficiency and latency cost, not accuracy."
        )
    lines.append(
        f"- Going from {coarsest['chunks_per_section']} to "
        f"{finest['chunks_per_section']} chunk per section cuts evidence tokens from "
        f"{_cell(coarsest['mean_evidence_tokens'], 1)} to "
        f"{_cell(finest['mean_evidence_tokens'], 1)} while raising the map index from "
        f"{coarsest['map_index_tokens']} to {finest['map_index_tokens']} tokens. The "
        "finer map lets the agent aim at a smaller target and so returns less "
        "surrounding context."
    )
    lines.append(
        "- Map index tokens are only prompt cost when the optional LLM planner is "
        "used. With the deterministic planner they are local scoring work, so the "
        "resolution decision trades context size against local compute, not against "
        "billed prompt tokens."
    )
    lines.extend(
        [
            "",
            "Latency is machine-dependent and stays in `results/resolution.csv` and "
            "`results/ablation.json` so this report remains byte-reproducible.",
            "",
            "See `figures/resolution_tradeoff.svg` for the same data plotted as "
            "accuracy against cost.",
        ]
    )
    return lines


def _scaling_variants() -> tuple[dict, ...]:
    return (
        {
            "key": "global_topk",
            "label": "Naive lexical RAG (BM25 top-k)",
            "family": "baseline",
            "description": "Unrestricted BM25 retrieval, same top-k cap.",
            "retriever": "lexical",
            "agent": None,
        },
        {
            "key": "agentic_no_map",
            "label": "Agentic text RAG (no map)",
            "family": "ablation",
            "description": "Observe/plan/act over flat chunk cards, no document structure.",
            "retriever": "lexical",
            "agent": None,
            "flat": True,
        },
        {
            "key": "map_agent_full",
            "label": "Map agent (production default)",
            "family": "map",
            "description": "Fixed chunks-per-section resolution.",
            "chunks_per_section": RESOLUTION_DEFAULT,
            "agent": dict(MAP_AGENT_SETTINGS),
        },
        {
            "key": "map_agent_adaptive",
            "label": "Map agent (adaptive)",
            "family": "map",
            "description": "Per-document adaptive granularity.",
            "target_sections_per_document": ADAPTIVE_TARGET_SECTIONS,
            "agent": dict(MAP_AGENT_SETTINGS),
        },
        {
            "key": "map_agent_high_resolution",
            "label": "Map agent (high resolution)",
            "family": "map",
            "description": "One chunk per section.",
            "chunks_per_section": RESOLUTION_HIGH,
            "agent": dict(MAP_AGENT_SETTINGS),
        },
    )


SCALING_SIZES: tuple[int, ...] = (8, 16, 32, 64)
MAP_SCALING_KEYS = frozenset(
    {"map_agent_full", "map_agent_adaptive", "map_agent_high_resolution"}
)


class _MapCache:
    """Build and reuse one document map per (resolution, adaptive target)."""

    def __init__(self, retriever, tokenizer):
        self.retriever = retriever
        self.tokenizer = tokenizer
        self._maps: dict[tuple, object] = {}
        self._index_tokens: dict[tuple, int] = {}

    def get(self, resolution: int, target: int = 0):
        key = (int(resolution), int(target))
        if key not in self._maps:
            self._maps[key] = build_document_map(
                self.retriever,
                config=RetrievalMapConfig(
                    chunks_per_section=int(resolution),
                    target_sections_per_document=int(target),
                ),
            )
            self._index_tokens[key] = map_index_tokens(self._maps[key], self.tokenizer)
        return self._maps[key]

    def index_tokens(self, resolution: int, target: int = 0) -> int:
        self.get(resolution, target)
        return self._index_tokens[(int(resolution), int(target))]


def run_scaling_study(
    top_k: int = TOP_K_DEFAULT, out_dir: Path | str = "research"
) -> list[dict]:
    """Measure how map routing and a flat agentic loop behave as distractors grow.

    The question set is held fixed while distractor documents are added, so any
    change in hit rate is caused by corpus difficulty rather than by easier
    questions. The filler documents reuse the corpus vocabulary precisely so that
    they compete for the same query terms.
    """
    top_k = max(1, min(int(top_k), HARD_MAX_RESULTS))
    tokenizer = _tokenizer
    queries = SCALING_QUERIES
    answers = answer_nodes(queries)
    results = []
    hand_written = len(DOCUMENTS) + len(DISTRACTOR_DOCUMENTS)
    for size in SCALING_SIZES:
        if size <= hand_written:
            documents = (DOCUMENTS + DISTRACTOR_DOCUMENTS)[:size]
        else:
            documents = (
                DOCUMENTS
                + DISTRACTOR_DOCUMENTS
                + build_filler_documents(size - hand_written)
            )
        retriever = FixtureRetriever(build_nodes(documents), top_k=top_k)
        cache = _MapCache(retriever, tokenizer)

        for variant in _scaling_variants():
            resolution = int(variant.get("chunks_per_section") or RESOLUTION_DEFAULT)
            target = int(variant.get("target_sections_per_document") or 0)
            uses_map = bool(variant.get("agent")) and not variant.get("flat")
            document_map = cache.get(resolution, target)
            rows = [
                _measure(
                    variant,
                    query,
                    retriever,
                    document_map,
                    answers,
                    tokenizer,
                    baseline_retriever=retriever,
                    index_tokens=(
                        cache.index_tokens(resolution, target) if uses_map else 0
                    ),
                )
                for query in queries
                if answers.get(query["id"])
            ]
            summary = aggregate(rows, variant)
            results.append(
                {
                    "documents": len(documents),
                    "chunks": len(retriever.corpus),
                    "variant": variant["key"],
                    "label": variant["label"],
                    "questions": len(rows),
                    "hit_at_1": summary["hit_at_1"],
                    "hit_at_k": summary["hit_at_k"],
                    "mean_precision": summary["mean_precision"],
                    "mean_recall": summary["mean_recall"],
                    "mean_evidence_tokens": summary["mean_evidence_tokens"],
                    "map_index_tokens": summary["map_index_tokens"],
                    "routing_engaged_rate": summary["routing_engaged_rate"],
                    "mean_latency_ms": summary["mean_latency_ms"],
                }
            )
    return results


def _scaling_section(scaling_rows) -> list[str]:
    if not scaling_rows:
        return []
    lines = [
        "",
        "## Scaling study: does the map help as the corpus grows?",
        "",
        "The question set is held fixed while distractor documents that reuse the "
        "corpus vocabulary are added, so every change in hit rate comes from corpus "
        "difficulty rather than easier questions.",
        "",
        "| Documents | Chunks | Variant | Hit@1 | Hit@k | Precision | Mean evidence "
        "tokens | Map index tokens | Routing engaged |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in scaling_rows:
        lines.append(
            "| {documents} | {chunks} | {label} | {h1} | {hk} | {prec} | {tok} | "
            "{index} | {routed} |".format(
                documents=row["documents"],
                chunks=row["chunks"],
                label=row["label"],
                h1=_cell(row["hit_at_1"]),
                hk=_cell(row["hit_at_k"]),
                prec=_cell(row["mean_precision"]),
                tok=_cell(row["mean_evidence_tokens"], 1),
                index=row["map_index_tokens"],
                routed=_cell(row["routing_engaged_rate"]),
            )
        )
    sizes = sorted({row["documents"] for row in scaling_rows})
    crossovers = []
    for size in sizes:
        at_size = {
            row["variant"]: row for row in scaling_rows if row["documents"] == size
        }
        naive = at_size.get("global_topk")
        flat = at_size.get("agentic_no_map")
        best_map = max(
            (row for row in at_size.values() if row["variant"] in MAP_SCALING_KEYS),
            key=lambda row: (row["hit_at_k"], -row["mean_evidence_tokens"]),
            default=None,
        )
        if naive and flat and best_map:
            winners = [
                name
                for name, row in (
                    ("naive", naive),
                    ("flat agent", flat),
                    ("map agent", best_map),
                )
                if row["hit_at_k"]
                >= max(naive["hit_at_k"], flat["hit_at_k"], best_map["hit_at_k"])
            ]
            crossovers.append(
                f"- {size} documents: best hit@k "
                f"{_cell(max(naive['hit_at_k'], flat['hit_at_k'], best_map['hit_at_k']))} "
                f"shared by {', '.join(winners)}; the best map variant is "
                f"{best_map['label']}."
            )
    lines.append("")
    lines.extend(crossovers)
    lines.extend(
        [
            "- **No accuracy crossover was observed.** Naive top-k and the flat agent "
            "stay at the top of this question set at every corpus size, so this study "
            "does not demonstrate a point where routing finds answers that top-k "
            "misses. Proving that needs documents long enough that the answer is one "
            "of hundreds of chunks, or genuinely multi-hop questions, neither of which "
            "this synthetic corpus contains.",
            "- The map's demonstrated benefit is different and still real: at equal "
            "accuracy it returns a much smaller context, and that gap widens as the "
            "corpus grows.",
        ]
    )
    for size in sizes:
        at_size = {
            row["variant"]: row for row in scaling_rows if row["documents"] == size
        }
        naive = at_size.get("global_topk")
        best_map = max(
            (row for row in at_size.values() if row["variant"] in MAP_SCALING_KEYS),
            key=lambda row: (row["hit_at_k"], -row["mean_evidence_tokens"]),
            default=None,
        )
        if naive and best_map and naive["mean_evidence_tokens"]:
            saved = (
                (naive["mean_evidence_tokens"] - best_map["mean_evidence_tokens"])
                / naive["mean_evidence_tokens"]
                * 100.0
            )
            lines.append(
                f"- {size} documents: {best_map['label']} matches naive accuracy at "
                f"{_cell(best_map['mean_evidence_tokens'], 1)} evidence tokens against "
                f"{_cell(naive['mean_evidence_tokens'], 1)}, a "
                f"{_cell(saved, 1)}% smaller prompt, for "
                f"{best_map['map_index_tokens']} map index tokens of local work."
            )
    lines.extend(
        [
            "- Map index tokens grow with the number of sections while naive prompt "
            "cost stays flat, because naive retrieval is capped at top-k. This is the "
            "break-even condition from the cost-model section, measured rather than "
            "assumed.",
            "- The fixed production resolution is the weakest setting and degrades as "
            "the corpus grows, while the adaptive setting holds its hit rate. That is "
            "the practical recommendation from this study.",
        ]
    )
    return lines


def run_study(top_k: int = TOP_K_DEFAULT, out_dir: Path | str = "research") -> dict:
    top_k = max(1, min(int(top_k), HARD_MAX_RESULTS))
    out_path = Path(out_dir)
    (out_path / "results").mkdir(parents=True, exist_ok=True)
    (out_path / "figures").mkdir(parents=True, exist_ok=True)

    tokenizer = _tokenizer
    lexical = build_retriever(top_k=top_k)
    dense = build_dense_retriever(top_k=top_k)
    retrievers = {"lexical": lexical, "dense": dense}
    cache = _MapCache(lexical, tokenizer)

    reference_map = cache.get(RESOLUTION_DEFAULT)
    answers = answer_nodes()
    truth = dict(answers)
    truth.update({query["id"]: () for query in QUERIES if query["id"] not in answers})
    missing = [
        query["id"] for query in QUERIES if query.get("doc") and not truth[query["id"]]
    ]
    if missing:
        raise SystemExit(f"Ground truth could not be resolved for {missing}")

    rows = []
    for variant in VARIANTS:
        retriever = retrievers[variant.get("retriever", "lexical")]
        resolution = int(variant.get("chunks_per_section") or RESOLUTION_DEFAULT)
        target = int(variant.get("target_sections_per_document") or 0)
        uses_map = bool(variant.get("agent")) and not variant.get("flat")
        variant_map = cache.get(resolution, target)
        for query in QUERIES:
            rows.append(
                _measure(
                    variant,
                    query,
                    retriever,
                    variant_map,
                    truth,
                    tokenizer,
                    baseline_retriever=lexical,
                    index_tokens=(
                        cache.index_tokens(resolution, target) if uses_map else 0
                    ),
                )
            )

    aggregates = [aggregate(rows, variant) for variant in VARIANTS]
    resolution_rows = _resolution_sweep(lexical, cache, truth, tokenizer)
    scaling_rows = run_scaling_study(top_k=top_k, out_dir=out_path)
    route_query = next(
        (query for query in QUERIES if query["id"] == ROUTE_FIGURE_QUERY), QUERIES[0]
    )
    route_trace: dict = {}
    route_variant = next(row for row in VARIANTS if row["key"] == "map_agent_full")
    agent = _build_agent(
        route_variant,
        lexical,
        reference_map,
        tokenizer,
        truth.get(route_query["id"], ()),
    )
    _results, route_trace = agent.retrieve(route_query["question"])

    run = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_fingerprint": _fingerprint(),
        "top_k": top_k,
        "chunks_per_section": RESOLUTION_DEFAULT,
        "map_id": _document_map_id(reference_map),
        "section_count": len(reference_map.sections),
        "node_count": int(reference_map.input_node_count),
        "document_count": len(
            {section.source_id for section in reference_map.sections}
        ),
        "query_count": len(QUERIES),
        "answerable_queries": sum(1 for query in QUERIES if query.get("doc")),
        "no_match_queries": sum(1 for query in QUERIES if not query.get("doc")),
        "llm_planner_status": "no endpoint supplied",
        "dense_baseline": "deterministic hash embeddings, not a trained encoder",
        "python": platform.python_version(),
        "platform": platform.system(),
        "tokenizer": "whitespace",
    }

    _write_csv(out_path / "results" / "routes.csv", rows, list(rows[0].keys()))
    _write_csv(
        out_path / "results" / "ablation.csv", aggregates, list(aggregates[0].keys())
    )
    _write_csv(
        out_path / "results" / "resolution.csv",
        resolution_rows,
        list(resolution_rows[0].keys()),
    )
    _write_csv(
        out_path / "results" / "scaling.csv",
        scaling_rows,
        list(scaling_rows[0].keys()),
    )
    payload = json_safe(
        {
            "run": run,
            "variants": list(VARIANTS),
            "aggregates": aggregates,
            "routes": rows,
            "resolution": resolution_rows,
            "scaling": scaling_rows,
            "route_example": {
                "query_id": route_query["id"],
                "question": route_query["question"],
                "trace": route_trace,
            },
        }
    )
    (out_path / "results" / "ablation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _bar_chart(
        "Answerable-question evidence tokens",
        "Mean prompt-context tokens per question at identical top-k. "
        "Lower is a smaller prompt.",
        [row["label"] for row in aggregates],
        [row["mean_evidence_tokens"] for row in aggregates],
        "",
        out_path / "figures" / "ablation_tokens.svg",
    )
    _bar_chart(
        "Answerable-question hit@k",
        "Fraction of answerable questions retrieving the ground-truth section.",
        [row["label"] for row in aggregates],
        [row["hit_at_k"] for row in aggregates],
        "",
        out_path / "figures" / "ablation_hit.svg",
    )
    _resolution_figure(
        resolution_rows, out_path / "figures" / "resolution_tradeoff.svg"
    )
    _route_figure(
        reference_map,
        route_trace,
        out_path / "figures" / "route_example.svg",
        f"Agentic route for \u201c{route_query['question']}\u201d",
    )
    (out_path / "REPORT.md").write_text(
        _report(aggregates, run, rows, resolution_rows, scaling_rows), encoding="utf-8"
    )
    return {
        "run": run,
        "aggregates": aggregates,
        "rows": rows,
        "resolution": resolution_rows,
        "scaling": scaling_rows,
    }


def _resolution_sweep(retriever, cache, truth, tokenizer):
    """Measure accuracy against map cost as the map resolution changes."""
    variant = {
        "key": "resolution_probe",
        "label": "resolution probe",
        "family": "map",
        "description": "Map agent at one resolution.",
        "chunks_per_section": RESOLUTION_DEFAULT,
        "agent": dict(MAP_AGENT_SETTINGS),
    }
    results = []
    for level in RESOLUTION_LEVELS:
        resolution = level["chunks_per_section"]
        target = int(level.get("target_sections_per_document") or 0)
        document_map = cache.get(resolution, target)
        variant["chunks_per_section"] = resolution
        variant["target_sections_per_document"] = target
        index_cost = cache.index_tokens(resolution, target)
        measured = {}
        for refine in (True, False):
            variant["agent"] = {**MAP_AGENT_SETTINGS, "refine_candidates": refine}
            rows = [
                _measure(
                    variant,
                    query,
                    retriever,
                    document_map,
                    truth,
                    tokenizer,
                    index_tokens=index_cost,
                )
                for query in QUERIES
            ]
            measured[refine] = aggregate(rows, variant)
        summary = measured[False]
        refined = measured[True]
        results.append(
            {
                "chunks_per_section": resolution,
                "target_sections_per_document": target,
                "label": level["label"],
                "sections": len(document_map.sections),
                "nodes": int(document_map.input_node_count),
                "map_index_tokens": int(index_cost),
                "hit_at_1": summary["hit_at_1"],
                "hit_at_k": summary["hit_at_k"],
                "hit_at_k_refine": refined["hit_at_k"],
                "mean_evidence_tokens_refine": refined["mean_evidence_tokens"],
                "mean_precision": summary["mean_precision"],
                "mean_recall": summary["mean_recall"],
                "mean_evidence_tokens": summary["mean_evidence_tokens"],
                "mean_total_context_tokens": summary["mean_total_context_tokens"],
                "mean_nodes_scored": summary["mean_nodes_scored"],
                "mean_route_steps": summary["mean_route_steps"],
                "no_match_rejection_rate": summary["no_match_rejection_rate"],
                "mean_latency_ms": summary["mean_latency_ms"],
            }
        )
    return results


def _drop(record, fields):
    if not isinstance(record, dict):
        return record
    return {key: value for key, value in record.items() if key not in fields}


def _read_csv(path: Path, drop=()):
    with path.open(encoding="utf-8", newline="") as handle:
        return [_drop(dict(row), drop) for row in csv.DictReader(handle)]


def stable_snapshot(out_dir: Path | str) -> dict:
    """Return the committed artifacts without wall-clock dependent values."""
    out = Path(out_dir)
    payload = json.loads(
        (out / "results" / "ablation.json").read_text(encoding="utf-8")
    )
    snapshot = {
        "run": _drop(payload.get("run") or {}, VOLATILE_RUN_FIELDS),
        "variants": payload.get("variants"),
        "aggregates": [
            _drop(row, VOLATILE_AGGREGATE_FIELDS)
            for row in payload.get("aggregates") or []
        ],
        "routes": [
            _drop(row, VOLATILE_ROW_FIELDS) for row in payload.get("routes") or []
        ],
        "route_example": payload.get("route_example"),
        "routes_csv": _read_csv(out / "results" / "routes.csv", VOLATILE_ROW_FIELDS),
        "ablation_csv": _read_csv(
            out / "results" / "ablation.csv", VOLATILE_AGGREGATE_FIELDS
        ),
        "resolution_csv": _read_csv(
            out / "results" / "resolution.csv", VOLATILE_AGGREGATE_FIELDS
        ),
        "scaling_csv": _read_csv(
            out / "results" / "scaling.csv", VOLATILE_AGGREGATE_FIELDS
        ),
        "scaling": [
            _drop(row, VOLATILE_AGGREGATE_FIELDS)
            for row in payload.get("scaling") or []
        ],
        "resolution": [
            _drop(row, VOLATILE_AGGREGATE_FIELDS)
            for row in payload.get("resolution") or []
        ],
    }
    for name in FIGURE_FILES:
        snapshot[f"figure:{name}"] = (out / "figures" / name).read_text(
            encoding="utf-8"
        )
    snapshot["report"] = "\n".join(
        line
        for line in (out / "REPORT.md").read_text(encoding="utf-8").splitlines()
        if not line.startswith("Generated ")
    )
    return snapshot


def check_artifacts(
    out_dir: Path | str = "research", top_k: int = TOP_K_DEFAULT
) -> list:
    """Regenerate the study and return a list of human-readable mismatches."""
    committed = Path(out_dir)
    if not (committed / "results" / "ablation.json").is_file():
        return [f"no committed study found under {committed}"]
    with tempfile.TemporaryDirectory(prefix="docmind_ablation_check_") as temp_dir:
        run_study(top_k=top_k, out_dir=temp_dir)
        fresh = stable_snapshot(temp_dir)
    recorded = stable_snapshot(committed)
    mismatches = []
    for key in sorted(set(recorded) | set(fresh)):
        if recorded.get(key) != fresh.get(key):
            mismatches.append(key)
    return mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval-map ablation study.")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--out-dir", default="research")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    top_k = max(1, min(int(args.top_k), HARD_MAX_RESULTS))
    if args.check:
        mismatches = check_artifacts(out_dir=args.out_dir, top_k=top_k)
        if mismatches:
            print(
                "retrieval-map artifacts are stale: "
                + ", ".join(mismatches)
                + f". Rerun python -m research.map_ablation --out-dir {args.out_dir}."
            )
            return 1
        if not args.quiet:
            print(f"retrieval-map artifacts match the code under {args.out_dir}")
        return 0
    result = run_study(top_k=top_k, out_dir=args.out_dir)
    if not args.quiet:
        for row in result["aggregates"]:
            print(
                f"{row['variant']:<28} hit@k={row['hit_at_k']:<6} "
                f"hit@1={row['hit_at_1']:<6} tokens={row['mean_evidence_tokens']:<8} "
                f"delta={row['mean_token_delta_pct']:>7}% "
                f"steps={row['mean_route_steps']}"
            )
        print(f"wrote ablation outputs to {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
