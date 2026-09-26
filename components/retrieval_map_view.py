import contextlib
import html as html_module
from collections.abc import Mapping, Sequence

import streamlit as st

try:
    import streamlit.components.v1 as components
except Exception:
    components = None

from utils.logs import log, safe_log_exception
from utils.retrieval_map import (
    MAP_PLANNER_DETERMINISTIC,
    MAP_PLANNER_LLM,
    MAP_PLANNER_LLM_FALLBACK_DETERMINISTIC,
)

MAX_VIEW_SECTIONS = 24
MAX_VIEW_DOCUMENTS = 12
MAX_OVERVIEW_SECTIONS = 8
MAX_OVERVIEW_DOCUMENTS = 4
MAX_OVERVIEW_HEIGHT = 330
MAX_VIEW_ACTIONS = 8
MAX_VIEW_STEPS = 12
MAX_VIEW_IDS = 2048
MAX_VIEW_CHARACTERS = 40000
MAX_VIEW_HEIGHT = 560
MIN_VIEW_HEIGHT = 190
MAX_ID_CHARS = 200
MAX_LABEL_CHARS = 90
MAX_SUMMARY_CHARS = 140
MAX_KEYWORD_CHARS = 28
MAX_VIEW_KEYWORDS = 6
MAX_REASON_CHARS = 120
MAX_ACTION_CHARS = 40

STATE_SELECTED = "selected"
STATE_PLANNED = "planned"
STATE_CONSIDERED = "considered"
STATE_UNCONSIDERED = "not considered"
STATE_LABELS = {
    STATE_SELECTED: "Selected (routed evidence)",
    STATE_PLANNED: "Planned, not selected",
    STATE_CONSIDERED: "Considered, not selected",
    STATE_UNCONSIDERED: "Not considered this question",
}

PLANNER_LABELS = {
    MAP_PLANNER_DETERMINISTIC: "Deterministic keyword scores",
    "injected": "LLM planner",
    "injected_fallback_deterministic": "LLM planner, deterministic fallback",
    MAP_PLANNER_LLM: "LLM planner",
    MAP_PLANNER_LLM_FALLBACK_DETERMINISTIC: "LLM planner, deterministic fallback",
}

STATE_SLUGS = {
    STATE_SELECTED: "selected",
    STATE_PLANNED: "planned",
    STATE_CONSIDERED: "considered",
    STATE_UNCONSIDERED: "unconsidered",
}

ACTION_PHASES = {
    "observe": "observe",
    "plan": "plan",
    "retrieve": "retrieve",
    "reflect": "reflect",
    "expand": "expand",
    "fallback": "fallback",
}

MAP_MISMATCH_NOTE = (
    "Map mismatch: this answer was routed on a different document map than the map "
    "currently loaded, so the route is shown as metrics only. Section states and route "
    "edges are not drawn for an old map."
)
TRUNCATION_NOTE = (
    "This map was built with a section/node limit, so only part of the index is drawn."
)
TOKEN_MEASUREMENT_NOTE = (
    "Tokenizer and prompt-context measurement only. It is not a guarantee of total "
    "compute, latency, GPU, embedding, retrieval-speed, or answer-quality savings."
)
BASELINE_DISABLED_NOTE = (
    "Baseline comparison is off, so no token delta is shown. Turn on "
    "Settings → Advanced → Agentic Map Retrieval → Measure global baseline to add one "
    "extra local retrieval pass per question. It adds retrieval work but no extra "
    "answer-model call."
)
MAP_MISSING_NOTE = (
    "No compact document map is available for the active index, so only route "
    "metrics are shown."
)
EMPTY_ROUTE_NOTE = "No map route was recorded for this answer."
MAP_FALLBACK_NOTE = (
    "The interactive map needs Streamlit components, which are unavailable here. "
    "The section table below carries the same information."
)
MAP_ASK_QUESTION_NOTE = (
    "This document map is ready. Ask a question to see which sections the map agent "
    "plans, selects, and reflects on."
)

_STYLE = """
<style>
.dm{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,
sans-serif;font-size:12px;line-height:1.35;color:#1b1b1b}
@media (prefers-color-scheme:dark){.dm{color:#e9e9e9}}
.dm *{box-sizing:border-box}
.dm .legend{display:flex;flex-wrap:wrap;gap:4px 10px;margin:0 0 6px;padding:0;
list-style:none}
.dm .legend li{display:flex;align-items:center;gap:4px;max-width:100%}
.dm .dot{width:9px;height:9px;border-radius:2px;border:1px solid rgba(0,0,0,.35);
display:inline-block;flex:0 0 auto}
.dm .dot-selected{background:rgba(33,140,80,.9)}
.dm .dot-planned{background:rgba(48,110,190,.85)}
.dm .dot-considered{background:rgba(120,120,130,.55)}
.dm .dot-unconsidered{background:transparent;border-style:dashed}
.dm .doc{margin:0 0 6px;border-left:3px solid rgba(90,90,110,.45);padding-left:6px}
.dm .doc-title{margin:0 0 3px;font-size:12px;font-weight:600;overflow-wrap:anywhere}
.dm .nodes{list-style:none;margin:0;padding:0}
.dm .node{position:relative;margin:0 0 4px;padding:3px 6px;border-radius:4px;
border:1px solid rgba(120,120,130,.45);background:rgba(120,120,130,.10);
overflow-wrap:anywhere}
.dm .node.s-selected{border-color:rgba(33,140,80,.9);background:rgba(33,140,80,.16)}
.dm .node.s-planned{border-color:rgba(48,110,190,.9);background:rgba(48,110,190,.14)}
.dm .node.s-considered{border-style:dashed;background:rgba(120,120,130,.08)}
.dm .node.s-unconsidered{border-style:dotted;background:transparent;opacity:.85}
.dm .title{font-weight:600}
.dm .order{display:inline-block;margin-left:5px;padding:0 4px;border-radius:7px;
font-size:10px;font-weight:700;background:rgba(33,140,80,.9);color:#fff}
.dm .meta{display:block;opacity:.75}
.dm .summary{display:block;opacity:.85}
.dm .kw{display:block;opacity:.7}
.dm .timeline{margin:2px 0 0;display:block}
.dm .tl-line{stroke:rgba(60,60,80,.75);stroke-width:1.5}
.dm .tl-head{fill:rgba(60,60,80,.75)}
.dm .tl-step{stroke:rgba(60,60,80,.75);stroke-width:1.5}
.dm .tl-fill-selected{fill:rgba(33,140,80,.9)}
.dm .tl-fill-planned{fill:rgba(48,110,190,.85)}
.dm .tl-fill-other{fill:rgba(120,120,130,.55)}
.dm .tl-num{font-size:9px;fill:#fff;text-anchor:middle;dominant-baseline:central}
.dm .tl-back{stroke:rgba(180,80,40,.9);stroke-width:1.2;stroke-dasharray:4 3;
fill:none}
.dm .tl-label{font-size:9px;fill:rgba(180,80,40,.95);text-anchor:middle}
.dm .note{font-size:11px;opacity:.8}
"""

_rendered_keys: set[str] = set()
_render_pass_open = False


def begin_render_pass() -> None:
    """Start one Streamlit script run so a route key renders at most once."""
    global _rendered_keys, _render_pass_open
    _rendered_keys = set()
    _render_pass_open = True


def _claim_render(key) -> bool:
    render_key = str(key or "current")
    if _render_pass_open and render_key in _rendered_keys:
        return False
    _rendered_keys.add(render_key)
    return True


def _text(value, limit: int) -> str:
    return " ".join(str("" if value is None else value).split())[: max(0, int(limit))]


def _esc(value, limit: int = MAX_LABEL_CHARS) -> str:
    return html_module.escape(_text(value, limit), quote=True)


def _mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _integer(value, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(fallback)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _list(value, limit: int) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)[: max(0, int(limit))]


def _ids(value, limit: int = MAX_VIEW_IDS) -> tuple[str, ...]:
    unique: list[str] = []
    for item in _list(value, max(0, int(limit)) * 2):
        candidate = _text(item, MAX_ID_CHARS)
        if candidate and candidate not in unique:
            unique.append(candidate)
        if len(unique) >= max(1, int(limit)):
            break
    return tuple(unique)


def _keywords(value) -> list[str]:
    keywords = []
    for item in _list(value, MAX_VIEW_KEYWORDS * 2):
        term = _text(item, MAX_KEYWORD_CHARS)
        if term and term not in keywords:
            keywords.append(term)
        if len(keywords) >= MAX_VIEW_KEYWORDS:
            break
    return keywords


def has_route(route) -> bool:
    """Return whether a stored route carries any local map resolution data."""
    return bool(_mapping(route))


def reflection_triggered(route) -> bool:
    """Return whether the route recorded a reflection step."""
    payload = _mapping(route)
    reflection = _mapping(payload.get("reflection"))
    return bool(payload.get("reflection_flag") or reflection.get("triggered"))


def _state_from_sets(considered: set, planned: set, selected: set, section_id) -> str:
    if section_id in selected:
        return STATE_SELECTED
    if section_id in planned:
        return STATE_PLANNED
    if section_id in considered:
        return STATE_CONSIDERED
    return STATE_UNCONSIDERED


def section_state(route, section_id) -> str:
    """Return the visual state of one section for the given route."""
    payload = _mapping(route)
    return _state_from_sets(
        set(_ids(payload.get("considered_section_ids"))),
        set(_ids(payload.get("planned_section_ids"))),
        set(_ids(payload.get("selected_section_ids"))),
        section_id,
    )


def _section_payload(section) -> dict:
    return {
        "section_id": _text(getattr(section, "section_id", ""), MAX_ID_CHARS),
        "source_id": _text(getattr(section, "source_id", ""), MAX_ID_CHARS),
        "title": _text(getattr(section, "title", ""), MAX_LABEL_CHARS),
        "source": _text(getattr(section, "source", ""), MAX_LABEL_CHARS),
        "sequence": _integer(getattr(section, "sequence", 0)),
        "token_count": _integer(getattr(section, "token_count", 0)),
        "chunk_count": _integer(getattr(section, "chunk_count", 0)),
        "keywords": _keywords(getattr(section, "keywords", ())),
        "summary": _text(getattr(section, "summary", ""), MAX_SUMMARY_CHARS),
    }


def build_map_view_model(
    document_map=None,
    route=None,
    *,
    max_sections: int = MAX_VIEW_SECTIONS,
    max_documents: int = MAX_VIEW_DOCUMENTS,
) -> dict:
    """Return a bounded, JSON-safe description of the map and its latest route."""
    payload = _mapping(route)
    section_limit = max(1, int(max_sections))
    document_limit = max(1, int(max_documents))
    raw_sections = getattr(document_map, "sections", ()) or ()
    all_sections = list(raw_sections) if isinstance(raw_sections, Sequence) else []
    total_sections = len(all_sections)
    map_id = _text(getattr(document_map, "map_id", ""), MAX_ID_CHARS)
    route_map_id = _text(payload.get("map_id"), MAX_ID_CHARS)
    mismatch = bool(map_id and route_map_id and map_id != route_map_id)
    selected_order = _ids(payload.get("selected_section_ids"))
    planned_order = _ids(payload.get("planned_section_ids"))
    considered_order = _ids(payload.get("considered_section_ids"))
    considered = set(considered_order)
    planned = set(planned_order)
    selected = set(selected_order)
    order_index = {
        section_id: index + 1 for index, section_id in enumerate(selected_order)
    }
    section_by_id = {
        _text(getattr(section, "section_id", ""), MAX_ID_CHARS): section
        for section in all_sections
    }
    priority = [*selected_order, *planned_order, *considered_order]
    priority.extend(
        _text(getattr(section, "section_id", ""), MAX_ID_CHARS)
        for section in all_sections
    )
    visible_sections = []
    seen_section_ids = set()
    for section_id in priority:
        section = section_by_id.get(section_id)
        if section is None or section_id in seen_section_ids:
            continue
        seen_section_ids.add(section_id)
        visible_sections.append(section)
        if len(visible_sections) >= section_limit:
            break

    def state_for(section_id) -> str:
        return _state_from_sets(considered, planned, selected, section_id)

    documents: list[dict] = []
    by_source: dict[str, dict] = {}
    for section in visible_sections:
        node = _section_payload(section)
        section_id = node["section_id"]
        if mismatch:
            node["state"] = STATE_UNCONSIDERED
            node["route_order"] = 0
        else:
            node["state"] = state_for(section_id)
            node["route_order"] = order_index.get(section_id, 0)
        source_id = node["source_id"] or "document"
        document = by_source.get(source_id)
        if document is None:
            if len(documents) >= document_limit:
                continue
            document = {
                "source_id": source_id,
                "label": node["source"] or "document",
                "sections": [],
            }
            by_source[source_id] = document
            documents.append(document)
        document["sections"].append(node)
    sections = [node for document in documents for node in document["sections"]]
    visible = {node["section_id"] for node in sections}
    steps = (
        [
            {
                "index": order_index[section_id],
                "section_id": section_id,
                "title": next(
                    (
                        node["title"]
                        for node in sections
                        if node["section_id"] == section_id
                    ),
                    "",
                ),
                "state": state_for(section_id),
            }
            for section_id in selected_order
            if section_id in visible
        ][:MAX_VIEW_STEPS]
        if not mismatch
        else []
    )
    return {
        "map_id": map_id,
        "route_map_id": route_map_id,
        "source_identity": _text(
            getattr(document_map, "source_identity", ""), MAX_ID_CHARS
        ),
        "map_available": bool(sections),
        "map_mismatch": mismatch,
        "truncated_map": bool(getattr(document_map, "truncated", False)),
        "section_total": int(total_sections),
        "shown_sections": len(sections),
        "documents": documents,
        "sections": sections,
        "steps": steps,
        "reflected": reflection_triggered(payload) if not mismatch else False,
        "has_route": bool(payload),
        "considered_count": len(considered),
        "planned_count": len(planned),
        "selected_count": len(selected),
        "route_steps": _integer(payload.get("step_count")),
        "planner_mode": _text(
            payload.get("planner_mode")
            or _mapping(payload.get("metrics")).get("planner_mode"),
            MAX_ACTION_CHARS,
        ),
    }


def _action_phase(action: str) -> str:
    name = _text(action, MAX_ACTION_CHARS).lower()
    if name in ACTION_PHASES:
        return ACTION_PHASES[name]
    for prefix, phase in (
        ("observe", "observe"),
        ("plan", "plan"),
        ("retrieve", "retrieve"),
        ("reflect", "reflect"),
        ("expand", "expand"),
    ):
        if name.startswith(prefix):
            return phase
    if "fallback" in name:
        return "fallback"
    return "other"


def build_action_rows(route) -> list[dict]:
    """Return the bounded observe/plan/retrieve/reflect/expand/fallback trace."""
    payload = _mapping(route)
    rows = []
    for action in _list(payload.get("actions"), MAX_VIEW_ACTIONS):
        entry = _mapping(action)
        name = _text(entry.get("action"), MAX_ACTION_CHARS)
        rows.append(
            {
                "Step": _integer(entry.get("step")),
                "Phase": _action_phase(name),
                "Action": name or "unknown",
                "Reason": _text(entry.get("reason"), MAX_REASON_CHARS) or "-",
                "Sections": len(_ids(entry.get("allowed_section_ids"))),
                "Queries": _integer(entry.get("attempted_queries")),
                "Results": _integer(entry.get("result_count")),
            }
        )
    return rows


def build_section_rows(view) -> list[dict]:
    """Return the accessible section table without document text or node ids."""
    return [
        {
            "Title": node["title"] or "(untitled section)",
            "Source": node["source"] or node["source_id"] or "document",
            "Sequence": node["sequence"],
            "Tokens": node["token_count"],
            "Keyword summary": ", ".join(node["keywords"]) or "-",
            "State": STATE_LABELS.get(node["state"], node["state"]),
        }
        for node in _list(_mapping(view).get("sections"), MAX_VIEW_SECTIONS)
    ]


def _signed_int(value, suffix: str = "") -> str:
    return f"{int(value):+,}{suffix}"


def _signed_percent(value) -> str:
    return f"{float(value):+,.2f}%"


def build_metric_rows(route) -> list[dict]:
    """Return the honest retrieval resolution metrics for one route."""
    payload = _mapping(route)
    metrics = _mapping(payload.get("metrics"))
    accounting = _mapping(payload.get("token_accounting"))
    considered = _ids(payload.get("considered_section_ids"))
    planned = _ids(payload.get("planned_section_ids"))
    selected = _ids(payload.get("selected_section_ids"))
    planner_mode = _text(
        payload.get("planner_mode") or metrics.get("planner_mode"), MAX_ACTION_CHARS
    )
    rows = [
        {"Metric": "Sections considered", "Value": str(len(considered))},
        {"Metric": "Sections planned", "Value": str(len(planned))},
        {"Metric": "Sections selected", "Value": str(len(selected))},
        {
            "Metric": "Selected evidence chunks",
            "Value": str(_integer(metrics.get("result_count"))),
        },
        {"Metric": "Route steps", "Value": str(_integer(payload.get("step_count")))},
        {
            "Metric": "Reflection",
            "Value": "Triggered" if reflection_triggered(payload) else "Not needed",
        },
        {
            "Metric": "Planner mode",
            "Value": PLANNER_LABELS.get(planner_mode, planner_mode or "unknown"),
        },
        {
            "Metric": "Selected evidence tokens",
            "Value": str(_integer(metrics.get("selected_evidence_tokens"))),
        },
        {
            "Metric": "RAG input tokens (actual)",
            "Value": str(_integer(accounting.get("actual_rag_input_prompt_tokens"))),
        },
    ]
    baseline = _number(accounting.get("baseline_input_prompt_tokens"))
    rows.append(
        {
            "Metric": "Baseline input tokens",
            "Value": "not measured" if baseline is None else f"{int(baseline):,}",
        }
    )
    delta = _number(accounting.get("selected_minus_baseline_input_tokens"))
    percent = _number(accounting.get("selected_minus_baseline_input_pct"))
    if delta is None:
        rows.append(
            {"Metric": "Token delta vs measured baseline", "Value": "not measured"}
        )
        return rows
    value = _signed_int(delta, " tokens")
    if percent is not None:
        value = f"{value} ({_signed_percent(percent)})"
    rows.append({"Metric": "Token delta vs measured baseline", "Value": value})
    return rows


def baseline_note(route) -> str:
    """Return the explanation shown when no baseline measurement exists."""
    accounting = _mapping(_mapping(route).get("token_accounting"))
    if (
        accounting.get("baseline_measured")
        or _number(accounting.get("baseline_input_prompt_tokens")) is not None
    ):
        return TOKEN_MEASUREMENT_NOTE
    return f"{BASELINE_DISABLED_NOTE} {TOKEN_MEASUREMENT_NOTE}"


def _state_slug(state) -> str:
    return STATE_SLUGS.get(state, "unconsidered")


def _node_html(node) -> str:
    state = _esc(node["state"], MAX_ACTION_CHARS)
    slug = _state_slug(node["state"])
    order = node["route_order"]
    badge = (
        f'<span class="order">{int(order)}</span>'
        if isinstance(order, int) and order
        else ""
    )
    keywords = _esc(", ".join(node["keywords"]), MAX_SUMMARY_CHARS)
    return (
        f'<li class="node s-{slug}" data-state="{state}" data-order="{int(order)}"'
        f' title="{_esc(node["title"], MAX_LABEL_CHARS)}">'
        f'<span class="title">{_esc(node["title"], MAX_LABEL_CHARS) or "Untitled section"}</span>'
        f"{badge}"
        f'<span class="meta">section {_esc(node["sequence"], 8)}'
        f' · {_esc(node["token_count"], 12)} tokens'
        f' · {_esc(node["chunk_count"], 12)} chunks</span>'
        f'<span class="summary">{_esc(node["summary"], MAX_SUMMARY_CHARS)}</span>'
        f'<span class="kw">{keywords}</span>'
        "</li>"
    )


def _legend_html() -> str:
    items = "".join(
        f'<li><span class="dot dot-{_state_slug(state)}" aria-hidden="true"></span>'
        f"{_esc(label, MAX_LABEL_CHARS)}</li>"
        for state, label in STATE_LABELS.items()
    )
    return f'<ul class="legend" aria-label="Map legend">{items}</ul>'


def _timeline_svg(steps, reflected) -> str:
    span = 32
    height = 58 if reflected and len(steps) > 1 else 34
    width = 20 + span * len(steps) + 12
    center = 17
    parts = [
        f'<svg class="timeline" viewBox="0 0 {width} {height}" width="100%"'
        f' height="{height}" preserveAspectRatio="xMidYMid meet" role="img"'
        ' aria-label="Route order across the selected map sections">'
    ]
    for index, step in enumerate(steps):
        x = 20 + span * index
        state = step["state"]
        if state == STATE_SELECTED:
            fill = "tl-fill-selected"
        elif state == STATE_PLANNED:
            fill = "tl-fill-planned"
        else:
            fill = "tl-fill-other"
        label = _esc(f"{step['index']}. {step['title']} ({state})", MAX_LABEL_CHARS)
        parts.append(
            f'<circle class="tl-step {fill}" cx="{x}" cy="{center}" r="9">'
            f"<title>{label}</title></circle>"
            f'<text class="tl-num" x="{x}" y="{center + 1}">{_esc(step["index"], 4)}</text>'
        )
        if index + 1 < len(steps):
            start = x + 11
            end = x + span - 11
            parts.append(
                f'<line class="tl-line" x1="{start}" y1="{center}"'
                f' x2="{end}" y2="{center}"></line>'
                f'<polygon class="tl-head" points="{end},{center} {end - 5},'
                f'{center - 4} {end - 5},{center + 4}"></polygon>'
            )
    if reflected and len(steps) > 1:
        first_x = 20
        last_x = 20 + span * (len(steps) - 1)
        baseline = height - 8
        middle = int((first_x + last_x) / 2)
        parts.append(
            f'<path class="tl-back" d="M {last_x} {center + 11} L {last_x} {baseline}'
            f' L {first_x} {baseline} L {first_x} {center + 11}"></path>'
            f'<text class="tl-label" x="{middle}" y="{baseline - 2}">reflect</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_map_html(view) -> str:
    """Return self-contained inline HTML/CSS/SVG for the compact document map."""
    model = _mapping(view)
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<style>{_STYLE}</style></head><body>",
        f'<div class="dm">{_legend_html()}',
    ]
    for document in _list(model.get("documents"), MAX_VIEW_DOCUMENTS):
        label = _esc(document.get("label"), MAX_LABEL_CHARS) or "document"
        parts.append(
            f'<section class="doc"><p class="doc-title">{label}</p>'
            '<ol class="nodes">'
        )
        for node in _list(document.get("sections"), MAX_VIEW_SECTIONS):
            parts.append(_node_html(_mapping(node)))
        parts.append("</ol></section>")
    if model.get("map_mismatch"):
        parts.append(f'<p class="note">{_esc(MAP_MISMATCH_NOTE, 400)}</p>')
    if model.get("truncated_map"):
        parts.append(f'<p class="note">{_esc(TRUNCATION_NOTE, 200)}</p>')
    steps = _list(model.get("steps"), MAX_VIEW_STEPS)
    if steps:
        parts.append(_timeline_svg(steps, bool(model.get("reflected"))))
    parts.append("</div></body></html>")
    return "".join(parts)


def build_map_text_fallback(view) -> str:
    """Return a plain-text map summary for environments without Streamlit components."""
    model = _mapping(view)
    lines = [
        f"Map {model.get('map_id') or 'unavailable'} — "
        f"{_integer(model.get('shown_sections'))} of "
        f"{_integer(model.get('section_total'))} sections shown"
    ]
    for document in _list(model.get("documents"), MAX_VIEW_DOCUMENTS):
        nodes = [_mapping(node) for node in _list(document.get("sections"), 64)]
        counts: dict[str, int] = {}
        for node in nodes:
            label = STATE_LABELS.get(node.get("state"), str(node.get("state")))
            counts[label] = counts.get(label, 0) + 1
        detail = ", ".join(f"{name} x{amount}" for name, amount in counts.items())
        label = _text(document.get("label"), MAX_LABEL_CHARS) or "document"
        lines.append(f"- {label}: {len(nodes)} sections ({detail})")
    if _integer(model.get("selected_count")):
        lines.append(
            f"Route: {_integer(model.get('selected_count'))} selected, "
            f"{_integer(model.get('planned_count'))} planned, "
            f"{_integer(model.get('considered_count'))} considered, "
            f"{_integer(model.get('route_steps'))} step(s)"
        )
    if model.get("map_mismatch"):
        lines.append(MAP_MISMATCH_NOTE)
    if model.get("truncated_map"):
        lines.append(TRUNCATION_NOTE)
    return "\n".join(lines)[:MAX_VIEW_CHARACTERS]


def map_html_height(view) -> int:
    """Return a bounded iframe height for the compact map."""
    model = _mapping(view)
    rows = sum(
        len(_list(document.get("sections"), MAX_VIEW_SECTIONS))
        for document in _list(model.get("documents"), MAX_VIEW_DOCUMENTS)
    )
    height = MIN_VIEW_HEIGHT + 20 * rows
    if _list(model.get("steps"), 1):
        height += 62
    return max(MIN_VIEW_HEIGHT, min(MAX_VIEW_HEIGHT, height))


def _log_failure(error: BaseException) -> None:
    with contextlib.suppress(Exception):
        log.warning("Retrieval map view unavailable: %s", safe_log_exception(error))


def _render_map_component(view, max_height: int = MAX_VIEW_HEIGHT) -> bool:
    model = _mapping(view)
    if not model.get("map_available"):
        return False
    if components is None or not callable(getattr(components, "html", None)):
        st.caption(MAP_FALLBACK_NOTE)
        st.write(build_map_text_fallback(model))
        return True
    markup = build_map_html(model)
    if len(markup) > MAX_VIEW_CHARACTERS:
        st.caption(MAP_FALLBACK_NOTE)
        st.write(build_map_text_fallback(model))
        return True
    try:
        components.html(
            markup,
            height=max(MIN_VIEW_HEIGHT, min(max_height, map_html_height(model))),
            scrolling=True,
        )
    except Exception as error:
        _log_failure(error)
        st.caption(MAP_FALLBACK_NOTE)
        st.write(build_map_text_fallback(model))
    return True


def render_map_section(document_map=None, route=None) -> bool:
    """Render the map, the honest metrics, the action trace, and the section table."""
    view = build_map_view_model(document_map, route)
    if view["map_mismatch"]:
        st.warning(MAP_MISMATCH_NOTE)
    elif not view["map_available"]:
        st.info(MAP_MISSING_NOTE)
    if view["truncated_map"]:
        st.caption(TRUNCATION_NOTE)
    _render_map_component(view)
    if view["has_route"]:
        rows = build_metric_rows(route)
        if rows:
            st.table(rows)
        st.caption(TOKEN_MEASUREMENT_NOTE)
        st.caption(baseline_note(route))
        actions = build_action_rows(route)
        if actions:
            st.table(actions)
    else:
        st.caption(EMPTY_ROUTE_NOTE)
    sections = build_section_rows(view)
    if sections:
        st.table(sections)
    return True


def render_retrieval_route(
    route, document_map=None, *, key=None, expanded=False
) -> bool:
    """Render a collapsed retrieval-route expander for one local answer."""
    if not has_route(route):
        return False
    if not _claim_render(key):
        return False
    try:
        with st.expander("🗺️ Retrieval route", expanded=bool(expanded)):
            render_map_section(document_map, route)
    except Exception as error:
        _log_failure(error)
        return False
    return True


def render_stored_route(message, document_map=None, *, key=None) -> bool:
    """Render the stored route of a historical assistant message, if it has one."""
    if not isinstance(message, Mapping):
        return False
    if message.get("r2r"):
        return False
    if str(message.get("role") or "").strip().lower() != "assistant":
        return False
    return render_retrieval_route(message.get("retrieval_route"), document_map, key=key)


def render_map_overview(document_map=None, route=None, *, key=None) -> bool:
    """Render the compact retrieval map overview used by the Data Sources tab."""
    view = build_map_view_model(
        document_map,
        route,
        max_sections=MAX_OVERVIEW_SECTIONS,
        max_documents=MAX_OVERVIEW_DOCUMENTS,
    )
    if not view["map_available"] and not view["has_route"]:
        return False
    if not _claim_render(key or "overview"):
        return False
    try:
        with st.container(border=True):
            st.markdown("**🗺️ Retrieval map**")
            if not _render_map_component(view, MAX_OVERVIEW_HEIGHT):
                st.caption(MAP_MISSING_NOTE)
            if view["map_mismatch"]:
                st.warning(MAP_MISMATCH_NOTE)
            elif view["truncated_map"]:
                st.caption(TRUNCATION_NOTE)
            if view["has_route"]:
                st.caption(
                    "Last route: "
                    f"{view['selected_count']} selected · "
                    f"{view['planned_count']} planned · "
                    f"{view['considered_count']} considered · "
                    f"{view['route_steps']} step(s) · "
                    f"{view['planner_mode'] or 'unknown planner'}"
                )
                st.caption(TOKEN_MEASUREMENT_NOTE)
            else:
                st.info(MAP_ASK_QUESTION_NOTE)
    except Exception as error:
        _log_failure(error)
        return False
    return True


__all__ = [
    "MAP_MISMATCH_NOTE",
    "MAX_OVERVIEW_SECTIONS",
    "MAX_VIEW_ACTIONS",
    "MAX_VIEW_CHARACTERS",
    "MAX_VIEW_SECTIONS",
    "MAX_VIEW_STEPS",
    "STATE_CONSIDERED",
    "STATE_LABELS",
    "STATE_PLANNED",
    "STATE_SELECTED",
    "STATE_UNCONSIDERED",
    "TOKEN_MEASUREMENT_NOTE",
    "begin_render_pass",
    "build_action_rows",
    "build_map_html",
    "build_map_text_fallback",
    "build_map_view_model",
    "build_metric_rows",
    "build_section_rows",
    "has_route",
    "map_html_height",
    "reflection_triggered",
    "render_map_overview",
    "render_map_section",
    "render_retrieval_route",
    "render_stored_route",
    "section_state",
]
