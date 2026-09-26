import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from components import chatbox
from components import retrieval_map_view as view
from components.tabs import settings as settings_tab
from utils.retrieval_map import (
    MAP_PLANNER_MODES,
    MAP_SETTING_DEFAULTS,
    MAP_SETTING_RANGES,
    DocumentMap,
    MapEdge,
    MapNode,
    MapSection,
    RetrievalMapConfig,
)

ROOT = Path(__file__).resolve().parents[1]
HOSTILE_TEXT = '<script>alert("x")</script> & "quoted" <b>bold</b>'
SECRET_NODE_ID = "member-node-secret-3"


class _Container:
    def __init__(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeStreamlit:
    def __init__(self, state=None):
        self.session_state = {} if state is None else state
        self.captions = []
        self.markdowns = []
        self.warnings = []
        self.infos = []
        self.writes = []
        self.tables = []
        self.expanders = []
        self.containers = 0
        self.toggles = []
        self.selectboxes = []
        self.sliders = []
        self.select_sliders = []
        self.radios = []
        self.text_inputs = []

    def caption(self, value, *args, **kwargs):
        self.captions.append(str(value))

    def markdown(self, value, *args, **kwargs):
        self.markdowns.append((str(value), kwargs))

    def warning(self, value, *args, **kwargs):
        self.warnings.append(str(value))

    def info(self, value, *args, **kwargs):
        self.infos.append(str(value))

    def write(self, value, *args, **kwargs):
        self.writes.append(str(value))

    def table(self, data, *args, **kwargs):
        self.tables.append([dict(row) for row in data])

    def expander(self, label, *args, **kwargs):
        self.expanders.append((str(label), kwargs))
        return _Container()

    def container(self, *args, **kwargs):
        self.containers += 1
        return _Container()

    def toggle(self, label, *args, **kwargs):
        self.toggles.append((str(label), kwargs))
        return self.session_state.get(kwargs.get("key"), False)

    def selectbox(self, label, options, *args, **kwargs):
        choices = list(options)
        self.selectboxes.append((str(label), choices, kwargs))
        return self.session_state.get(
            kwargs.get("key"), choices[0] if choices else None
        )

    def slider(self, label, *args, **kwargs):
        self.sliders.append((str(label), kwargs))
        return self.session_state.get(kwargs.get("key"), kwargs.get("value"))

    def select_slider(self, label, options, *args, **kwargs):
        self.select_sliders.append((str(label), kwargs))
        return self.session_state.get(kwargs.get("key"), list(options)[0])

    def radio(self, label, options, *args, **kwargs):
        choices = list(options)
        self.radios.append((str(label), choices, kwargs))
        return self.session_state.get(
            kwargs.get("key"), choices[min(1, len(choices) - 1)] if choices else None
        )

    def text_input(self, label, *args, **kwargs):
        self.text_inputs.append((str(label), kwargs))
        return self.session_state.get(kwargs.get("key"), "")

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _FakeComponents:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def html(self, markup, *args, **kwargs):
        self.calls.append((str(markup), kwargs))
        if self.error is not None:
            raise self.error
        return None


@contextmanager
def _render(state=None, components=None, error=None):
    streamlit = _FakeStreamlit(state)
    rendered = _FakeComponents(error)
    with (
        patch.object(view, "st", streamlit),
        patch.object(
            view, "components", rendered if components is None else components
        ),
    ):
        view.begin_render_pass()
        yield streamlit, rendered


def _section(
    index,
    source_index=0,
    title=None,
    source=None,
    summary=None,
    keywords=None,
    tokens=None,
    state="s",
):
    return MapSection(
        section_id=f"map_sec_{source_index}_{index}",
        source_id=f"map_doc_{source_index}",
        source=source if source is not None else f"doc{source_index}.txt",
        source_identity=f"doc{source_index}:0",
        title=title if title is not None else f"Doc {source_index} Section {index}",
        summary=summary if summary is not None else f"summary for section {index}",
        keywords=tuple(keywords if keywords is not None else (f"kw{index}", "alpha")),
        member_node_ids=(SECRET_NODE_ID, f"{SECRET_NODE_ID}-{index}"),
        chunk_count=2,
        token_count=120 if tokens is None else tokens,
        sequence=index,
    )


def _document_map(sections=None, map_id="map_current", truncated=False):
    items = (
        list(sections)
        if sections is not None
        else [_section(index) for index in range(3)]
    )
    nodes = []
    edges = []
    seen = set()
    for section in items:
        if section.source_id not in seen:
            seen.add(section.source_id)
            nodes.append(
                MapNode(
                    node_id=section.source_id,
                    kind="document",
                    label=section.source,
                    source_id=section.source_id,
                )
            )
        nodes.append(
            MapNode(
                node_id=section.section_id,
                kind="section",
                label=section.title,
                source_id=section.source_id,
            )
        )
        edges.append(
            MapEdge(
                source_id=section.source_id,
                target_id=section.section_id,
                kind="contains",
            )
        )
    return DocumentMap(
        map_id=map_id,
        version=1,
        source_identity="doc0:0",
        sections=tuple(items),
        nodes=tuple(nodes),
        edges=tuple(edges),
        input_node_count=len(items) * 2,
        mapped_node_count=len(items) * 2,
        skipped_node_count=0,
        truncated=truncated,
        config=RetrievalMapConfig(),
    )


def _route(
    considered=None,
    planned=None,
    selected=None,
    map_id="map_current",
    actions=None,
    reflected=False,
    baseline=True,
):
    considered_ids = list(considered if considered is not None else ["map_sec_0_0"])
    planned_ids = list(planned if planned is not None else considered_ids[:1])
    selected_ids = list(selected if selected is not None else planned_ids[:1])
    accounting = {
        "token_basis": "tokenizer_prompt_and_selected_evidence_text",
        "baseline_measured": baseline,
        "actual_selected_evidence_tokens": 640,
        "actual_rag_input_prompt_tokens": 1000,
        "baseline_input_prompt_tokens": 2000 if baseline else None,
        "selected_minus_baseline_input_tokens": -1000 if baseline else None,
        "selected_minus_baseline_input_pct": -50.0 if baseline else None,
    }
    return {
        "map_id": map_id,
        "map_source_identity": "doc0:0",
        "considered_section_ids": considered_ids,
        "planned_section_ids": planned_ids,
        "selected_section_ids": selected_ids,
        "step_count": 2,
        "reflection_flag": reflected,
        "reflection": {"performed": True, "triggered": reflected},
        "planner_mode": "deterministic",
        "actions": (
            actions
            if actions is not None
            else [
                {"step": 0, "action": "observe", "reason": "bounded_map_ready"},
                {"step": 0, "action": "plan", "reason": "deterministic_keyword_scores"},
                {
                    "step": 1,
                    "action": "retrieve_sections",
                    "reason": "routed_evidence",
                    "allowed_section_ids": selected_ids,
                    "attempted_queries": 1,
                    "result_count": 2,
                },
                {"step": 1, "action": "reflect", "reason": "weak_evidence"},
                {
                    "step": 2,
                    "action": "expand_sections",
                    "reason": "positive_unrouted_sections",
                    "allowed_section_ids": [],
                    "attempted_queries": 1,
                    "result_count": 3,
                },
            ]
        ),
        "metrics": {
            "planner_mode": "deterministic",
            "selected_evidence_tokens": 640,
            "result_count": 3,
        },
        "token_accounting": accounting,
    }


def _metric(rows, label):
    for row in rows:
        if row["Metric"] == label:
            return row["Value"]
    raise AssertionError(f"missing metric {label}")


def _captions(streamlit):
    return " \n".join(streamlit.captions)


class MapViewEscapingTests(unittest.TestCase):
    def test_every_map_field_is_html_escaped(self):
        sections = [
            _section(
                0,
                title=HOSTILE_TEXT,
                source=HOSTILE_TEXT,
                summary=HOSTILE_TEXT,
                keywords=[HOSTILE_TEXT],
            ),
            _section(1, title=HOSTILE_TEXT, source=HOSTILE_TEXT),
        ]
        route = _route(
            considered=[item.section_id for item in sections],
            planned=[sections[0].section_id],
            selected=[sections[0].section_id],
            actions=[
                {
                    "step": 0,
                    "action": "plan",
                    "reason": HOSTILE_TEXT,
                    "allowed_section_ids": [sections[0].section_id],
                }
            ],
        )
        model = view.build_map_view_model(_document_map(sections), route)
        markup = view.build_map_html(model)

        self.assertNotIn("<script>", markup)
        self.assertNotIn('alert("x")', markup)
        self.assertNotIn("<b>bold</b>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertIn("&amp;", markup)
        self.assertIn("&quot;", markup)
        self.assertNotIn(SECRET_NODE_ID, markup)

        rendered_rows = " ".join(
            str(value)
            for row in view.build_action_rows(route)
            for value in row.values()
        )
        self.assertIn(HOSTILE_TEXT, rendered_rows)
        self.assertNotIn(SECRET_NODE_ID, rendered_rows)

        fallback = view.build_map_text_fallback(model)
        self.assertNotIn(SECRET_NODE_ID, fallback)

    def test_render_helpers_never_pass_user_html_to_streamlit(self):
        sections = [_section(0, title=HOSTILE_TEXT, source=HOSTILE_TEXT)]
        with _render() as (streamlit, rendered):
            self.assertTrue(view.render_map_overview(_document_map(sections), None))
        for _text, kwargs in streamlit.markdowns:
            self.assertNotIn("unsafe_allow_html", kwargs)
        self.assertTrue(rendered.calls)
        for markup, _kwargs in rendered.calls:
            self.assertNotIn("<script>", markup)

    def test_hostile_route_shapes_do_not_raise(self):
        malformed = {
            "map_id": ["not", "a", "string"],
            "considered_section_ids": "map_sec_0_0",
            "planned_section_ids": 17,
            "selected_section_ids": {"bad": "shape"},
            "step_count": "many",
            "actions": "nope",
            "metrics": 5,
            "token_accounting": [],
            "reflection": "yes",
        }
        with _render() as (streamlit, rendered):
            self.assertTrue(view.render_retrieval_route(malformed, _document_map()))
        self.assertTrue(rendered.calls)
        self.assertTrue(streamlit.tables)


class MapViewBoundTests(unittest.TestCase):
    def test_sections_documents_steps_and_actions_stay_bounded(self):
        sections = [_section(index, source_index=index % 20) for index in range(120)]
        considered = [section.section_id for section in sections]
        selected = considered[:40]
        route = _route(
            considered=considered,
            planned=selected[:10],
            selected=selected,
            actions=[
                {
                    "step": index,
                    "action": "retrieve_sections",
                    "reason": "r",
                    "allowed_section_ids": considered,
                }
                for index in range(50)
            ],
        )
        model = view.build_map_view_model(_document_map(sections), route)
        markup = view.build_map_html(model)

        self.assertEqual(model["section_total"], 120)
        self.assertLessEqual(len(model["sections"]), view.MAX_VIEW_SECTIONS)
        self.assertLessEqual(len(model["documents"]), view.MAX_VIEW_DOCUMENTS)
        self.assertLessEqual(len(model["steps"]), view.MAX_VIEW_STEPS)
        self.assertLessEqual(len(view.build_action_rows(route)), view.MAX_VIEW_ACTIONS)
        self.assertLessEqual(len(markup), view.MAX_VIEW_CHARACTERS)
        self.assertLessEqual(view.map_html_height(model), view.MAX_VIEW_HEIGHT)
        self.assertLessEqual(
            len(view.build_section_rows(model)), view.MAX_VIEW_SECTIONS
        )

    def test_bounded_view_prioritizes_routed_sections_over_early_noise(self):
        sections = [_section(index) for index in range(200)]
        late_selected = [
            sections[190].section_id,
            sections[195].section_id,
            sections[199].section_id,
        ]
        route = _route(
            considered=[section.section_id for section in sections],
            planned=[sections[195].section_id],
            selected=late_selected,
        )
        model = view.build_map_view_model(_document_map(sections), route)
        visible = {node["section_id"] for node in model["sections"]}
        self.assertLessEqual(len(model["sections"]), view.MAX_VIEW_SECTIONS)
        for section_id in late_selected:
            self.assertIn(section_id, visible)
        step_ids = [step["section_id"] for step in model["steps"]]
        self.assertEqual(step_ids, late_selected)

    def test_long_fields_are_truncated_per_field(self):
        sections = [
            _section(
                0,
                title="T" * 5000,
                source="S" * 5000,
                summary="U" * 5000,
                keywords=[f"K{index}" * 40 for index in range(30)],
            )
        ]
        model = view.build_map_view_model(_document_map(sections), _route())
        node = model["sections"][0]
        self.assertLessEqual(len(node["title"]), view.MAX_LABEL_CHARS)
        self.assertLessEqual(len(node["source"]), view.MAX_LABEL_CHARS)
        self.assertLessEqual(len(node["summary"]), view.MAX_SUMMARY_CHARS)
        self.assertLessEqual(len(node["keywords"]), 6)
        row = view.build_section_rows(model)[0]
        self.assertLessEqual(len(row["Title"]), view.MAX_LABEL_CHARS)
        self.assertLessEqual(len(row["Source"]), view.MAX_LABEL_CHARS)
        self.assertNotIn(SECRET_NODE_ID, view.build_map_html(model))

    def test_oversized_markup_falls_back_to_a_text_map(self):
        sections = [_section(index) for index in range(view.MAX_VIEW_SECTIONS)]
        with (
            patch.object(view, "MAX_VIEW_CHARACTERS", 64),
            _render() as (streamlit, rendered),
        ):
            self.assertTrue(
                view.render_retrieval_route(_route(), _document_map(sections))
            )
        self.assertEqual(rendered.calls, [])
        self.assertIn(view.MAP_FALLBACK_NOTE, _captions(streamlit))
        self.assertTrue(streamlit.writes)


class MapMismatchTests(unittest.TestCase):
    def test_stale_map_id_keeps_metrics_and_drops_edges(self):
        sections = [
            _section(0),
            _section(1),
        ]
        route = _route(
            considered=[item.section_id for item in sections],
            planned=[sections[0].section_id],
            selected=[sections[0].section_id, sections[1].section_id],
            map_id="map_previous_generation",
            reflected=True,
        )
        model = view.build_map_view_model(_document_map(sections), route)
        markup = view.build_map_html(model)

        self.assertTrue(model["map_mismatch"])
        self.assertEqual(model["map_id"], "map_current")
        self.assertEqual(model["route_map_id"], "map_previous_generation")
        self.assertEqual(model["steps"], [])
        self.assertFalse(model["reflected"])
        self.assertEqual(
            {node["state"] for node in model["sections"]}, {view.STATE_UNCONSIDERED}
        )
        self.assertEqual({node["route_order"] for node in model["sections"]}, {0})
        self.assertNotIn("<circle", markup)
        self.assertIn("Map mismatch", markup)
        self.assertEqual(
            _metric(view.build_metric_rows(route), "Sections selected"), "2"
        )

    def test_mismatch_note_is_rendered_for_a_stored_answer(self):
        route = _route(map_id="map_previous_generation")
        with _render() as (streamlit, rendered):
            self.assertTrue(view.render_retrieval_route(route, _document_map()))
        self.assertIn(view.MAP_MISMATCH_NOTE, streamlit.warnings)
        self.assertTrue(rendered.calls)
        self.assertTrue(streamlit.tables)

    def test_matching_map_id_draws_the_route(self):
        model = view.build_map_view_model(_document_map(), _route())
        self.assertFalse(model["map_mismatch"])
        self.assertEqual(len(model["steps"]), 1)
        self.assertIn("<circle", view.build_map_html(model))


class RouteMetricTests(unittest.TestCase):
    def test_signed_delta_and_honest_caption(self):
        route = _route(reflected=True)
        rows = view.build_metric_rows(route)
        self.assertEqual(_metric(rows, "Sections considered"), "1")
        self.assertEqual(_metric(rows, "Sections planned"), "1")
        self.assertEqual(_metric(rows, "Sections selected"), "1")
        self.assertEqual(_metric(rows, "Selected evidence chunks"), "3")
        self.assertEqual(_metric(rows, "Route steps"), "2")
        self.assertEqual(_metric(rows, "Reflection"), "Triggered")
        self.assertEqual(_metric(rows, "Planner mode"), "Deterministic keyword scores")
        self.assertEqual(_metric(rows, "Selected evidence tokens"), "640")
        self.assertEqual(_metric(rows, "RAG input tokens (actual)"), "1000")
        self.assertEqual(_metric(rows, "Baseline input tokens"), "2,000")
        self.assertEqual(
            _metric(rows, "Token delta vs measured baseline"),
            "-1,000 tokens (-50.00%)",
        )
        delta = _metric(rows, "Token delta vs measured baseline")
        self.assertNotIn("sav", delta.casefold())
        self.assertNotIn("guarantee", delta.casefold())

    def test_positive_delta_keeps_its_sign(self):
        route = _route()
        route["token_accounting"]["baseline_input_prompt_tokens"] = 400
        route["token_accounting"]["selected_minus_baseline_input_tokens"] = 600
        route["token_accounting"]["selected_minus_baseline_input_pct"] = 150.0
        self.assertEqual(
            _metric(view.build_metric_rows(route), "Token delta vs measured baseline"),
            "+600 tokens (+150.00%)",
        )

    def test_measurement_caption_refuses_wider_claims(self):
        rendered = _captions_for_route(_route())
        self.assertIn("Tokenizer and prompt-context measurement only", rendered)
        for excluded in (
            "total compute",
            "latency",
            "GPU",
            "embedding",
            "answer-quality",
        ):
            self.assertIn(excluded, rendered)
        self.assertIn("not a guarantee", rendered)

    def test_baseline_disabled_state_explains_how_to_enable_it(self):
        route = _route(baseline=False)
        rows = view.build_metric_rows(route)
        self.assertEqual(_metric(rows, "Baseline input tokens"), "not measured")
        self.assertEqual(
            _metric(rows, "Token delta vs measured baseline"), "not measured"
        )
        self.assertEqual(_metric(rows, "RAG input tokens (actual)"), "1000")
        note = view.baseline_note(route)
        self.assertIn("Baseline comparison is off", note)
        self.assertIn("Agentic Map Retrieval", note)
        self.assertIn("Measure global baseline", note)
        self.assertIn("no extra answer-model call", note)
        self.assertIn("Tokenizer and prompt-context measurement only", note)
        self.assertEqual(view.baseline_note(_route()), view.TOKEN_MEASUREMENT_NOTE)

    def test_missing_map_still_reports_the_route(self):
        with _render() as (streamlit, rendered):
            self.assertTrue(view.render_retrieval_route(_route(), None))
        self.assertEqual(rendered.calls, [])
        self.assertIn(view.MAP_MISSING_NOTE, streamlit.infos)
        self.assertTrue(streamlit.tables)
        self.assertEqual(
            _metric(view.build_metric_rows(_route()), "Selected evidence chunks"), "3"
        )

    def test_overview_uses_compact_bounds(self):
        sections = [_section(index, source_index=index % 6) for index in range(40)]
        with _render() as (_streamlit, rendered):
            self.assertTrue(view.render_map_overview(_document_map(sections), None))
        self.assertLessEqual(
            len(
                view.build_map_view_model(
                    _document_map(sections),
                    None,
                    max_sections=view.MAX_OVERVIEW_SECTIONS,
                    max_documents=view.MAX_OVERVIEW_DOCUMENTS,
                )["sections"]
            ),
            view.MAX_OVERVIEW_SECTIONS,
        )
        _markup, kwargs = rendered.calls[0]
        self.assertLessEqual(kwargs["height"], view.MAX_OVERVIEW_HEIGHT)


class RouteTraceTests(unittest.TestCase):
    def test_action_rows_cover_every_route_phase(self):
        route = _route(
            actions=[
                {"step": 0, "action": "observe", "reason": "bounded_map_ready"},
                {"step": 0, "action": "plan", "reason": "deterministic_keyword_scores"},
                {
                    "step": 1,
                    "action": "retrieve_global",
                    "reason": "routed_evidence_weak",
                    "attempted_queries": 2,
                    "result_count": 4,
                },
                {"step": 1, "action": "reflect", "reason": "weak_evidence"},
                {
                    "step": 2,
                    "action": "expand_sections",
                    "reason": "positive_unrouted_sections",
                    "allowed_section_ids": ["a", "b"],
                },
                {"step": 2, "action": "global_fallback", "reason": "fallback"},
            ]
        )
        rows = view.build_action_rows(route)
        self.assertEqual(
            [row["Phase"] for row in rows],
            ["observe", "plan", "retrieve", "reflect", "expand", "fallback"],
        )
        self.assertEqual(rows[2]["Queries"], 2)
        self.assertEqual(rows[2]["Results"], 4)
        self.assertEqual(rows[4]["Sections"], 2)
        self.assertNotIn(
            "member", " ".join(str(value) for row in rows for value in row.values())
        )

    def test_section_table_exposes_only_bounded_metadata(self):
        sections = [
            _section(0, keywords=("alpha", "beta")),
            _section(1, keywords=()),
        ]
        route = _route(
            considered=[item.section_id for item in sections],
            planned=[sections[0].section_id],
            selected=[sections[0].section_id],
        )
        rows = view.build_section_rows(
            view.build_map_view_model(_document_map(sections), route)
        )
        self.assertEqual(
            list(rows[0]),
            [
                "Title",
                "Source",
                "Sequence",
                "Tokens",
                "Keyword summary",
                "State",
            ],
        )
        self.assertEqual(rows[0]["Keyword summary"], "alpha, beta")
        self.assertEqual(rows[1]["Keyword summary"], "-")
        self.assertEqual(rows[0]["State"], view.STATE_LABELS[view.STATE_SELECTED])
        self.assertEqual(rows[1]["State"], view.STATE_LABELS[view.STATE_CONSIDERED])
        flat = " ".join(str(value) for row in rows for value in row.values())
        self.assertNotIn(SECRET_NODE_ID, flat)
        self.assertNotIn("map_sec_0_0", flat)

    def test_truncated_map_is_announced(self):
        with _render() as (streamlit, _rendered):
            self.assertTrue(
                view.render_retrieval_route(_route(), _document_map(truncated=True))
            )
        self.assertIn(view.TRUNCATION_NOTE, _captions(streamlit))
        self.assertIn(
            view.TRUNCATION_NOTE,
            view.build_map_html(
                view.build_map_view_model(_document_map(truncated=True), _route())
            ),
        )

    def test_reflection_and_traversal_are_drawn(self):
        sections = [_section(0), _section(1), _section(2)]
        route = _route(
            considered=[item.section_id for item in sections],
            planned=[sections[0].section_id],
            selected=[item.section_id for item in sections],
            reflected=True,
        )
        markup = view.build_map_html(
            view.build_map_view_model(_document_map(sections), route)
        )
        self.assertIn("tl-back", markup)
        self.assertIn("reflect", markup)
        self.assertEqual(markup.count("<circle"), 3)
        self.assertIn('class="order"', markup)


def _captions_for_route(route):
    with _render() as (streamlit, _rendered):
        view.render_retrieval_route(route, _document_map())
    return _captions(streamlit)


class RenderHelperTests(unittest.TestCase):
    def test_route_renders_one_collapsed_expander_with_a_map(self):
        with _render() as (streamlit, rendered):
            self.assertTrue(
                view.render_retrieval_route(_route(), _document_map(), key="one")
            )
        self.assertEqual(len(streamlit.expanders), 1)
        label, kwargs = streamlit.expanders[0]
        self.assertIn("Retrieval route", label)
        self.assertFalse(kwargs.get("expanded", True))
        self.assertEqual(len(rendered.calls), 1)
        markup, kwargs = rendered.calls[0]
        self.assertIn("<!DOCTYPE html>", markup)
        self.assertLessEqual(kwargs["height"], view.MAX_VIEW_HEIGHT)
        self.assertEqual(len(streamlit.tables), 3)

    def test_empty_or_missing_route_renders_nothing(self):
        with _render() as (streamlit, rendered):
            self.assertFalse(view.render_retrieval_route({}, _document_map()))
            self.assertFalse(view.render_retrieval_route(None, _document_map()))
            self.assertFalse(view.render_retrieval_route([], _document_map()))
        self.assertEqual(streamlit.expanders, [])
        self.assertEqual(rendered.calls, [])

    def test_component_module_absence_falls_back_to_text_and_table(self):
        with (
            _render(components=None) as (streamlit, _rendered),
            patch.object(view, "components", None),
        ):
            self.assertTrue(view.render_retrieval_route(_route(), _document_map()))
        self.assertIn(view.MAP_FALLBACK_NOTE, _captions(streamlit))
        self.assertTrue(streamlit.writes)
        self.assertTrue(streamlit.tables)

    def test_component_failure_falls_back_without_raising(self):
        failing = _FakeComponents(RuntimeError("boom"))
        with _render(components=failing) as (streamlit, _rendered):
            self.assertTrue(view.render_retrieval_route(_route(), _document_map()))
        self.assertTrue(failing.calls)
        self.assertIn(view.MAP_FALLBACK_NOTE, _captions(streamlit))
        self.assertTrue(streamlit.writes)

    def test_streamlit_failure_is_contained(self):
        class _Broken:
            def __getattr__(self, name):
                def _boom(*args, **kwargs):
                    raise RuntimeError("streamlit down")

                return _boom

        with patch.object(view, "st", _Broken()):
            self.assertFalse(
                view.render_retrieval_route(_route(), _document_map(), key="broken")
            )
            self.assertFalse(view.render_map_overview(_document_map(), _route()))

    def test_overview_panel_prompts_for_a_question_without_a_route(self):
        with _render() as (streamlit, rendered):
            self.assertTrue(view.render_map_overview(_document_map(), None))
        self.assertEqual(streamlit.containers, 1)
        self.assertTrue(rendered.calls)
        self.assertIn(view.MAP_ASK_QUESTION_NOTE, " \n".join(streamlit.infos))

    def test_overview_panel_summarises_the_latest_route(self):
        with _render() as (streamlit, rendered):
            self.assertTrue(view.render_map_overview(_document_map(), _route()))
        self.assertIn(
            "Last route: 1 selected · 1 planned · 1 considered · 2 step(s) · deterministic",
            _captions(streamlit),
        )
        self.assertIn(view.TOKEN_MEASUREMENT_NOTE, _captions(streamlit))
        self.assertTrue(rendered.calls)
        self.assertEqual(streamlit.infos, [])

    def test_overview_is_skipped_without_a_map_or_route(self):
        with _render() as (streamlit, rendered):
            self.assertFalse(view.render_map_overview(None, None))
            self.assertFalse(view.render_map_overview(None, {}))
        self.assertEqual(streamlit.containers, 0)
        self.assertEqual(rendered.calls, [])


class MessageRouteTests(unittest.TestCase):
    def test_stored_local_assistant_route_renders(self):
        message = {
            "role": "assistant",
            "content": "answer",
            "evidence": [{"source": "doc0.txt"}],
            "retrieval_route": _route(),
        }
        with _render() as (streamlit, rendered):
            self.assertTrue(
                view.render_stored_route(message, _document_map(), key="stored-route-1")
            )
        self.assertEqual(len(streamlit.expanders), 1)
        self.assertEqual(len(rendered.calls), 1)

    def test_r2r_and_direct_messages_show_no_local_map(self):
        with _render() as (streamlit, rendered):
            self.assertFalse(
                view.render_stored_route(
                    {"role": "assistant", "r2r": {"citations": []}, "content": "a"},
                    _document_map(),
                )
            )
            self.assertFalse(
                view.render_stored_route(
                    {
                        "role": "assistant",
                        "content": "a",
                        "evidence": [],
                        "retrieval_route": {},
                    },
                    _document_map(),
                )
            )
            self.assertFalse(
                view.render_stored_route(
                    {"role": "assistant", "content": "a"}, _document_map()
                )
            )
            self.assertFalse(view.render_stored_route("not-a-message", _document_map()))
            self.assertFalse(
                view.render_stored_route(
                    {"role": "user", "content": "q", "retrieval_route": _route()},
                    _document_map(),
                )
            )
        self.assertEqual(streamlit.expanders, [])
        self.assertEqual(rendered.calls, [])

    def test_every_stored_message_renders_once_per_run(self):
        first = {"role": "assistant", "content": "a", "retrieval_route": _route()}
        second = {"role": "assistant", "content": "b", "retrieval_route": _route()}
        with _render() as (streamlit, rendered):
            for index, message in enumerate((first, second)):
                self.assertTrue(
                    view.render_stored_route(
                        message, _document_map(), key=f"stored-route-{index}"
                    )
                )
            self.assertFalse(
                view.render_stored_route(first, _document_map(), key="stored-route-0")
            )
        self.assertEqual(len(streamlit.expanders), 2)
        self.assertEqual(len(rendered.calls), 2)

    def test_a_new_render_pass_allows_rendering_again(self):
        with _render() as (streamlit, rendered):
            self.assertTrue(
                view.render_retrieval_route(_route(), _document_map(), key="same")
            )
        with _render() as (streamlit, rendered):
            self.assertTrue(
                view.render_retrieval_route(_route(), _document_map(), key="same")
            )
        self.assertEqual(len(streamlit.expanders), 1)
        self.assertEqual(len(rendered.calls), 1)

    def test_a_stored_route_does_not_hide_the_current_answer(self):
        stored = {
            "role": "assistant",
            "content": "a",
            "retrieval_route": _route(),
        }
        with _render() as (_streamlit, rendered):
            view.render_stored_route(stored, _document_map(), key="stored-route-3")
            self.assertTrue(
                view.render_retrieval_route(
                    _route(), _document_map(), key="current-route"
                )
            )
        self.assertEqual(len(rendered.calls), 2)


class ChatboxRouteTests(unittest.TestCase):
    def _chat_streamlit(self, state):
        class _ChatStreamlit(_FakeStreamlit):
            def chat_message(self, *args, **kwargs):
                return _Container()

            def spinner(self, *args, **kwargs):
                return _Container()

            def write_stream(self, stream):
                return "".join(str(item) for item in stream)

            def button(self, *args, **kwargs):
                return False

            def rerun(self):
                return None

        return _ChatStreamlit(state)

    def test_local_answer_renders_a_collapsed_route_expander(self):
        state = {
            "messages": [],
            "selected_model": "llama3:8b",
            "query_engine": "engine",
            "last_doc_sources": [],
            "last_rag_evidence": [],
            "last_retrieval_route": {},
            "last_rag_no_result": False,
            "last_rag_question": None,
            "retrieval_map": _document_map(),
        }
        route = _route()
        streamlit = self._chat_streamlit(state)
        view_streamlit = _FakeStreamlit(state)
        rendered = _FakeComponents()

        def _context_chat(prompt, query_engine, evidence_sink=None, route_sink=None):
            route_sink.update(route)
            evidence_sink.append({"source": "doc0.txt", "score": 0.9})
            yield "grounded answer"

        with (
            patch.object(chatbox, "st", streamlit),
            patch.object(chatbox, "_local_index_ready", return_value=True),
            patch.object(chatbox, "is_openai_compatible_backend", return_value=False),
            patch.object(chatbox.r2r, "r2r_is_ready", return_value=False),
            patch.object(chatbox, "context_chat", _context_chat),
            patch.object(view, "st", view_streamlit),
            patch.object(view, "components", rendered),
        ):
            view.begin_render_pass()
            chatbox._process_prompt("question about section one")

        self.assertEqual(len(view_streamlit.expanders), 1)
        self.assertIn("Retrieval route", view_streamlit.expanders[0][0])
        self.assertEqual(len(rendered.calls), 1)
        self.assertTrue(
            state["messages"][-1]["retrieval_route"]["selected_section_ids"]
        )

    def test_r2r_answer_renders_no_local_route(self):
        state = {
            "messages": [],
            "query_engine": "engine",
            "last_doc_sources": [],
            "last_rag_evidence": [],
            "last_retrieval_route": {"selected_section_ids": ["stale"]},
            "last_rag_no_result": False,
            "last_rag_question": None,
            "retrieval_map": _document_map(),
        }
        streamlit = self._chat_streamlit(state)
        view_streamlit = _FakeStreamlit(state)
        rendered = _FakeComponents()
        with (
            patch.object(chatbox, "st", streamlit),
            patch.object(chatbox, "_local_index_ready", return_value=True),
            patch.object(chatbox.r2r, "r2r_is_ready", return_value=True),
            patch.object(chatbox.r2r, "r2r_chat", return_value=iter(["r2r answer"])),
            patch.object(view, "st", view_streamlit),
            patch.object(view, "components", rendered),
        ):
            view.begin_render_pass()
            chatbox._process_prompt("question")

        self.assertEqual(view_streamlit.expanders, [])
        self.assertEqual(rendered.calls, [])
        self.assertNotIn("retrieval_route", state["messages"][-1])


class MapSettingsControlTests(unittest.TestCase):
    def _state(self, **overrides):
        state = {
            "advanced": True,
            "llm_backend": "Ollama",
            "ollama_endpoint": "http://localhost:11434",
            "ollama_models": ["llama3:8b"],
            "selected_model": "llama3:8b",
            "ollama_embedding_models": ["nomic-embed-text"],
            "ollama_embedding_model": "nomic-embed-text",
            "top_k": 3,
            "similarity_cutoff": 0.3,
            "chunk_size": 256,
            "chunk_overlap_pct": 12,
            "temperature": 0.4,
            "messages": [],
        }
        state.update(overrides)
        return state

    def _run(self, state):
        streamlit = _FakeStreamlit(state)
        with patch.object(settings_tab, "st", streamlit):
            settings_tab.settings()
        return streamlit

    def test_group_uses_exact_keys_defaults_and_ranges(self):
        streamlit = self._run(self._state())
        toggles = {kwargs.get("key"): label for label, kwargs in streamlit.toggles}
        self.assertIn("retrieval_map_enabled", toggles)
        self.assertIn("retrieval_map_measure_baseline", toggles)
        sliders = {
            kwargs.get("key"): (label, kwargs) for label, kwargs in streamlit.sliders
        }
        for key, bounds in MAP_SETTING_RANGES.items():
            with self.subTest(key=key):
                self.assertIn(key, sliders)
                label, kwargs = sliders[key]
                self.assertEqual(kwargs["min_value"], int(bounds[0]))
                self.assertLessEqual(kwargs["max_value"], int(bounds[1]))
                if (
                    key in MAP_SETTING_DEFAULTS
                    and key != "retrieval_map_initial_sections"
                ):
                    self.assertEqual(kwargs["value"], MAP_SETTING_DEFAULTS[key])
                self.assertTrue(label)
                self.assertTrue(kwargs.get("help"))
        planner = next(
            (label, choices, kwargs)
            for label, choices, kwargs in streamlit.selectboxes
            if kwargs.get("key") == "retrieval_map_planner_mode"
        )
        self.assertEqual(planner[1], list(MAP_PLANNER_MODES))
        self.assertNotIn("index", planner[2])
        self.assertTrue(planner[2].get("help"))
        self.assertTrue(
            any("Agentic Map Retrieval" in text for text, _ in streamlit.markdowns)
        )
        self.assertIn("not accuracy claims", " \n".join(streamlit.captions))

    def test_planned_sections_never_exceed_selected_sections(self):
        state = self._state(
            retrieval_map_max_selected_sections=1,
            retrieval_map_initial_sections=9,
        )
        streamlit = self._run(state)
        sliders = {kwargs.get("key"): kwargs for _label, kwargs in streamlit.sliders}
        self.assertEqual(sliders["retrieval_map_initial_sections"]["min_value"], 1)
        self.assertEqual(
            sliders["retrieval_map_initial_sections"]["max_value"],
            state["retrieval_map_max_selected_sections"],
        )
        self.assertEqual(state["retrieval_map_initial_sections"], 1)
        self.assertLessEqual(
            state["retrieval_map_initial_sections"],
            state["retrieval_map_max_selected_sections"],
        )
        wide = self._run(
            self._state(
                retrieval_map_max_selected_sections=16,
                retrieval_map_initial_sections=16,
            )
        )
        wide_sliders = {kwargs.get("key"): kwargs for _label, kwargs in wide.sliders}
        self.assertEqual(
            wide_sliders["retrieval_map_initial_sections"]["max_value"],
            MAP_SETTING_RANGES["retrieval_map_initial_sections"][1],
        )

    def test_baseline_toggle_explains_extra_retrieval_and_no_extra_model_call(self):
        off = self._run(self._state(retrieval_map_measure_baseline=False))
        captions = " \n".join(off.captions)
        self.assertIn("no extra retrieval work", captions)
        self.assertIn("baseline comparison", captions)
        on = self._run(self._state(retrieval_map_measure_baseline=True))
        captions_on = " \n".join(on.captions)
        self.assertIn("one unrestricted local retrieval pass", captions_on)
        self.assertIn("No extra answer-model call", captions_on)
        toggles = {kwargs.get("key"): kwargs for _label, kwargs in on.toggles}
        self.assertIn(
            "no extra answer-model call",
            toggles["retrieval_map_measure_baseline"]["help"],
        )

    def test_map_group_is_hidden_until_advanced_is_enabled(self):
        streamlit = self._run(self._state(advanced=False))
        keys = {kwargs.get("key") for _label, kwargs in streamlit.toggles}
        self.assertNotIn("retrieval_map_enabled", keys)
        self.assertFalse(
            any("retrieval_map" in str(key) for key in keys),
        )

    def test_layout_change_caption_mentions_rebuild_without_reingestion(self):
        streamlit = self._run(self._state())
        captions = " \n".join(streamlit.captions)
        self.assertIn("rebuild the compact map on your next question", captions)
        self.assertIn("do not need to be ingested again", captions)


class SourcesOverviewTests(unittest.TestCase):
    def test_sources_renders_the_overview_only_when_a_map_or_route_exists(self):
        from components.tabs import sources as sources_tab

        for document_map, route, expected in (
            (None, None, False),
            (None, _route(), True),
            (_document_map(), None, True),
            (_document_map(), _route(), True),
        ):
            with self.subTest(document_map=bool(document_map), route=bool(route)):
                state = {
                    "retrieval_map": document_map,
                    "last_retrieval_route": route,
                }
                streamlit = _FakeStreamlit(state)
                with (
                    patch.object(sources_tab, "st", streamlit),
                    patch.object(view, "st", streamlit),
                    patch.object(view, "components", _FakeComponents()),
                    patch.object(
                        sources_tab, "ingestion_is_configured", return_value=True
                    ),
                    patch.object(sources_tab, "local_files"),
                    patch.object(sources_tab, "github_repo"),
                    patch.object(sources_tab, "website"),
                ):
                    view.begin_render_pass()
                    sources_tab.sources()
                self.assertEqual(streamlit.containers, 1 if expected else 0)


def _map_route_app():
    import streamlit as st

    from components import retrieval_map_view as route_view
    from components.page_state import WELCOME_MESSAGE
    from utils.retrieval_map import (
        DocumentMap,
        MapNode,
        MapSection,
        RetrievalMapConfig,
    )

    section = MapSection(
        section_id="map_sec_0_0",
        source_id="map_doc_0",
        source="doc0.txt",
        source_identity="doc0:0",
        title="Doc 0 Section 0",
        summary="bounded section summary",
        keywords=("alpha", "beta"),
        member_node_ids=("hidden-member-node",),
        chunk_count=2,
        token_count=120,
        sequence=1,
    )
    document_map = DocumentMap(
        map_id="map_current",
        version=1,
        source_identity="doc0:0",
        sections=(section,),
        nodes=(
            MapNode(
                node_id="map_doc_0",
                kind="document",
                label="doc0.txt",
                source_id="map_doc_0",
            ),
            MapNode(
                node_id=section.section_id,
                kind="section",
                label=section.title,
                source_id="map_doc_0",
            ),
        ),
        edges=(),
        input_node_count=2,
        mapped_node_count=2,
        skipped_node_count=0,
        truncated=False,
        config=RetrievalMapConfig(),
    )
    route = {
        "map_id": "map_current",
        "considered_section_ids": ["map_sec_0_0"],
        "planned_section_ids": ["map_sec_0_0"],
        "selected_section_ids": ["map_sec_0_0"],
        "step_count": 1,
        "reflection_flag": False,
        "planner_mode": "deterministic",
        "actions": [{"step": 0, "action": "observe", "reason": "bounded_map_ready"}],
        "metrics": {
            "planner_mode": "deterministic",
            "selected_evidence_tokens": 240,
            "result_count": 2,
        },
        "token_accounting": {
            "baseline_measured": False,
            "actual_selected_evidence_tokens": 240,
            "actual_rag_input_prompt_tokens": 900,
            "baseline_input_prompt_tokens": None,
            "selected_minus_baseline_input_tokens": None,
            "selected_minus_baseline_input_pct": None,
        },
    }
    st.session_state.update(
        {
            "messages": [dict(WELCOME_MESSAGE)],
            "last_doc_sources": [],
            "last_rag_evidence": [],
            "last_retrieval_route": {},
            "last_rag_no_result": False,
            "last_rag_question": None,
            "system_prompt": "system",
            "answer_style": "Balanced (default)",
            "quick_answer_style": "Balanced (default)",
            "retrieval_map": document_map,
        }
    )
    messages = [
        *st.session_state["messages"],
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "stored answer",
            "evidence": [{"source": "doc0.txt", "score": 0.9}],
            "retrieval_route": route,
        },
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "r2r answer", "r2r": {"citations": []}},
    ]
    route_view.begin_render_pass()
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("role") == "assistant":
                route_view.render_stored_route(
                    message,
                    st.session_state["retrieval_map"],
                    key=f"stored-route-{index}",
                )
    route_view.render_map_overview(
        st.session_state["retrieval_map"], st.session_state["last_retrieval_route"]
    )


class StreamlitAppTestTests(unittest.TestCase):
    def test_apptest_renders_stored_routes_without_exceptions(self):
        script = (
            "from streamlit.testing.v1 import AppTest\n"
            "from tests.test_retrieval_map_view import _map_route_app\n"
            "app = AppTest.from_function(_map_route_app).run(timeout=30)\n"
            "assert not app.exception, [str(item.value) for item in app.exception]\n"
            "assert len(app.expander) == 1, len(app.expander)\n"
            "assert 'Retrieval route' in app.expander[0].label\n"
            "assert len(app.chat_message) == 5, len(app.chat_message)\n"
            "assert len(app.table) == 3, len(app.table)\n"
            "captions = ' '.join(item.value for item in app.caption)\n"
            "assert 'Tokenizer and prompt-context measurement only' in captions\n"
            "assert 'Baseline comparison is off' in captions\n"
            "assert any('Ask a question' in item.value for item in app.info)\n"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
