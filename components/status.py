"""One shared description of what the application is currently doing.

The header strip and the sidebar card used to disagree, because each computed the
answering mode on its own. They now both read this module so the mode, provider,
source, and map state can never be rendered two different ways.
"""

from __future__ import annotations

import html

import streamlit as st

from utils.provider_config import get_chat_profile
from utils.source_state import active_index_matches_settings, ensure_active_source

MODE_DIRECT = "direct"
MODE_RAG = "rag"
MODE_R2R = "r2r"

MODE_LABELS = {
    MODE_DIRECT: "Direct chat",
    MODE_RAG: "Local RAG",
    MODE_R2R: "R2R",
}

MODE_HINTS = {
    MODE_DIRECT: "No document index. Answers come from the model only.",
    MODE_RAG: "Answers are grounded in the active local index.",
    MODE_R2R: "Answers come from documents on the configured R2R server.",
}

SOURCE_LABELS = {
    "local": "Local files",
    "github": "GitHub repository",
    "website": "Website",
    "r2r": "R2R documents",
}


def _clip(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _model_label(state) -> str:
    backend = state.get("llm_backend", "Ollama")
    try:
        profile = get_chat_profile(state, backend)
    except Exception:
        return ""
    model = _clip(profile.get("model"), 42)
    return f"{backend} · {model}" if model else str(backend)


def _map_summary(state) -> dict:
    document_map = state.get("retrieval_map")
    empty = {"built": False, "sections": 0, "nodes": 0, "truncated": False}
    if document_map is None or not hasattr(document_map, "sections"):
        return empty
    sections = getattr(document_map, "sections", ()) or ()
    try:
        nodes = int(getattr(document_map, "input_node_count", 0) or 0)
    except (TypeError, ValueError):
        nodes = 0
    return {
        "built": True,
        "sections": len(sections),
        "nodes": nodes,
        "truncated": bool(getattr(document_map, "truncated", False)),
    }


def resolve_status(state=None) -> dict:
    """Return the current mode, provider, source, and map state."""
    state = st.session_state if state is None else state
    source = ensure_active_source(state)
    kind = source.get("kind")
    ready = source.get("status") == "ready"
    has_engine = bool(state.get("query_engine"))
    r2r_ids = state.get("r2r_document_ids")

    if state.get("r2r_enabled") and r2r_ids:
        mode = MODE_R2R if kind == "r2r" and ready else MODE_DIRECT
        note = (
            ""
            if mode == MODE_R2R
            else "R2R is enabled but its documents are not ready."
        )
    elif has_engine and kind in {None, "local", "github", "website"} and ready:
        mode = MODE_RAG if active_index_matches_settings(state) else MODE_DIRECT
        note = (
            ""
            if mode == MODE_RAG
            else "The index was built with different settings. Re-add the source."
        )
    else:
        mode = MODE_DIRECT
        note = ""

    return {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "mode_hint": MODE_HINTS[mode],
        "note": note,
        "source_kind": kind or "",
        "source_label": SOURCE_LABELS.get(kind or "", "No source"),
        "source_display": _clip(source.get("display_name"), 48),
        "model": _model_label(state),
        "index_ready": bool(has_engine and ready),
        "map": _map_summary(state),
        "map_enabled": bool(state.get("retrieval_map_enabled", True)),
    }


def _badge(text: str, tone: str) -> str:
    # The model name comes from user-editable settings, so it is escaped before
    # it reaches unsafe_allow_html rather than trusted to be well formed.
    return f'<span class="dm-badge dm-{tone}">{html.escape(str(text))}</span>'


def render_status_strip(status: dict | None = None) -> None:
    """Render the orientation strip under the brand row."""
    status = status or resolve_status()
    tone = "ok" if status["mode"] != MODE_DIRECT else "idle"
    parts = [_badge(status["mode_label"], tone)]
    if status["source_label"] != "No source":
        parts.append(_badge(status["source_label"], "idle"))
    if status["model"]:
        parts.append(_badge(status["model"], "idle"))
    map_state = status["map"]
    if status["map_enabled"] and map_state["built"]:
        suffix = "+" if map_state["truncated"] else ""
        parts.append(_badge(f"Map {map_state['sections']} sections{suffix}", "ok"))
    elif status["map_enabled"] and status["index_ready"]:
        parts.append(_badge("Map not built", "warn"))
    st.markdown(
        '<div class="dm-strip">' + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )
    if status["note"]:
        st.caption(status["note"])


def render_mode_card(status: dict | None = None) -> None:
    """Render the sidebar status card above the tabs."""
    status = status or resolve_status()
    if status["mode"] == MODE_R2R:
        st.success("R2R Mode: using documents on the external R2R server")
    elif status["mode"] == MODE_RAG:
        st.success("RAG Mode: using the active local document index")
    else:
        st.info("Chat Mode: direct model conversation without a document index")
    if status["source_display"]:
        st.caption(f"Active source: {status['source_display']}")
    map_state = status["map"]
    if status["map_enabled"] and status["index_ready"]:
        if map_state["built"]:
            st.caption(
                f"Retrieval map: {map_state['sections']} sections over "
                f"{map_state['nodes']} chunks"
                + (" (truncated)" if map_state["truncated"] else "")
            )
        else:
            st.caption("Retrieval map: not built for the active source yet.")
