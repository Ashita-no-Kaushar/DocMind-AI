import copy

import streamlit as st

import utils.r2r as r2r
from components.retrieval_map_view import render_retrieval_route
from utils.ollama import (
    chat,
    context_chat,
    get_embedding_models,
    get_models,
    is_openai_compatible_backend,
)
from utils.source_state import active_index_matches_settings, ensure_active_source

ANSWER_STYLE_OPTIONS = [
    "Concise",
    "Balanced (default)",
    "Detailed",
    "Bulleted",
    "Technical",
    "Simple / ELI5",
]


def _style_to_prompt(style: str) -> str:
    """Return the system prompt for a given answer style preset."""
    base = (
        "You are DocMind AI, a helpful and accurate virtual assistant. "
        "When document context is provided, answer strictly from that context "
        "and do not invent information. Otherwise answer from your general "
        "knowledge. Be factual and conversational."
    )
    style_instructions = {
        "Concise": " Keep answers as short as possible — one or two sentences.",
        "Balanced (default)": " Keep answers concise but complete.",
        "Detailed": " Provide thorough explanations with context and examples.",
        "Bulleted": " Structure answers as bullet points for readability.",
        "Technical": " Use precise terminology; assume a technical audience.",
        "Simple / ELI5": " Explain simply, like you're talking to a 12-year-old. Avoid jargon.",
    }
    return base + style_instructions.get(
        style, style_instructions["Balanced (default)"]
    )


def _sync_answer_style(source_key: str):
    """Keep the two style selectors (chatbox + Settings) in sync."""
    # Settings uses 'answer_style', chatbox uses 'quick_answer_style' — keep them identical
    style = st.session_state.get(source_key, "Balanced (default)")
    st.session_state["answer_style"] = style
    st.session_state["quick_answer_style"] = style
    st.session_state["system_prompt"] = _style_to_prompt(style)


def _apply_quick_answer_style():
    """Rewrite the system prompt when the quick tone selector changes."""
    _sync_answer_style("quick_answer_style")


def _apply_answer_style_from_settings():
    _sync_answer_style("answer_style")


def _local_index_ready():
    source = ensure_active_source(st.session_state)
    if source.get("kind") is None:
        return bool(st.session_state.get("query_engine"))
    return bool(
        st.session_state.get("query_engine")
        and source.get("kind") in {"local", "github", "website"}
        and source.get("status") == "ready"
        and active_index_matches_settings(st.session_state)
    )


def _suggested_questions():
    """Starter questions shown while the conversation is still empty."""
    if _local_index_ready() or r2r.r2r_is_ready(st.session_state):
        return [
            "Summarize my documents",
            "What are the key points?",
            "Explain the main terms",
        ]
    return [
        "What is RAG in simple words?",
        "Help me write a polite email",
        "Give me 3 study tips",
    ]


def _clear_turn_state():
    st.session_state["last_doc_sources"] = []
    st.session_state["last_rag_evidence"] = []
    st.session_state["last_retrieval_route"] = {}
    st.session_state["last_rag_no_result"] = False
    st.session_state["last_rag_question"] = None


def _render_sources(sources):
    best = {}
    for item in sources or []:
        if isinstance(item, dict):
            name = str(item.get("source", "document"))
            score = item.get("score", 0.0)
        else:
            try:
                name, score = item
            except (TypeError, ValueError):
                continue
            name = str(name)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if name not in best or score > best[name]:
            best[name] = score
    labels = []
    for name, score in best.items():
        if score >= 0.15:
            labels.append(f"`{name}` ({score:.0%})")
        else:
            labels.append(f"`{name}` (keyword match)")
    if labels:
        st.caption("📄 **Sources:** " + ", ".join(labels))


def _route_summary(route) -> str:
    """Return a one-line description of what the route actually did."""
    if not isinstance(route, dict) or not route:
        return ""
    selected = route.get("selected_section_ids")
    if not isinstance(selected, (list, tuple)) or not selected:
        return ""
    metrics = route.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    try:
        steps = int(route.get("step_count") or 0)
    except (TypeError, ValueError):
        steps = 0
    tokens = metrics.get("selected_evidence_tokens")
    parts = [f"🗺 {len(selected)} section{'s' if len(selected) != 1 else ''} routed"]
    if steps:
        parts.append(f"{steps} step{'s' if steps != 1 else ''}")
    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and tokens > 0:
        parts.append(f"{int(tokens)} context tokens")
    if route.get("reflection_flag"):
        parts.append("reflected")
    return " · ".join(parts)


def _render_turn_footer(sources, route):
    """Render the compact answer footer: sources, then what retrieval did."""
    _render_sources(sources)
    summary = _route_summary(route)
    if summary:
        st.caption(summary)


def _process_prompt(prompt):
    _clear_turn_state()
    use_r2r = r2r.r2r_is_ready(st.session_state)
    if not use_r2r and is_openai_compatible_backend():
        if not st.session_state.get("openai_model"):
            st.warning(
                "⚠️ No chat model configured. Please go to **Settings → Chat** "
                "and enter a Chat Model for the OpenAI-compatible backend.",
                icon=None,
            )
            return
    elif not use_r2r and not st.session_state.get("selected_model"):
        try:
            models = get_models()
            if models:
                st.session_state["ollama_models"] = models
                st.session_state["selected_model"] = models[0]
                get_embedding_models()
                st.rerun()
        except Exception:
            pass
        if not st.session_state.get("selected_model"):
            st.warning(
                "⚠️ No chat model available. Please go to **Settings → Chat** "
                "and click **Refresh Models**, then select a model.",
                icon=None,
            )
            return

    st.session_state.pop("last_r2r_metadata", None)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    turn_evidence = []
    turn_route = {}
    with st.chat_message("assistant"):
        if use_r2r:
            st.caption("R2R response (complete, non-streaming)")
        with st.spinner("Thinking..."):
            if use_r2r:
                stream = r2r.r2r_chat(prompt=prompt)
            elif _local_index_ready():
                stream = context_chat(
                    prompt=prompt,
                    query_engine=st.session_state["query_engine"],
                    evidence_sink=turn_evidence,
                    route_sink=turn_route,
                )
            else:
                stream = chat(prompt=prompt)

            response = st.write_stream(stream)

        if use_r2r:
            turn_sources = st.session_state.get("last_doc_sources") or []
        else:
            turn_evidence = [
                dict(item)
                for item in (
                    turn_evidence or st.session_state.get("last_rag_evidence") or []
                )
                if isinstance(item, dict)
            ]
            turn_sources = [
                (item.get("source", "document"), item.get("score", 0.0))
                for item in turn_evidence
            ]
        _render_turn_footer(
            turn_sources,
            turn_route or st.session_state.get("last_retrieval_route") or {},
        )
        if not use_r2r:
            render_retrieval_route(
                turn_route or st.session_state.get("last_retrieval_route") or {},
                st.session_state.get("retrieval_map"),
                key="current-route",
            )

    if st.session_state.get("last_rag_no_result") and st.button(
        "💬 Ask without documents", key="ask_without_docs_btn"
    ):
        st.session_state["ask_without_docs"] = True
        st.rerun()

    if response:
        message = {"role": "assistant", "content": response}
        r2r_metadata = st.session_state.get("last_r2r_metadata")
        if use_r2r and r2r_metadata:
            message["r2r"] = copy.deepcopy(r2r_metadata)
        elif not use_r2r:
            message["evidence"] = copy.deepcopy(turn_evidence)
            message["retrieval_route"] = copy.deepcopy(
                turn_route or st.session_state.get("last_retrieval_route") or {}
            )
        st.session_state["messages"].append(message)


def chatbox():
    # Sync tone between Settings (answer_style) and chatbox (quick_answer_style) — single source
    if st.session_state.get("answer_style") is not None and st.session_state.get(
        "quick_answer_style"
    ) != st.session_state.get("answer_style"):
        st.session_state["quick_answer_style"] = st.session_state["answer_style"]
        st.session_state["system_prompt"] = _style_to_prompt(
            st.session_state["answer_style"]
        )
    elif (
        st.session_state.get("quick_answer_style") is not None
        and st.session_state.get("answer_style") is None
    ):
        st.session_state["answer_style"] = st.session_state["quick_answer_style"]

    if st.session_state.get("system_prompt") is None:
        _apply_quick_answer_style()

    if st.session_state.pop("ask_without_docs", False):
        prompt = st.session_state.get("last_rag_question")
        st.session_state["last_doc_sources"] = []
        st.session_state["last_rag_evidence"] = []
        st.session_state["last_retrieval_route"] = {}
        st.session_state["last_rag_no_result"] = False
        st.session_state["last_rag_question"] = None
        if prompt:
            with st.chat_message("assistant"), st.spinner("Thinking..."):
                response = st.write_stream(chat(prompt=prompt))
            if response:
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": response,
                        "evidence": [],
                        "retrieval_route": {},
                    }
                )

    messages = st.session_state.get("messages", [])
    is_empty_chat = (
        len(messages) == 1
        and messages[0].get("role") == "assistant"
        and not st.session_state.get("last_doc_sources")
    )
    if is_empty_chat:
        selected = st.pills(
            "Try asking",
            _suggested_questions(),
            label_visibility="collapsed",
            selection_mode="single",
        )
        if selected:
            _process_prompt(selected)

    if prompt := st.chat_input("Ask about your documents or just chat..."):
        _process_prompt(prompt)

    if is_empty_chat:
        with st.expander("Answer style", expanded=False):
            st.caption(
                "How should answers sound? You can also change this anytime in Settings."
            )
            st.selectbox(
                "Answer tone",
                options=ANSWER_STYLE_OPTIONS,
                key="quick_answer_style",
                label_visibility="collapsed",
                on_change=_apply_quick_answer_style,
            )
