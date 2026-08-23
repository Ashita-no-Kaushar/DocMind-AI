import streamlit as st

import utils.r2r as r2r
from utils.ollama import chat, context_chat, get_models, get_embedding_models

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
    return base + style_instructions.get(style, style_instructions["Balanced (default)"])


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


def _suggested_questions():
    """Starter questions shown while the conversation is still empty."""
    if st.session_state.get("query_engine") or r2r.r2r_is_ready(st.session_state):
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


def _process_prompt(prompt):
    """Handle one user turn end-to-end: render, stream the answer, and store it."""
    if st.session_state.get("llm_backend", "Ollama") == "OpenAI":
        if not st.session_state.get("openai_model"):
            st.warning(
                "⚠️ No chat model configured. Please go to **Settings → Chat** "
                "and enter a Chat Model for the OpenAI-compatible backend.",
                icon=None,
            )
            return
    elif not st.session_state.get("selected_model"):
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

    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["last_rag_no_result"] = False
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if r2r.r2r_is_ready(st.session_state):
                st.session_state["last_doc_sources"] = []
                stream = r2r.r2r_chat(prompt=prompt)
            elif st.session_state.get("query_engine"):
                stream = context_chat(
                    prompt=prompt,
                    query_engine=st.session_state["query_engine"],
                )
            else:
                st.session_state["last_doc_sources"] = []
                stream = chat(prompt=prompt)

            response = st.write_stream(stream)

        sources = st.session_state.get("last_doc_sources") or []
        if sources:
            best = {}
            for name, score in sources:
                if name not in best or score > best[name]:
                    best[name] = score
            labels = []
            for name, score in best.items():
                if score >= 0.15:
                    labels.append(f"`{name}` ({score:.0%})")
                else:
                    labels.append(f"`{name}` (keyword match)")
            st.caption("📄 **Sources:** " + ", ".join(labels))

    if st.session_state.get("last_rag_no_result"):
        if st.button("💬 Ask without documents", key="ask_without_docs_btn"):
            st.session_state["ask_without_docs"] = True
            st.rerun()

    if response:
        st.session_state["messages"].append({"role": "assistant", "content": response})


def chatbox():
    # Sync tone between Settings (answer_style) and chatbox (quick_answer_style) — single source
    if st.session_state.get("answer_style") is not None and st.session_state.get("quick_answer_style") != st.session_state.get("answer_style"):
        st.session_state["quick_answer_style"] = st.session_state["answer_style"]
        st.session_state["system_prompt"] = _style_to_prompt(st.session_state["answer_style"])
    elif st.session_state.get("quick_answer_style") is not None and st.session_state.get("answer_style") is None:
        st.session_state["answer_style"] = st.session_state["quick_answer_style"]

    if st.session_state.get("system_prompt") is None:
        _apply_quick_answer_style()

    if st.session_state.pop("ask_without_docs", False):
        prompt = st.session_state.get("last_rag_question")
        st.session_state["last_rag_no_result"] = False
        st.session_state["last_rag_question"] = None
        if prompt:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.write_stream(chat(prompt=prompt))
            if response:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
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
            st.caption("How should answers sound? You can also change this anytime in Settings.")
            st.selectbox(
                "Answer tone",
                options=ANSWER_STYLE_OPTIONS,
                key="quick_answer_style",
                label_visibility="collapsed",
                on_change=_apply_quick_answer_style,
            )
