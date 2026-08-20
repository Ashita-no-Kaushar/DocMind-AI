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


def _apply_quick_answer_style():
    """Rewrite the system prompt when the quick tone selector changes."""
    style = st.session_state.get("quick_answer_style", "Balanced (default)")
    st.session_state["system_prompt"] = _style_to_prompt(style)


def chatbox():
    quick_style = st.selectbox(
        "Answer tone",
        options=ANSWER_STYLE_OPTIONS,
        key="quick_answer_style",
        label_visibility="collapsed",
        help="Choose how the assistant formats its answers for this conversation. "
        "Overrides the Answer Style preset in Settings.",
        on_change=_apply_quick_answer_style,
    )
    # Keep the system prompt in sync with the visible quick selector. The
    # Settings page writes the same derived prompt via its own radio widget.
    if st.session_state.get("system_prompt") is None:
        _apply_quick_answer_style()

    # The last RAG question found no matches and the user clicked
    # "Ask without documents": answer the same question from general
    # knowledge so a failed retrieval never leaves the user hanging.
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

    if prompt := st.chat_input("How can I help?"):
        if st.session_state.get("llm_backend", "Ollama") == "OpenAI":
            if not st.session_state.get("openai_model"):
                st.warning(
                    "⚠️ No chat model configured. Please go to **Settings → Chat** "
                    "and enter a Chat Model for the OpenAI-compatible backend.",
                    icon=None,
                )
                return
        elif not st.session_state.get("selected_model"):
            # Try to auto-discover models before giving up
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

        # Add the user input to messages state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["last_rag_no_result"] = False
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate stream with user input (Context RAG vs General Chat)
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

            # Show the document sources the answer was grounded in.
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

        # If RAG found nothing in the documents, offer an ungrounded answer.
        if st.session_state.get("last_rag_no_result"):
            if st.button("💬 Ask without documents", key="ask_without_docs_btn"):
                st.session_state["ask_without_docs"] = True
                st.rerun()

        # Add the final response to messages state
        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})

