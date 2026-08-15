import json
from datetime import datetime

import streamlit as st

import utils.ollama as ollama
from components.page_state import default_chat_model
from utils.browser_settings import ensure_ollama_endpoint


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


def _refresh_models():
    ensure_ollama_endpoint(st.session_state)
    ollama.get_models()
    ollama.get_embedding_models()
    if st.session_state.get("selected_model") not in st.session_state["ollama_models"]:
        st.session_state["selected_model"] = default_chat_model(
            st.session_state["ollama_models"]
        )
    st.session_state["ollama_models_endpoint"] = st.session_state["ollama_endpoint"]
    st.session_state["ollama_embedding_models_endpoint"] = st.session_state["ollama_endpoint"]


def _refresh_embedding_models():
    ensure_ollama_endpoint(st.session_state)
    ollama.get_embedding_models()
    st.session_state["ollama_embedding_models_endpoint"] = st.session_state["ollama_endpoint"]


def settings():
    st.header("Settings")
    st.caption("Configure DocMind AI settings and integrations")

    st.subheader("Chat")
    chat_settings = st.container(border=True)
    with chat_settings:
        st.text_input(
            "Ollama Endpoint",
            key="ollama_endpoint",
            placeholder="http://localhost:11434",
            on_change=_refresh_models,
        )
        st.selectbox(
            "Chat Model",
            st.session_state["ollama_models"],
            key="selected_model",
            disabled= len(st.session_state["ollama_models"])==0,
            placeholder= "Select Chat Model" if len(st.session_state["ollama_models"])>0 else "No Models Available",
        )
        st.button(
            "Refresh Models",
            key="refresh_chat_models",
            on_click=_refresh_models,
        )
        if len(st.session_state["ollama_models"]) == 0:
            st.info(
                "💡 **No chat models found.** Run in terminal:\n`ollama pull qwen2.5:0.5b`"
            )
        if st.session_state["advanced"] == True:
            st.select_slider(
                "Top K",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                help="The number of most similar document chunks to retrieve in response to a query.",
                value=st.session_state["top_k"],
                key="top_k",
            )
            st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=st.session_state.get("similarity_cutoff", 0.3),
                help="Minimum similarity score (0-1) for a chunk to be considered. "
                "Lower = more recall, higher = only very relevant chunks. 0 = disabled.",
                key="similarity_cutoff",
            )
            # st.text_area(
            #     "System Prompt",
            #     value=st.session_state["system_prompt"],
            #     key="system_prompt",
            # )

    st.subheader("Answer Style")
    style_settings = st.container(border=True)
    with style_settings:
        st.caption("Choose how the assistant formats its answers. This adjusts the system prompt.")
        style = st.radio(
            "Style",
            options=[
                "Concise",
                "Balanced (default)",
                "Detailed",
                "Bulleted",
                "Technical",
                "Simple / ELI5",
                "Custom",
            ],
            index=1,
            horizontal=True,
            key="answer_style",
            help="Presets inject a style directive into the system prompt. "
            "Pick 'Custom' to write your own.",
        )
        if style == "Custom":
            st.text_area(
                "Custom System Prompt",
                key="system_prompt",
                height=120,
                help="Write your own system prompt. Overrides the selected style.",
            )
        else:
            # Apply the preset immediately (a keyed preview widget would hold a
            # stale value and silently block style changes).
            st.session_state["system_prompt"] = _style_to_prompt(style)
            st.text_area(
                "System Prompt (preview)",
                value=_style_to_prompt(style),
                height=120,
                disabled=True,
                help="Uneditable preview. Switch to 'Custom' to edit.",
            )

    st.write("")

    st.subheader(
        "Embeddings",
        help="Embeddings are numerical representations of data, useful for tasks like document clustering and similarity detection when processing files, as they encode semantic meaning for efficient manipulation and retrieval.",
    )
    embedding_settings = st.container(border=True)
    with embedding_settings:
        st.selectbox(
            "Embedding Model",
            st.session_state["ollama_embedding_models"],
            key="ollama_embedding_model",
            disabled=len(st.session_state["ollama_embedding_models"]) == 0,
            placeholder=(
                "Select Model"
                if len(st.session_state["ollama_embedding_models"]) > 0
                else "No Embedding Models Available"
            ),
        )
        st.button(
            "Refresh Models",
            key="refresh_embedding_models",
            on_click=_refresh_embedding_models,
        )
        if len(st.session_state["ollama_embedding_models"]) == 0:
            st.caption("Need one? Pull an Ollama embedding model first, e.g. `ollama pull embeddinggemma`.")
        if st.session_state["advanced"] == True:
            st.caption(
                "View the [MTEB Embeddings Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)"
            )
            st.text_input(
                "Chunk Size (tokens)",
                help="Reducing `chunk_size` improves embedding precision by focusing on smaller text portions. "
                "This enhances information retrieval accuracy but escalates computational demands due to "
                "processing more chunks. In tokens (~4 characters each); 256 is a good balance.",
                key="chunk_size",
                placeholder="256",
                value=st.session_state["chunk_size"],
            )
            st.text_input(
                "Chunk Overlap (tokens)",
                help="The amount of overlap between two consecutive chunks. A higher overlap value helps "
                "maintain continuity and context across chunks.",
                key="chunk_overlap",
                placeholder="32",
                value=st.session_state["chunk_overlap"],
            )

    st.subheader("Export Data")
    export_data_settings = st.container(border=True)
    with export_data_settings:
        st.write("Chat History")
        st.download_button(
            label="Download",
            data=json.dumps(st.session_state["messages"]),
            file_name=f"docmind-chat-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json",
            mime="application/json",
        )

    st.toggle("Advanced Settings", key="advanced")

    if st.session_state["advanced"] == True:
        with st.expander("Current Application State"):
            state = dict(sorted(st.session_state.items()))
            st.write(state)
