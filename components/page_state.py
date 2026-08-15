import os
import shutil
import time

import streamlit as st

import utils.logs as logs

from utils.llama_index import INDEX_CACHE_DIR
from utils.ollama import default_embedding_model, get_models, get_embedding_models
from utils.browser_settings import (
    PERSISTED_SETTINGS_HASH_STATE_KEY,
    ensure_ollama_endpoint,
    restore_settings_from_browser_storage,
    should_refresh_models_for_endpoint,
)

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Welcome to **DocMind AI**! 👋\n\nYou can:\n- 💬 **Chat directly** with the LLM — just type a question below\n- 📂 **Import documents** (files, GitHub repo, or website) from the sidebar for grounded RAG answers\n\nHow can I help you today?",
}


def _remove_dir_retry(path, attempts=5, delay=0.5):
    """Delete a directory tree, retrying transient Windows file locks."""
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(delay)


def perform_project_reset(state):
    """Wipe all stored data and restore the app to a fresh-project state.

    Must run BEFORE any widget with a session key is instantiated in the
    current script run, otherwise Streamlit rejects modifying widget keys.
    """
    _remove_dir_retry(INDEX_CACHE_DIR)
    _remove_dir_retry(os.path.join(os.getcwd(), "data"))

    state["query_engine"] = None
    state["retriever"] = None
    state["llm"] = None
    state["documents"] = None
    state["file_list"] = []
    state["processed_file_signature"] = None
    state["processing_file_signature"] = None
    state["processed_github_repo"] = None
    state["processed_website_urls"] = None
    state["github_ingestion_stages"] = []
    state["website_ingestion_stages"] = []
    state["file_ingestion_stages"] = []
    state["messages"] = [dict(WELCOME_MESSAGE)]

    # Widget-backed keys: safe here only because no widgets exist yet.
    state["top_k"] = 3
    state["chat_mode"] = "compact"
    state["chunk_size"] = 256
    state["chunk_overlap"] = 32
    state["advanced"] = False

    # Force the next persist round to write clean defaults to localStorage
    # instead of the stale pre-reset values.
    state[PERSISTED_SETTINGS_HASH_STATE_KEY] = None

    logs.log.info("Project reset: caches and data cleared")


def default_chat_model(models):
    """Return the preferred default chat model from discovered Ollama models."""
    preferred_models = (
        "gemma4:latest",
        "llama3:8b",
        "llama2:7b",
    )

    for model in preferred_models:
        if model in models:
            return model

    if models:
        return models[0]

    return None


def ensure_valid_model_selections(state):
    """Keep selected model values consistent with discovered model lists."""
    chat_models = state.get("ollama_models", [])
    if chat_models:
        if state.get("selected_model") not in chat_models:
            state["selected_model"] = default_chat_model(chat_models)
    elif "selected_model" in state:
        state["selected_model"] = None

    if state.get("embedding_backend") == "Ollama":
        embedding_models = state.get("ollama_embedding_models", [])
        if embedding_models:
            if state.get("ollama_embedding_model") not in embedding_models:
                state["ollama_embedding_model"] = default_embedding_model(embedding_models)
        elif "ollama_embedding_model" in state:
            state["ollama_embedding_model"] = None


def set_initial_state():
    restore_settings_from_browser_storage()

    # A pending project reset (requested from the sidebar button) must be
    # executed here: widgets are not instantiated yet, so resetting
    # widget-backed keys is still legal.
    if st.session_state.get("reset_requested"):
        perform_project_reset(st.session_state)
        st.session_state["reset_requested"] = False

    ###########
    # General #
    ###########

    if "sidebar_state" not in st.session_state:
        st.session_state["sidebar_state"] = "expanded"

    ensure_ollama_endpoint(st.session_state)

    if "embedding_backend" not in st.session_state:
        st.session_state["embedding_backend"] = "Ollama"

    if "ollama_embedding_model" not in st.session_state:
        st.session_state["ollama_embedding_model"] = "nomic-embed-text:latest"

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = "Default (gte-modernbert-base)"

    if should_refresh_models_for_endpoint(st.session_state, "ollama_models"):
        try:
            models = get_models()
            st.session_state["ollama_models"] = models
        except Exception:
            st.session_state["ollama_models"] = []
            pass
        st.session_state["ollama_models_endpoint"] = st.session_state["ollama_endpoint"]

    if should_refresh_models_for_endpoint(st.session_state, "ollama_embedding_models"):
        try:
            models = get_embedding_models()
            st.session_state["ollama_embedding_models"] = models
        except Exception:
            st.session_state["ollama_embedding_models"] = []
            pass
        st.session_state["ollama_embedding_models_endpoint"] = st.session_state["ollama_endpoint"]

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = default_chat_model(
            st.session_state.get("ollama_models", [])
        )

    ensure_valid_model_selections(st.session_state)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [dict(WELCOME_MESSAGE)]

    ################################
    #  Files, Documents & Websites #
    ################################

    if "file_list" not in st.session_state:
        st.session_state["file_list"] = []

    if "processed_file_signature" not in st.session_state:
        st.session_state["processed_file_signature"] = None

    if "processing_file_signature" not in st.session_state:
        st.session_state["processing_file_signature"] = None

    if "file_ingestion_stages" not in st.session_state:
        st.session_state["file_ingestion_stages"] = []

    if "github_ingestion_stages" not in st.session_state:
        st.session_state["github_ingestion_stages"] = []

    if "website_ingestion_stages" not in st.session_state:
        st.session_state["website_ingestion_stages"] = []

    if "github_repo" not in st.session_state:
        st.session_state["github_repo"] = ""
    elif st.session_state["github_repo"] is None:
        st.session_state["github_repo"] = ""

    if "processed_github_repo" not in st.session_state:
        st.session_state["processed_github_repo"] = None

    if "websites" not in st.session_state:
        st.session_state["websites"] = []

    if "new_website" not in st.session_state:
        st.session_state["new_website"] = ""

    if "website_input_error" not in st.session_state:
        st.session_state["website_input_error"] = None

    ###############
    # Llama-Index #
    ###############

    if "llm" not in st.session_state:
        st.session_state["llm"] = None

    if "documents" not in st.session_state:
        st.session_state["documents"] = None

    if "query_engine" not in st.session_state:
        st.session_state["query_engine"] = None

    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    if "chat_mode" not in st.session_state:
        st.session_state["chat_mode"] = "compact"

    #####################
    # Advanced Settings #
    #####################

    if "advanced" not in st.session_state:
        st.session_state["advanced"] = False

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = (
            "You are DocMind AI, a helpful and accurate virtual assistant. "
            "When document context is provided, answer strictly from that context "
            "and do not invent information. If you are unsure, say so directly. "
            "Otherwise answer from your general knowledge. "
            "Be concise, factual, and conversational."
        )

    if "top_k" not in st.session_state:
        st.session_state["top_k"] = 3

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = None

    if "other_embedding_model" not in st.session_state:
        st.session_state["other_embedding_model"] = None

    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = 256

    if "similarity_cutoff" not in st.session_state:
        st.session_state["similarity_cutoff"] = 0.3
