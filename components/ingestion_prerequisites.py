import streamlit as st


def missing_ingestion_settings():
    """Return human-readable missing settings that block ingestion."""
    missing = []

    # R2R hosts its own LLM and embedding models: local settings are not
    # required while the R2R backend is enabled.
    if st.session_state.get("r2r_enabled"):
        return missing

    backend = st.session_state.get("llm_backend", "Ollama")

    if backend == "OpenAI":
        if not st.session_state.get("openai_base_url"):
            missing.append("an OpenAI-compatible Base URL")
        if not st.session_state.get("openai_model"):
            missing.append("an OpenAI-compatible chat model")
        if not st.session_state.get("openai_embedding_model"):
            missing.append("an OpenAI-compatible embedding model")
        return missing

    chat_model = st.session_state.get("selected_model")
    chat_models = st.session_state.get("ollama_models", [])
    if not chat_model or chat_model not in chat_models:
        missing.append("a valid Ollama chat model")

    embedding_model = st.session_state.get("ollama_embedding_model")
    embedding_models = st.session_state.get("ollama_embedding_models", [])
    if not embedding_model or embedding_model not in embedding_models:
        missing.append("a valid Ollama embedding model")

    return missing


def ingestion_is_configured():
    """Return whether the app has the model settings required for ingestion."""
    return len(missing_ingestion_settings()) == 0


def render_ingestion_settings_warning():
    """Explain why ingestion is unavailable and where to fix it."""
    missing = missing_ingestion_settings()
    if not missing:
        return

    missing_str = " and ".join(missing)
    st.warning(
        f"⚠️ **Ingestion unavailable.** Missing: {missing_str}.\n\n"
        "Go to **Settings → Chat** and click **Refresh Models** to load your Ollama models, "
        "then select a Chat Model and an Embedding Model.",
        icon=None,
    )

