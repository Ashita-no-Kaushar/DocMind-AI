import streamlit as st


def missing_ingestion_settings():
    """Return human-readable missing settings that block ingestion."""
    missing = []

    chat_model = st.session_state.get("selected_model")
    chat_models = st.session_state.get("ollama_models", [])
    if not chat_model or chat_model not in chat_models:
        missing.append("a valid Ollama chat model")

    if st.session_state.get("embedding_backend") == "Ollama":
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

