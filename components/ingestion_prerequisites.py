import streamlit as st

from utils.provider_config import (
    OLLAMA,
    OPENAI_OFFICIAL,
    get_chat_profile,
    get_embedding_profile,
)


def missing_ingestion_settings(allow_r2r: bool = False):
    """Return missing model settings for local or R2R ingestion."""
    missing = []

    if allow_r2r and st.session_state.get("r2r_enabled"):
        return missing

    backend = st.session_state.get("llm_backend", "Ollama")
    try:
        chat = get_chat_profile(st.session_state, backend)
    except ValueError as err:
        return [str(err)]
    if not chat.get("model"):
        missing.append("a valid chat model")
    if chat["provider_kind"] == OPENAI_OFFICIAL and not chat.get("api_key"):
        missing.append("an explicit OpenAI API key")
    if chat["provider_kind"] == OLLAMA:
        chat_models = st.session_state.get("ollama_models", [])
        if chat.get("model") not in chat_models:
            missing.append("a valid Ollama chat model")

    try:
        embedding = get_embedding_profile(st.session_state)
    except ValueError as err:
        missing.append(str(err))
        return missing
    if not embedding.get("model"):
        missing.append("a valid embedding model")
    if embedding["provider_kind"] == OPENAI_OFFICIAL and not embedding.get("api_key"):
        missing.append("an explicit OpenAI embedding API key")
    if embedding["provider_kind"] == OLLAMA:
        embedding_models = st.session_state.get("ollama_embedding_models", [])
        if embedding.get("model") not in embedding_models:
            missing.append("a valid Ollama embedding model")

    return missing


def ingestion_is_configured(allow_r2r: bool = False):
    """Return whether the required model settings are available."""
    return len(missing_ingestion_settings(allow_r2r=allow_r2r)) == 0


def render_ingestion_settings_warning():
    """Explain why ingestion is unavailable and where to fix it."""
    missing = missing_ingestion_settings()
    if not missing:
        return

    missing_str = " and ".join(missing)
    st.warning(
        f"Local document ingestion is unavailable. Missing: {missing_str}.\n\n"
        "Open **Settings**, configure the active provider, and refresh model lists when needed. "
        "R2R local uploads can still be used when R2R is enabled.",
        icon=None,
    )
