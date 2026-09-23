import hashlib
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from utils.endpoint_policy import normalize_provider_endpoint

PERSISTED_SETTING_TYPES = {
    "ollama_endpoint": str,
    "ollama_embedding_model": str,
    "selected_model": str,
    "top_k": int,
    "candidate_depth": int,
    "vector_candidate_depth": int,
    "bm25_candidate_depth": int,
    "chunk_size": int,
    "chunk_overlap": int,
    "chunk_overlap_pct": int,
    "similarity_cutoff": float,
    "advanced": bool,
    "temperature": float,
    "eco_mode": bool,
    "llm_backend": str,
    "openai_base_url": str,
    "openai_model": str,
    "openai_embedding_model": str,
    "lm_studio_base_url": str,
    "lm_studio_model": str,
    "tabby_base_url": str,
    "tabby_model": str,
    "openai_compatible_base_url": str,
    "openai_compatible_model": str,
    "embedding_backend": str,
    "embedding_base_url": str,
    "embedding_model": str,
    "r2r_enabled": bool,
    "r2r_base_url": str,
    "r2r_workspace_id": str,
}
BROWSER_STORAGE_KEY = "docmind:settings"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
PERSISTED_SETTINGS_HASH_STATE_KEY = "browser_settings_persisted_hash"
SENSITIVE_SETTING_KEYS = frozenset(
    {
        "openai_api_key",
        "lm_studio_api_key",
        "tabby_api_key",
        "openai_compatible_api_key",
        "embedding_api_key",
        "r2r_api_key",
    }
)
PROVIDER_URL_SETTING_KEYS = frozenset(
    {
        "ollama_endpoint",
        "openai_base_url",
        "lm_studio_base_url",
        "tabby_base_url",
        "openai_compatible_base_url",
        "embedding_base_url",
    }
)
_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "browser_storage_component")
_browser_storage_component = components.declare_component(
    "browser_storage", path=_COMPONENT_PATH
)


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError("invalid boolean")


def normalize_ollama_endpoint(value):
    """Return a validated Ollama endpoint, using the safe local default when blank."""
    return normalize_provider_endpoint(
        value,
        default=DEFAULT_OLLAMA_ENDPOINT,
        label="Ollama endpoint",
    )


def ensure_ollama_endpoint(state):
    """Keep the live Ollama endpoint usable even if a widget submits an empty value."""
    state["ollama_endpoint"] = normalize_ollama_endpoint(state.get("ollama_endpoint"))
    return state["ollama_endpoint"]


def _normalize_persisted_url(key, value):
    if value is None or str(value).strip() == "":
        return None
    return normalize_provider_endpoint(
        str(value).strip(),
        label=f"{key} URL",
    )


def apply_persisted_settings(state, raw_settings):
    """Apply valid browser-persisted settings before defaults initialize."""
    for key, expected_type in PERSISTED_SETTING_TYPES.items():
        if key not in raw_settings or key in SENSITIVE_SETTING_KEYS:
            continue
        raw_value = raw_settings[key]
        try:
            if key in PROVIDER_URL_SETTING_KEYS:
                value = _normalize_persisted_url(key, raw_value)
                if value is None:
                    continue
            elif expected_type is bool:
                value = _coerce_bool(raw_value)
            else:
                value = expected_type(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if key in {"candidate_depth", "vector_candidate_depth", "bm25_candidate_depth"}:
            if not 1 <= value <= 50:
                continue
        state[key] = value


def serialize_persisted_settings(state):
    """Return only browser-persistable user settings from session state."""
    serialized = {}
    for key in PERSISTED_SETTING_TYPES:
        if key not in state or key in SENSITIVE_SETTING_KEYS:
            continue
        value = state[key]
        if value is None:
            continue
        try:
            if key in PROVIDER_URL_SETTING_KEYS:
                value = _normalize_persisted_url(key, value)
                if value is None:
                    continue
            if key in {"candidate_depth", "vector_candidate_depth", "bm25_candidate_depth"}:
                if not 1 <= int(value) <= 50:
                    continue
        except (TypeError, ValueError, OverflowError):
            continue
        serialized[key] = value
    return serialized


def deserialize_persisted_settings(payload):
    """Return persisted settings from a browser localStorage JSON payload."""
    if not payload:
        return {}
    try:
        settings = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(settings, dict):
        return {}
    return settings


def browser_storage_payload(settings):
    """Return the JSON string stored in browser localStorage."""
    return json.dumps(settings)


def option_index(options, selected_value):
    """Return the selectbox index for a restored value, or a stable fallback."""
    if not options:
        return None
    try:
        return options.index(selected_value)
    except ValueError:
        return 0


def should_refresh_models_for_endpoint(state, models_key, endpoint=None):
    """Return whether cached Ollama models belong to a different endpoint or are empty."""
    endpoint = normalize_ollama_endpoint(
        endpoint if endpoint is not None else state.get("ollama_endpoint")
    )
    endpoint_key = f"{models_key}_endpoint"
    return (
        models_key not in state
        or state.get(endpoint_key) != endpoint
        or len(state.get(models_key, [])) == 0
    )


def restore_settings_from_browser_storage():
    """Hydrate session state from browser localStorage once per session."""
    if st.session_state.get("browser_settings_restored"):
        return

    payload = _browser_storage_component(
        action="get",
        storage_key=BROWSER_STORAGE_KEY,
        key="restore_browser_settings",
    )
    if payload is None:
        st.stop()

    storage_payload = payload
    if isinstance(payload, dict) and "value" in payload:
        storage_payload = payload["value"]

    raw_settings = deserialize_persisted_settings(storage_payload)
    apply_persisted_settings(st.session_state, raw_settings)
    st.session_state["browser_settings_restored"] = True


def persist_settings_to_browser_storage():
    """Keep supported settings in browser localStorage so app restarts restore them."""
    if not st.session_state.get("browser_settings_restored"):
        return

    serialized = serialize_persisted_settings(st.session_state)
    storage_hash = hashlib.sha256(
        json.dumps(serialized, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    if st.session_state.get(PERSISTED_SETTINGS_HASH_STATE_KEY) == storage_hash:
        return

    st.session_state[PERSISTED_SETTINGS_HASH_STATE_KEY] = storage_hash
    _browser_storage_component(
        action="set",
        storage_key=BROWSER_STORAGE_KEY,
        value=browser_storage_payload(serialized),
        key=f"persist_browser_settings_{storage_hash}",
    )
