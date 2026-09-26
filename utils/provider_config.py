import hashlib
from collections.abc import Mapping

from utils.endpoint_policy import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OPENAI_BASE_URL,
    endpoint_host,
    normalize_provider_endpoint,
)

OPENAI_OFFICIAL = "openai_official"
OPENAI_COMPATIBLE = "openai_compatible"
OLLAMA = "ollama"
PROVIDER_KINDS = frozenset({OPENAI_OFFICIAL, OPENAI_COMPATIBLE, OLLAMA})

BACKEND_LABELS = {
    OLLAMA: "Ollama",
    OPENAI_OFFICIAL: "OpenAI",
    OPENAI_COMPATIBLE: "OpenAI-compatible",
    "lm_studio": "LM Studio (Local AI)",
    "tabbyapi": "TabbyAPI",
}

PROFILE_DEFAULTS = {
    "openai": {
        "base_url": DEFAULT_OPENAI_BASE_URL,
        "model": "gpt-4o-mini",
    },
    "lm_studio": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
    },
    "tabbyapi": {
        "base_url": "http://localhost:5000/v1",
        "model": "local-model",
    },
    "openai_compatible": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
    },
}

PROFILE_KEYS = {
    "ollama": {
        "base_url": "ollama_endpoint",
        "api_key": "",
        "model": "selected_model",
        "models": "ollama_models",
    },
    "openai": {
        "base_url": "openai_base_url",
        "api_key": "openai_api_key",
        "model": "openai_model",
        "models": "openai_models",
    },
    "lm_studio": {
        "base_url": "lm_studio_base_url",
        "api_key": "lm_studio_api_key",
        "model": "lm_studio_model",
        "models": "lm_studio_models",
    },
    "tabbyapi": {
        "base_url": "tabby_base_url",
        "api_key": "tabby_api_key",
        "model": "tabby_model",
        "models": "tabby_models",
    },
    "openai_compatible": {
        "base_url": "openai_compatible_base_url",
        "api_key": "openai_compatible_api_key",
        "model": "openai_compatible_model",
        "models": "openai_compatible_models",
    },
}


def _text(value) -> str:
    return str(value or "").strip()


def normalize_provider_kind(value=None) -> str:
    text = _text(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"ollama", "ollama_api"}:
        return OLLAMA
    if text in {OPENAI_OFFICIAL, "openai", "openai_api", "official_openai"}:
        return OPENAI_OFFICIAL
    if text in {
        OPENAI_COMPATIBLE,
        "openai_like",
        "openai_compatible_api",
        "compatible",
        "custom",
        "custom_openai",
    }:
        return OPENAI_COMPATIBLE
    if "lmstudio" in text or "lm_studio" in text:
        return OPENAI_COMPATIBLE
    if "tabby" in text:
        return OPENAI_COMPATIBLE
    if "openai" in text:
        return OPENAI_COMPATIBLE
    return OLLAMA


normalize_backend = normalize_provider_kind
backend_provider_kind = normalize_provider_kind


def provider_slug(value=None) -> str:
    text = _text(value).lower().replace("-", "_").replace(" ", "_")
    kind = normalize_provider_kind(value)
    if kind == OPENAI_OFFICIAL:
        return "openai"
    if kind == OLLAMA:
        return "ollama"
    if "lmstudio" in text or "lm_studio" in text:
        return "lm_studio"
    if "tabby" in text:
        return "tabbyapi"
    return "openai_compatible"


def provider_label(value=None) -> str:
    slug = provider_slug(value)
    return BACKEND_LABELS.get(slug, "OpenAI-compatible")


def profile_keys(value=None) -> dict[str, str]:
    return dict(PROFILE_KEYS[provider_slug(value)])


def _is_compatible_label(value) -> bool:
    return normalize_provider_kind(value) == OPENAI_COMPATIBLE


def _legacy_chat_values(state, backend) -> tuple[str | None, str | None, str | None]:
    if not _is_compatible_label(backend):
        return None, None, None
    return (
        state.get("openai_base_url"),
        state.get("openai_model"),
        None,
    )


def initialize_provider_state(state) -> dict:
    backend = state.get("llm_backend", OLLAMA)
    if not state.get("llm_backend"):
        state["llm_backend"] = "Ollama"
    legacy_base, legacy_model, _ = _legacy_chat_values(state, backend)
    legacy_base_allowed = False
    if _is_compatible_label(backend):
        slug = provider_slug(backend)
        keys = PROFILE_KEYS[slug]
        if legacy_base and keys["base_url"] not in state:
            try:
                legacy_endpoint = normalize_provider_endpoint(legacy_base)
            except ValueError:
                legacy_endpoint = ""
            if legacy_endpoint and endpoint_host(legacy_endpoint) != "api.openai.com":
                state[keys["base_url"]] = legacy_endpoint
                legacy_base_allowed = True
        if (
            legacy_model
            and keys["model"] not in state
            and (not legacy_base or legacy_base_allowed)
        ):
            state[keys["model"]] = legacy_model
    for slug, defaults in PROFILE_DEFAULTS.items():
        keys = PROFILE_KEYS[slug]
        state.setdefault(keys["base_url"], defaults["base_url"])
        state.setdefault(keys["api_key"], "")
        state.setdefault(keys["model"], defaults["model"])
        state.setdefault(keys["models"], [])

    if not state.get("embedding_profile_migrated"):
        chat = get_chat_profile(state, backend)
        if "embedding_backend" not in state:
            state["embedding_backend"] = backend
        if "embedding_base_url" not in state:
            state["embedding_base_url"] = chat["base_url"]
        if "embedding_model" not in state:
            legacy_embedding = state.get("openai_embedding_model")
            if legacy_embedding is None and normalize_provider_kind(backend) == OLLAMA:
                legacy_embedding = state.get("ollama_embedding_model")
            state["embedding_model"] = legacy_embedding or chat["model"]
        if "embedding_api_key" not in state:
            state["embedding_api_key"] = chat["api_key"]
        state["embedding_profile_migrated"] = True
    if normalize_provider_kind(backend) != OLLAMA:
        keys = profile_keys(backend)
        state["openai_model"] = state[keys["model"]]
        if state.get(keys["api_key"]) and f"{keys['api_key']}_endpoint" not in state:
            state[f"{keys['api_key']}_endpoint"] = get_chat_profile(state, backend)[
                "base_url"
            ]
    if state.get("embedding_api_key"):
        key_provider = state.get("embedding_api_key_provider")
        current_embedding_kind = normalize_provider_kind(
            state.get("embedding_backend") or state.get("llm_backend", "Ollama")
        )
        if key_provider and key_provider != current_embedding_kind:
            state["embedding_api_key"] = ""
            state["embedding_api_key_endpoint"] = None
            state["embedding_api_key_provider"] = None
        else:
            embedding = get_embedding_profile(state)
            state.setdefault("embedding_api_key_provider", embedding["provider_kind"])
            if "embedding_api_key_endpoint" not in state:
                state["embedding_api_key_endpoint"] = embedding["base_url"]
    if normalize_provider_kind(backend) != OLLAMA:
        profile = get_chat_profile(state, backend)
        keys = profile["keys"]
        models_key = keys["models"]
        catalog_known = any(
            f"{models_key}_{suffix}" in state
            for suffix in ("endpoint", "key_fingerprint", "provider")
        )
        if catalog_known and (
            state.get(f"{models_key}_endpoint") != profile["base_url"]
            or state.get(f"{models_key}_key_fingerprint")
            != credential_fingerprint(profile["api_key"])
            or state.get(f"{models_key}_provider") != profile["provider_kind"]
        ):
            state[models_key] = []
            state[f"{models_key}_endpoint"] = None
            state[f"{models_key}_key_fingerprint"] = None
            state[f"{models_key}_provider"] = None
    embedding = get_embedding_profile(state)
    if (
        state.get("embedding_models")
        and any(
            f"embedding_models_{suffix}" in state
            for suffix in ("endpoint", "key_fingerprint", "provider")
        )
        and (
            state.get("embedding_models_endpoint") != embedding["base_url"]
            or state.get("embedding_models_key_fingerprint")
            != credential_fingerprint(embedding["api_key"])
            or state.get("embedding_models_provider") != embedding["provider_kind"]
        )
    ):
        state["embedding_models"] = []
        state["embedding_models_endpoint"] = None
        state["embedding_models_key_fingerprint"] = None
        state["embedding_models_provider"] = None
    return state


def get_chat_profile(state, backend=None) -> dict:
    backend = backend or state.get("llm_backend", "Ollama")
    slug = provider_slug(backend)
    keys = profile_keys(backend)
    if slug == OLLAMA:
        endpoint = normalize_provider_endpoint(
            state.get("ollama_endpoint"),
            default=DEFAULT_OLLAMA_ENDPOINT,
            label="Ollama endpoint",
        )
        return {
            "backend": backend,
            "provider_kind": OLLAMA,
            "profile": slug,
            "base_url": endpoint,
            "api_key": "",
            "model": state.get("selected_model"),
            "models": list(state.get("ollama_models", [])),
            "keys": keys,
        }
    defaults = PROFILE_DEFAULTS[slug]
    base_url = state.get(keys["base_url"]) or defaults["base_url"]
    model = state.get(keys["model"]) or defaults["model"]
    if slug != "openai" and keys["base_url"] not in state:
        legacy_base = state.get("openai_base_url")
        try:
            legacy_endpoint = normalize_provider_endpoint(legacy_base)
        except ValueError:
            legacy_endpoint = ""
        if legacy_endpoint and endpoint_host(legacy_endpoint) != "api.openai.com":
            base_url = legacy_endpoint
    if slug != "openai" and keys["model"] not in state:
        model = state.get("openai_model") or model
    endpoint = normalize_provider_endpoint(
        base_url,
        default=defaults["base_url"],
        label=f"{provider_label(backend)} endpoint",
    )
    api_key = _text(state.get(keys["api_key"]))
    binding = state.get(f"{keys['api_key']}_endpoint")
    if api_key and binding and binding != endpoint:
        raise ValueError(
            "The provider API key is bound to a different endpoint; clear and re-enter it."
        )
    return {
        "backend": backend,
        "provider_kind": normalize_provider_kind(backend),
        "profile": slug,
        "base_url": endpoint,
        "api_key": api_key,
        "model": _text(model),
        "models": list(state.get(keys["models"], [])),
        "keys": keys,
    }


def get_embedding_profile(state) -> dict:
    backend = state.get("embedding_backend") or state.get("llm_backend", "Ollama")
    kind = normalize_provider_kind(backend)
    chat = get_chat_profile(state, state.get("llm_backend", "Ollama"))
    if kind == OLLAMA:
        base_url = state.get("embedding_base_url") or state.get("ollama_endpoint")
        model = state.get("embedding_model") or state.get("ollama_embedding_model")
        if base_url is None or str(base_url).strip() == "":
            base_url = DEFAULT_OLLAMA_ENDPOINT
        return {
            "backend": backend,
            "provider_kind": OLLAMA,
            "profile": "ollama",
            "base_url": normalize_provider_endpoint(
                base_url,
                default=DEFAULT_OLLAMA_ENDPOINT,
                label="Ollama embedding endpoint",
            ),
            "api_key": "",
            "model": _text(model),
            "models": list(state.get("ollama_embedding_models", [])),
            "keys": profile_keys(backend),
        }
    base_url = state.get("embedding_base_url")
    if base_url is None or str(base_url).strip() == "":
        base_url = (
            chat["base_url"]
            if chat["provider_kind"] == kind
            else PROFILE_DEFAULTS[provider_slug(backend)]["base_url"]
        )
    model = state.get("embedding_model")
    if model is None or str(model).strip() == "":
        model = chat["model"] if chat["provider_kind"] == kind else ""
    endpoint = normalize_provider_endpoint(
        base_url,
        default=PROFILE_DEFAULTS[provider_slug(backend)]["base_url"],
        label=f"{provider_label(backend)} embedding endpoint",
    )
    api_key = _text(state.get("embedding_api_key"))
    key_provider = state.get("embedding_api_key_provider")
    if api_key and key_provider and key_provider != kind:
        raise ValueError(
            "The embedding API key belongs to a different provider; clear and re-enter it."
        )
    binding = state.get("embedding_api_key_endpoint")
    if api_key and binding and binding != endpoint:
        raise ValueError(
            "The embedding API key is bound to a different endpoint; clear and re-enter it."
        )
    return {
        "backend": backend,
        "provider_kind": kind,
        "profile": provider_slug(backend),
        "base_url": endpoint,
        "api_key": api_key,
        "model": _text(model),
        "models": [],
        "keys": profile_keys(backend),
    }


def credential_fingerprint(value) -> str | None:
    if value in (None, ""):
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def profile_cache_signature(profile: Mapping) -> str:
    return "|".join(
        (
            str(profile.get("provider_kind", "")),
            str(profile.get("base_url", "")),
            str(credential_fingerprint(profile.get("api_key"))),
        )
    )


def model_catalog_is_current(state, profile) -> bool:
    keys = profile.get("keys") or {}
    models_key = keys.get("models")
    if not models_key:
        return False
    return (
        state.get(f"{models_key}_endpoint") == profile.get("base_url")
        and state.get(f"{models_key}_key_fingerprint")
        == credential_fingerprint(profile.get("api_key"))
        and state.get(f"{models_key}_provider") == profile.get("provider_kind")
        and bool(state.get(models_key))
    )


def clear_model_catalog(state, profile) -> None:
    keys = profile.get("keys") or {}
    models_key = keys.get("models")
    if models_key:
        state[models_key] = []
        state[f"{models_key}_endpoint"] = None
        state[f"{models_key}_key_fingerprint"] = None
    state["openai_models_endpoint"] = None
    state["openai_models_key_fingerprint"] = None
    state["openai_models_provider"] = None


__all__ = [
    "BACKEND_LABELS",
    "OLLAMA",
    "OPENAI_COMPATIBLE",
    "OPENAI_OFFICIAL",
    "PROFILE_DEFAULTS",
    "PROFILE_KEYS",
    "PROVIDER_KINDS",
    "backend_provider_kind",
    "clear_model_catalog",
    "credential_fingerprint",
    "get_chat_profile",
    "get_embedding_profile",
    "initialize_provider_state",
    "model_catalog_is_current",
    "normalize_backend",
    "normalize_provider_kind",
    "profile_cache_signature",
    "profile_keys",
    "provider_label",
    "provider_slug",
]
