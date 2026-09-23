import re
import time
from collections.abc import Mapping

import ollama
import requests
import streamlit as st
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama

import utils.logs as logs
from utils.endpoint_policy import (
    DEFAULT_OPENAI_BASE_URL,
    is_loopback_endpoint,
    normalize_ollama_endpoint,
    normalize_provider_endpoint,
    validate_credential_endpoint,
    validate_local_endpoint,
    validate_provider_endpoint,
)
from utils.llama_index import (
    EVIDENCE_EXCERPT_MAX_CHARS,
    TEXT_QA_TEMPLATE,
    build_retrieval_query,
    normalize_evidence,
    retrieval_query_candidates,
)
from utils.provider_config import (
    OLLAMA,
    OPENAI_COMPATIBLE,
    OPENAI_OFFICIAL,
    credential_fingerprint,
    get_chat_profile,
    get_embedding_profile,
    initialize_provider_state,
    normalize_provider_kind,
)

CHAT_HISTORY_TOKEN_BUDGET = 1200
DEFAULT_TEMPERATURE = 0.4
LOCAL_COMPATIBLE_API_KEY = "local"
OPENAI_COMPATIBLE_BACKENDS = {
    "LM Studio (Local AI)",
    "TabbyAPI",
    "OpenAI-compatible",
}
OPENAI_OFFICIAL_BACKENDS = {"OpenAI"}
OLLAMA_REQUEST_TIMEOUT = 300
MODEL_DISCOVERY_TTL_SECONDS = 30
RAG_CONTEXT_WINDOW = 2048
RAG_TOTAL_CONTEXT_TOKENS = RAG_CONTEXT_WINDOW
RAG_OUTPUT_RESERVE_TOKENS = 512
RAG_SAFETY_RESERVE_TOKENS = 64
RAG_MESSAGE_OVERHEAD_TOKENS = 4
validate_endpoint = validate_provider_endpoint
validate_local = validate_local_endpoint
is_loopback = is_loopback_endpoint


def provider_kind(backend: str | None = None) -> str:
    value = backend
    if value is None:
        value = st.session_state.get("llm_backend", "Ollama")
    return normalize_provider_kind(value)


def normalize_backend(backend: str | None = None) -> str:
    return provider_kind(backend)


def is_openai_compatible_backend(backend: str | None = None) -> bool:
    if backend is None:
        return provider_kind() != OLLAMA
    return provider_kind(backend) == OPENAI_COMPATIBLE


def is_openai_like_backend(backend: str | None = None) -> bool:
    return provider_kind(backend) == OPENAI_COMPATIBLE


def is_openai_official_backend(backend: str | None = None) -> bool:
    return provider_kind(backend) == OPENAI_OFFICIAL


def _active_backend() -> str:
    return st.session_state.get("llm_backend", "Ollama")


def _active_chat_profile() -> dict:
    initialize_provider_state(st.session_state)
    return get_chat_profile(st.session_state, _active_backend())


def _active_embedding_profile() -> dict:
    initialize_provider_state(st.session_state)
    return get_embedding_profile(st.session_state)


def _active_chat_model() -> str:
    return _active_chat_profile().get("model")


def _active_embedding_model() -> str:
    return _active_embedding_profile().get("model")


def _active_base_url() -> str:
    return _active_chat_profile().get("base_url")


def _active_embedding_base_url() -> str:
    return _active_embedding_profile().get("base_url")


def _active_api_key() -> str:
    return _active_chat_profile().get("api_key", "")


def _fallback_tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", str(text or ""), flags=re.UNICODE)


def _resolve_tokenizer(llm=None):
    candidates = [
        getattr(llm, "_tokenizer", None),
        getattr(llm, "tokenizer", None),
    ]
    candidates.append(getattr(Settings, "tokenizer", None))
    for candidate in candidates:
        if hasattr(candidate, "encode"):
            return candidate.encode
        if callable(candidate):
            return candidate
    try:
        from llama_index.core import get_tokenizer

        return get_tokenizer()
    except Exception:
        return _fallback_tokenize


def _token_count(text: str, tokenizer=None) -> int:
    value = str(text or "")
    if not value:
        return 0
    try:
        counter = tokenizer or _resolve_tokenizer()
        return len(counter(value))
    except Exception:
        return len(_fallback_tokenize(value))


def _estimate_tokens(text: str) -> int:
    return max(1, _token_count(text))


def _message_text(message) -> str:
    if isinstance(message, Mapping):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _message_role(message) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role", "")).lower()
    role = getattr(message, "role", "")
    return str(getattr(role, "value", role)).lower()


def _trim_history(messages, budget_tokens: int = CHAT_HISTORY_TOKEN_BUDGET):
    recent = []
    used = 0
    for message in reversed(messages):
        estimated = _token_count(_message_text(message))
        if recent and used + estimated > max(0, int(budget_tokens)):
            break
        recent.append(message)
        used += estimated
    recent.reverse()
    return recent


def _is_eco_mode() -> bool:
    try:
        return bool(st.session_state.get("eco_mode"))
    except Exception:
        return False


def _num_predict(eco: bool | None = None) -> int:
    if eco is None:
        eco = _is_eco_mode()
    return RAG_OUTPUT_RESERVE_TOKENS // 2 if eco else RAG_OUTPUT_RESERVE_TOKENS


def _embed_batch_size() -> int:
    return 4 if _is_eco_mode() else 16


def _rag_history_budget() -> int:
    return 300 if _is_eco_mode() else 500


def _normalize_history(history) -> list:
    normalized = []
    for message in history or []:
        if isinstance(message, ChatMessage):
            if _message_role(message) in {"user", "assistant"}:
                normalized.append(message)
            continue
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "") or "")
        if not content or role not in {"user", "assistant"}:
            continue
        normalized.append(
            ChatMessage(
                role=MessageRole.ASSISTANT if role == "assistant" else MessageRole.USER,
                content=content,
            )
        )
    return normalized


def _messages_token_count(messages, tokenizer=None) -> int:
    total = 0
    for message in messages:
        role = _message_role(message)
        total += _token_count(
            f"{role}\n{_message_text(message)}",
            tokenizer,
        ) + RAG_MESSAGE_OVERHEAD_TOKENS
    return total


def _truncate_text_to_budget(text: str, max_tokens: int, tokenizer=None) -> str:
    value = str(text or "")
    limit = max(0, int(max_tokens))
    if limit == 0:
        return ""
    if _token_count(value, tokenizer) <= limit:
        return value
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if _token_count(value[:middle], tokenizer) <= limit:
            low = middle
        else:
            high = middle - 1
    return value[:low].rstrip()


def _chunk_content(chunk) -> str:
    if isinstance(chunk, Mapping):
        for key in ("content", "text", "excerpt"):
            if key in chunk and chunk[key]:
                return str(chunk[key])
        return ""
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "get_content"):
        return str(chunk.get_content() or "")
    node = getattr(chunk, "node", None)
    if node is not None:
        return _chunk_content(node)
    return str(getattr(chunk, "text", "") or "")


def _format_rag_context(chunks, formatted: bool = False) -> str:
    values = [_chunk_content(chunk) for chunk in chunks]
    if formatted:
        return "\n\n".join(values)
    return "\n\n".join(
        f"[{index}]:\n{content}"
        for index, content in enumerate(values, start=1)
    )


def _base_rag_messages(
    prompt: str,
    context: str,
    history: list,
    system_prompt: str,
) -> list:
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
    messages.extend(history)
    messages.append(
        ChatMessage(
            role=MessageRole.USER,
            content=TEXT_QA_TEMPLATE.format(context_str=context, query_str=prompt),
        )
    )
    return messages


def _fit_rag_base(
    prompt: str,
    history: list,
    system_prompt: str,
    budget: int,
    tokenizer,
):
    query = str(prompt or "")
    system = str(system_prompt or "")
    history = _normalize_history(history)
    original_history = list(history)
    for _ in range(4):
        empty_context_messages = _base_rag_messages(
            query, "", history, system
        )
        if _messages_token_count(empty_context_messages, tokenizer) <= budget:
            return system, original_history, query, empty_context_messages
        fixed = _base_rag_messages("", "", [], system)
        fixed_tokens = _messages_token_count(fixed, tokenizer)
        query_budget = max(0, budget - fixed_tokens)
        shortened_query = _truncate_text_to_budget(query, query_budget, tokenizer)
        if shortened_query != query:
            query = shortened_query
            continue
        if system:
            system = _truncate_text_to_budget(
                system,
                max(0, _token_count(system, tokenizer) // 2),
                tokenizer,
            )
            continue
        break
    for _ in range(8):
        candidate = _base_rag_messages(query, "", history, system)
        if _messages_token_count(candidate, tokenizer) <= budget:
            return system, original_history, query, candidate
        if system:
            system = ""
            continue
        if query:
            query = _truncate_text_to_budget(
                query,
                max(0, budget - _messages_token_count(
                    _base_rag_messages("", "", [], system), tokenizer
                )),
                tokenizer,
            )
            if query:
                continue
        history = []
        continue
    fallback = _base_rag_messages(query, "", history, system)
    if _messages_token_count(fallback, tokenizer) <= budget:
        return system, original_history, query, fallback
    if budget <= 0:
        return "", original_history, query, []
    minimal = _base_rag_messages("", "", [], "")
    if _messages_token_count(minimal, tokenizer) <= budget:
        return "", original_history, query, minimal
    available = max(0, budget - RAG_MESSAGE_OVERHEAD_TOKENS)
    minimal_content = _truncate_text_to_budget(
        _message_text(minimal[-1]),
        available,
        tokenizer,
    )
    fallback = [ChatMessage(role=MessageRole.USER, content=minimal_content)]
    if _messages_token_count(fallback, tokenizer) > budget:
        return "", original_history, query, []
    return "", original_history, query, fallback


def _fit_rag_history(
    prompt: str,
    history: list,
    system_prompt: str,
    budget: int,
    tokenizer,
):
    system, normalized_history, effective_prompt, base = _fit_rag_base(
        prompt, history, system_prompt, budget, tokenizer
    )
    selected = []
    for message in reversed(normalized_history):
        candidate = [message, *selected]
        candidate_messages = _base_rag_messages(
            effective_prompt, "", candidate, system
        )
        if _messages_token_count(candidate_messages, tokenizer) > budget:
            break
        selected = candidate
    final_base = _base_rag_messages(effective_prompt, "", selected, system)
    if _messages_token_count(final_base, tokenizer) > budget:
        final_base = base
    return system, selected, effective_prompt, final_base


def _llm_context_window(llm) -> int:
    values = []
    metadata = getattr(llm, "metadata", None)
    values.append(
        metadata.get("context_window")
        if isinstance(metadata, Mapping)
        else getattr(metadata, "context_window", None)
    )
    values.append(getattr(llm, "context_window", None))
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number > 0:
            return min(RAG_CONTEXT_WINDOW, number)
    return RAG_CONTEXT_WINDOW


def rag_input_budget(
    output_tokens: int | None = None,
    *,
    safety_tokens: int = RAG_SAFETY_RESERVE_TOKENS,
    context_window: int = RAG_CONTEXT_WINDOW,
) -> int:
    output = _num_predict() if output_tokens is None else max(0, int(output_tokens))
    return max(0, int(context_window) - output - max(0, int(safety_tokens)))


_rag_input_budget = rag_input_budget


def _minimal_chunk_messages(
    chunk,
    prompt: str,
    system: str,
    budget: int,
    tokenizer,
    formatted: bool,
):
    context = _format_rag_context([chunk], formatted=formatted)
    messages = _base_rag_messages(prompt, context, [], system)
    if _messages_token_count(messages, tokenizer) <= budget:
        return prompt, messages
    fixed_tokens = _messages_token_count(
        _base_rag_messages("", context, [], system), tokenizer
    )
    shortened_prompt = _truncate_text_to_budget(
        prompt,
        max(0, budget - fixed_tokens),
        tokenizer,
    )
    if not shortened_prompt:
        return None, None
    shortened_messages = _base_rag_messages(
        shortened_prompt,
        context,
        [],
        system,
    )
    if _messages_token_count(shortened_messages, tokenizer) <= budget:
        return shortened_prompt, shortened_messages
    return None, None


def plan_rag_prompt(
    prompt: str,
    context_chunks=None,
    history=None,
    system_prompt: str = "",
    *,
    output_tokens: int | None = None,
    safety_tokens: int = RAG_SAFETY_RESERVE_TOKENS,
    context_window: int = RAG_CONTEXT_WINDOW,
    tokenizer=None,
    context_is_formatted: bool = False,
) -> dict:
    if context_chunks is None:
        chunks = []
    elif isinstance(context_chunks, (str, bytes)):
        chunks = [context_chunks]
    else:
        chunks = list(context_chunks)
    output = _num_predict() if output_tokens is None else max(0, int(output_tokens))
    budget = rag_input_budget(
        output,
        safety_tokens=safety_tokens,
        context_window=context_window,
    )
    system, fitted_history, effective_prompt, base_messages = _fit_rag_history(
        prompt,
        history,
        system_prompt,
        budget,
        tokenizer,
    )
    selected = []
    selected_context = ""
    selected_messages = base_messages
    for chunk in chunks:
        candidate_chunks = [*selected, chunk]
        candidate_context = _format_rag_context(
            candidate_chunks,
            formatted=context_is_formatted,
        )
        candidate_messages = _base_rag_messages(
            effective_prompt,
            candidate_context,
            fitted_history,
            system,
        )
        if _messages_token_count(candidate_messages, tokenizer) <= budget:
            selected = candidate_chunks
            selected_context = candidate_context
            selected_messages = candidate_messages
            continue
        if not selected:
            minimal_prompt, minimal_messages = _minimal_chunk_messages(
                chunk,
                effective_prompt,
                system,
                budget,
                tokenizer,
                context_is_formatted,
            )
            if minimal_messages is not None:
                effective_prompt = minimal_prompt
                fitted_history = []
                base_messages = minimal_messages
                selected = [chunk]
                selected_context = _format_rag_context(
                    selected,
                    formatted=context_is_formatted,
                )
                selected_messages = minimal_messages
    return {
        "messages": selected_messages,
        "context": selected_context,
        "selected_chunks": selected,
        "input_tokens": _messages_token_count(selected_messages, tokenizer),
        "input_budget": budget,
        "output_reserve": output,
        "safety_reserve": max(0, int(safety_tokens)),
        "context_window": int(context_window),
    }


build_rag_prompt = plan_rag_prompt
plan_rag_context = plan_rag_prompt


def _build_rag_messages(
    prompt: str,
    context: str,
    history: list,
    system_prompt: str,
    **kwargs,
) -> list:
    chunks = list(context) if isinstance(context, (list, tuple)) else [context]
    plan = plan_rag_prompt(
        prompt,
        chunks,
        history,
        system_prompt,
        context_is_formatted=True,
        **kwargs,
    )
    return plan["messages"]


def _clear_rag_state() -> None:
    try:
        st.session_state["last_rag_evidence"] = []
        st.session_state["last_doc_sources"] = []
    except Exception:
        pass


def set_rag_evidence(evidence, evidence_sink=None) -> list:
    normalized = [dict(item) for item in (evidence or [])]
    try:
        st.session_state["last_rag_evidence"] = [dict(item) for item in normalized]
        st.session_state["last_doc_sources"] = [
            (item.get("source", "document"), float(item.get("score", 0.0)))
            for item in normalized
        ]
    except Exception:
        pass
    if isinstance(evidence_sink, list):
        evidence_sink.clear()
        evidence_sink.extend([dict(item) for item in normalized])
    return normalized


def evidence_sources(evidence) -> list[tuple[str, float]]:
    sources = []
    for item in evidence or []:
        if not isinstance(item, Mapping):
            continue
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        sources.append((str(item.get("source", "document")), score))
    return sources


_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,80})\]")
_CITATION_EMPTY_BODY_RE = re.compile(r"\[\s*\]")
_CITATION_UNCLOSED_BRACKET_RE = re.compile(
    r"\[\s*\d(?![^\]\n]*\])[^\]\n]{0,80}(?=$|[\n.,!?;])"
)
_CITATION_PREFIXED_RE = re.compile(
    r"\b(from|source|citation|evidence|reference|ref)\s*(\[[^\[\]\n]{1,80}\])",
    flags=re.IGNORECASE,
)
_CITATION_UNCLOSED_RE = re.compile(
    r"\(\s*(?:from|source|citation|evidence)\s*\[\s*(?![^\]\n]*\])[^\]\n]{1,80}(?:\)|(?=$|[\s.,!?;]))",
    flags=re.IGNORECASE,
)
_CITATION_EMPTY_WRAPPER_RE = re.compile(
    r"\(\s*(?:from|source|citation|evidence)\s*(?:\[\s*\]|\s*)\s*\)",
    flags=re.IGNORECASE,
)
_CITATION_LEFTOVER_WRAPPER_RE = re.compile(
    r"\(\s*(?:from|source|citation|evidence)\s*[\[\]]+\s*\)",
    flags=re.IGNORECASE,
)
def _citation_values(body: str, maximum: int):
    value = str(body or "").strip()
    if re.fullmatch(r"\d+(?:\s*[,;]\s*\d+)*", value):
        values = [int(item) for item in re.findall(r"\d+", value)]
    else:
        match = re.fullmatch(
            r"(?:from|source|citation|evidence|reference|ref)\s+(.+)", value, re.I
        )
        if not match or not re.fullmatch(
            r"\d+(?:\s*[,;]\s*\d+)*", match.group(1).strip()
        ):
            return None
        values = [int(item) for item in re.findall(r"\d+", match.group(1))]
    if not values:
        return None
    return [item for item in values if 1 <= item <= max(0, int(maximum))]


def sanitize_citations(text: str, evidence_count: int) -> str:
    value = str(text or "")
    maximum = max(0, int(evidence_count))

    def replace(match):
        body = match.group(1).strip()
        values = _citation_values(body, maximum)
        if values is not None:
            raw_values = [int(item) for item in re.findall(r"\d+", body)]
            if len(values) == len(raw_values):
                return match.group(0)
            return f"[{', '.join(str(item) for item in values)}]" if values else ""
        if value[match.end() :].lstrip().startswith("("):
            return match.group(0)
        if any(character.isdigit() for character in body):
            return ""
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,31}", body):
            return ""
        return match.group(0)

    def replace_prefixed(match):
        values = _citation_values(match.group(2)[1:-1], maximum)
        if values is None:
            return match.group(0)
        if not values:
            return ""
        raw_values = [int(item) for item in re.findall(r"\d+", match.group(2))]
        if len(values) == len(raw_values):
            return match.group(0)
        return f"{match.group(1)} [{', '.join(str(item) for item in values)}]"

    cleaned = _CITATION_UNCLOSED_RE.sub("", value)
    cleaned = _CITATION_UNCLOSED_BRACKET_RE.sub("", cleaned)
    cleaned = _CITATION_PREFIXED_RE.sub(replace_prefixed, cleaned)
    cleaned = _CITATION_EMPTY_BODY_RE.sub("", cleaned)
    cleaned = _CITATION_BRACKET_RE.sub(replace, cleaned)
    cleaned = _CITATION_EMPTY_WRAPPER_RE.sub("", cleaned)
    cleaned = _CITATION_LEFTOVER_WRAPPER_RE.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    return cleaned.strip()


_sanitize_citations = sanitize_citations
postprocess_citations = sanitize_citations


def build_evidence(nodes, max_excerpt_chars: int = EVIDENCE_EXCERPT_MAX_CHARS):
    return normalize_evidence(nodes, max_excerpt_chars=max_excerpt_chars)


def retrieval_history_messages(messages, current_prompt: str) -> list:
    history = []
    current = str(current_prompt or "").strip()
    for message in messages or []:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")).lower() != "user":
            continue
        content = str(message.get("content", "") or "").strip()
        if not content or content == current:
            continue
        history.append(message)
    return history


def expanded_retrieval_query(prompt: str, history=None) -> str:
    return build_retrieval_query(prompt, history)


build_followup_query = build_retrieval_query
build_conversation_query = build_retrieval_query


###################################
#
# Create Client
#
###################################


def create_client(host: str):
    endpoint = normalize_ollama_endpoint(host)
    try:
        client = ollama.Client(
            host=endpoint,
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
        logs.log.info("Ollama client created successfully")
        return client
    except Exception as err:
        logs.log.error(f"Failed to create Ollama client: {err}")
        return False


def _get_installed_model_names(chat_client):
    data = chat_client.list()
    if isinstance(data, dict):
        models = data.get("models", [])
    else:
        models = getattr(data, "models", [])
    names = []
    for model in models or []:
        if isinstance(model, dict):
            model_name = model.get("model") or model.get("name")
        else:
            model_name = getattr(model, "model", None) or getattr(model, "name", None)
        if model_name and str(model_name) not in names:
            names.append(str(model_name))
    return names


def _capabilities(details):
    if isinstance(details, dict):
        values = details.get("capabilities", [])
    else:
        values = getattr(details, "capabilities", [])
    if isinstance(values, str):
        return {values.lower()}
    return {str(value).lower() for value in values or []}


def _ollama_discovery(endpoint: str) -> dict:
    state = st.session_state
    now = time.monotonic()
    cached = state.get("_ollama_model_discovery")
    if (
        isinstance(cached, dict)
        and cached.get("endpoint") == endpoint
        and now - float(cached.get("created_at", 0)) < MODEL_DISCOVERY_TTL_SECONDS
    ):
        return cached
    client = create_client(endpoint)
    if not client:
        raise RuntimeError("Ollama client could not be created")
    names = _get_installed_model_names(client)
    capabilities = {}
    errors = {}
    for name in names:
        try:
            capabilities[name] = _capabilities(client.show(name))
        except Exception as err:
            errors[name] = str(err)
    discovery = {
        "endpoint": endpoint,
        "created_at": now,
        "names": names,
        "capabilities": capabilities,
        "errors": errors,
    }
    state["_ollama_model_discovery"] = discovery
    return discovery


def _models_from_discovery(state, discovery, kind):
    names = discovery.get("names", [])
    capabilities = discovery.get("capabilities", {})
    previous_key = "ollama_models" if kind == "chat" else "ollama_embedding_models"
    previous = set(state.get(previous_key, []) or [])
    result = []
    for name in names:
        values = capabilities.get(name)
        if values is None:
            if name in previous or kind == "chat":
                result.append(name)
            continue
        if kind == "chat" and ("completion" in values or not values):
            result.append(name)
        elif kind == "embedding" and "embedding" in values:
            result.append(name)
    return result


def default_embedding_model(models):
    preferred_models = (
        "embeddinggemma:latest",
        "nomic-embed-text:latest",
        "mxbai-embed-large:latest",
    )
    for model in preferred_models:
        if model in models:
            return model
    return models[0] if models else None


def default_chat_model(models):
    preferred_models = ("gemma4:latest", "llama3:8b", "llama2:7b")
    for model in preferred_models:
        if model in models:
            return model
    return models[0] if models else None


def _model_capabilities(entry) -> set[str]:
    if not isinstance(entry, dict):
        return set()
    values = []
    for key in ("capabilities", "capability", "task", "type", "object"):
        value = entry.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value)
    return {value.lower() for value in values}


def _looks_like_embedding_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(
        marker in lowered
        for marker in (
            "embed",
            "embedding",
            "bge",
            "e5-",
            "gte-",
            "minilm",
            "jina-embeddings",
            "nomic-embed",
        )
    )


def _split_openai_model_ids(entries) -> tuple[list[str], list[str], list[str]]:
    all_models = []
    chat_models = []
    embedding_models = []
    for entry in entries:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if not model_id:
            continue
        model_id = str(model_id)
        if model_id in all_models:
            continue
        all_models.append(model_id)
        capabilities = _model_capabilities(entry)
        if "embedding" in capabilities or _looks_like_embedding_model(model_id):
            embedding_models.append(model_id)
        else:
            chat_models.append(model_id)
    return sorted(chat_models), sorted(embedding_models), sorted(all_models)


def get_openai_model_catalog(
    base_url: str,
    api_key: str = "",
    *,
    provider_kind: str = OPENAI_COMPATIBLE,
) -> dict:
    provider_kind = normalize_provider_kind(provider_kind)
    api_key = str(api_key).strip() if api_key else ""
    endpoint = normalize_provider_endpoint(
        base_url,
        default=(
            DEFAULT_OPENAI_BASE_URL
            if provider_kind == OPENAI_OFFICIAL
            else None
        ),
        label="OpenAI-compatible endpoint",
    )
    endpoint = validate_credential_endpoint(api_key, endpoint, provider_kind)
    if provider_kind == OPENAI_OFFICIAL and not api_key:
        raise ValueError("Official OpenAI model discovery requires an explicit API key.")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.get(
            f"{endpoint}/models",
            headers=headers,
            timeout=(5, 10),
            allow_redirects=False,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as err:
        logs.log.error(f"Failed to fetch OpenAI-compatible models: {err}")
        return {
            "all": [],
            "chat": [],
            "embedding": [],
            "endpoint": endpoint,
            "key_fingerprint": credential_fingerprint(api_key),
            "provider_kind": provider_kind,
        }
    entries = data.get("data", []) if isinstance(data, dict) else []
    chat_models, embedding_models, all_models = _split_openai_model_ids(entries)
    return {
        "all": all_models,
        "chat": chat_models,
        "embedding": embedding_models,
        "endpoint": endpoint,
        "key_fingerprint": credential_fingerprint(api_key),
        "provider_kind": provider_kind,
    }


def get_openai_models(
    base_url: str,
    api_key: str = "",
    *,
    provider_kind: str = OPENAI_COMPATIBLE,
    model_type: str | None = None,
) -> list:
    catalog = get_openai_model_catalog(
        base_url,
        api_key,
        provider_kind=provider_kind,
    )
    if model_type == "chat":
        return catalog["chat"]
    if model_type == "embedding":
        return catalog["embedding"]
    return catalog["all"]


def get_openai_chat_models(base_url: str, api_key: str = "", provider_kind=OPENAI_COMPATIBLE):
    return get_openai_models(
        base_url,
        api_key,
        provider_kind=provider_kind,
        model_type="chat",
    )


def get_openai_embedding_models(
    base_url: str,
    api_key: str = "",
    provider_kind=OPENAI_COMPATIBLE,
):
    return get_openai_models(
        base_url,
        api_key,
        provider_kind=provider_kind,
        model_type="embedding",
    )


def get_models():
    state = st.session_state
    endpoint = normalize_ollama_endpoint(state.get("ollama_endpoint"))
    state["ollama_endpoint"] = endpoint
    try:
        discovery = _ollama_discovery(endpoint)
        models = _models_from_discovery(state, discovery, "chat")
        had_selection = bool(state.get("selected_model"))
        state["ollama_models"] = models
        if state.get("selected_model") not in models:
            state["selected_model"] = default_chat_model(models)
        if not had_selection and state.get("selected_model"):
            preferred = state["selected_model"]
            models = [preferred] + [model for model in models if model != preferred]
            state["ollama_models"] = models
        if discovery.get("errors"):
            logs.log.warning(
                "Ollama metadata was unavailable for: "
                + ", ".join(sorted(discovery["errors"]))
            )
        if models:
            logs.log.info("Ollama chat models loaded successfully")
        else:
            logs.log.warning("Ollama did not return any chat-capable models")
        return models
    except Exception as err:
        logs.log.error(f"Failed to retrieve Ollama model list: {err}")
        state["ollama_models"] = []
        return []


def get_embedding_models(endpoint: str | None = None):
    state = st.session_state
    endpoint = normalize_ollama_endpoint(endpoint or state.get("ollama_endpoint"))
    if endpoint == state.get("ollama_endpoint"):
        state["ollama_endpoint"] = endpoint
    try:
        discovery = _ollama_discovery(endpoint)
        embedding_models = _models_from_discovery(state, discovery, "embedding")
        state["ollama_embedding_models"] = embedding_models
        if embedding_models:
            if state.get("ollama_embedding_model") not in embedding_models:
                state["ollama_embedding_model"] = default_embedding_model(
                    embedding_models
                )
            logs.log.info("Ollama embedding models loaded successfully")
        else:
            logs.log.warning("Ollama did not return any embedding-capable models")
        return embedding_models
    except Exception as err:
        logs.log.error(f"Failed to retrieve Ollama embedding model list: {err}")
        state["ollama_embedding_models"] = []
        return []


###################################
#
# Create Ollama LLM instance
#
###################################


def verify_chat_model(model: str, base_url: str) -> bool:
    try:
        endpoint = normalize_ollama_endpoint(base_url)
        client = create_client(endpoint)
        if not client or model not in _get_installed_model_names(client):
            return False
        details = client.show(model)
        return "completion" in _capabilities(details)
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _create_ollama_llm_cached(
    model: str,
    base_url: str,
    request_timeout: int,
    temperature: float,
    eco_mode: bool,
    max_tokens: int,
) -> Ollama:
    Settings.llm = Ollama(
        model=model,
        base_url=base_url,
        request_timeout=request_timeout,
        context_window=RAG_CONTEXT_WINDOW,
        temperature=temperature,
        keep_alive="2m",
        additional_kwargs={"num_predict": max_tokens},
    )
    return Settings.llm


def create_ollama_llm(
    model: str,
    base_url: str,
    system_prompt: str = None,
    request_timeout: int = OLLAMA_REQUEST_TIMEOUT,
    temperature: float = None,
    eco_mode: bool | None = None,
) -> Ollama:
    if not model:
        raise ValueError("An Ollama chat model is required.")
    endpoint = normalize_ollama_endpoint(base_url)
    if temperature is None:
        temperature = float(st.session_state.get("temperature", DEFAULT_TEMPERATURE))
    if eco_mode is None:
        eco_mode = _is_eco_mode()
    if request_timeout is None:
        request_timeout = OLLAMA_REQUEST_TIMEOUT
    bounded_timeout = min(max(int(request_timeout), 1), OLLAMA_REQUEST_TIMEOUT)
    try:
        result = _create_ollama_llm_cached(
            model=model,
            base_url=endpoint,
            request_timeout=bounded_timeout,
            temperature=float(temperature),
            eco_mode=bool(eco_mode),
            max_tokens=_num_predict(eco_mode),
        )
        Settings.llm = result
        logs.log.info("Ollama LLM instance created successfully")
        return result
    except Exception as err:
        logs.log.error(f"Error creating Ollama language model: {err}")
        raise


def _load_openai_adapter(compatible: bool):
    if compatible:
        try:
            from llama_index.llms.openai_like import OpenAILike
        except (ImportError, ModuleNotFoundError) as err:
            raise RuntimeError(
                "OpenAI-compatible providers require "
                "llama-index-llms-openai-like==0.8.0. Install the pinned "
                "Pipfile dependency and retry."
            ) from err
        return OpenAILike
    try:
        from llama_index.llms.openai import OpenAI
    except (ImportError, ModuleNotFoundError) as err:
        raise RuntimeError(
            "Official OpenAI support requires llama-index-llms-openai. "
            "Install the declared dependency and retry."
        ) from err
    return OpenAI


@st.cache_resource(show_spinner=False)
def _create_openai_llm_cached(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    provider_kind: str,
):
    adapter = _load_openai_adapter(provider_kind == OPENAI_COMPATIBLE)
    kwargs = {
        "model": model,
        "api_key": api_key,
        "api_base": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": 1,
        "timeout": 300.0,
    }
    if provider_kind == OPENAI_COMPATIBLE:
        kwargs.update(
            {
                "context_window": RAG_CONTEXT_WINDOW,
                "is_chat_model": True,
                "is_function_calling_model": False,
            }
        )
    try:
        Settings.llm = adapter(**kwargs)
    except (AttributeError, TypeError) as err:
        if provider_kind == OPENAI_COMPATIBLE:
            raise RuntimeError(
                "The installed OpenAI-compatible adapter does not expose the "
                "OpenAILike API required by this provider. Install "
                "llama-index-llms-openai-like==0.8.0 and retry."
            ) from err
        raise
    return Settings.llm


def create_openai_llm(
    model: str,
    base_url: str,
    api_key: str = "",
    temperature: float = None,
    eco_mode: bool | None = None,
    backend: str = None,
):
    kind = provider_kind(backend)
    api_key = str(api_key).strip() if api_key else ""
    if kind == OLLAMA:
        raise ValueError("The Ollama provider cannot use the OpenAI adapter.")
    if not model:
        raise ValueError("An OpenAI provider chat model is required.")
    endpoint = normalize_provider_endpoint(
        base_url,
        default=(
            DEFAULT_OPENAI_BASE_URL
            if kind == OPENAI_OFFICIAL
            else None
        ),
        label="OpenAI endpoint",
    )
    endpoint = validate_credential_endpoint(api_key, endpoint, kind)
    if kind == OPENAI_OFFICIAL and not api_key:
        raise ValueError("Official OpenAI requires an explicit API key.")
    if temperature is None:
        temperature = float(st.session_state.get("temperature", DEFAULT_TEMPERATURE))
    if eco_mode is None:
        eco_mode = _is_eco_mode()
    try:
        result = _create_openai_llm_cached(
            model=model,
            base_url=endpoint,
            api_key=api_key or LOCAL_COMPATIBLE_API_KEY,
            temperature=float(temperature),
            max_tokens=_num_predict(eco_mode),
            provider_kind=kind,
        )
        Settings.llm = result
        logs.log.info(f"OpenAI provider LLM created successfully ({endpoint})")
        return result
    except Exception as err:
        logs.log.error(f"Error creating OpenAI provider language model: {err}")
        raise


def create_openai_compatible_llm(
    model: str,
    base_url: str,
    api_key: str = "",
    temperature: float = None,
    eco_mode: bool | None = None,
    backend: str = None,
):
    if backend is not None and provider_kind(backend) != OPENAI_COMPATIBLE:
        raise ValueError("The selected backend is not OpenAI-compatible.")
    return create_openai_llm(
        model,
        base_url,
        api_key,
        temperature=temperature,
        eco_mode=eco_mode,
        backend=backend or "LM Studio (Local AI)",
    )


def create_llm(
    model: str,
    base_url: str,
    api_key: str = "",
    system_prompt: str = None,
    temperature: float = None,
    backend: str = None,
):
    backend = backend or st.session_state.get("llm_backend", "Ollama")
    kind = provider_kind(backend)
    eco_mode = st.session_state.get("eco_mode", False)
    if temperature is None:
        temperature = float(st.session_state.get("temperature", DEFAULT_TEMPERATURE))
    if kind == OPENAI_OFFICIAL:
        return create_openai_llm(
            model,
            base_url,
            api_key,
            temperature,
            eco_mode=eco_mode,
            backend=backend,
        )
    if kind == OPENAI_COMPATIBLE:
        return create_openai_compatible_llm(
            model,
            base_url,
            api_key,
            temperature,
            eco_mode=eco_mode,
            backend=backend,
        )
    return create_ollama_llm(
        model,
        base_url,
        system_prompt,
        temperature=temperature,
        eco_mode=eco_mode,
    )


###################################
#
# Chat (no context)
#
###################################


def chat(prompt: str):
    """
    Initiates a chat with the active LLM backend using multi-turn conversational history.

    Parameters:
        - prompt (str): The starting prompt for the conversation.

    Yields:
        - str: Successive chunks of conversation from the model.
    """
    try:
        llm = create_llm(
            _active_chat_model(),
            _active_base_url(),
            _active_api_key(),
            system_prompt=st.session_state.get("system_prompt"),
        )

        chat_messages = []
        system_prompt = st.session_state.get("system_prompt")
        if system_prompt:
            chat_messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )
        for msg in st.session_state.get("messages", []):
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            if role_str == "assistant":
                chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
            elif role_str == "user":
                chat_messages.append(ChatMessage(role=MessageRole.USER, content=content))
            elif role_str == "system":
                chat_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=content))

        # Trim the conversational history (not the system message) so it fits
        # comfortably inside the model's context window. Oversized histories
        # silently truncate and degrade response quality.
        history = _trim_history(chat_messages[1:]) if system_prompt else _trim_history(chat_messages)
        if not history or history[-1].content != prompt:
            history.append(ChatMessage(role=MessageRole.USER, content=prompt))

        recent_messages = ([chat_messages[0]] + history) if system_prompt else history
        stream = llm.stream_chat(recent_messages)
        for chunk in stream:
            yield chunk.delta
    except Exception as err:
        logs.log.error(f"Ollama chat stream error: {err}")
        if provider_kind() != OLLAMA:
            yield (
                f"⚠️ **Error during chat:** {err}. Please ensure the "
                f"configured provider is running and model "
                f"'{_active_chat_model()}' is available on it."
            )
        else:
            yield f"⚠️ **Error during chat:** {err}. Please ensure Ollama is running and model '{st.session_state.get('selected_model')}' is installed."
        return


###################################
#
# Document Chat (with context)
#
###################################


def context_chat(
    prompt: str,
    query_engine: RetrieverQueryEngine,
    evidence_sink: list | None = None,
):
    _clear_rag_state()
    try:
        system_prompt = st.session_state.get("system_prompt", "")
        llm = create_llm(
            _active_chat_model(),
            _active_base_url(),
            _active_api_key(),
            system_prompt=system_prompt,
        )

        retriever = st.session_state.get("retriever")
        if retriever is None:
            retriever = getattr(query_engine, "_retriever", None)
        if retriever is None:
            st.session_state["last_rag_no_result"] = False
            st.session_state["last_rag_question"] = None
            yield "⚠️ **No retriever available.** Please re-ingest your documents."
            return

        messages_state = st.session_state.get("messages", [])
        query_candidates = retrieval_query_candidates(prompt, messages_state)
        t0 = time.time()
        nodes = retriever.retrieve(query_candidates[0])
        if not nodes and len(query_candidates) > 1:
            nodes = retriever.retrieve(query_candidates[1])
        nodes = [
            node
            for node in list(nodes)
            if _chunk_content(node).strip()
        ]
        if not nodes:
            _clear_rag_state()
            st.session_state["last_rag_no_result"] = True
            st.session_state["last_rag_question"] = prompt
            yield "I could not find this information in the documents."
            return

        history = []
        for message in messages_state:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "")).lower()
            content = str(message.get("content", "") or "")
            if not content or role not in {"user", "assistant"}:
                continue
            history.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT if role == "assistant" else MessageRole.USER,
                    content=content,
                )
            )
        if history and _message_text(history[-1]) == str(prompt or "").strip():
            history = history[:-1]

        plan = plan_rag_prompt(
            prompt,
            nodes,
            history,
            system_prompt,
            output_tokens=_num_predict(),
            context_window=_llm_context_window(llm),
            tokenizer=_resolve_tokenizer(llm),
        )
        selected_nodes = list(plan.get("selected_chunks") or [])
        if not selected_nodes:
            _clear_rag_state()
            st.session_state["last_rag_no_result"] = True
            st.session_state["last_rag_question"] = prompt
            yield "I could not find this information in the documents."
            return

        evidence = set_rag_evidence(
            normalize_evidence(selected_nodes),
            evidence_sink=evidence_sink,
        )
        st.session_state["last_rag_no_result"] = False
        st.session_state["last_rag_question"] = None
        top_score = getattr(selected_nodes[0], "score", 0.0)
        try:
            top_score = float(top_score)
        except (TypeError, ValueError):
            top_score = 0.0
        logs.log.info(
            f"Doc query: {len(selected_nodes)} chunks | top score {top_score:.3f}"
        )

        full = []
        for chunk in llm.stream_chat(plan["messages"]):
            delta = getattr(chunk, "delta", chunk)
            if delta:
                full.append(str(delta))
        answer = sanitize_citations("".join(full), len(evidence))
        if answer:
            yield answer
        logs.log.info(f"Doc query answered in {time.time() - t0:.1f}s")
    except Exception as err:
        _clear_rag_state()
        try:
            st.session_state["last_rag_no_result"] = False
            st.session_state["last_rag_question"] = None
        except Exception:
            pass
        logs.log.error(f"Ollama chat stream error: {err}")
        if provider_kind() != OLLAMA:
            yield (
                f"⚠️ **Error generating response:** {err}. Please ensure the "
                f"configured provider is running and model "
                f"'{_active_chat_model()}' is available on it."
            )
        else:
            yield f"⚠️ **Error generating response:** {err}. If the model is taking longer to respond on CPU, please try again."
        return
