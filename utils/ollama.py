import contextlib
import copy
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
    sanitize_untrusted_context,
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
from utils.retrieval_map import (
    MAP_PLANNER_LLM,
    DocumentMap,
    LlmMapPlanner,
    MapRetrievalAgent,
    annotate_planner_trace,
    build_map_retrieval_agent,
    count_map_tokens,
    json_safe,
    map_agent_config,
    map_document_config,
    map_source_identity,
    normalize_map_settings,
    retriever_supports_map,
    state_map_source_identity,
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
OPENAI_COMPATIBLE_QA_TEMPLATE = TEXT_QA_TEMPLATE
TOKEN_ACCOUNTING_BASIS = "tokenizer_prompt_and_selected_evidence_text"
TOKEN_ACCOUNTING_NOTE = (
    "Tokenizer and prompt-context measurements only. They are not total compute, "
    "latency, GPU, embedding, retrieval-speed, or answer-quality measurements."
)
TOKEN_ACCOUNTING_DELTA_FIELDS = (
    "actual_rag_input_prompt_tokens",
    "actual_selected_evidence_tokens",
    "baseline_input_prompt_tokens",
    "baseline_selected_evidence_tokens",
    "selected_minus_baseline_input_tokens",
    "selected_minus_baseline_input_pct",
)
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


def _session_llm():
    try:
        llm = st.session_state.get("llm")
        if llm is None:
            return None
        profile = _active_chat_profile()
    except Exception:
        return None
    model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    endpoint = getattr(llm, "base_url", None) or getattr(llm, "api_base", None)
    if model and profile.get("model") and str(model) != str(profile["model"]):
        return None
    if endpoint and profile.get("base_url"):
        try:
            if normalize_provider_endpoint(endpoint) != normalize_provider_endpoint(
                profile["base_url"]
            ):
                return None
        except (TypeError, ValueError):
            return None
    return llm


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


def resolve_tokenizer(llm=None):
    """Return the active tokenizer callable for token and context measurements."""
    return _resolve_tokenizer(llm)


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
        total += (
            _token_count(
                f"{role}\n{_message_text(message)}",
                tokenizer,
            )
            + RAG_MESSAGE_OVERHEAD_TOKENS
        )
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
            if chunk.get(key):
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


def _neutralize_context_markers(value: str) -> str:
    return sanitize_untrusted_context(value)


def _format_rag_context(chunks, formatted: bool = False) -> str:
    values = [_neutralize_context_markers(_chunk_content(chunk)) for chunk in chunks]
    if formatted:
        return "\n\n".join(values)
    return "\n\n".join(
        f"[{index}]:\n{content}" for index, content in enumerate(values, start=1)
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
        empty_context_messages = _base_rag_messages(query, "", history, system)
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
                max(
                    0,
                    budget
                    - _messages_token_count(
                        _base_rag_messages("", "", [], system), tokenizer
                    ),
                ),
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
        candidate_messages = _base_rag_messages(effective_prompt, "", candidate, system)
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


def _selected_evidence_tokens(plan, tokenizer=None) -> int:
    return sum(
        count_map_tokens(_chunk_content(chunk), tokenizer)
        for chunk in list((plan or {}).get("selected_chunks") or [])
    )


def measure_rag_token_accounting(
    prompt: str,
    plan,
    *,
    tokenizer=None,
    history=None,
    system_prompt: str = "",
    baseline_chunks=None,
    output_tokens: int | None = None,
    safety_tokens: int = RAG_SAFETY_RESERVE_TOKENS,
    context_window: int = RAG_CONTEXT_WINDOW,
) -> dict:
    """Measure the tokens of the real plan and an optional hypothetical baseline."""
    accounting = {
        "token_basis": TOKEN_ACCOUNTING_BASIS,
        "notes": TOKEN_ACCOUNTING_NOTE,
        "baseline_measured": False,
        "baseline_result_count": None,
        "actual_selected_evidence_tokens": _selected_evidence_tokens(plan, tokenizer),
        "actual_rag_input_prompt_tokens": max(
            0, int((plan or {}).get("input_tokens") or 0)
        ),
        "rag_input_budget_tokens": max(0, int((plan or {}).get("input_budget") or 0)),
        "output_reserve_tokens": max(0, int((plan or {}).get("output_reserve") or 0)),
        "context_window_tokens": max(0, int(context_window or 0)),
        "baseline_input_prompt_tokens": None,
        "baseline_selected_evidence_tokens": None,
        "selected_minus_baseline_input_tokens": None,
        "selected_minus_baseline_input_pct": None,
    }
    if not baseline_chunks:
        return accounting
    baseline_plan = plan_rag_prompt(
        prompt,
        list(baseline_chunks),
        history,
        system_prompt,
        output_tokens=output_tokens,
        safety_tokens=safety_tokens,
        context_window=context_window,
        tokenizer=tokenizer,
    )
    baseline_input = max(0, int(baseline_plan.get("input_tokens") or 0))
    accounting["baseline_measured"] = True
    accounting["baseline_result_count"] = len(
        list(baseline_plan.get("selected_chunks") or [])
    )
    accounting["baseline_input_prompt_tokens"] = baseline_input
    accounting["baseline_selected_evidence_tokens"] = _selected_evidence_tokens(
        baseline_plan, tokenizer
    )
    if baseline_input > 0:
        actual_input = accounting["actual_rag_input_prompt_tokens"]
        delta = actual_input - baseline_input
        accounting["selected_minus_baseline_input_tokens"] = delta
        accounting["selected_minus_baseline_input_pct"] = round(
            delta * 100.0 / baseline_input,
            2,
        )
    return accounting


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


def set_retrieval_route(route, route_sink=None) -> dict:
    payload = json_safe(route) if isinstance(route, Mapping) else {}
    with contextlib.suppress(Exception):
        st.session_state["last_retrieval_route"] = copy.deepcopy(payload)
    if isinstance(route_sink, dict):
        route_sink.clear()
        route_sink.update(copy.deepcopy(payload))
    elif isinstance(route_sink, list):
        route_sink.clear()
        if payload:
            route_sink.append(copy.deepcopy(payload))
    return payload


def clear_retrieval_route(route_sink=None) -> dict:
    return set_retrieval_route({}, route_sink=route_sink)


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
        logs.log.error(
            "Failed to create Ollama client: %s", logs.safe_log_exception(err)
        )
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
            errors[name] = logs.safe_log_exception(err)
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
        if (kind == "chat" and ("completion" in values or not values)) or (
            kind == "embedding" and "embedding" in values
        ):
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
        default=(DEFAULT_OPENAI_BASE_URL if provider_kind == OPENAI_OFFICIAL else None),
        label="OpenAI-compatible endpoint",
    )
    endpoint = validate_credential_endpoint(api_key, endpoint, provider_kind)
    if provider_kind == OPENAI_OFFICIAL and not api_key:
        raise ValueError(
            "Official OpenAI model discovery requires an explicit API key."
        )
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
        logs.log.error(
            "Failed to fetch OpenAI-compatible models: %s",
            logs.safe_log_exception(err),
        )
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


def get_openai_chat_models(
    base_url: str, api_key: str = "", provider_kind=OPENAI_COMPATIBLE
):
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
        logs.log.error(
            "Failed to retrieve Ollama model list: %s",
            logs.safe_log_exception(err),
        )
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
        logs.log.error(
            "Failed to retrieve Ollama embedding model list: %s",
            logs.safe_log_exception(err),
        )
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
    system_prompt: str | None = None,
    request_timeout: int = OLLAMA_REQUEST_TIMEOUT,
    temperature: float | None = None,
    eco_mode: bool | None = None,
    set_global: bool = True,
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
    previous_llm = getattr(Settings, "_llm", None)
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
        logs.log.error(
            "Error creating Ollama language model: %s",
            logs.safe_log_exception(err),
        )
        raise
    finally:
        if not set_global:
            with contextlib.suppress(Exception):
                Settings.llm = previous_llm


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
    temperature: float | None = None,
    eco_mode: bool | None = None,
    backend: str | None = None,
    set_global: bool = True,
):
    kind = provider_kind(backend)
    api_key = str(api_key).strip() if api_key else ""
    if kind == OLLAMA:
        raise ValueError("The Ollama provider cannot use the OpenAI adapter.")
    if not model:
        raise ValueError("An OpenAI provider chat model is required.")
    endpoint = normalize_provider_endpoint(
        base_url,
        default=(DEFAULT_OPENAI_BASE_URL if kind == OPENAI_OFFICIAL else None),
        label="OpenAI endpoint",
    )
    endpoint = validate_credential_endpoint(api_key, endpoint, kind)
    if kind == OPENAI_OFFICIAL and not api_key:
        raise ValueError("Official OpenAI requires an explicit API key.")
    if temperature is None:
        temperature = float(st.session_state.get("temperature", DEFAULT_TEMPERATURE))
    if eco_mode is None:
        eco_mode = _is_eco_mode()
    previous_llm = getattr(Settings, "_llm", None)
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
        logs.log.info(
            "OpenAI provider LLM created successfully (%s)", logs.redact_text(endpoint)
        )
        return result
    except Exception as err:
        logs.log.error(
            "Error creating OpenAI provider language model: %s",
            logs.safe_log_exception(err),
        )
        raise
    finally:
        if not set_global:
            with contextlib.suppress(Exception):
                Settings.llm = previous_llm


def create_openai_compatible_llm(
    model: str,
    base_url: str,
    api_key: str = "",
    temperature: float | None = None,
    eco_mode: bool | None = None,
    backend: str | None = None,
    set_global: bool = True,
):
    if backend is not None and provider_kind(backend) != OPENAI_COMPATIBLE:
        raise ValueError("The selected backend is not OpenAI-compatible.")
    dispatch_global = {} if set_global else {"set_global": False}
    return create_openai_llm(
        model,
        base_url,
        api_key,
        temperature=temperature,
        eco_mode=eco_mode,
        backend=backend or "LM Studio (Local AI)",
        **dispatch_global,
    )


def create_llm(
    model: str,
    base_url: str,
    api_key: str = "",
    system_prompt: str | None = None,
    temperature: float | None = None,
    backend: str | None = None,
    set_global: bool = True,
):
    backend = backend or st.session_state.get("llm_backend", "Ollama")
    kind = provider_kind(backend)
    eco_mode = st.session_state.get("eco_mode", False)
    if temperature is None:
        temperature = float(st.session_state.get("temperature", DEFAULT_TEMPERATURE))
    dispatch_global = {} if set_global else {"set_global": False}
    if kind == OPENAI_OFFICIAL:
        return create_openai_llm(
            model,
            base_url,
            api_key,
            temperature,
            eco_mode=eco_mode,
            backend=backend,
            **dispatch_global,
        )
    if kind == OPENAI_COMPATIBLE:
        return create_openai_compatible_llm(
            model,
            base_url,
            api_key,
            temperature,
            eco_mode=eco_mode,
            backend=backend,
            **dispatch_global,
        )
    return create_ollama_llm(
        model,
        base_url,
        system_prompt,
        temperature=temperature,
        eco_mode=eco_mode,
        **dispatch_global,
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
        llm = _session_llm()
        if llm is None:
            llm = create_llm(
                _active_chat_model(),
                _active_base_url(),
                _active_api_key(),
                system_prompt=st.session_state.get("system_prompt"),
                set_global=False,
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
                chat_messages.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=content)
                )
            elif role_str == "user":
                chat_messages.append(
                    ChatMessage(role=MessageRole.USER, content=content)
                )
            elif role_str == "system":
                chat_messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=content)
                )

        # Trim the conversational history (not the system message) so it fits
        # comfortably inside the model's context window. Oversized histories
        # silently truncate and degrade response quality.
        history = (
            _trim_history(chat_messages[1:])
            if system_prompt
            else _trim_history(chat_messages)
        )
        if not history or history[-1].content != prompt:
            history.append(ChatMessage(role=MessageRole.USER, content=prompt))

        recent_messages = [chat_messages[0], *history] if system_prompt else history
        stream = llm.stream_chat(recent_messages)
        for chunk in stream:
            yield chunk.delta
    except Exception as err:
        logs.log.error("Chat stream failed: %s", logs.safe_log_exception(err))
        if provider_kind() != OLLAMA:
            yield "⚠️ **Error during chat:** The configured provider could not complete the request."
        else:
            yield "⚠️ **Error during chat:** Ollama could not complete the request."
        return


###################################
#
# Document Chat (with context)
#
###################################


def _map_retrieval_planner(llm, tokenizer, settings):
    if settings.get("retrieval_map_planner_mode") != MAP_PLANNER_LLM:
        return None
    try:
        return LlmMapPlanner(
            llm,
            tokenizer=tokenizer,
            max_selected_sections=settings["retrieval_map_max_selected_sections"],
        )
    except (TypeError, ValueError):
        logs.log.warning("Map planner was unavailable; using deterministic routing")
        return None


def _active_document_map(state, identity):
    stored = state.get("retrieval_map")
    if (
        isinstance(stored, DocumentMap)
        and identity
        and stored.source_identity == identity
    ):
        return stored
    return None


def _build_map_agent(state, retriever, llm, tokenizer, settings, identity):
    planner = _map_retrieval_planner(llm, tokenizer, settings)
    agent_config = map_agent_config(settings)
    map_config = map_document_config(settings)
    document_map = _active_document_map(state, identity)
    if document_map is not None and document_map.config == map_config:
        return MapRetrievalAgent(
            retriever,
            document_map,
            tokenizer=tokenizer,
            planner=planner,
            agent_config=agent_config,
        )
    agent = build_map_retrieval_agent(
        retriever,
        tokenizer=tokenizer,
        planner=planner,
        map_config=map_config,
        agent_config=agent_config,
        source_identity=identity,
    )
    with contextlib.suppress(Exception):
        state["retrieval_map"] = agent.document_map
    return agent


def map_routed_retrieval(prompt, retriever, history, llm=None, tokenizer=None):
    """Return bounded map-routed results, or None when raw retrieval must be used."""
    state = st.session_state
    settings = normalize_map_settings(state)
    if not settings["retrieval_map_enabled"]:
        return None
    if not retriever_supports_map(retriever):
        return None
    identity = state_map_source_identity(state) or map_source_identity(
        "local-session", state.get("index_generation")
    )
    try:
        agent = _build_map_agent(state, retriever, llm, tokenizer, settings, identity)
        results, trace = agent.retrieve(prompt, history=history)
    except Exception as err:
        logs.log.warning(
            "Map retrieval routing was unavailable: %s",
            logs.safe_log_exception(err),
        )
        return None
    return results, trace, agent


def _raw_retrieval(prompt, retriever, messages_state):
    query_candidates = retrieval_query_candidates(prompt, messages_state)
    nodes = retriever.retrieve(query_candidates[0])
    if not nodes and len(query_candidates) > 1:
        nodes = retriever.retrieve(query_candidates[1])
    return list(nodes or [])


def _baseline_retrieval(prompt, agent, history, limit):
    if agent is None or not limit:
        return []
    try:
        return agent.global_baseline(prompt, history=history, limit=limit)
    except Exception as err:
        logs.log.warning(
            "Global retrieval baseline was unavailable: %s",
            logs.safe_log_exception(err),
        )
        return []


def _record_retrieval_route(trace, planner=None, accounting=None, route_sink=None):
    route = annotate_planner_trace(trace, planner)
    if accounting:
        route["token_accounting"] = json_safe(accounting)
    return set_retrieval_route(route, route_sink=route_sink)


def context_chat(
    prompt: str,
    query_engine: RetrieverQueryEngine,
    evidence_sink: list | None = None,
    route_sink=None,
):
    _clear_rag_state()
    clear_retrieval_route(route_sink)
    try:
        system_prompt = st.session_state.get("system_prompt", "")
        llm = _session_llm()
        if llm is None:
            llm = create_llm(
                _active_chat_model(),
                _active_base_url(),
                _active_api_key(),
                system_prompt=system_prompt,
                set_global=False,
            )

        retriever = st.session_state.get("retriever")
        if retriever is None:
            retriever = getattr(query_engine, "_retriever", None)
        if retriever is None:
            clear_retrieval_route(route_sink)
            st.session_state["last_rag_no_result"] = False
            st.session_state["last_rag_question"] = None
            yield "⚠️ **No retriever available.** Please re-ingest your documents."
            return

        messages_state = st.session_state.get("messages", [])
        t0 = time.time()
        tokenizer = _resolve_tokenizer(llm)
        context_window = _llm_context_window(llm)
        output_tokens = _num_predict()

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
                    role=(
                        MessageRole.ASSISTANT
                        if role == "assistant"
                        else MessageRole.USER
                    ),
                    content=content,
                )
            )
        if history and _message_text(history[-1]) == str(prompt or "").strip():
            history = history[:-1]

        routed = map_routed_retrieval(prompt, retriever, history, llm, tokenizer)
        if routed is None:
            nodes = _raw_retrieval(prompt, retriever, messages_state)
            trace = None
            planner = None
            agent = None
        else:
            nodes, trace, agent = routed
            if isinstance(trace, dict) and agent is not None:
                trace["map_id"] = agent.document_map.map_id
                trace["map_source_identity"] = agent.document_map.source_identity
            planner = (
                agent.planner if isinstance(agent.planner, LlmMapPlanner) else None
            )
        nodes = [node for node in list(nodes) if _chunk_content(node).strip()]
        if not nodes:
            _clear_rag_state()
            clear_retrieval_route(route_sink)
            st.session_state["last_rag_no_result"] = True
            st.session_state["last_rag_question"] = prompt
            yield "I could not find this information in the documents."
            return

        plan = plan_rag_prompt(
            prompt,
            nodes,
            history,
            system_prompt,
            output_tokens=output_tokens,
            context_window=context_window,
            tokenizer=tokenizer,
        )
        selected_nodes = list(plan.get("selected_chunks") or [])
        if not selected_nodes:
            _clear_rag_state()
            clear_retrieval_route(route_sink)
            st.session_state["last_rag_no_result"] = True
            st.session_state["last_rag_question"] = prompt
            yield "I could not find this information in the documents."
            return

        evidence = set_rag_evidence(
            normalize_evidence(selected_nodes),
            evidence_sink=evidence_sink,
        )
        if trace is not None:
            try:
                settings = normalize_map_settings(st.session_state)
                accounting = measure_rag_token_accounting(
                    prompt,
                    plan,
                    tokenizer=tokenizer,
                    history=history,
                    system_prompt=system_prompt,
                    baseline_chunks=_baseline_retrieval(
                        prompt,
                        agent if settings["retrieval_map_measure_baseline"] else None,
                        history,
                        settings["retrieval_map_max_results"],
                    ),
                    output_tokens=output_tokens,
                    context_window=context_window,
                )
                _record_retrieval_route(
                    trace, planner=planner, accounting=accounting, route_sink=route_sink
                )
            except Exception as err:
                clear_retrieval_route(route_sink)
                logs.log.warning(
                    "Retrieval route metrics were unavailable: %s",
                    logs.safe_log_exception(err),
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
        clear_retrieval_route(route_sink)
        try:
            st.session_state["last_rag_no_result"] = False
            st.session_state["last_rag_question"] = None
        except Exception:
            pass
        logs.log.error("Document chat stream failed: %s", logs.safe_log_exception(err))
        if provider_kind() != OLLAMA:
            yield "⚠️ **Error generating response:** The configured provider could not complete the request."
        else:
            yield "⚠️ **Error generating response:** Ollama could not complete the request."
        return
