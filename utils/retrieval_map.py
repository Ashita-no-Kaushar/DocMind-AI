import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

from llama_index.core.schema import NodeWithScore

from utils.llama_index import (
    _bm25_expanded_tokens,
    _bm25_tokens,
    _rewrite_query,
    retrieval_query_candidates,
    sanitize_untrusted_context,
)
from utils.logs import redact_text

RETRIEVAL_MAP_VERSION = 1
DEFAULT_CHUNKS_PER_SECTION = 6
HARD_MAX_CHUNKS_PER_SECTION = 32
DEFAULT_MAX_SECTIONS = 128
HARD_MAX_SECTIONS = 1024
DEFAULT_MAX_NODES = 2048
HARD_MAX_NODES = 16384
DEFAULT_MAX_KEYWORDS = 12
HARD_MAX_KEYWORDS = 32
DEFAULT_MAX_SUMMARY_CHARS = 240
HARD_MAX_SUMMARY_CHARS = 512
DEFAULT_MAX_TITLE_CHARS = 80
HARD_MAX_TITLE_CHARS = 160
DEFAULT_MAX_SELECTED_SECTIONS = 4
HARD_MAX_SELECTED_SECTIONS = 16
DEFAULT_INITIAL_SECTIONS = 2
DEFAULT_NEIGHBOR_SECTIONS = 1
HARD_MAX_NEIGHBOR_SECTIONS = 4
DEFAULT_MAX_STEPS = 2
HARD_MAX_STEPS = 2
DEFAULT_MAX_RESULTS = 8
HARD_MAX_RESULTS = 50
DEFAULT_MAX_BASELINE_RESULTS = 8
DEFAULT_MAX_PLANNER_CARDS = 64
HARD_MAX_PLANNER_CARDS = 256
DEFAULT_REFINE_POOL = 12
HARD_MAX_REFINE_POOL = 64
HARD_MAX_REFINE_NODES = 8
HARD_MAX_REFINE_CHARS = 4000
HARD_MAX_PLANNER_OUTPUT_CHARS = 1024
HARD_MAX_PLANNER_OUTPUT_TOKENS = 256
HARD_MAX_PLANNER_QUERY_CHARS = 400
MAP_PLANNER_DETERMINISTIC = "deterministic"
MAP_PLANNER_LLM = "llm"
MAP_PLANNER_LLM_FALLBACK_DETERMINISTIC = "llm_fallback_deterministic"
MAP_PLANNER_MODES = (MAP_PLANNER_DETERMINISTIC, MAP_PLANNER_LLM)
MAP_PLANNER_OUTPUT_NOT_JSON = "planner_output_not_json"
MAP_PLANNER_OUTPUT_NOT_LIST = "planner_output_not_list"
MAP_PLANNER_CALL_FAILED = "planner_call_failed"
MAP_PLANNER_INSTRUCTION = (
    "You are the retrieval router for DocMind AI. Select the candidate document "
    "sections that should be read to answer the user query.\n"
    "Output rules:\n"
    "- Reply with a JSON array of section ids and nothing else.\n"
    "- No prose, no markdown, no code fences, no object keys, no trailing commas.\n"
    "- Use only ids that appear in the candidate sections.\n"
    "- Order ids from most to least relevant and return at most {limit} of them.\n"
    "- Return [] only when no candidate section can help.\n"
)
MAP_SETTING_DEFAULTS = {
    "retrieval_map_enabled": True,
    "retrieval_map_planner_mode": MAP_PLANNER_DETERMINISTIC,
    "retrieval_map_measure_baseline": False,
    "retrieval_map_chunks_per_section": DEFAULT_CHUNKS_PER_SECTION,
    "retrieval_map_target_sections": 0,
    "retrieval_map_max_selected_sections": DEFAULT_MAX_SELECTED_SECTIONS,
    "retrieval_map_initial_sections": DEFAULT_INITIAL_SECTIONS,
    "retrieval_map_neighbor_sections": DEFAULT_NEIGHBOR_SECTIONS,
    "retrieval_map_max_results": DEFAULT_MAX_RESULTS,
}
MAP_SETTING_RANGES = {
    "retrieval_map_chunks_per_section": (1, HARD_MAX_CHUNKS_PER_SECTION),
    "retrieval_map_target_sections": (0, 12),
    "retrieval_map_max_selected_sections": (1, HARD_MAX_SELECTED_SECTIONS),
    "retrieval_map_initial_sections": (1, HARD_MAX_SELECTED_SECTIONS),
    "retrieval_map_neighbor_sections": (0, HARD_MAX_NEIGHBOR_SECTIONS),
    "retrieval_map_max_results": (1, HARD_MAX_RESULTS),
}
MAP_BOOL_SETTINGS = frozenset(
    {"retrieval_map_enabled", "retrieval_map_measure_baseline"}
)
MAP_CHOICE_SETTINGS = {"retrieval_map_planner_mode": MAP_PLANNER_MODES}
HARD_MAX_TRACE_ACTIONS = 8
HARD_MAX_CACHE_ENTRIES = 8
HARD_MAX_CACHE_SIGNATURE_NODES = 20000
HARD_MAX_KEYWORD_VOCAB = 128
_MAP_CACHE_ATTRIBUTE = "_docmind_retrieval_map_cache"
_MAP_ATTRIBUTE = "retrieval_map"
_MAP_AGENT_ATTRIBUTE = "retrieval_map_agent"
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MAP_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "but",
        "for",
        "from",
        "has",
        "have",
        "into",
        "not",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "was",
        "were",
        "which",
        "with",
        "you",
        "your",
    }
)


def _bounded_int(value, low: int, high: int, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = int(fallback)
    return max(low, min(high, number))


def _bounded_float(value, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = float(fallback)
    if not math.isfinite(number):
        number = float(fallback)
    return max(low, min(high, number))


def _bounded_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(fallback)


def _bounded_choice(value, choices, fallback: str) -> str:
    if isinstance(value, str) and value.strip().lower() in choices:
        return value.strip().lower()
    return str(fallback)


def _clean_text(value, max_chars: int = 512) -> str:
    return " ".join(str(value or "").split())[: max(0, int(max_chars))]


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key)[:120]: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return _clean_text(value, 200)


def count_map_tokens(text: str, tokenizer=None) -> int:
    value = str(text or "")
    if not value:
        return 0
    counter = tokenizer
    if counter is None:
        counter = _WORD_RE
    if not callable(counter) and hasattr(counter, "encode"):
        counter = counter.encode
    try:
        result = counter(value) if callable(counter) else None
        if isinstance(result, bool):
            raise TypeError
        if isinstance(result, int):
            return max(0, result)
        if isinstance(result, float):
            if not math.isfinite(result):
                raise TypeError
            return max(0, int(result))
        if hasattr(result, "__len__") and not isinstance(
            result, (str, bytes, bytearray)
        ):
            return len(result)
    except Exception:
        pass
    return len(_WORD_RE.findall(value))


@dataclass(frozen=True, slots=True)
class RetrievalMapConfig:
    chunks_per_section: int = DEFAULT_CHUNKS_PER_SECTION
    target_sections_per_document: int = 0
    max_sections: int = DEFAULT_MAX_SECTIONS
    max_nodes: int = DEFAULT_MAX_NODES
    max_keywords: int = DEFAULT_MAX_KEYWORDS
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS
    max_title_chars: int = DEFAULT_MAX_TITLE_CHARS

    def __post_init__(self):
        object.__setattr__(
            self,
            "chunks_per_section",
            _bounded_int(
                self.chunks_per_section,
                1,
                HARD_MAX_CHUNKS_PER_SECTION,
                DEFAULT_CHUNKS_PER_SECTION,
            ),
        )
        object.__setattr__(
            self,
            "target_sections_per_document",
            _bounded_int(
                self.target_sections_per_document,
                0,
                HARD_MAX_SECTIONS,
                0,
            ),
        )
        object.__setattr__(
            self,
            "max_sections",
            _bounded_int(self.max_sections, 1, HARD_MAX_SECTIONS, DEFAULT_MAX_SECTIONS),
        )
        object.__setattr__(
            self,
            "max_nodes",
            _bounded_int(self.max_nodes, 1, HARD_MAX_NODES, DEFAULT_MAX_NODES),
        )
        object.__setattr__(
            self,
            "max_keywords",
            _bounded_int(self.max_keywords, 1, HARD_MAX_KEYWORDS, DEFAULT_MAX_KEYWORDS),
        )
        object.__setattr__(
            self,
            "max_summary_chars",
            _bounded_int(
                self.max_summary_chars,
                1,
                HARD_MAX_SUMMARY_CHARS,
                DEFAULT_MAX_SUMMARY_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "max_title_chars",
            _bounded_int(
                self.max_title_chars,
                8,
                HARD_MAX_TITLE_CHARS,
                DEFAULT_MAX_TITLE_CHARS,
            ),
        )

    def to_dict(self) -> dict:
        return json_safe(asdict(self))

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class MapAgentConfig:
    max_selected_sections: int = DEFAULT_MAX_SELECTED_SECTIONS
    initial_sections: int = DEFAULT_INITIAL_SECTIONS
    neighbor_sections: int = DEFAULT_NEIGHBOR_SECTIONS
    max_steps: int = DEFAULT_MAX_STEPS
    max_results: int = DEFAULT_MAX_RESULTS
    max_baseline_results: int = DEFAULT_MAX_BASELINE_RESULTS
    max_planner_cards: int = DEFAULT_MAX_PLANNER_CARDS
    refine_candidates: bool = False
    refine_pool: int = DEFAULT_REFINE_POOL
    weak_score: float = 0.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "max_selected_sections",
            _bounded_int(
                self.max_selected_sections,
                1,
                HARD_MAX_SELECTED_SECTIONS,
                DEFAULT_MAX_SELECTED_SECTIONS,
            ),
        )
        object.__setattr__(
            self,
            "initial_sections",
            _bounded_int(
                self.initial_sections,
                1,
                self.max_selected_sections,
                DEFAULT_INITIAL_SECTIONS,
            ),
        )
        object.__setattr__(
            self,
            "neighbor_sections",
            _bounded_int(
                self.neighbor_sections,
                0,
                HARD_MAX_NEIGHBOR_SECTIONS,
                DEFAULT_NEIGHBOR_SECTIONS,
            ),
        )
        object.__setattr__(
            self,
            "max_steps",
            _bounded_int(self.max_steps, 1, HARD_MAX_STEPS, DEFAULT_MAX_STEPS),
        )
        object.__setattr__(
            self,
            "max_results",
            _bounded_int(self.max_results, 1, HARD_MAX_RESULTS, DEFAULT_MAX_RESULTS),
        )
        object.__setattr__(
            self,
            "max_baseline_results",
            _bounded_int(
                self.max_baseline_results,
                1,
                HARD_MAX_RESULTS,
                DEFAULT_MAX_BASELINE_RESULTS,
            ),
        )
        object.__setattr__(
            self,
            "max_planner_cards",
            _bounded_int(
                self.max_planner_cards,
                1,
                HARD_MAX_PLANNER_CARDS,
                DEFAULT_MAX_PLANNER_CARDS,
            ),
        )
        object.__setattr__(
            self,
            "refine_candidates",
            _bounded_bool(self.refine_candidates, False),
        )
        object.__setattr__(
            self,
            "refine_pool",
            _bounded_int(
                self.refine_pool,
                1,
                HARD_MAX_REFINE_POOL,
                DEFAULT_REFINE_POOL,
            ),
        )
        object.__setattr__(
            self,
            "weak_score",
            _bounded_float(self.weak_score, 0.0, 1.0, 0.0),
        )

    def to_dict(self) -> dict:
        return json_safe(asdict(self))

    as_dict = to_dict


def normalize_map_settings(settings=None) -> dict:
    """Return bounded, secret-free retrieval map settings for a session mapping."""
    source = settings if isinstance(settings, Mapping) else {}
    normalized = {}
    for key, default in MAP_SETTING_DEFAULTS.items():
        value = source.get(key, default)
        if key in MAP_BOOL_SETTINGS:
            normalized[key] = _bounded_bool(value, default)
        elif key in MAP_CHOICE_SETTINGS:
            normalized[key] = _bounded_choice(value, MAP_CHOICE_SETTINGS[key], default)
        else:
            bounds = MAP_SETTING_RANGES.get(key)
            if bounds is None:
                normalized[key] = _bounded_int(value, 0, 0, 0)
            else:
                normalized[key] = _bounded_int(value, bounds[0], bounds[1], default)
    return normalized


def map_agent_config(settings=None) -> MapAgentConfig:
    """Return the bounded map agent configuration for the current settings."""
    normalized = normalize_map_settings(settings)
    selected = normalized["retrieval_map_max_selected_sections"]
    results = normalized["retrieval_map_max_results"]
    return MapAgentConfig(
        max_selected_sections=selected,
        initial_sections=min(normalized["retrieval_map_initial_sections"], selected),
        neighbor_sections=normalized["retrieval_map_neighbor_sections"],
        max_steps=DEFAULT_MAX_STEPS,
        max_results=results,
        max_baseline_results=results,
        max_planner_cards=DEFAULT_MAX_PLANNER_CARDS,
    )


def map_document_config(settings=None) -> RetrievalMapConfig:
    """Return the bounded document map configuration for the current settings."""
    normalized = normalize_map_settings(settings)
    return RetrievalMapConfig(
        chunks_per_section=normalized["retrieval_map_chunks_per_section"],
        target_sections_per_document=normalized["retrieval_map_target_sections"],
    )


def map_source_identity(source=None, generation=None) -> str | None:
    """Return the stable source id and generation key used for map identity."""
    if isinstance(source, Mapping):
        identity = source.get("id") or source.get("source_id")
        if generation is None:
            generation = source.get("index_generation")
    else:
        identity = source
    if identity in (None, ""):
        return None
    try:
        number = int(generation or 0)
    except (TypeError, ValueError, OverflowError):
        number = 0
    return f"{_clean_text(identity, 200)}:{max(0, number)}"


def state_map_source_identity(state=None) -> str | None:
    """Return the map identity for the active source and index generation."""
    if not isinstance(state, Mapping):
        return None
    source = state.get("active_source")
    if not isinstance(source, Mapping):
        return None
    generation = state.get("index_generation", source.get("index_generation"))
    return map_source_identity(source, generation)


def retriever_supports_map(retriever) -> bool:
    """Return whether a retriever exposes the corpus the map builder requires."""
    if retriever is None or not callable(getattr(retriever, "retrieve", None)):
        return False
    docstore = getattr(retriever, "docstore", None)
    if not callable(getattr(docstore, "get_node", None)):
        return False
    corpus = _retriever_corpus(retriever)
    return len(corpus) > 0


@dataclass(frozen=True, slots=True)
class MapSection:
    section_id: str
    source_id: str
    source: str
    source_identity: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    member_node_ids: tuple[str, ...]
    chunk_count: int
    token_count: int
    sequence: int

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "source_id": self.source_id,
            "source": self.source,
            "source_identity": self.source_identity,
            "title": self.title,
            "summary": self.summary,
            "keywords": list(self.keywords),
            "member_node_ids": list(self.member_node_ids),
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "sequence": self.sequence,
        }

    as_dict = to_dict

    def to_card(self, score: float = 0.0) -> dict:
        return {
            "id": self.section_id,
            "title": self.title,
            "keywords": list(self.keywords[:HARD_MAX_KEYWORDS]),
            "summary": self.summary[:HARD_MAX_SUMMARY_CHARS],
            "score": _bounded_float(score, 0.0, 1_000_000.0, 0.0),
        }


@dataclass(frozen=True, slots=True)
class MapNode:
    node_id: str
    kind: str
    label: str
    source_id: str

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "source_id": self.source_id,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class MapEdge:
    source_id: str
    target_id: str
    kind: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class DocumentMap:
    map_id: str
    version: int
    source_identity: str
    sections: tuple[MapSection, ...]
    nodes: tuple[MapNode, ...]
    edges: tuple[MapEdge, ...]
    input_node_count: int
    mapped_node_count: int
    skipped_node_count: int
    truncated: bool
    config: RetrievalMapConfig

    def section(self, section_id: str) -> MapSection | None:
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None

    def member_node_ids(self) -> tuple[str, ...]:
        return tuple(
            node_id for section in self.sections for node_id in section.member_node_ids
        )

    def to_dict(self) -> dict:
        return {
            "map_id": self.map_id,
            "version": self.version,
            "source_identity": self.source_identity,
            "sections": [section.to_dict() for section in self.sections],
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "stats": {
                "input_node_count": self.input_node_count,
                "mapped_node_count": self.mapped_node_count,
                "skipped_node_count": self.skipped_node_count,
                "section_count": len(self.sections),
                "truncated": self.truncated,
            },
            "config": self.config.to_dict(),
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class RouteAction:
    step: int
    action: str
    reason: str
    allowed_section_ids: tuple[str, ...] = ()
    attempted_queries: int = 0
    result_count: int = 0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "reason": self.reason,
            "allowed_section_ids": list(self.allowed_section_ids),
            "attempted_queries": self.attempted_queries,
            "result_count": self.result_count,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    planner_mode: str
    planner_card_token_estimate: int
    selected_evidence_tokens: int
    result_count: int
    token_basis: str = "selected_evidence_text"
    global_baseline_tokens: int | None = None
    selected_minus_baseline_tokens: int | None = None
    global_baseline_result_count: int | None = None

    def to_dict(self) -> dict:
        return {
            "planner_mode": self.planner_mode,
            "planner_card_token_estimate": self.planner_card_token_estimate,
            "selected_evidence_tokens": self.selected_evidence_tokens,
            "result_count": self.result_count,
            "token_basis": self.token_basis,
            "global_baseline_tokens": self.global_baseline_tokens,
            "selected_minus_baseline_tokens": self.selected_minus_baseline_tokens,
            "global_baseline_result_count": self.global_baseline_result_count,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class RouteTrace:
    considered_section_ids: tuple[str, ...]
    selected_section_ids: tuple[str, ...]
    planned_section_ids: tuple[str, ...]
    actions: tuple[RouteAction, ...]
    step_count: int
    reflection_flag: bool
    planner_mode: str
    planner_card_token_estimate: int
    metrics: RetrievalMetrics
    result_metadata: tuple[dict, ...]
    result_metadata_truncated: bool

    def to_dict(self) -> dict:
        return {
            "considered_section_ids": list(self.considered_section_ids),
            "selected_section_ids": list(self.selected_section_ids),
            "planned_section_ids": list(self.planned_section_ids),
            "actions": [action.to_dict() for action in self.actions],
            "step_count": self.step_count,
            "reflection_flag": self.reflection_flag,
            "reflection": {
                "performed": True,
                "triggered": self.reflection_flag,
            },
            "planner_mode": self.planner_mode,
            "planner_card_token_estimate": self.planner_card_token_estimate,
            "metrics": self.metrics.to_dict(),
            "result_metadata": json_safe(list(self.result_metadata)),
            "result_metadata_truncated": self.result_metadata_truncated,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class _ChunkDescriptor:
    node_id: str
    source_key: str
    source_display: str
    source_identity: str
    heading: str
    token_count: int
    excerpt: str
    terms: tuple[tuple[str, int], ...]


def _digest(payload) -> str:
    serialized = json.dumps(
        json_safe(payload), sort_keys=True, ensure_ascii=False, default=str
    )
    return sha256(serialized.encode("utf-8", errors="replace")).hexdigest()


def _node_metadata(node) -> Mapping:
    metadata = getattr(node, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _node_content(node) -> str:
    if hasattr(node, "get_content"):
        return str(node.get_content() or "")
    return str(getattr(node, "text", "") or "")


def _node_identifier(raw, node) -> str | None:
    candidates = [node, raw]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            value = candidate.get("node_id") or candidate.get("id")
            if value not in (None, ""):
                return _clean_text(value, 200)
        value = getattr(candidate, "node_id", None)
        if value not in (None, ""):
            return _clean_text(value, 200)
    if isinstance(raw, (str, int)) and not isinstance(raw, bool):
        return _clean_text(raw, 200)
    return None


def _basename(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    if "://" in value and not name:
        return value.split("://", 1)[-1].split("/", 1)[0]
    return name


def _human_title(value: str, max_chars: int) -> str:
    name = _basename(_clean_text(value, 400))
    name = os.path.splitext(name)[0]
    title = " ".join(name.replace("_", " ").replace("-", " ").split())[:max_chars]
    return title.title() or _clean_text(value, max_chars) or "Document"


def _source_details(node) -> tuple[str, str, str, str]:
    metadata = _node_metadata(node)
    group_raw = None
    for key in (
        "file_name",
        "file_path",
        "source",
        "url",
        "title",
        "document_id",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            group_raw = value
            break
    group_key = _clean_text(group_raw, 512) or "document"
    display = _basename(group_key) if group_key != "document" else "document"
    display = _clean_text(display, 160) or "document"
    identity = _clean_text(metadata.get("source_id"), 200)
    generation = _clean_text(metadata.get("index_generation"), 40)
    if identity and generation:
        identity = f"{identity}:{generation}"
    heading = ""
    for key in ("section_title", "heading", "title"):
        value = _clean_text(metadata.get(key), 120)
        if value:
            heading = value
            break
    return group_key, display, identity, heading


def _map_terms(text: str):
    for term in _bm25_tokens(str(text or "")):
        if len(term) > 1 and term not in _MAP_STOPWORDS:
            yield term


def _add_terms(aggregate: dict[str, int], terms, limit: int) -> None:
    for term, amount in terms:
        if term in aggregate:
            aggregate[term] += max(1, int(amount))
        elif len(aggregate) < limit:
            aggregate[term] = max(1, int(amount))
        else:
            weakest = min(aggregate, key=lambda item: (aggregate[item], item))
            if aggregate[weakest] < max(1, int(amount)) or (
                aggregate[weakest] == max(1, int(amount)) and term < weakest
            ):
                aggregate[term] = aggregate.pop(weakest)


def _bounded_excerpt(text: str, limit: int, title: str = "") -> str:
    """Return a length-bounded excerpt that covers as much of the text as fits.

    Taking only the leading sentence made every card describe how a chunk starts
    rather than what it contains, so a discriminative term late in a chunk was
    invisible to the map scorer. The budget is still a hard bound; the excerpt now
    simply spends it on as many whole sentences as will fit.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    sentences = [part.strip() for part in _SENTENCE_RE.split(cleaned) if part.strip()]
    title_key = " ".join(title.lower().split())
    budget = max(1, int(limit))
    parts: list[str] = []
    used = 0
    for sentence in sentences[:4]:
        if title_key and sentence.lower().rstrip(".") == title_key.rstrip("."):
            continue
        addition = len(sentence) + (1 if parts else 0)
        if used + addition > budget:
            break
        parts.append(sentence)
        used += addition
    if parts:
        return " ".join(parts)
    fallback = sentences[0] if sentences else cleaned
    if len(fallback) <= budget:
        return fallback
    clipped = fallback[:budget]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip()


def _merge_section_summary(excerpts: Sequence[str], limit: int) -> str:
    parts = []
    used = 0
    for excerpt in excerpts:
        value = _clean_text(excerpt, limit)
        if not value:
            continue
        remaining = limit - used
        if remaining <= 0:
            break
        if len(value) > remaining:
            value = value[:remaining]
            if " " in value:
                value = value.rsplit(" ", 1)[0]
            value = value.strip()
        if not value:
            break
        parts.append(value)
        used += len(value) + 1
    return " ".join(parts)


def _top_keywords(aggregate: Mapping[str, int], limit: int) -> tuple[str, ...]:
    ordered = sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))
    return tuple(term for term, _count in ordered[:limit])


def _retriever_corpus(retriever) -> list:
    corpus = getattr(retriever, "corpus", None)
    if isinstance(corpus, Sequence) and not isinstance(corpus, (str, bytes)):
        return list(corpus)
    if corpus is None:
        return []
    try:
        return list(corpus)
    except TypeError:
        return []


def _section_step(run_length: int, map_config: RetrievalMapConfig) -> int:
    """Return how many consecutive chunks belong to one map section.

    A fixed ``chunks_per_section`` degenerates on real corpora: a short document
    collapses into a single section, so the agent can only choose whole
    documents and cannot aim inside one, while a long document is cut into very
    coarse blocks. When ``target_sections_per_document`` is set, the step is
    derived per document so every document is divided into roughly that many
    sections, bounded to keep sections non-empty and sections-per-document sane.
    """
    fixed = _bounded_int(
        map_config.chunks_per_section,
        1,
        HARD_MAX_CHUNKS_PER_SECTION,
        DEFAULT_CHUNKS_PER_SECTION,
    )
    target = _bounded_int(
        map_config.target_sections_per_document, 0, HARD_MAX_SECTIONS, 0
    )
    if target <= 0 or run_length <= 0:
        return fixed
    adaptive = -(-run_length // target)
    return _bounded_int(adaptive, 1, HARD_MAX_CHUNKS_PER_SECTION, fixed)


def _configured_map(
    config: RetrievalMapConfig | None = None,
    chunks_per_section=None,
    max_sections=None,
    max_nodes=None,
    max_keywords=None,
    max_summary_chars=None,
    max_title_chars=None,
) -> RetrievalMapConfig:
    base = config or RetrievalMapConfig()
    overrides = {
        "chunks_per_section": chunks_per_section,
        "max_sections": max_sections,
        "max_nodes": max_nodes,
        "max_keywords": max_keywords,
        "max_summary_chars": max_summary_chars,
        "max_title_chars": max_title_chars,
    }
    values = base.to_dict()
    values.update({key: value for key, value in overrides.items() if value is not None})
    return RetrievalMapConfig(**values)


def build_document_map(
    retriever,
    *,
    tokenizer=None,
    config: RetrievalMapConfig | None = None,
    source_identity=None,
    chunks_per_section=None,
    max_sections=None,
    max_nodes=None,
    max_keywords=None,
    max_summary_chars=None,
    max_title_chars=None,
) -> DocumentMap:
    map_config = _configured_map(
        config,
        chunks_per_section,
        max_sections,
        max_nodes,
        max_keywords,
        max_summary_chars,
        max_title_chars,
    )
    corpus = _retriever_corpus(retriever)
    descriptors: list[_ChunkDescriptor] = []
    skipped = 0
    truncated = False
    vocabulary_limit = min(
        HARD_MAX_KEYWORD_VOCAB,
        max(map_config.max_keywords * 3, map_config.max_keywords + 4),
    )
    for raw in corpus:
        if len(descriptors) >= map_config.max_nodes:
            truncated = True
            break
        node = None
        try:
            if isinstance(raw, (str, int)) and not isinstance(raw, bool):
                node = retriever.docstore.get_node(raw)
            else:
                node = raw
        except Exception:
            node = None
        if node is None:
            skipped += 1
            continue
        node_id = _node_identifier(raw, node)
        content = _node_content(node)
        if not node_id or not content.strip():
            skipped += 1
            continue
        group_key, display, identity, heading = _source_details(node)
        aggregate: dict[str, int] = {}
        for term in _map_terms(content):
            if term in aggregate:
                aggregate[term] += 1
            elif len(aggregate) < vocabulary_limit:
                aggregate[term] = 1
            else:
                weakest = min(aggregate, key=lambda item: (aggregate[item], item))
                if aggregate[weakest] < 1 or term < weakest:
                    aggregate[weakest] = term
                    aggregate[term] = 1
        descriptors.append(
            _ChunkDescriptor(
                node_id=node_id,
                source_key=group_key,
                source_display=display,
                source_identity=identity,
                heading=heading,
                token_count=count_map_tokens(content, tokenizer),
                excerpt=_bounded_excerpt(
                    content,
                    map_config.max_summary_chars,
                    _human_title(display, map_config.max_title_chars),
                ),
                terms=tuple(sorted(aggregate.items())),
            )
        )

    sections: list[MapSection] = []
    nodes: list[MapNode] = []
    edges: list[MapEdge] = []
    seen_source_nodes: set[str] = set()
    last_section_by_source: dict[str, str] = {}
    sequences: dict[str, int] = {}
    source_identities: dict[str, str] = {}
    source_titles: dict[str, str] = {}
    index = 0
    while index < len(descriptors) and len(sections) < map_config.max_sections:
        source_key = descriptors[index].source_key
        run_end = index + 1
        while (
            run_end < len(descriptors) and descriptors[run_end].source_key == source_key
        ):
            run_end += 1
        sequence = sequences.get(source_key, 0) + 1
        sequences[source_key] = sequence
        run = descriptors[index:run_end]
        index = run_end
        step = _section_step(len(run), map_config)
        for start in range(0, len(run), step):
            if len(sections) >= map_config.max_sections:
                truncated = True
                break
            members = run[start : start + step]
            member_ids = tuple(member.node_id for member in members)
            if not member_ids:
                continue
            source_display = members[0].source_display
            source_title = source_titles.get(source_key) or _human_title(
                source_display, map_config.max_title_chars
            )
            source_titles[source_key] = source_title
            section_source_identity = (
                members[0].source_identity or _digest({"source": source_key})[:20]
            )
            source_identities.setdefault(source_key, section_source_identity)
            source_id = "map_doc_" + _digest({"source": source_key})[:20]
            section_id = (
                "map_sec_"
                + _digest(
                    {
                        "source": source_key,
                        "sequence": sequence,
                        "members": list(member_ids),
                    }
                )[:24]
            )
            aggregate: dict[str, int] = {}
            for member in members:
                _add_terms(aggregate, member.terms, vocabulary_limit)
            for term in _map_terms(source_title):
                if term in aggregate:
                    aggregate[term] += 2
                elif len(aggregate) < vocabulary_limit:
                    aggregate[term] = 2
            summary = _merge_section_summary(
                [member.excerpt for member in members],
                map_config.max_summary_chars,
            )
            heading = next(
                (member.heading for member in members if member.heading),
                "",
            )
            title_parts = [source_title]
            if heading and heading.casefold() != source_title.casefold():
                title_parts.append(heading)
            title_parts.append(f"Section {sequence}")
            title = " - ".join(title_parts)[: map_config.max_title_chars].strip()
            section = MapSection(
                section_id=section_id,
                source_id=source_id,
                source=source_display,
                source_identity=section_source_identity,
                title=title,
                summary=summary,
                keywords=_top_keywords(aggregate, map_config.max_keywords),
                member_node_ids=member_ids,
                chunk_count=len(member_ids),
                token_count=sum(member.token_count for member in members),
                sequence=sequence,
            )
            sections.append(section)
            if source_id not in seen_source_nodes:
                seen_source_nodes.add(source_id)
                nodes.append(
                    MapNode(
                        node_id=source_id,
                        kind="document",
                        label=source_title,
                        source_id=source_id,
                    )
                )
            nodes.append(
                MapNode(
                    node_id=section_id,
                    kind="section",
                    label=title,
                    source_id=source_id,
                )
            )
            edges.append(
                MapEdge(source_id=source_id, target_id=section_id, kind="contains")
            )
            previous = last_section_by_source.get(source_key)
            if previous:
                edges.append(
                    MapEdge(
                        source_id=previous,
                        target_id=section_id,
                        kind="next",
                    )
                )
            last_section_by_source[source_key] = section_id

    if len(sections) >= map_config.max_sections and index < len(descriptors):
        truncated = True

    explicit_identity = _clean_text(source_identity, 200)
    if explicit_identity:
        map_source_identity = explicit_identity
    elif source_identities:
        map_source_identity = _digest(
            {"sources": [source_identities[key] for key in sorted(source_identities)]}
        )[:24]
    else:
        map_source_identity = "unknown-source"
    map_id = (
        "map_"
        + _digest(
            {
                "version": RETRIEVAL_MAP_VERSION,
                "source_identity": map_source_identity,
                "sections": [section.section_id for section in sections],
            }
        )[:24]
    )
    return DocumentMap(
        map_id=map_id,
        version=RETRIEVAL_MAP_VERSION,
        source_identity=map_source_identity,
        sections=tuple(sections),
        nodes=tuple(nodes),
        edges=tuple(edges),
        input_node_count=len(corpus),
        mapped_node_count=sum(section.chunk_count for section in sections),
        skipped_node_count=skipped,
        truncated=truncated,
        config=map_config,
    )


def _card_terms(card: Mapping) -> tuple[set[str], set[str], set[str]]:
    title = set(_map_terms(card.get("title", "")))
    keywords = {
        str(term)
        for term in card.get("keywords", [])
        if str(term) not in _MAP_STOPWORDS
    }
    summary = set(_map_terms(card.get("summary", "")))
    return title, keywords, summary


def _corpus_term_stats(term_sets: Sequence[set]) -> tuple[int, dict]:
    """Return document count and term frequencies for a corpus."""
    frequencies: dict[str, int] = {}
    for terms in term_sets:
        for term in terms:
            frequencies[term] = frequencies.get(term, 0) + 1
    return max(1, len(term_sets)), frequencies


def _score_term_sets(
    entries: Sequence[tuple],
    queries: Sequence[str],
    stats: tuple[int, dict] | None = None,
) -> dict[str, float]:
    """Score (key, title, keywords, summary) term sets against the queries.

    This is the single scoring implementation used both for the map cards and for
    the full-text refinement pass, so the two stages cannot drift apart. A caller
    that scores only a shortlist must pass corpus-wide ``stats`` for the document
    count and term frequencies, otherwise term rarity is measured on the subset
    and the ranking is meaningless. The length normalisation is always derived
    from the entries actually being scored, so swapping a card summary for full
    text cannot silently rescale every score.
    """
    scores: dict[str, float] = dict.fromkeys((str(entry[0]) for entry in entries), 0.0)
    prepared = []
    for query in queries:
        rewritten = _rewrite_query(query)
        original = set(_bm25_tokens(rewritten))
        expanded = set(_bm25_expanded_tokens(_bm25_tokens(rewritten)))
        prepared.append((original, expanded))
    if not prepared or not entries:
        return dict.fromkeys(scores, 0.0)
    compact = [
        (str(key), set(title), set(keywords), set(summary))
        for key, title, keywords, summary in entries
    ]
    if stats is None:
        stats = _corpus_term_stats(
            [title | keywords | summary for _k, title, keywords, summary in compact]
        )
    document_count, corpus_frequencies = stats
    average_length = max(
        1.0,
        sum(
            max(1, len(title | keywords | summary))
            for _key, title, keywords, summary in compact
        )
        / max(1, len(compact)),
    )
    for original, expanded in prepared:
        document_frequency = {
            term: corpus_frequencies.get(term, 0) for term in original | expanded
        }
        for key, title, keywords, summary in compact:
            available = title | keywords | summary
            score = 0.0
            for term in original:
                if term not in available:
                    continue
                frequency = document_count - document_frequency.get(term, 0) + 0.5
                idf = math.log(
                    1.0 + (frequency / (document_frequency.get(term, 0) + 0.5))
                )
                term_frequency = 1.0
                if term in title:
                    term_frequency += 2.0
                if term in keywords:
                    term_frequency += 1.0
                if term in summary:
                    term_frequency += 0.25
                length = max(1, len(available))
                denominator = term_frequency + 1.2 * (
                    1.0 - 0.5 + 0.5 * (length / average_length)
                )
                score += idf * ((term_frequency * 2.2) / denominator)
                if term in title:
                    score += 1.5 * idf
                if term in keywords:
                    score += 0.75 * idf
            if original:
                coverage = len(original & available) / len(original)
                score *= 0.5 + coverage
            score += 0.15 * len((expanded - original) & available)
            scores[key] = max(scores[key], score)
    return {
        key: round(_bounded_float(score, 0.0, 1_000_000.0, 0.0), 6)
        for key, score in scores.items()
    }


def _score_cards(sections: Sequence[MapSection], queries: Sequence[str]) -> list[dict]:
    cards = [section.to_card(0.0) for section in sections]
    entries = []
    for card, section in zip(cards, sections, strict=False):
        title, keywords, summary = _card_terms(card)
        entries.append((section.section_id, title, keywords, summary))
    scores = _score_term_sets(entries, queries)
    for card, section in zip(cards, sections, strict=False):
        card["score"] = scores.get(section.section_id, 0.0)
    return cards


def _result_node(result):
    node = getattr(result, "node", None)
    if node is not None:
        return node
    return result


def _result_node_id(result) -> str:
    node = _result_node(result)
    value = getattr(node, "node_id", None)
    if value in (None, ""):
        value = getattr(result, "node_id", None)
    return _clean_text(value, 200)


def _result_content(result) -> str:
    return _node_content(_result_node(result))


def _result_score(result) -> float:
    try:
        score = float(getattr(result, "score", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _normalized_results(results, limit: int) -> list:
    normalized = []
    seen: set[str] = set()
    for result in list(results or [])[: max(0, int(limit))]:
        node = _result_node(result)
        if isinstance(result, NodeWithScore):
            candidate = result
        else:
            try:
                candidate = NodeWithScore(node=node, score=_result_score(result))
            except Exception:
                candidate = result
        node_id = _result_node_id(candidate)
        if node_id and node_id in seen:
            continue
        if node_id:
            seen.add(node_id)
        normalized.append(candidate)
    return normalized


def _planner_cards(cards: Sequence[dict], limit: int) -> list[dict]:
    selected = [
        {
            "id": str(card["id"]),
            "title": str(card["title"]),
            "keywords": [str(term) for term in card["keywords"]],
            "summary": str(card["summary"]),
            "score": float(card["score"]),
        }
        for card in cards[: max(1, int(limit))]
    ]
    return selected


def _deterministic_plan(cards: Sequence[dict], limit: int) -> list[str]:
    ordered = sorted(
        range(len(cards)),
        key=lambda index: (-float(cards[index]["score"]), index),
    )
    return [str(cards[index]["id"]) for index in ordered[: max(1, int(limit))]]


def _validated_planner_ids(value, known_ids: set[str], limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError
    selected: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in known_ids:
            raise ValueError
        if item in selected:
            continue
        selected.append(item)
    return selected[: max(1, int(limit))]


def _result_metadata(results, tokenizer, limit: int) -> tuple[tuple[dict, ...], bool]:
    entries = []
    for result in list(results or []):
        node = _result_node(result)
        metadata = _node_metadata(node)
        source = _clean_text(
            metadata.get("file_name")
            or metadata.get("file_path")
            or metadata.get("source")
            or metadata.get("url")
            or "document",
            200,
        )
        entries.append(
            {
                "node_id": _result_node_id(result),
                "source": source,
                "score": round(_result_score(result), 6),
                "token_count": count_map_tokens(_result_content(result), tokenizer),
            }
        )
    bounded = entries[: max(0, int(limit))]
    return tuple(bounded), len(entries) > len(bounded)


def _completion_text(response) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("text", "content", "response", "message"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    for attribute in ("text", "content"):
        value = getattr(response, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


class _PlannerOutputError(ValueError):
    pass


def _planner_json_payload(text: str):
    candidates = [_clean_text(text, HARD_MAX_PLANNER_OUTPUT_CHARS)]
    start = candidates[0].find("[")
    end = candidates[0].rfind("]")
    if start >= 0 and end > start:
        candidates.append(candidates[0][start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (TypeError, ValueError):
            continue
    raise _PlannerOutputError(MAP_PLANNER_OUTPUT_NOT_JSON)


def _planner_failure_reason(error) -> str:
    if isinstance(error, _PlannerOutputError):
        return str(error)
    return _clean_text(f"{MAP_PLANNER_CALL_FAILED}: {redact_text(error)}", 200)


@dataclass
class LlmMapPlannerStatus:
    calls: int = 0
    mode: str = MAP_PLANNER_LLM
    card_token_estimate: int = 0
    output_token_estimate: int = 0
    output_truncated: bool = False
    output_text: str = ""
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "planner_mode": self.mode,
            "planner_card_token_estimate": max(0, int(self.card_token_estimate)),
            "planner_output_token_estimate": max(
                0, min(int(self.output_token_estimate), HARD_MAX_PLANNER_OUTPUT_TOKENS)
            ),
            "planner_output_truncated": bool(self.output_truncated),
            "planner_failure_reason": self.failure_reason,
        }


class LlmMapPlanner:
    """Route section selection with the active LlamaIndex LLM complete() API."""

    def __init__(
        self,
        llm,
        *,
        tokenizer=None,
        max_selected_sections=None,
        max_planner_cards=None,
    ):
        if llm is None or not callable(getattr(llm, "complete", None)):
            raise ValueError("An LLM exposing complete() is required for planning.")
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_selected_sections = _bounded_int(
            max_selected_sections,
            1,
            HARD_MAX_SELECTED_SECTIONS,
            DEFAULT_MAX_SELECTED_SECTIONS,
        )
        self.max_planner_cards = _bounded_int(
            max_planner_cards,
            1,
            HARD_MAX_PLANNER_CARDS,
            DEFAULT_MAX_PLANNER_CARDS,
        )
        self.status = LlmMapPlannerStatus()

    def _payload(self, query: str, cards: Sequence[Mapping]) -> str:
        sections = []
        for card in list(cards or [])[: self.max_planner_cards]:
            sections.append(
                {
                    "id": _clean_text(card.get("id", ""), 200),
                    "title": sanitize_untrusted_context(
                        _clean_text(card.get("title", ""), HARD_MAX_TITLE_CHARS)
                    ),
                    "keywords": [
                        sanitize_untrusted_context(_clean_text(term, 40))
                        for term in list(card.get("keywords") or [])[:HARD_MAX_KEYWORDS]
                    ],
                    "summary": sanitize_untrusted_context(
                        _clean_text(card.get("summary", ""), HARD_MAX_SUMMARY_CHARS)
                    ),
                    "score": _bounded_float(
                        card.get("score", 0.0), 0.0, 1_000_000.0, 0.0
                    ),
                }
            )
        request = json.dumps(
            {
                "query": sanitize_untrusted_context(
                    _clean_text(query, HARD_MAX_PLANNER_QUERY_CHARS)
                ),
                "candidate_sections": sections,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        instruction = MAP_PLANNER_INSTRUCTION.format(limit=self.max_selected_sections)
        return f"{instruction}\n{request}\n"

    def _record_output(self, text: str) -> None:
        bounded = _clean_text(text, HARD_MAX_PLANNER_OUTPUT_CHARS)
        self.status.output_text = bounded
        self.status.output_truncated = len(str(text or "")) > len(bounded)
        self.status.output_token_estimate = min(
            count_map_tokens(bounded, self.tokenizer), HARD_MAX_PLANNER_OUTPUT_TOKENS
        )

    def __call__(self, query: str, cards: Sequence[Mapping]):
        payload = self._payload(query, cards)
        self.status.calls += 1
        self.status.card_token_estimate = count_map_tokens(payload, self.tokenizer)
        try:
            response = self.llm.complete(payload)
            self._record_output(_completion_text(response))
            parsed = _planner_json_payload(self.status.output_text)
            if not isinstance(parsed, (list, tuple)):
                raise _PlannerOutputError(MAP_PLANNER_OUTPUT_NOT_LIST)
        except Exception as err:
            self.status.mode = MAP_PLANNER_LLM_FALLBACK_DETERMINISTIC
            self.status.failure_reason = _planner_failure_reason(err)
            raise
        finally:
            self.status.output_text = ""
        self.status.mode = MAP_PLANNER_LLM
        self.status.failure_reason = None
        return list(parsed)


def _plan_reason(payload: Mapping):
    for action in payload.get("actions") or []:
        if isinstance(action, Mapping) and action.get("action") == "plan":
            return _clean_text(action.get("reason"), 120) or None
    return None


def annotate_planner_trace(trace, planner=None) -> dict:
    """Return a JSON-safe route trace annotated with optional LLM planner state."""
    payload = json_safe(trace) if isinstance(trace, Mapping) else {}
    if not isinstance(planner, LlmMapPlanner) or not planner.status.calls:
        return payload
    status = planner.status
    core_mode = str(payload.get("planner_mode") or "")
    if core_mode == "injected":
        mode = MAP_PLANNER_LLM
    elif core_mode == "injected_fallback_deterministic":
        mode = MAP_PLANNER_LLM_FALLBACK_DETERMINISTIC
    else:
        mode = core_mode or status.mode
    failure_reason = status.failure_reason
    if failure_reason is None and mode == MAP_PLANNER_LLM_FALLBACK_DETERMINISTIC:
        failure_reason = _plan_reason(payload)
    recorded = status.to_dict()
    recorded["planner_mode"] = mode
    recorded["planner_failure_reason"] = failure_reason
    payload.update(recorded)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        metrics["planner_mode"] = mode
    return payload


class MapRetrievalAgent:
    def __init__(
        self,
        retriever,
        document_map: DocumentMap | None = None,
        *,
        tokenizer=None,
        planner: Callable[[str, list[dict]], Sequence[str]] | None = None,
        map_config: RetrievalMapConfig | None = None,
        agent_config: MapAgentConfig | None = None,
        source_identity=None,
        max_selected_sections=None,
        initial_sections=None,
        include_neighbors=True,
        neighbor_sections=None,
        max_steps=None,
        result_limit=None,
        max_results=None,
        baseline_limit=None,
        weak_score=None,
        max_planner_cards=None,
        refine_candidates=None,
        refine_pool=None,
        **map_overrides,
    ):
        if retriever is None or not callable(getattr(retriever, "retrieve", None)):
            raise ValueError("A retriever with a retrieve method is required.")
        if not planner or not callable(planner):
            self.planner = None
        else:
            self.planner = planner
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.document_map = document_map or build_document_map(
            retriever,
            tokenizer=tokenizer,
            config=map_config,
            source_identity=source_identity,
            **map_overrides,
        )
        base_config = (agent_config or MapAgentConfig()).to_dict()
        overrides = {
            "max_selected_sections": max_selected_sections,
            "initial_sections": initial_sections,
            "neighbor_sections": (
                0
                if neighbor_sections is None and include_neighbors is False
                else neighbor_sections
            ),
            "max_steps": max_steps,
            "max_results": result_limit if result_limit is not None else max_results,
            "max_baseline_results": baseline_limit,
            "max_planner_cards": max_planner_cards,
            "weak_score": weak_score,
        }
        if refine_candidates is not None:
            overrides["refine_candidates"] = _bounded_bool(refine_candidates, False)
        if refine_pool is not None:
            overrides["refine_pool"] = refine_pool
        base_config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        self.config = MapAgentConfig(**base_config)
        self._corpus_stats_cache = None

    def _corpus_stats(self):
        """Corpus-wide term statistics, built once and only if refinement is used.

        Computing this walks and tokenises every section card, so it must not be
        paid on the default path where refinement is disabled.
        """
        if self._corpus_stats_cache is None:
            term_sets = []
            for section in self.document_map.sections:
                title, keywords, summary = _card_terms(section.to_card(0.0))
                term_sets.append(title | keywords | summary)
            self._corpus_stats_cache = _corpus_term_stats(term_sets)
        return self._corpus_stats_cache

    def _cards(self, query: str, history=None) -> list[dict]:
        queries = retrieval_query_candidates(query, history)[:2]
        return _score_cards(self.document_map.sections, queries)

    def _node_text(self, node_id: str) -> str:
        docstore = getattr(self.retriever, "docstore", None)
        if docstore is None:
            return ""
        try:
            node = docstore.get_node(node_id)
        except Exception:
            return ""
        if node is None:
            return ""
        getter = getattr(node, "get_content", None)
        if callable(getter):
            try:
                return str(getter() or "")
            except Exception:
                return ""
        return str(getattr(node, "text", "") or "")

    def _section_text(self, section, limit: int) -> str:
        parts = []
        for node_id in list(section.member_node_ids)[: max(1, int(limit))]:
            text = self._node_text(node_id)
            if text:
                parts.append(text)
            if sum(len(part) for part in parts) >= HARD_MAX_REFINE_CHARS:
                break
        return " ".join(parts)[:HARD_MAX_REFINE_CHARS]

    def _refine_candidates(
        self, queries: Sequence[str], cards: Sequence[dict], limit: int
    ) -> list[str]:
        """Re-score shortlisted sections on their full text instead of their cards.

        Section cards are lossy: a short title, a few keywords, and a bounded
        summary. Re-scoring only the shortlist with the same scorer, but with the
        real chunk text added, keeps the map's structural targeting without paying
        the information cost of the card. Only a bounded number of sections is
        re-read, so this stays local work.
        """
        pool = list(cards)[: max(1, int(limit))]
        entries = []
        for card in pool:
            section = self.document_map.section(str(card["id"]))
            if section is None:
                continue
            title, keywords, _card_summary = _card_terms(section.to_card(0.0))
            text = self._section_text(section, HARD_MAX_REFINE_NODES)
            if not text.strip():
                continue
            entries.append((section.section_id, title, keywords, set(_map_terms(text))))
        if not entries:
            return []
        scores = _score_term_sets(
            entries, list(queries)[:2], stats=self._corpus_stats()
        )
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [section_id for section_id, score in ordered if score > 0.0]

    def _deterministic_plan_for(
        self, queries: Sequence[str], cards: Sequence[dict]
    ) -> list[str]:
        planned = _deterministic_plan(cards, self.config.initial_sections)
        if not self.config.refine_candidates:
            return planned
        pool = max(
            self.config.refine_pool,
            self.config.initial_sections,
        )
        refined = self._refine_candidates(queries, cards, pool)
        selected = refined[: self.config.initial_sections]
        return selected or planned

    def _plan(
        self, query: str, cards: Sequence[dict], queries: Sequence[str] | None = None
    ) -> tuple[list[str], str, int, str]:
        planner_cards = _planner_cards(cards, self.config.max_planner_cards)
        card_tokens = 0
        if self.planner is None:
            return (
                self._deterministic_plan_for(list(queries or [query]), cards),
                "deterministic",
                0,
                "deterministic_keyword_scores"
                + ("_refined" if self.config.refine_candidates else ""),
            )
        serialized = json.dumps(planner_cards, sort_keys=True, ensure_ascii=False)
        card_tokens = count_map_tokens(serialized, self.tokenizer)
        known_ids = {str(card["id"]) for card in planner_cards}
        fallback = self._deterministic_plan_for(list(queries or [query]), cards)
        try:
            value = self.planner(query, planner_cards)
            selected = _validated_planner_ids(
                value, known_ids, self.config.max_selected_sections
            )
        except ValueError:
            return (
                fallback,
                "injected_fallback_deterministic",
                card_tokens,
                "planner_rejected_output",
            )
        except Exception:
            return (
                fallback,
                "injected_fallback_deterministic",
                card_tokens,
                "planner_failed",
            )
        return selected, "injected", card_tokens, "injected_selection"

    def _neighbor_ids(self, planned: Sequence[str]) -> list[str]:
        window = self.config.neighbor_sections
        if not planned:
            return []
        if window <= 0:
            return list(dict.fromkeys(str(item) for item in planned))[
                : self.config.max_selected_sections
            ]
        by_source: dict[str, list[str]] = {}
        for section in self.document_map.sections:
            by_source.setdefault(section.source_id, []).append(section.section_id)
        positions = {
            section_id: index
            for section_ids in by_source.values()
            for index, section_id in enumerate(section_ids)
        }
        selected: list[str] = []
        seen: set[str] = set()
        for section_id in planned:
            section = self.document_map.section(section_id)
            if section is None:
                continue
            siblings = by_source.get(section.source_id, [])
            center = positions.get(section_id, 0)
            neighbors = []
            for distance in range(1, window + 1):
                if center - distance >= 0:
                    neighbors.append(siblings[center - distance])
                if center + distance < len(siblings):
                    neighbors.append(siblings[center + distance])
            for candidate in [section_id, *neighbors]:
                if candidate in seen:
                    continue
                seen.add(candidate)
                selected.append(candidate)
                if len(selected) >= self.config.max_selected_sections:
                    return selected
        return selected

    def _fetch(self, query: str, history, allowed_node_ids) -> tuple[list, int]:
        candidates = retrieval_query_candidates(query, history)[:2]
        if not candidates or not str(candidates[0] or "").strip():
            return [], 0
        for candidate in candidates:
            if allowed_node_ids is None:
                results = self.retriever.retrieve(candidate)
            else:
                try:
                    results = self.retriever.retrieve(
                        candidate,
                        allowed_node_ids=allowed_node_ids,
                    )
                except TypeError as error:
                    if "allowed_node_ids" not in str(error):
                        raise
                    allowed = set(allowed_node_ids)
                    results = [
                        result
                        for result in self.retriever.retrieve(candidate) or []
                        if _result_node_id(result) in allowed
                    ]
            bounded = _normalized_results(results, self.config.max_results)
            if bounded:
                return bounded, 1
        return [], len(candidates)

    def _is_weak(self, results: Sequence) -> bool:
        if not results:
            return True
        return not any(
            _result_score(result) > self.config.weak_score for result in results
        )

    def _selected_member_ids(self, section_ids: Sequence[str]) -> tuple[str, ...]:
        node_ids: list[str] = []
        seen: set[str] = set()
        for section_id in section_ids:
            section = self.document_map.section(section_id)
            if section is None:
                continue
            for node_id in section.member_node_ids:
                if node_id in seen:
                    continue
                seen.add(node_id)
                node_ids.append(node_id)
        return tuple(node_ids)

    def _metrics(
        self,
        results: Sequence,
        planner_mode: str,
        card_tokens: int,
        baseline=None,
    ) -> RetrievalMetrics:
        selected_tokens = sum(
            count_map_tokens(_result_content(result), self.tokenizer)
            for result in list(results or [])[: self.config.max_results]
        )
        baseline_results = None
        baseline_tokens = None
        if baseline is not None:
            baseline_results = list(baseline)[: self.config.max_baseline_results]
            baseline_tokens = sum(
                count_map_tokens(_result_content(result), self.tokenizer)
                for result in baseline_results
            )
        return RetrievalMetrics(
            planner_mode=planner_mode,
            planner_card_token_estimate=max(0, int(card_tokens)),
            selected_evidence_tokens=selected_tokens,
            result_count=len(list(results or [])[: self.config.max_results]),
            global_baseline_tokens=baseline_tokens,
            selected_minus_baseline_tokens=(
                selected_tokens - baseline_tokens
                if baseline_tokens is not None
                else None
            ),
            global_baseline_result_count=(
                len(baseline_results) if baseline_results is not None else None
            ),
        )

    def _trace(
        self,
        cards: Sequence[dict],
        planned: Sequence[str],
        selected: Sequence[str],
        actions: Sequence[RouteAction],
        step_count: int,
        reflection_flag: bool,
        planner_mode: str,
        card_tokens: int,
        results: Sequence,
    ) -> dict:
        metadata, truncated = _result_metadata(
            results,
            self.tokenizer,
            min(self.config.max_results, HARD_MAX_RESULTS),
        )
        metrics = self._metrics(results, planner_mode, card_tokens)
        trace = RouteTrace(
            considered_section_ids=tuple(str(card["id"]) for card in cards),
            selected_section_ids=tuple(selected),
            planned_section_ids=tuple(planned),
            actions=tuple(actions[:HARD_MAX_TRACE_ACTIONS]),
            step_count=max(0, int(step_count)),
            reflection_flag=bool(reflection_flag),
            planner_mode=planner_mode,
            planner_card_token_estimate=max(0, int(card_tokens)),
            metrics=metrics,
            result_metadata=metadata,
            result_metadata_truncated=truncated,
        )
        payload = trace.to_dict()
        payload["routed"] = bool(selected) and not any(
            action.action in ("retrieve_global", "global_fallback")
            for action in actions
        )
        return payload

    def retrieve(
        self,
        query: str,
        history=None,
    ) -> tuple[list[NodeWithScore], dict]:
        actions: list[RouteAction] = []
        cards = self._cards(query, history)
        actions.append(
            RouteAction(
                step=0,
                action="observe",
                reason="bounded_map_ready",
                result_count=len(self.document_map.sections),
            )
        )
        if not str(query or "").strip() or not cards:
            actions.append(
                RouteAction(
                    step=0,
                    action="plan",
                    reason="empty_query_or_map",
                )
            )
            results, attempts = self._fetch(query, history, None)
            if attempts:
                actions.append(
                    RouteAction(
                        step=1,
                        action="retrieve_global",
                        reason="empty_query_or_map",
                        attempted_queries=attempts,
                        result_count=len(results),
                    )
                )
            trace = self._trace(
                cards,
                (),
                (),
                actions,
                1 if attempts else 0,
                False,
                "deterministic",
                0,
                [],
            )
            return [], trace

        planned, planner_mode, card_tokens, plan_reason = self._plan(
            query, cards, retrieval_query_candidates(query, history)[:2]
        )
        actions.append(
            RouteAction(
                step=0,
                action="plan",
                reason=plan_reason,
                allowed_section_ids=tuple(planned),
            )
        )
        selected = self._neighbor_ids(planned)
        if not selected:
            results, attempts = self._fetch(query, history, None)
            actions.append(
                RouteAction(
                    step=1,
                    action="retrieve_global",
                    reason="planner_selected_no_sections",
                    attempted_queries=attempts,
                    result_count=len(results),
                )
            )
            trace = self._trace(
                cards,
                planned,
                (),
                actions,
                1,
                False,
                planner_mode,
                card_tokens,
                results if not self._is_weak(results) else [],
            )
            return (
                results if not self._is_weak(results) else [],
                trace,
            )

        allowed = self._selected_member_ids(selected)
        results, attempts = self._fetch(query, history, allowed)
        step_count = 1
        actions.append(
            RouteAction(
                step=1,
                action="retrieve_sections",
                reason="routed_evidence",
                allowed_section_ids=tuple(selected),
                attempted_queries=attempts,
                result_count=len(results),
            )
        )
        weak = self._is_weak(results)
        actions.append(
            RouteAction(
                step=1,
                action="reflect",
                reason="weak_evidence" if weak else "evidence_sufficient",
            )
        )
        if weak and self.config.max_steps >= 2:
            selected_set = set(selected)
            expansion = [
                str(card["id"])
                for card in cards
                if str(card["id"]) not in selected_set and float(card["score"]) > 0.0
            ][: max(0, self.config.max_selected_sections - len(selected))]
            if expansion:
                selected = [*selected, *expansion]
                allowed = self._selected_member_ids(selected)
                results, attempts = self._fetch(query, history, allowed)
                step_count = 2
                actions.append(
                    RouteAction(
                        step=2,
                        action="expand_sections",
                        reason="positive_unrouted_sections",
                        allowed_section_ids=tuple(expansion),
                        attempted_queries=attempts,
                        result_count=len(results),
                    )
                )
            else:
                results, attempts = self._fetch(query, history, None)
                step_count = 2
                actions.append(
                    RouteAction(
                        step=2,
                        action="global_fallback",
                        reason="routed_evidence_weak",
                        attempted_queries=attempts,
                        result_count=len(results),
                    )
                )
        if self._is_weak(results):
            results = []
        return results, self._trace(
            cards,
            planned,
            selected,
            actions,
            step_count,
            weak,
            planner_mode,
            card_tokens,
            results,
        )

    def global_baseline(self, query: str, history=None, limit=None) -> list:
        """Return one bounded unrestricted retrieval used for baseline comparison."""
        results, _attempts = self._fetch(query, history, None)
        return _normalized_results(
            results,
            _bounded_int(
                limit if limit is not None else self.config.max_baseline_results,
                1,
                HARD_MAX_RESULTS,
                DEFAULT_MAX_BASELINE_RESULTS,
            ),
        )

    def compare_global_context(
        self,
        query: str,
        selected_results=None,
        *,
        history=None,
        baseline_results=None,
        baseline_limit=None,
    ) -> RetrievalMetrics:
        selected = (
            _normalized_results(selected_results, self.config.max_results)
            if selected_results is not None
            else self.retrieve(query, history=history)[0]
        )
        if baseline_results is None:
            baseline, _attempts = self._fetch(query, history, None)
        else:
            baseline = _normalized_results(
                baseline_results,
                baseline_limit or self.config.max_baseline_results,
            )
        limit = _bounded_int(
            baseline_limit or self.config.max_baseline_results,
            1,
            HARD_MAX_RESULTS,
            DEFAULT_MAX_BASELINE_RESULTS,
        )
        return self._metrics(selected, "deterministic", 0, baseline=baseline[:limit])


def _cache_signature(retriever, source_identity, config: RetrievalMapConfig) -> str:
    digest = sha256()
    digest.update(RETRIEVAL_MAP_VERSION.__str__().encode("utf-8"))
    digest.update(_clean_text(source_identity, 200).encode("utf-8", errors="replace"))
    digest.update(
        json.dumps(config.to_dict(), sort_keys=True).encode("utf-8", errors="replace")
    )
    corpus = _retriever_corpus(retriever)
    for raw in corpus[:HARD_MAX_CACHE_SIGNATURE_NODES]:
        digest.update(
            _clean_text(
                _node_identifier(raw, raw) or getattr(raw, "node_id", "") or raw,
                200,
            ).encode("utf-8", errors="replace")
        )
        digest.update(b"\x00")
    digest.update(str(len(corpus)).encode("utf-8"))
    return digest.hexdigest()


def _retriever_cache(retriever) -> dict:
    cache = getattr(retriever, _MAP_CACHE_ATTRIBUTE, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(retriever, _MAP_CACHE_ATTRIBUTE, cache)
    return cache


def build_map_retrieval_agent(
    retriever,
    *,
    tokenizer=None,
    planner: Callable[[str, list[dict]], Sequence[str]] | None = None,
    map_config: RetrievalMapConfig | None = None,
    agent_config: MapAgentConfig | None = None,
    source_identity=None,
    attach: bool = True,
    **kwargs,
) -> MapRetrievalAgent:
    if retriever is None or not callable(getattr(retriever, "retrieve", None)):
        raise ValueError("A retriever with a retrieve method is required.")
    agent_keys = (
        "max_selected_sections",
        "initial_sections",
        "include_neighbors",
        "neighbor_sections",
        "max_steps",
        "result_limit",
        "max_results",
        "baseline_limit",
        "weak_score",
        "max_planner_cards",
    )
    agent_overrides = {key: kwargs.pop(key) for key in agent_keys if key in kwargs}
    config = _configured_map(map_config, **kwargs)
    cache = _retriever_cache(retriever)
    cache_key = _cache_signature(retriever, source_identity, config)
    document_map = cache.get(cache_key)
    if not isinstance(document_map, DocumentMap):
        document_map = build_document_map(
            retriever,
            tokenizer=tokenizer,
            config=config,
            source_identity=source_identity,
        )
        cache[cache_key] = document_map
        while len(cache) > HARD_MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
    agent = MapRetrievalAgent(
        retriever,
        document_map,
        tokenizer=tokenizer,
        planner=planner,
        agent_config=agent_config,
        **agent_overrides,
    )
    if attach:
        setattr(retriever, _MAP_ATTRIBUTE, document_map)
        setattr(retriever, _MAP_AGENT_ATTRIBUTE, agent)
    return agent


def build_retrieval_map_agent(*args, **kwargs) -> MapRetrievalAgent:
    return build_map_retrieval_agent(*args, **kwargs)


def ensure_retrieval_map_agent(retriever, **kwargs) -> MapRetrievalAgent:
    return build_map_retrieval_agent(retriever, **kwargs)


def build_retrieval_map(retriever, **kwargs) -> DocumentMap:
    return build_document_map(retriever, **kwargs)


__all__ = [
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_SECTIONS",
    "DEFAULT_MAX_SELECTED_SECTIONS",
    "HARD_MAX_CACHE_ENTRIES",
    "HARD_MAX_KEYWORDS",
    "HARD_MAX_NODES",
    "HARD_MAX_SECTIONS",
    "HARD_MAX_SELECTED_SECTIONS",
    "HARD_MAX_STEPS",
    "HARD_MAX_SUMMARY_CHARS",
    "MAP_CHOICE_SETTINGS",
    "MAP_PLANNER_DETERMINISTIC",
    "MAP_PLANNER_LLM",
    "MAP_PLANNER_LLM_FALLBACK_DETERMINISTIC",
    "MAP_PLANNER_MODES",
    "MAP_SETTING_DEFAULTS",
    "MAP_SETTING_RANGES",
    "RETRIEVAL_MAP_VERSION",
    "DocumentMap",
    "LlmMapPlanner",
    "LlmMapPlannerStatus",
    "MapAgentConfig",
    "MapEdge",
    "MapNode",
    "MapRetrievalAgent",
    "MapSection",
    "RetrievalMapConfig",
    "RetrievalMetrics",
    "RouteAction",
    "RouteTrace",
    "annotate_planner_trace",
    "build_document_map",
    "build_map_retrieval_agent",
    "build_retrieval_map",
    "build_retrieval_map_agent",
    "count_map_tokens",
    "ensure_retrieval_map_agent",
    "json_safe",
    "map_agent_config",
    "map_document_config",
    "map_source_identity",
    "normalize_map_settings",
    "retriever_supports_map",
    "state_map_source_identity",
]
