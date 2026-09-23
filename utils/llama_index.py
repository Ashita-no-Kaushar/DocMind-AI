import csv
import fnmatch
import hashlib
import io
import json
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

# Transformers 5.x can emit a large volume of non-actionable alias warnings
# during startup. Keep app logs focused on failures we can act on.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import ollama
import streamlit as st
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import run_transformations
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi

import utils.logs as logs
from utils.endpoint_policy import (
    DEFAULT_OPENAI_BASE_URL,
    normalize_ollama_endpoint,
    normalize_provider_endpoint,
    validate_credential_endpoint,
)
from utils.format_ingestion import (
    CAPABILITY_REGISTRY,
    PARSER_CAPABILITIES,
    SUPPORTED_EXTENSIONS,
    LegacyFormatError,
    build_file_extractors,
    get_parser_capability,
    legacy_conversion_message,
)
from utils.provider_config import (
    OLLAMA,
    OPENAI_OFFICIAL,
    normalize_provider_kind,
)
from utils.source_state import (
    PARSER_CACHE_VERSION,
    tag_report,
)

FORMAT_CAPABILITY_REGISTRY = CAPABILITY_REGISTRY
FORMAT_CAPABILITIES = PARSER_CAPABILITIES
FORMAT_EXTENSIONS = SUPPORTED_EXTENSIONS

# On-disk cache of built indexes, keyed by source identity, document text,
# embedding model, endpoint, and chunk settings.
INDEX_CACHE_DIR = os.path.join(os.getcwd(), ".index_cache")
INDEX_CACHE_KEEP = 5
INGESTION_LOCK = threading.RLock()


def ingestion_lock():
    return INGESTION_LOCK


def _check_ingestion_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Website ingestion deadline exceeded.")


# Explicit, document-grounded QA instruction. Small local models (0.5B-8B)
# answer far more reliably when the prompt tells them to use ONLY the context
# and to refuse to answer from prior knowledge.
TEXT_QA_TEMPLATE = PromptTemplate(
    "You are DocMind AI, a document-grounded assistant.\n"
    "Answer the query using ONLY the context provided below.\n"
    "Do not use prior knowledge. Do not invent information.\n"
    "Quote exact numbers, dates and names from the context when present.\n"
    "Citations are optional. If you cite a claim, use (from [n]) only when the cited chunk directly supports that claim.\n"
    "If no chunk directly supports a claim, omit its citation. Never add a citation merely because context sources were provided.\n"
    "If the context does not contain the answer, reply exactly:\n"
    "\"I could not find this information in the documents.\"\n"
    "Follow any style guidance from the system message and stay factual.\n\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)


# Documents with less content than this are skipped during ingestion: too
# short to embed usefully, they only add embedding work and retrieval noise.
# Kept low (15) so small but meaningful files (CSV rows, JSON snippets, short
# notes) are still indexed — the user sees "rest types not working" if this
# is too high.
MIN_CHUNK_CHARS = 15

# Rough token budget for the retrieved context sent to the LLM (chars/4).
# Excess chunks are trimmed from the bottom of the ranking, keeping the
# prompt comfortably inside the 2048-token window.
CONTEXT_CHAR_BUDGET = 4800
EVIDENCE_EXCERPT_MAX_CHARS = 1200
DEFAULT_CANDIDATE_DEPTH = 10
MAX_CANDIDATE_DEPTH = 50
RRF_K = 60
BM25_MIN_EVIDENCE_SCORE = 0.01
_BM25_GENERIC_TOKENS = frozenset(
    {
        "about",
        "detail",
        "document",
        "file",
        "information",
        "manual",
        "paper",
        "polici",
        "report",
        "rule",
        "term",
    }
)

# Below this vector score a chunk is only trusted when BM25 also found an
# exact keyword match. Small local embedding models score nearly everything
# in the 0.3-0.5 range, so the similarity cutoff alone cannot tell a real
# match from a random one. Requiring keyword evidence prevents unrelated
# questions from pulling irrelevant context (which makes small models
# hallucinate) and lets the "could not find" fallback actually fire.
VECTOR_EVIDENCE_FLOOR = 0.5

# Bump when retrieval/ingestion logic changes so stale caches rebuild once.
INDEX_CACHE_VERSION = 10

# Lazy Porter stemmer (pure Python, no model data): normalizes word endings
# ("running" -> "run") so BM25 keyword matching understands inflected forms
# of the query without any GPU work.
_STEMMER = None


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()
    return _STEMMER


def _text_overlap_ratio(a: str, b: str) -> float:
    """Overlap coefficient of stemmed token sets for near-duplicate filtering."""
    wa = set(_bm25_tokens(a))
    wb = set(_bm25_tokens(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _dedupe_near_duplicate_nodes(nodes, threshold: float = 0.95):
    """Drop chunks whose stemmed-token overlap exceeds the configured threshold."""
    deduped = []
    for node in nodes:
        content = node.get_content()
        if any(
            _text_overlap_ratio(content, other.get_content()) > threshold
            for other in deduped
        ):
            continue
        deduped.append(node)
    return deduped


def _document_title(node, max_chars: int = 60) -> str:
    """Return a readable title for a node's source document.

    Derived from the file name ("annual_report.md" -> "Annual Report"), used
    as a lightweight title prefix on every chunk so chunks carry their
    document's identity into BM25 and embedding search.
    """
    file_name = (node.metadata or {}).get("file_name") or ""
    base = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ")
    title = " ".join(base.split())[:max_chars].strip()
    if title:
        title = title.title()
    return title or file_name


def _prepend_document_title(nodes):
    """Prefix each chunk with its document's title.

    Small local models answer "what is this document about" questions far
    better when every chunk carries the title line, and queries containing
    the title words ("annual report", "refund policy") match every chunk of
    that document instead of just the first one.
    """
    for node in nodes:
        title = _document_title(node)
        if title and not node.get_content().lstrip().lower().startswith(
            title.lower()
        ):
            node.text = f"{title}.\n\n{node.get_content()}"
    return nodes


def _set_document_text(document, text: str) -> None:
    try:
        document.set_content(text)
    except Exception:
        document.text = text


def _verbalize_tabular_docs(documents):
    """Rewrite delimited and JSON records without losing their source data."""
    for doc in documents:
        metadata = doc.metadata or {}
        fname = str(metadata.get("file_name") or "").lower()
        text = doc.text or ""
        if not text.strip():
            continue
        if fname.endswith(".csv") or fname.endswith(".tsv"):
            if metadata.get("tabular_verbalized"):
                continue
            try:
                delimiter = "\t" if fname.endswith(".tsv") else ","
                rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
                if not rows:
                    continue
                headers = [
                    str(value).strip() or f"Column {index + 1}"
                    for index, value in enumerate(rows[0])
                ]
                lines = ["Columns: " + ", ".join(headers)]
                for row_number, row in enumerate(rows[1:], start=1):
                    values = []
                    for index, value in enumerate(row):
                        value = str(value).strip()
                        if not value:
                            continue
                        header = (
                            headers[index]
                            if index < len(headers)
                            else f"Extra column {index + 1}"
                        )
                        values.append(f"{header} is {value}")
                    lines.append(
                        f"Row {row_number}: "
                        + (", ".join(values) if values else "(empty row)")
                    )
                _set_document_text(
                    doc,
                    "\n".join(lines) + "\n\nOriginal table:\n" + text,
                )
            except (csv.Error, ValueError):
                continue
            continue
        if not fname.endswith(".json") and not fname.endswith(".jsonl"):
            continue
        records = []
        record_numbers = []
        source_lines = []
        if fname.endswith(".jsonl"):
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                source_lines.append(line)
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append(line)
                record_numbers.append(int(metadata.get("record_index") or line_number))
        else:
            source_lines.append(text)
            try:
                data = json.loads(text)
                records = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                continue
            record_numbers = list(range(1, len(records) + 1))
        verbalized = []
        for position, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            parts = []
            for key, value in record.items():
                if value is None or str(value).strip() == "":
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                parts.append(f"{key} is {value}")
            if parts:
                index = record_numbers[position]
                verbalized.append(f"Record {index}: " + ", ".join(parts) + ".")
        if not verbalized:
            continue
        label = "Original JSONL" if fname.endswith(".jsonl") else "Original JSON"
        source = "\n".join(source_lines)
        _set_document_text(
            doc,
            "\n".join(verbalized) + f"\n\n{label}:\n{source}",
        )
    return documents


def _merge_code_blocks(nodes):
    """Merge chunks split inside a markdown code fence (```).

    LlamaIndex splitters can cut a ```python block in half, leaving an
    unclosed fence that embeddings and small LLMs mis-handle. If a chunk
    contains an odd number of ``` fences, it is inside a code block that
    was split — merge it with the next chunk.
    """
    if not nodes:
        return nodes
    merged = []
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        text = node.get_content()
        # Odd number of fences means we are inside an unclosed block
        if text.count("```") % 2 == 1 and idx + 1 < len(nodes):
            nxt = nodes[idx + 1]
            # Merge the two and keep the first node's metadata
            node.text = text + "\n\n" + nxt.get_content()
            # Preserve file_name, etc. from first node (already in metadata)
            merged.append(node)
            idx += 2  # skip next as merged
        else:
            merged.append(node)
            idx += 1
    return merged


def _is_eco_mode() -> bool:
    """Return whether Eco Mode is enabled (light load / weak hardware)."""
    try:
        return bool(st.session_state.get("eco_mode"))
    except Exception:
        return False


def _context_char_budget() -> int:
    """Rough token budget for retrieved context; Eco Mode trims it (~800 chars)."""
    return 3200 if _is_eco_mode() else CONTEXT_CHAR_BUDGET


def _is_oom_error(err) -> bool:
    message = str(err).lower()
    return ("out of memory" in message or "oom" in message) or (
        "cuda" in message and "memory" in message
    )


def _is_embedding_capacity_error(err) -> bool:
    message = str(err).lower()
    oom_markers = (
        "out of memory",
        "outofmemory",
        "memoryerror",
        "oom",
        "insufficient memory",
        "not enough memory",
        "memory allocation",
        "resource exhausted",
        "resource_exhausted",
    )
    capacity_markers = (
        "capacity",
        "model is too large",
        "batch too large",
        "payload too large",
        "request entity too large",
        "server is busy",
        "overloaded",
        "too many requests",
        "service unavailable",
    )
    return (
        "cuda" in message and "memory" in message
    ) or any(marker in message for marker in oom_markers + capacity_markers)


def _bm25_tokens(text):
    """Lowercase word tokens without punctuation, stemmed for BM25 matching.

    Hyphens are split ("30-days" -> "30", "day") and single-letter tokens
    dropped ("company's" -> "company"). Applied symmetrically to the corpus
    and queries, so "30-days" and "30 days" match each other.
    """
    stemmer = _stemmer()
    text = text.lower().replace("-", " ")
    words = re.findall(r"[a-z0-9]+", text)
    return [stemmer.stem(word) for word in words if len(word) > 1]


# Curated keyword synonyms (stemmed forms) used to expand SHORT queries for
# BM25 matching only. The vector search still uses the cleaned query, so the
# expansion cannot dilute semantics: it only widens exact keyword recall.
# Kept deliberately small: every entry must be high-confidence.
_QUERY_SYNONYMS = {
    "refund": {"return", "money", "back"},
    "money": {"refund", "return"},
    "back": {"refund", "return"},
    "polici": {"rule", "guideline", "term", "condition"},
    "rule": {"polici", "guideline"},
    "purchas": {"buy", "order", "acquir"},
    "buy": {"purchas", "order"},
    "salary": {"pay", "wage", "income", "compens"},
    "vacat": {"leav", "holiday", "timeoff"},
    "job": {"work", "employ", "position"},
    "cost": {"price", "charge", "fee", "expens"},
    "revenue": {"income", "sale", "earning"},
    "customer": {"client", "user", "buyer", "consum"},
    "document": {"file", "report", "paper", "manual"},
    "help": {"assist", "support"},
    "problem": {"issue", "error", "bug"},
    "fix": {"repair", "solv", "correct"},
}


def _bm25_expanded_tokens(tokens):
    """Widen short queries with curated synonyms for BM25 matching.

    Long, specific queries are left untouched: they already contain enough
    exact terms, and expanding them would add noise.
    """
    if len(tokens) > 4:
        return tokens
    expanded = list(tokens)
    for token in tokens:
        for synonym in _QUERY_SYNONYMS.get(token, ()):
            if synonym not in expanded:
                expanded.append(synonym)
    return expanded


# Numeric words → digits so "thirty days" matches "30 days" without embedding help.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}


# Filler words that carry no retrieval value: removing them helps BM25
# weight the real terms and slightly improves vector matching too.
# Includes common Hinglish/Hindi question words (batao, kya, hai, ...) so
# mixed-language queries retrieve the right chunks as well.
_QUERY_FILLER_WORDS = {
    "please", "tell", "me", "about", "can", "you", "could", "would",
    "i", "want", "to", "know", "what", "is", "are", "was", "were",
    "the", "a", "an", "of", "for", "with", "and", "or", "do", "does",
    "batao", "bata", "kya", "kyaa", "hai", "hain", "kaise", "karke",
    "karne", "mein", "ho", "hoga",
}


def _rewrite_query(query):
    """Light, rule-based query cleanup: no LLM calls, no GPU cost.

    Stems words so "documents", "documented" and "documenting" all match
    "document" in the index, which materially helps small local models answer.
    """
    # Normalize numeric words before tokenization so "thirty days" == "30 days"
    lowered = query.lower()
    for word, digit in _NUMBER_WORDS.items():
        lowered = re.sub(rf"\b{re.escape(word)}\b", digit, lowered)
    stemmer = _stemmer()
    text = " ".join(stemmer.stem(word) for word in _bm25_tokens(lowered))
    words = [word for word in text.split() if word not in _QUERY_FILLER_WORDS]
    return " ".join(words) or text


def _history_role(message) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role", "")).lower()
    role = getattr(message, "role", "")
    return str(getattr(role, "value", role)).lower()


def _history_content(message) -> str:
    if isinstance(message, Mapping):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _recent_user_turns(
    history,
    current_query: str = "",
    max_turns: int = 3,
    max_chars: int = 1200,
) -> list[str]:
    current = str(current_query or "").strip()
    turns: list[str] = []
    for message in reversed(list(history or [])):
        if _history_role(message) != "user":
            continue
        content = _history_content(message).strip()
        if not content:
            continue
        if current and content == current and not turns:
            continue
        turns.append(content)
        if len(turns) >= max(1, int(max_turns)):
            break
    turns.reverse()
    result = []
    used = 0
    for turn in reversed(turns):
        remaining = max_chars - used
        if remaining <= 0:
            break
        value = turn[:remaining].strip()
        if value:
            result.append(value)
            used += len(value)
    result.reverse()
    return result


def is_followup_query(query: str) -> bool:
    text = " ".join(str(query or "").strip().lower().split())
    if not text:
        return False
    words = re.findall(r"[a-z0-9]+", text)
    if re.match(
        r"^(?:and|also|then|so|but|what about|how about)\b",
        text,
    ):
        return True
    if re.search(
        r"\b(?:it|that|this|those|these|they|them|there|then|same|previous|above|one|two|three|first|second|third|last)\b",
        text,
    ) and len(words) <= 14:
        return True
    if text.endswith(("...", "…")):
        return True
    return len(words) <= 2 and text not in {"no", "yes", "ok", "okay"}


def build_retrieval_query(
    query: str,
    history=None,
    *,
    max_user_turns: int = 3,
    max_history_chars: int = 1200,
) -> str:
    current = str(query or "").strip()
    if not current or not is_followup_query(current):
        return current
    prior = _recent_user_turns(
        history,
        current_query=current,
        max_turns=max_user_turns,
        max_chars=max_history_chars,
    )
    if not prior:
        return current
    return f"{' '.join(prior)} {current}".strip()


def retrieval_query_candidates(
    query: str,
    history=None,
    *,
    max_user_turns: int = 3,
    max_history_chars: int = 1200,
) -> list[str]:
    standalone = str(query or "").strip()
    expanded = build_retrieval_query(
        standalone,
        history,
        max_user_turns=max_user_turns,
        max_history_chars=max_history_chars,
    )
    if not expanded or expanded == standalone:
        return [standalone]
    return [standalone, expanded]


build_followup_query = build_retrieval_query
build_conversation_query = build_retrieval_query


def _bounded_excerpt(content: str, max_chars: int = EVIDENCE_EXCERPT_MAX_CHARS) -> str:
    text = str(content or "")
    limit = max(0, int(max_chars))
    return text if len(text) <= limit else text[:limit]


def normalize_evidence(nodes, max_excerpt_chars: int = EVIDENCE_EXCERPT_MAX_CHARS) -> list[dict]:
    evidence = []
    for node_score in list(nodes or []):
        node = getattr(node_score, "node", node_score)
        if hasattr(node, "get_content"):
            content = str(node.get_content() or "")
        else:
            content = str(getattr(node, "text", "") or "")
        if not content:
            continue
        citation_index = len(evidence) + 1
        metadata = getattr(node, "metadata", {}) or {}
        source = metadata.get("file_name") or metadata.get("source") or metadata.get("url") or "document"
        try:
            score = float(getattr(node_score, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        evidence.append(
            {
                "citation_index": citation_index,
                "source": str(source),
                "score": score,
                "excerpt": _bounded_excerpt(content, max_excerpt_chars),
            }
        )
    return evidence


class HybridRetriever:
    def __init__(
        self,
        vector_retriever,
        docstore,
        corpus,
        top_k,
        similarity_cutoff=None,
        candidate_depth=None,
        vector_candidate_depth=None,
        bm25_candidate_depth=None,
    ):
        self.vector_retriever = vector_retriever
        self.docstore = docstore
        self.corpus = list(corpus)
        self.top_k = self._bounded_depth(top_k, 3)
        self.similarity_cutoff = (
            float(similarity_cutoff) if similarity_cutoff is not None else 0.3
        )
        default_depth = self._bounded_depth(
            candidate_depth, DEFAULT_CANDIDATE_DEPTH
        )
        self.candidate_depth = default_depth
        self.vector_candidate_depth = self._bounded_depth(
            vector_candidate_depth, default_depth
        )
        self.bm25_candidate_depth = self._bounded_depth(
            bm25_candidate_depth, default_depth
        )
        self._document_tokens = {}
        tokenized_corpus = []
        for node_id in self.corpus:
            tokens = _bm25_tokens(
                self._node_content(self.docstore.get_node(node_id))
            )
            self._document_tokens[node_id] = set(tokens)
            tokenized_corpus.append(tokens)
        self._bm25 = BM25Okapi(tokenized_corpus)
        self.intro_node_ids = self._intro_node_ids()

    @staticmethod
    def _bounded_depth(value, fallback):
        try:
            depth = int(value)
        except (TypeError, ValueError, OverflowError):
            depth = int(fallback)
        return max(1, min(MAX_CANDIDATE_DEPTH, depth))

    @staticmethod
    def _node_content(node) -> str:
        if hasattr(node, "get_content"):
            return str(node.get_content() or "")
        return str(getattr(node, "text", "") or "")

    def _tokenize(self, node_id):
        return _bm25_tokens(self._node_content(self.docstore.get_node(node_id)))

    def _intro_node_ids(self, count=2):
        try:
            def sort_key(node_id):
                node = self.docstore.get_node(node_id)
                metadata = getattr(node, "metadata", {}) or {}
                start = getattr(node, "start_char_idx", None) or 0
                return (metadata.get("file_name", ""), start)

            return sorted(self.corpus, key=sort_key)[:count]
        except Exception:
            return []

    @staticmethod
    def _is_doc_level_query(query: str, rewritten: str) -> bool:
        lower = str(query or "").lower()
        if "summar" in lower or "overview" in lower:
            return True
        return "document" in lower and len(rewritten.split()) <= 3

    def _apply_live_settings(self):
        try:
            state = st.session_state
            top_k = max(int(state.get("top_k", self.top_k)), 1)
            cutoff = float(state.get("similarity_cutoff", self.similarity_cutoff))
        except (TypeError, ValueError, OverflowError, AttributeError):
            return
        common = state.get(
            "candidate_depth",
            state.get("retrieval_candidate_depth", self.candidate_depth),
        )
        vector_depth = state.get(
            "vector_candidate_depth",
            self.vector_candidate_depth,
        )
        bm25_depth = state.get(
            "bm25_candidate_depth",
            self.bm25_candidate_depth,
        )
        self.candidate_depth = self._bounded_depth(common, self.candidate_depth)
        self.vector_candidate_depth = min(
            self.candidate_depth,
            self._bounded_depth(vector_depth, self.candidate_depth),
        )
        self.bm25_candidate_depth = min(
            self.candidate_depth,
            self._bounded_depth(bm25_depth, self.candidate_depth),
        )
        if _is_eco_mode():
            top_k = min(top_k, 3)
            self.vector_candidate_depth = min(self.vector_candidate_depth, 5)
            self.bm25_candidate_depth = min(self.bm25_candidate_depth, 5)
        self.top_k = self._bounded_depth(top_k, self.top_k)
        self.similarity_cutoff = cutoff
        try:
            self.vector_retriever._similarity_top_k = self.vector_candidate_depth
        except AttributeError:
            pass

    def _strong_bm25_evidence(self, node_id, original_tokens, expanded_tokens, score):
        if score < BM25_MIN_EVIDENCE_SCORE:
            return False
        content_tokens = self._document_tokens.get(node_id) or set(
            _bm25_tokens(self._node_content(self.docstore.get_node(node_id)))
        )
        original = set(original_tokens)
        expanded = set(expanded_tokens)
        exact = original & content_tokens
        expanded_match = expanded & content_tokens
        if not exact and not expanded_match:
            return False
        meaningful = [
            token
            for token in original
            if token not in _BM25_GENERIC_TOKENS and len(token) > 1
        ]
        if not meaningful:
            meaningful = [
                token
                for token in expanded
                if token not in _BM25_GENERIC_TOKENS and len(token) > 1
            ]
        matched = (exact or expanded_match) & set(meaningful)
        synonym_targets = {
            synonym
            for token in original
            for synonym in _QUERY_SYNONYMS.get(token, ())
        }
        if meaningful and expanded_match & synonym_targets:
            return True
        if len(matched) >= 2:
            return True
        if len(matched) == 1:
            token = next(iter(matched))
            document_frequency = sum(
                token in self._document_tokens.get(candidate_id, set())
                for candidate_id in self.corpus
            )
            return (
                token.isdigit()
                or len(token) >= 4
                or document_frequency <= max(2, len(self.corpus) // 3)
                or score >= 1.5
            )
        return False

    def _vector_candidates(self, query):
        results = list(self.vector_retriever.retrieve(query) or [])
        candidates = []
        seen = set()
        for node_score in results[: self.vector_candidate_depth]:
            node = getattr(node_score, "node", node_score)
            node_id = getattr(node, "node_id", None)
            if node_id is None or node_id in seen:
                continue
            seen.add(node_id)
            try:
                score = float(getattr(node_score, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            candidates.append((node_id, node_score, score))
        return candidates

    def _bm25_candidates(self, tokens):
        if not tokens or not self.corpus:
            return [], {}
        expanded = _bm25_expanded_tokens(tokens)
        depth = min(self.bm25_candidate_depth, len(self.corpus))
        try:
            bm25_ranked_ids = list(
                self._bm25.get_top_n(expanded, self.corpus, n=depth)
            )
        except Exception:
            bm25_ranked_ids = []
        raw_scores = self._bm25.get_scores(expanded)
        bm25_scores = {
            node_id: float(score)
            for node_id, score in zip(self.corpus, raw_scores)
            if math.isfinite(float(score)) and float(score) > 0.0
        }
        document_tokens = self._document_tokens
        document_frequency = {
            token: sum(token in values for values in document_tokens.values())
            for token in set(expanded)
        }
        lexical_scores = {}
        for node_id, values in document_tokens.items():
            matched = set(expanded) & values
            if not matched:
                continue
            lexical_scores[node_id] = sum(
                1.0 + math.log((len(self.corpus) + 1) / (document_frequency[token] + 1))
                for token in matched
            )
        score_by_id = {
            node_id: bm25_scores.get(node_id, lexical_scores.get(node_id, 0.0))
            for node_id in set(bm25_scores) | set(lexical_scores)
            if bm25_scores.get(node_id, 0.0) > 0.0
            or lexical_scores.get(node_id, 0.0) > 0.0
        }
        corpus_order = {node_id: index for index, node_id in enumerate(self.corpus)}
        bm25_rank = {node_id: rank for rank, node_id in enumerate(bm25_ranked_ids)}
        ranked_ids = sorted(
            score_by_id,
            key=lambda node_id: (
                bm25_rank.get(node_id, depth),
                -score_by_id[node_id],
                corpus_order[node_id],
            ),
        )[:depth]
        return [(node_id, score_by_id[node_id]) for node_id in ranked_ids], score_by_id

    def retrieve(self, query: str):
        self._apply_live_settings()
        rewritten = _rewrite_query(query)
        rrf_scores = {}
        vector_scores = {}
        for rank, (node_id, _node_score, score) in enumerate(self._vector_candidates(rewritten)):
            vector_scores[node_id] = max(vector_scores.get(node_id, 0.0), score)
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (RRF_K + rank)

        tokens = _bm25_tokens(rewritten)
        bm25_candidates, bm25_scores = self._bm25_candidates(tokens)
        for rank, (node_id, _score) in enumerate(bm25_candidates):
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (RRF_K + rank)

        def credible(node_id):
            if self._strong_bm25_evidence(
                node_id,
                tokens,
                _bm25_expanded_tokens(tokens),
                bm25_scores.get(node_id, 0.0),
            ):
                return True
            vector = vector_scores.get(node_id, 0.0)
            if vector >= VECTOR_EVIDENCE_FLOOR and vector >= self.similarity_cutoff:
                return True
            if self.similarity_cutoff <= 0:
                return vector > 0.0
            return False

        filtered = {
            node_id: score
            for node_id, score in rrf_scores.items()
            if credible(node_id)
        }
        if not filtered:
            return []

        phrases = re.findall(r'"([^"]+)"', str(query or ""))
        phrases += re.findall(r"'([^']+)'", str(query or ""))
        for node_id in list(filtered):
            try:
                content = self._node_content(self.docstore.get_node(node_id)).lower()
                if any(
                    phrase.strip() and phrase.strip().lower() in content
                    for phrase in phrases
                ):
                    filtered[node_id] += 0.5
            except Exception:
                continue

        ranked = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))
        selected = []
        selected_ids = set()
        for node_id, rrf_score in ranked:
            if node_id in selected_ids:
                continue
            content = self._node_content(self.docstore.get_node(node_id))
            if any(
                _text_overlap_ratio(
                    content,
                    self._node_content(self.docstore.get_node(selected_id)),
                )
                > 0.95
                for selected_id in selected_ids
            ):
                continue
            selected.append((node_id, rrf_score))
            selected_ids.add(node_id)
            if len(selected) >= self.top_k:
                break

        if self._is_doc_level_query(query, rewritten):
            existing_ids = set(selected_ids)
            intro = [
                (node_id, filtered.get(node_id, 0.0))
                for node_id in self.intro_node_ids
                if node_id in filtered and node_id not in existing_ids
            ]
            selected = sorted(
                [*intro, *selected],
                key=lambda item: (-item[1], item[0]),
            )[: self.top_k + len(intro)]

        return [
            NodeWithScore(
                node=self.docstore.get_node(node_id),
                score=vector_scores.get(node_id, rrf_score),
            )
            for node_id, rrf_score in selected
        ]


def build_hybrid_retriever(
    vector_retriever,
    index,
    top_k,
    similarity_cutoff=None,
    candidate_depth=None,
    vector_candidate_depth=None,
    bm25_candidate_depth=None,
):
    try:
        docstore = index.docstore
        corpus = list(docstore.docs.keys())
        if similarity_cutoff is None:
            similarity_cutoff = st.session_state.get("similarity_cutoff", 0.3)
        state = st.session_state
        common = state.get(
            "candidate_depth",
            state.get("retrieval_candidate_depth", candidate_depth),
        )
        return HybridRetriever(
            vector_retriever=vector_retriever,
            docstore=docstore,
            corpus=corpus,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            candidate_depth=common,
            vector_candidate_depth=vector_candidate_depth,
            bm25_candidate_depth=bm25_candidate_depth,
        )
    except Exception as err:
        logs.log.warning(f"Hybrid retriever unavailable, falling back to vector: {err}")
        return vector_retriever


class ProgressReportingEmbedding(BaseEmbedding):
    """Delegate embeddings while reporting exact batch progress."""

    wrapped_model: BaseEmbedding
    progress_callback: object = Field(exclude=True)
    total_texts: int = Field(default=0)
    completed_texts: int = Field(default=0)
    _deadline: Optional[float] = PrivateAttr(default=None)

    def _get_query_embedding(self, query: str):
        return self.wrapped_model.get_query_embedding(query)

    async def _aget_query_embedding(self, query: str):
        return await self.wrapped_model.aget_query_embedding(query)

    def _get_text_embedding(self, text: str):
        return self.wrapped_model.get_text_embedding(text)

    def get_text_embedding_batch(self, texts, show_progress=False, **kwargs):
        self.total_texts = len(texts)
        result = []
        batch_size = self.wrapped_model.embed_batch_size
        for start in range(0, len(texts), batch_size):
            _check_ingestion_deadline(self._deadline)
            batch = texts[start:start + batch_size]
            result.extend(
                self.wrapped_model.get_text_embedding_batch(
                    batch,
                    show_progress=False,
                    **kwargs,
                )
            )
            _check_ingestion_deadline(self._deadline)
            self.completed_texts += len(batch)
            self.progress_callback(self.completed_texts, self.total_texts)
        return result


class OllamaEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter backed by an Ollama server.
    
    Uses batched embed requests for significantly faster ingestion.
    """

    base_url: str = Field(description="Ollama server base URL")
    embed_batch_size: int = Field(default=16, description="Chunks per embed request")

    _client: Optional[ollama.Client] = PrivateAttr(default=None)
    _deadline: Optional[float] = PrivateAttr(default=None)

    def _client_inst(self):
        _check_ingestion_deadline(self._deadline)
        client = self._client
        replace_client = (
            client is not None
            and self._deadline is not None
            and isinstance(client, ollama.Client)
        )
        if client is None or replace_client:
            if replace_client:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            timeout = 300
            if self._deadline is not None:
                timeout = max(0.001, self._deadline - time.monotonic())
            client = ollama.Client(
                host=normalize_ollama_endpoint(self.base_url),
                timeout=timeout,
            )
            self._client = client
        return client

    def _get_query_embedding(self, query: str):
        return self._client_inst().embed(model=self.model_name, input=[query]).embeddings[0]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str):
        return self._client_inst().embed(model=self.model_name, input=[text]).embeddings[0]

    def get_text_embedding_batch(self, texts, show_progress=False, **kwargs):
        """Send chunks to Ollama, shrinking the batch on failure.

        A batch that is too large can exhaust GPU memory on the Ollama server
        (CUDA OOM). The failed batch is retried with a smaller size so ingestion
        finishes instead of crashing mid-way; the smaller size is kept for the
        rest of the run to avoid repeated failures.
        """
        result = []
        index = 0
        batch_size = self.embed_batch_size
        while index < len(texts):
            _check_ingestion_deadline(self._deadline)
            client = self._client_inst()
            batch = texts[index : index + batch_size]
            try:
                response = client.embed(model=self.model_name, input=batch)
                _check_ingestion_deadline(self._deadline)
                result.extend(response.embeddings)
                index += len(batch)
            except Exception as err:
                _check_ingestion_deadline(self._deadline)
                if not _is_embedding_capacity_error(err) or len(batch) <= 1:
                    raise
                logs.log.warning(
                    f"Embedding batch of {len(batch)} failed due to capacity; "
                    f"shrinking batch size and retrying"
                )
                batch_size = max(1, batch_size // 2)
                self.embed_batch_size = batch_size
        return result



###################################
#
# Setup Embedding Model
#
###################################


def _ollama_model_names(client) -> list[str]:
    """Return the installed model names from an Ollama client.

    Handles both the dict-based API responses and the pydantic model objects
    returned by newer versions of the `ollama` library.
    """
    try:
        data = client.list()
    except Exception:
        return []
    models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names = []
    for model in models:
        try:
            name = model.get("model") or model.get("name")
        except AttributeError:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(name)
    return names


def verify_embedding_model(model: str, base_url: str, timeout=30) -> bool:
    try:
        endpoint = normalize_ollama_endpoint(base_url)
        client = ollama.Client(host=endpoint, timeout=timeout)
        if model not in _ollama_model_names(client):
            return False
        details = client.show(model)
        capabilities = getattr(details, "capabilities", None)
        if capabilities is None and isinstance(details, dict):
            capabilities = details.get("capabilities", [])
        if isinstance(capabilities, str):
            capabilities = {capabilities}
        return "embedding" in {str(value).lower() for value in capabilities or []}
    except Exception:
        return False


def setup_embedding_model(
    model: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    backend: str = "Ollama",
    ollama_endpoint: Optional[str] = None,
    api_key: str = "",
    base_url: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    provider_kind: Optional[str] = None,
    deadline=None,
):
    with INGESTION_LOCK:
        return _setup_embedding_model(
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            backend=backend,
            ollama_endpoint=ollama_endpoint,
            api_key=api_key,
            base_url=base_url,
            embedding_backend=embedding_backend,
            provider_kind=provider_kind,
            deadline=deadline,
        )


def _setup_embedding_model(
    model: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    backend: str = "Ollama",
    ollama_endpoint: Optional[str] = None,
    api_key: str = "",
    base_url: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    provider_kind: Optional[str] = None,
    deadline=None,
):
    _check_ingestion_deadline(deadline)
    kind = normalize_provider_kind(
        provider_kind or embedding_backend or backend
    )
    api_key = str(api_key).strip() if api_key else ""
    endpoint_value = base_url or ollama_endpoint
    if not model:
        raise ValueError("An embedding model is required.")
    if not endpoint_value:
        if kind == OPENAI_OFFICIAL:
            endpoint_value = DEFAULT_OPENAI_BASE_URL
        else:
            raise ValueError("A server endpoint is required for embeddings")
    try:
        from utils.ollama import _embed_batch_size

        if kind == OLLAMA:
            endpoint = normalize_ollama_endpoint(endpoint_value)
            verification_timeout = 30
            if deadline is None:
                embedding_available = verify_embedding_model(model, endpoint)
            else:
                _check_ingestion_deadline(deadline)
                verification_timeout = max(0.001, deadline - time.monotonic())
                embedding_available = verify_embedding_model(
                    model,
                    endpoint,
                    timeout=verification_timeout,
                )
            _check_ingestion_deadline(deadline)
            if not embedding_available:
                raise ValueError(
                    f"Embedding model '{model}' is not available on the Ollama server "
                    f"at {endpoint}. Pull it first with: ollama pull {model}"
                )
            _check_ingestion_deadline(deadline)
            embedding_model = OllamaEmbedding(
                model_name=model,
                base_url=endpoint,
                embed_batch_size=_embed_batch_size(),
            )
            embedding_model._deadline = deadline
            Settings.embed_model = embedding_model
            logs.log.info(f"Using Ollama model {model} to generate embeddings (batched)")
        else:
            endpoint = normalize_provider_endpoint(
                endpoint_value,
                default=DEFAULT_OPENAI_BASE_URL if kind == OPENAI_OFFICIAL else None,
                label="Embedding endpoint",
            )
            endpoint = validate_credential_endpoint(api_key, endpoint, kind)
            if kind == OPENAI_OFFICIAL and not api_key:
                raise ValueError("Official OpenAI embeddings require an explicit API key.")
            from llama_index.embeddings.openai import OpenAIEmbedding

            embedding_timeout = 300.0
            if deadline is not None:
                _check_ingestion_deadline(deadline)
                embedding_timeout = max(0.001, deadline - time.monotonic())
            Settings.embed_model = OpenAIEmbedding(
                model_name=model,
                api_key=api_key or "local",
                api_base=endpoint,
                embed_batch_size=_embed_batch_size(),
                timeout=embedding_timeout,
                max_retries=1,
            )
            logs.log.info(
                f"Using configured {kind} embedding model {model} "
                f"(batched, {endpoint})"
            )

        if chunk_size is not None:
            Settings.chunk_size = chunk_size
        if chunk_overlap is not None:
            Settings.chunk_overlap = chunk_overlap

        _check_ingestion_deadline(deadline)
        logs.log.info("Embedding model created successfully")
    except Exception as err:
        logs.log.error(f"Failed to setup the embedding model: {err}")
        raise


###################################
#
# Load Documents
#
###################################


EXCLUDED_FILE_PATTERNS = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.ico", "*.svg", "*.webp",
    "*.zip", "*.tar", "*.gz", "*.bz2", "*.7z", "*.rar",
    "*.exe", "*.dll", "*.so", "*.dylib", "*.bin", "*.iso", "*.msi",
    "*.pyc", "*.pyo", "*.pyd", "*.class",
    "*.mp3", "*.mp4", "*.avi", "*.mov", "*.wav", "*.mkv",
    "*.git*", "*.venv*", "*node_modules*",
]


def _path_is_excluded(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    if any(part.startswith(".") for part in relative.parts):
        return True
    candidates = [path.name, relative.as_posix(), *relative.parts[:-1]]
    return any(
        fnmatch.fnmatch(candidate.lower(), pattern.lower())
        for pattern in EXCLUDED_FILE_PATTERNS
        for candidate in candidates
    )


def _safe_input_files(data_dir: str, input_files: list | None = None) -> list[str]:
    root = Path(data_dir).resolve()
    paths = [Path(path) for path in input_files] if input_files is not None else []
    if input_files is None:
        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name
                for name in directories
                if not (current_path / name).is_symlink()
                and not _path_is_excluded(current_path / name, root)
            ]
            paths.extend(current_path / name for name in filenames)
    safe_paths = []
    for path in paths:
        try:
            if path.is_symlink() or _path_is_excluded(path, root):
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            safe_paths.append(str(resolved))
        except OSError:
            continue
    return sorted(set(safe_paths))


def _report_entry(
    filename: str,
    status: str,
    document_count: int = 0,
    extracted_characters: int = 0,
    warning: str = "",
    error: str = "",
) -> dict:
    return {
        "filename": filename,
        "status": status,
        "document_count": document_count,
        "extracted_characters": extracted_characters,
        "warning": warning,
        "error": error,
    }


def _store_extraction_report(
    report: list[dict],
    source_id: str | None = None,
    index_generation: int | None = None,
) -> None:
    safe_report = tag_report(report, source_id, index_generation)
    safe_report = [
        {
            "filename": str(entry.get("filename", "")),
            "status": str(entry.get("status", "skipped")),
            "document_count": int(entry.get("document_count", 0) or 0),
            "extracted_characters": int(entry.get("extracted_characters", 0) or 0),
            "warning": str(entry.get("warning", "") or ""),
            "error": str(entry.get("error", "") or ""),
            **(
                {"source_id": entry["source_id"]}
                if entry.get("source_id") is not None
                else {}
            ),
            **(
                {"index_generation": entry["index_generation"]}
                if entry.get("index_generation") is not None
                else {}
            ),
        }
        for entry in safe_report
    ]
    try:
        st.session_state["extraction_report"] = [dict(entry) for entry in safe_report]
    except Exception:
        pass
    try:
        st.session_state["file_extraction_report"] = [
            dict(entry) for entry in safe_report
        ]
    except Exception:
        pass


def _path_key(path: str | Path) -> str:
    try:
        return os.path.normcase(str(Path(path).resolve()))
    except (OSError, RuntimeError):
        return os.path.normcase(str(path))


def _input_path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _order_extraction_report(report: list[dict], paths: list[Path]) -> list[dict]:
    order = {}
    for index, path in enumerate(paths):
        order.setdefault(path.name, index)
    return sorted(
        report,
        key=lambda entry: order.get(str(entry.get("filename", "")), len(order)),
    )


def _document_source_key(document) -> str | None:
    metadata = document.metadata or {}
    source = metadata.get("file_path") or metadata.get("source")
    return _path_key(source) if source else None


def _parser_error_message(path: str, error: Exception) -> str:
    capability = get_parser_capability(Path(path).suffix)
    parser = capability.parser if capability else "File parser"
    if isinstance(error, LegacyFormatError):
        return legacy_conversion_message(Path(path).suffix)
    if isinstance(error, ImportError):
        dependency = capability.dependency if capability else "required parser packages"
        return f"{parser} requires {dependency}; install the declared dependency and retry."
    return f"{parser} could not extract this file ({type(error).__name__})."


def _load_one(path: str, extractors: dict) -> list:
    reader = SimpleDirectoryReader(
        input_files=[path],
        file_extractor=extractors,
        encoding="utf-8",
        errors="replace",
        raise_on_error=True,
    )
    return reader.load_data()


def _load_paths(paths: list[str], extractors: dict) -> tuple[dict[str, list], dict[str, Exception]]:
    documents_by_path = {path: [] for path in paths}
    errors: dict[str, Exception] = {}
    if not paths:
        return documents_by_path, errors
    path_lookup = {_path_key(path): path for path in paths}
    try:
        reader = SimpleDirectoryReader(
            input_files=paths,
            file_extractor=extractors,
            encoding="utf-8",
            errors="replace",
            raise_on_error=True,
        )
        bulk_documents = reader.load_data()
        for document in bulk_documents:
            source = _document_source_key(document)
            source_path = path_lookup.get(source) if source else None
            if source_path:
                documents_by_path[source_path].append(document)
        for path in paths:
            if documents_by_path[path]:
                continue
            try:
                documents_by_path[path].extend(_load_one(path, extractors))
            except Exception as error:
                errors[path] = error
    except Exception:
        logs.log.warning("Bulk parser load failed; retrying files individually")
        for path in paths:
            try:
                documents_by_path[path].extend(_load_one(path, extractors))
            except Exception as error:
                errors[path] = error
    return documents_by_path, errors


def load_documents(
    data_dir: str,
    input_files: list = None,
    return_report: bool = False,
    store_report: bool = True,
    source_id: str | None = None,
    index_generation: int | None = None,
):
    """Load safe files and retain a content-free report for every input."""
    report = []
    if input_files is not None and len(input_files) == 0:
        report.append(
            _report_entry(
                "",
                "skipped",
                warning="No files were selected for ingestion.",
            )
        )
        if store_report:
            _store_extraction_report(report, source_id, index_generation)
        raise ValueError("No files were selected for ingestion.")
    requested_paths = [Path(path) for path in input_files] if input_files is not None else []
    safe_files = _safe_input_files(data_dir, input_files)
    safe_keys = {_input_path_key(path) for path in safe_files}
    paths_for_report = requested_paths if input_files is not None else [Path(path) for path in safe_files]
    seen_paths = set()
    for path in paths_for_report:
        key = _input_path_key(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if _input_path_key(path) not in safe_keys:
            report.append(
                _report_entry(
                    path.name,
                    "skipped",
                    warning="File was excluded by local path safety checks.",
                )
            )
    loadable_paths = []
    for path in safe_files:
        capability = get_parser_capability(Path(path).suffix)
        if capability is None:
            report.append(
                _report_entry(
                    Path(path).name,
                    "unsupported",
                    error="No registered parser is available for this file type.",
                )
            )
        elif capability.category == "unsupported_without_converter":
            report.append(
                _report_entry(
                    Path(path).name,
                    "unsupported",
                    error=legacy_conversion_message(Path(path).suffix),
                )
            )
        else:
            loadable_paths.append(path)
    if not loadable_paths:
        report = _order_extraction_report(report, paths_for_report)
        if store_report:
            _store_extraction_report(report, source_id, index_generation)
        if not report:
            report.append(
                _report_entry(
                    "",
                    "skipped",
                    warning="No safe readable files were found in the selected source.",
                )
            )
            if store_report:
                _store_extraction_report(report, source_id, index_generation)
        message = next(
            (
                entry["error"] or entry["warning"]
                for entry in report
                if entry["error"] or entry["warning"]
            ),
            "No files could be loaded; see the extraction report.",
        )
        logs.log.warning("No documents loaded; see the extraction report")
        raise ValueError(message)
    try:
        extractors = build_file_extractors()
    except Exception as error:
        message = "File parser dependencies are unavailable; install the declared format dependencies and retry."
        report.extend(
            _report_entry(Path(path).name, "skipped", error=message)
            for path in loadable_paths
        )
        report = _order_extraction_report(report, paths_for_report)
        if store_report:
            _store_extraction_report(report, source_id, index_generation)
        logs.log.warning("File parser dependencies are unavailable")
        raise ValueError(message) from error
    documents_by_path, errors = _load_paths(loadable_paths, extractors)
    loaded_documents = []
    paths_with_documents = set()
    for path in loadable_paths:
        raw_documents = documents_by_path.get(path, [])
        if raw_documents:
            paths_with_documents.add(path)
        path_documents = [
            document for document in raw_documents if _document_text(document).strip()
        ]
        documents_by_path[path] = path_documents
        loaded_documents.extend(path_documents)
    loaded_documents = _verbalize_tabular_docs(loaded_documents)
    for path in loadable_paths:
        path_documents = documents_by_path.get(path, [])
        extracted_characters = sum(
            len(_document_text(document)) for document in path_documents
        )
        if not path_documents:
            if path in errors:
                report.append(
                    _report_entry(
                        Path(path).name,
                        "skipped",
                        error=_parser_error_message(path, errors[path]),
                    )
                )
            elif path in paths_with_documents:
                capability = get_parser_capability(Path(path).suffix)
                warning = "Parser returned no extractable text."
                if capability and capability.category == "optional_dependency":
                    warning += f" {capability.note}"
                report.append(
                    _report_entry(
                        Path(path).name,
                        "skipped",
                        warning=warning,
                    )
                )
            else:
                report.append(
                    _report_entry(
                        Path(path).name,
                        "skipped",
                        warning="Parser returned no documents.",
                    )
                )
            continue
        warnings = []
        capability = get_parser_capability(Path(path).suffix)
        if capability and capability.category == "optional_dependency":
            warnings.append(capability.note)
        for document in path_documents:
            warning = (document.metadata or {}).get("parser_warning")
            if warning and warning not in warnings:
                warnings.append(str(warning))
        if extracted_characters == 0:
            warning = "Parser returned no extractable text."
            if capability and capability.category == "optional_dependency":
                warning += f" {capability.note}"
            report.append(
                _report_entry(
                    Path(path).name,
                    "skipped",
                    warning=warning,
                )
            )
            continue
        report.append(
            _report_entry(
                Path(path).name,
                "loaded",
                len(path_documents),
                extracted_characters,
                warning=" ".join(warnings),
            )
        )
    report = _order_extraction_report(report, paths_for_report)
    if store_report:
        _store_extraction_report(report, source_id, index_generation)
    if not loaded_documents:
        message = next(
            (
                entry["error"] or entry["warning"]
                for entry in report
                if entry["status"] != "loaded" and (entry["error"] or entry["warning"])
            ),
            "No files could be loaded; see the extraction report.",
        )
        logs.log.warning("No documents loaded; see the extraction report")
        raise ValueError(message)
    logs.log.info(
        "Loaded %s documents; %s files skipped or unsupported",
        len(loaded_documents),
        sum(entry["status"] != "loaded" for entry in report),
    )
    if return_report:
        return loaded_documents, report
    return loaded_documents


def load_documents_with_report(
    data_dir: str,
    input_files: list = None,
    store_report: bool = True,
    source_id: str | None = None,
    index_generation: int | None = None,
):
    return load_documents(
        data_dir,
        input_files=input_files,
        return_report=True,
        store_report=store_report,
        source_id=source_id,
        index_generation=index_generation,
    )


###################################
#
# Create Document Index
#
###################################


def create_index(documents, progress_callback=None, deadline=None):
    with INGESTION_LOCK:
        return _create_index_unlocked(
            documents,
            progress_callback=progress_callback,
            deadline=deadline,
        )


def _create_index_unlocked(documents, progress_callback=None, deadline=None):
    """
    Creates an index from the provided documents and service context.

    Args:
        documents (list[str]): A list of strings representing the content of the documents to be indexed.

    Returns:
        An instance of `VectorStoreIndex`, containing the indexed data.

    Raises:
        Exception: If there is an error creating the index.

    Notes:
        The `documents` parameter should be a list of strings representing the content of the documents to be indexed.
    """

    try:
        _check_ingestion_deadline(deadline)
        nodes = run_transformations(
            documents,
            Settings.transformations,
            show_progress=True,
        )
        _check_ingestion_deadline(deadline)

        # Keep code fences intact before any filtering — a ``` block split
        # across two chunks leaves an unclosed fence that confuses embeddings.
        nodes = _merge_code_blocks(nodes)

        # Skip too-short / empty chunks: they add embedding work and retrieval
        # noise without carrying any useful facts.
        nodes = [
            node for node in nodes if len(node.get_content().strip()) >= MIN_CHUNK_CHARS
        ]
        # Drop near-duplicate chunks before embedding. Real documents often
        # repeat headers, footers or boilerplate across chunks; embedding them
        # wastes GPU work and heats the machine. Cheap word-overlap check, no
        # extra embedding calls.
        nodes = _dedupe_near_duplicate_nodes(nodes)
        # Carry each chunk's document title into the chunk text itself, so
        # title-word queries match every chunk and small models can tell
        # which document a chunk belongs to.
        nodes = _prepend_document_title(nodes)
        if not nodes:
            raise ValueError(
                "No usable content was extracted from the documents. "
                "The files may be empty or contain only images."
            )

        if progress_callback is not None:
            progress_callback(0, len(nodes))
            embed_model = ProgressReportingEmbedding(
                wrapped_model=Settings.embed_model,
                progress_callback=progress_callback,
                model_name=Settings.embed_model.model_name,
                embed_batch_size=Settings.embed_model.embed_batch_size,
            )
            embed_model._deadline = deadline
        else:
            embed_model = Settings.embed_model

        _check_ingestion_deadline(deadline)
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            show_progress=False,
        )
        _check_ingestion_deadline(deadline)

        logs.log.info("Index created from loaded documents successfully")

        return index
    except TimeoutError:
        raise
    except Exception as err:
        logs.log.error(f"Index creation failed: {err}")
        if _is_oom_error(err):
            raise Exception(
                "Not enough GPU memory to embed these documents. "
                "Try a smaller chunk size, fewer documents, or a smaller "
                "embedding model."
            ) from err
        raise Exception(f"Index creation failed: {err}")


###################################
#
# Index Cache (disk persistence)
#
###################################


def _document_text(document):
    if hasattr(document, "get_content"):
        return document.get_content() or ""
    if hasattr(document, "text"):
        return document.text or ""
    return str(document)


def index_cache_key(
    documents,
    settings=None,
    source_identity=None,
    settings_signature=None,
) -> str:
    """Return a stable key for source identity and active indexing settings."""
    try:
        embedding = Settings.embed_model
    except Exception:
        embedding = None
    embedding = getattr(embedding, "wrapped_model", embedding)
    model_name = str(getattr(embedding, "model_name", "unknown"))
    model_type = embedding.__class__.__name__ if embedding is not None else "unconfigured"
    endpoint = str(
        getattr(embedding, "base_url", None)
        or getattr(embedding, "api_base", None)
        or ""
    )
    api_key = getattr(embedding, "api_key", None)
    credential_fingerprint = (
        hashlib.sha256(str(api_key).encode()).hexdigest() if api_key else None
    )
    if settings is None and settings_signature is not None:
        settings = {"settings_signature": settings_signature}
    if settings is None:
        chunk_size = Settings.chunk_size
        chunk_overlap = Settings.chunk_overlap
        settings = {
            "version": INDEX_CACHE_VERSION,
            "parser_cache_version": PARSER_CACHE_VERSION,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_overlap_pct": round(chunk_overlap / chunk_size * 100)
            if chunk_size
            else 0,
            "embedding_provider": model_type,
            "embedding_model": model_name,
            "embedding_endpoint": endpoint,
            "embedding_credential_fingerprint": credential_fingerprint,
        }
    elif isinstance(settings, dict) and "chunk_overlap_pct" not in settings:
        chunk_size = settings.get("chunk_size")
        chunk_overlap = settings.get("chunk_overlap")
        if chunk_size and chunk_overlap is not None:
            settings = {
                **settings,
                "chunk_overlap_pct": round(chunk_overlap / chunk_size * 100),
            }
    records = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        metadata_identity = {
            key: metadata.get(key)
            for key in ("file_name", "source", "url", "title")
            if metadata.get(key) is not None
        }
        records.append(
            {
                "text": _document_text(document),
                "metadata": metadata_identity,
            }
        )
    records.sort(key=lambda record: json.dumps(record, sort_keys=True, default=str))
    payload = {
        "version": INDEX_CACHE_VERSION,
        "settings": dict(settings),
        "source_identity": source_identity,
        "documents": records,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def index_cache_dir(
    documents,
    settings=None,
    source_identity=None,
    settings_signature=None,
) -> Optional[str]:
    """Return the cache directory for these documents, or None on failure."""
    try:
        return os.path.join(
            INDEX_CACHE_DIR,
            index_cache_key(
                documents,
                settings=settings,
                source_identity=source_identity,
                settings_signature=settings_signature,
            ),
        )
    except Exception:
        return None


def load_index_from_cache(cache_dir: str):
    """Load a persisted index, or return None if it can't be restored."""
    try:
        if not os.path.isdir(cache_dir):
            return None
        storage_context = StorageContext.from_defaults(persist_dir=cache_dir)
        index = load_index_from_storage(storage_context)
        logs.log.info(f"Index loaded from cache: {cache_dir}")
        return index
    except Exception as err:
        logs.log.warning(f"Failed to load cached index, rebuilding: {err}")
        return None


def persist_index_to_cache(index, cache_dir: str):
    """Persist an index to its content-addressed cache directory."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=cache_dir)
        logs.log.info(f"Index persisted to cache: {cache_dir}")
        return True
    except Exception as err:
        logs.log.warning(f"Failed to persist index cache: {err}")
        return False


def _prune_index_cache(keep: int = INDEX_CACHE_KEEP):
    """Remove the oldest cache entries beyond `keep`."""
    try:
        if not os.path.isdir(INDEX_CACHE_DIR):
            return
        entries = sorted(
            (
                os.path.getmtime(os.path.join(INDEX_CACHE_DIR, name)),
                os.path.join(INDEX_CACHE_DIR, name),
            )
            for name in os.listdir(INDEX_CACHE_DIR)
            if os.path.isdir(os.path.join(INDEX_CACHE_DIR, name))
        )
        for _, path in entries[:-keep]:
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


###################################
#
# Create Query Engine
#
###################################


class QueryEngineBundle(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err


QueryEngineCandidate = QueryEngineBundle


# @st.cache_resource(show_spinner=False)
def create_query_engine(
    documents,
    progress_callback=None,
    *,
    settings=None,
    source_identity=None,
    settings_signature=None,
    deadline=None,
):
    """Build and return a query bundle without changing session state."""
    with INGESTION_LOCK:
        try:
            _check_ingestion_deadline(deadline)
            cache_dir = index_cache_dir(
                documents,
                settings=settings,
                source_identity=source_identity,
                settings_signature=settings_signature,
            )
            index = load_index_from_cache(cache_dir) if cache_dir else None
            _check_ingestion_deadline(deadline)

            if index is None:
                index_kwargs = {"progress_callback": progress_callback}
                if deadline is not None:
                    index_kwargs["deadline"] = deadline
                index = create_index(documents, **index_kwargs)
                if cache_dir:
                    persist_index_to_cache(index, cache_dir)
                _check_ingestion_deadline(deadline)

            try:
                similarity_top_k = max(int(st.session_state.get("top_k", 3)), 1)
            except (TypeError, ValueError, OverflowError):
                similarity_top_k = 3
            query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode="compact",
                streaming=True,
            )
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": TEXT_QA_TEMPLATE}
            )
            vector_retriever = index.as_retriever(
                similarity_top_k=similarity_top_k
            )
            retriever = build_hybrid_retriever(
                vector_retriever,
                index,
                similarity_top_k,
                st.session_state.get("similarity_cutoff", 0.3),
                candidate_depth=st.session_state.get(
                    "candidate_depth",
                    st.session_state.get("retrieval_candidate_depth"),
                ),
                vector_candidate_depth=st.session_state.get("vector_candidate_depth"),
                bm25_candidate_depth=st.session_state.get("bm25_candidate_depth"),
            )
            bundle = QueryEngineBundle(
                query_engine=query_engine,
                retriever=retriever,
                index=index,
                cache_key=os.path.basename(cache_dir) if cache_dir else None,
            )
            _check_ingestion_deadline(deadline)
            logs.log.info("Query Engine candidate created successfully")
            return bundle
        except TimeoutError:
            raise
        except Exception as err:
            logs.log.error(f"Error when creating Query Engine: {err}")
            raise Exception(f"Error when creating Query Engine: {err}")


def create_query_engine_candidate(*args, **kwargs):
    return create_query_engine(*args, **kwargs)


create_candidate_bundle = create_query_engine
