import hashlib
import os
import re
import shutil
from typing import Optional

# Transformers 5.x can emit a large volume of non-actionable alias warnings
# during startup. Keep app logs focused on failures we can act on.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
import ollama
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi

import utils.logs as logs

from llama_index.core.embeddings import BaseEmbedding

# This is not used but required by llama-index and must be set FIRST
os.environ["OPENAI_API_KEY"] = "sk-abc123"

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.ingestion import run_transformations
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore

# On-disk cache of built indexes, keyed by document content + embedding
# settings. Re-ingesting the same documents loads the saved index instead of
# re-embedding everything: no repeated GPU work, no repeated heat.
INDEX_CACHE_DIR = os.path.join(os.getcwd(), ".index_cache")
INDEX_CACHE_KEEP = 5

# Explicit, document-grounded QA instruction. Small local models (0.5B-8B)
# answer far more reliably when the prompt tells them to use ONLY the context
# and to refuse to answer from prior knowledge.
TEXT_QA_TEMPLATE = PromptTemplate(
    "You are DocMind AI, a document-grounded assistant.\n"
    "Answer the query using ONLY the context provided below.\n"
    "Do not use prior knowledge. Do not invent information.\n"
    "Quote exact numbers, dates and names from the context when present.\n"
    "When a fact comes from a numbered chunk, cite it like this: (from [n]).\n"
    "If the context does not contain the answer, reply exactly:\n"
    "\"I could not find this information in the documents.\"\n"
    "Keep the answer concise and factual: 1-3 short sentences.\n\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)


# Documents with less content than this are skipped during ingestion: too
# short to embed usefully, they only add embedding work and retrieval noise.
MIN_CHUNK_CHARS = 50

# Rough token budget for the retrieved context sent to the LLM (chars/4).
# Excess chunks are trimmed from the bottom of the ranking, keeping the
# prompt comfortably inside the 2048-token window.
CONTEXT_CHAR_BUDGET = 4800

# Below this vector score a chunk is only trusted when BM25 also found an
# exact keyword match. Small local embedding models score nearly everything
# in the 0.3-0.5 range, so the similarity cutoff alone cannot tell a real
# match from a random one. Requiring keyword evidence prevents unrelated
# questions from pulling irrelevant context (which makes small models
# hallucinate) and lets the "could not find" fallback actually fire.
VECTOR_EVIDENCE_FLOOR = 0.5

# Bump when retrieval/ingestion logic changes so stale caches rebuild once.
INDEX_CACHE_VERSION = 5

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
    """Jaccard-style overlap of stemmed word sets, used for near-duplicate detection.

    Approximates embedding cosine similarity (>0.95) with pure text overlap so
    no extra embedding calls (and no GPU/heat) are spent finding duplicates.
    """
    wa = set(_bm25_tokens(a))
    wb = set(_bm25_tokens(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _dedupe_near_duplicate_nodes(nodes, threshold: float = 0.95):
    """Drop chunks nearly identical to an earlier chunk (headers, footers, boilerplate).

    Embedding near-duplicates wastes embedding work and heats the machine, so
    they are filtered out before the expensive step. Order is preserved.
    """
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
    """Return whether an exception looks like a GPU out-of-memory failure.

    Ollama proxies GPU failures from its runner with messages such as
    "CUDA out of memory" or plain "out of memory". Recognising them lets the
    pipeline give a targeted hint instead of a raw stack trace.
    """
    message = str(err).lower()
    return ("out of memory" in message or "oom" in message) or (
        "cuda" in message and "memory" in message
    )


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
    "polici": {"rule", "guideline", "term", "condition"},
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
    stemmer = _stemmer()
    text = " ".join(stemmer.stem(word) for word in _bm25_tokens(query))
    words = [word for word in text.split() if word not in _QUERY_FILLER_WORDS]
    return " ".join(words) or text


class HybridRetriever:
    """Fuse vector similarity with BM25 keyword scores.

    BM25 runs entirely on the CPU in a few milliseconds per query and catches
    exact keyword matches (names, codes, numbers) that vector search misses.
    Fusion uses Reciprocal Rank Fusion (RRF): no extra GPU work, no LLM calls.
    Includes similarity cutoff, duplicate filtering, and context budgeting.
    """

    def __init__(self, vector_retriever, docstore, corpus, top_k, similarity_cutoff=None):
        self.vector_retriever = vector_retriever
        self.docstore = docstore
        self.corpus = corpus
        self.top_k = top_k
        self.similarity_cutoff = similarity_cutoff if similarity_cutoff is not None else 0.3
        self._bm25 = BM25Okapi([self._tokenize(node_id) for node_id in corpus])
        self.intro_node_ids = self._intro_node_ids()

    def _tokenize(self, node_id):
        node = self.docstore.get_node(node_id)
        return _bm25_tokens(node.get_content())

    def _intro_node_ids(self, count=2):
        """Return the opening chunks of the corpus (document introductions)."""
        try:
            def sort_key(node_id):
                node = self.docstore.get_node(node_id)
                metadata = node.metadata or {}
                start = getattr(node, "start_char_idx", None) or 0
                return (metadata.get("file_name", ""), start)

            return sorted(self.corpus, key=sort_key)[:count]
        except Exception:
            return []

    @staticmethod
    def _is_doc_level_query(query: str, rewritten: str) -> bool:
        """Heuristically detect questions about the document itself.

        "What is this document about?", "summarize the document", "tell me
        about the uploaded file" are answered best by the document's opening
        chunks, which state the topic explicitly. Specific fact questions keep
        many meaningful tokens after query cleaning and are left untouched.
        """
        lower = query.lower()
        if "summar" in lower or "overview" in lower:
            return True
        if "document" in lower and len(rewritten.split()) <= 3:
            return True
        return False

    def _apply_live_settings(self):
        """Refresh Top K / similarity cutoff from session state.

        The retriever is cached for the session, but the Advanced Settings
        sliders change `session_state` directly. Reading them here makes slider
        changes take effect on the very next query instead of after a re-ingest.
        """
        try:
            top_k = max(int(st.session_state.get("top_k", self.top_k)), 1)
            cutoff = float(
                st.session_state.get("similarity_cutoff", self.similarity_cutoff)
            )
        except (TypeError, ValueError):
            return
        if _is_eco_mode():
            # Eco Mode: fewer chunks per query = less prompt = less heat.
            top_k = min(top_k, 3)
        self.top_k = top_k
        self.similarity_cutoff = cutoff
        # The underlying vector retriever caps its candidate list too; raise it
        # so a larger Top K can actually surface more chunks.
        try:
            if self.vector_retriever._similarity_top_k != top_k:
                self.vector_retriever._similarity_top_k = top_k
        except AttributeError:
            pass

    def retrieve(self, query: str):
        self._apply_live_settings()
        rewritten = _rewrite_query(query)
        rrf_scores = {}
        vector_scores = {}

        # Vector search ranks
        vector_nodes = self.vector_retriever.retrieve(rewritten)
        for rank, node_score in enumerate(vector_nodes):
            node_id = node_score.node.node_id
            vector_scores[node_id] = max(
                vector_scores.get(node_id, 0.0), node_score.score
            )
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (60 + rank)

        # BM25 keyword ranks (exact term matches). Short queries are widened with
        # curated synonyms; the vector search above still used the clean query.
        tokens = _bm25_tokens(rewritten)
        expanded_tokens = _bm25_expanded_tokens(tokens)
        bm25_hit_ids = set()
        if expanded_tokens:
            bm25_scores = self._bm25.get_scores(expanded_tokens)
            bm25_hit_ids = {
                node_id
                for node_id, score in zip(self.corpus, bm25_scores)
                if score > 0.0
            }
            bm25_ids = self._bm25.get_top_n(
                expanded_tokens, self.corpus, n=len(self.corpus)
            )
            for rank, node_id in enumerate(bm25_ids):
                rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (60 + rank)

        # Drop weak matches: below the similarity cutoff they are more likely
        # to cause hallucinated answers than to help. Chunks with a weak
        # vector score additionally need a BM25 keyword hit as evidence,
        # otherwise an unrelated question could still pull irrelevant context
        # (small local embedding models score everything in a tight band).
        def _is_credible(node_id):
            vector = vector_scores.get(node_id, 0.0)
            if vector < self.similarity_cutoff:
                return False
            if (
                self.similarity_cutoff > 0
                and vector < VECTOR_EVIDENCE_FLOOR
                and node_id not in bm25_hit_ids
            ):
                return False
            return True

        filtered = {
            node_id: rrf
            for node_id, rrf in rrf_scores.items()
            if _is_credible(node_id)
        }
        if not filtered:
            return []

        # Keep the top chunks within a token budget that fits the model's
        # context window (light context compression).
        ranked = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
        selected = []
        used_chars = 0
        context_budget = _context_char_budget()

        for node_id, rrf_score in ranked:
            content = self.docstore.get_node(node_id).get_content()
            # Duplicate check against already-selected chunks
            if any(_text_overlap_ratio(content, self.docstore.get_node(sid).get_content()) > 0.95 for sid, _ in selected):
                continue
            if selected and used_chars + len(content) > context_budget:
                break
            used_chars += len(content)
            selected.append((node_id, rrf_score))

        max_nodes = self.top_k
        if self._is_doc_level_query(query, rewritten):
            # Document-level questions ("what is this document about?", "summarize
            # it") are best answered from the document's introduction, which states
            # the topic explicitly. Prepending the opening chunks gives the small
            # local model the summary material it needs instead of mid-document
            # fragments (which made it refuse). Ranked results are trimmed from
            # the tail to make room inside the context budget.
            existing_ids = {node_id for node_id, _ in selected}
            intro = []
            for node_id in self.intro_node_ids:
                if node_id in existing_ids:
                    continue
                intro.append((node_id, rrf_scores.get(node_id, 0.0)))
            selected = intro + selected
            total_chars = used_chars + sum(
                len(self.docstore.get_node(node_id).get_content())
                for node_id, _ in intro
            )
            while selected and total_chars > context_budget:
                removed_id, _ = selected.pop()
                total_chars -= len(self.docstore.get_node(removed_id).get_content())
            max_nodes = self.top_k + len(intro)

        return [
            NodeWithScore(
                node=self.docstore.get_node(node_id),
                score=vector_scores.get(node_id, rrf_score),
            )
            for node_id, rrf_score in selected[:max_nodes]
        ]


def build_hybrid_retriever(vector_retriever, index, top_k, similarity_cutoff=None):
    """Return a vector+BM25 hybrid retriever over the index's docstore."""
    try:
        docstore = index.docstore
        corpus = list(docstore.docs.keys())
        if similarity_cutoff is None:
            similarity_cutoff = st.session_state.get("similarity_cutoff", 0.3)
        return HybridRetriever(
            vector_retriever=vector_retriever,
            docstore=docstore,
            corpus=corpus,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
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
            batch = texts[start:start + batch_size]
            result.extend(
                self.wrapped_model.get_text_embedding_batch(
                    batch,
                    show_progress=False,
                    **kwargs,
                )
            )
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

    def _client_inst(self):
        # Reuse a single persistent client. Creating a fresh client per call
        # opens a new connection each time, which on Windows can cost ~2s per
        # embed due to IPv6 -> IPv4 fallback against a 127.0.0.1-bound server.
        client = self._client
        if client is None:
            client = ollama.Client(host=self.base_url, timeout=300)
            self._client = client
        return client

    def _get_query_embedding(self, query: str):
        return self._client_inst().embed(model=self.model_name, input=[query]).embeddings[0]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str):
        return self._client_inst().embed(model=self.model_name, input=[text]).embeddings[0]

    def get_text_embedding_batch(self, texts, show_progress=False, **kwargs):
        """Send chunks in batches to Ollama, shrinking the batch on failure.

        A batch that is too large can exhaust GPU memory on the Ollama server
        (CUDA OOM). The failed batch is retried with a smaller size so ingestion
        finishes instead of crashing mid-way; the smaller size is kept for the
        rest of the run to avoid repeated failures.
        """
        result = []
        client = self._client_inst()
        index = 0
        batch_size = self.embed_batch_size
        while index < len(texts):
            batch = texts[index : index + batch_size]
            try:
                response = client.embed(model=self.model_name, input=batch)
                result.extend(response.embeddings)
                index += len(batch)
            except Exception as err:
                if len(batch) <= 1:
                    raise
                logs.log.warning(
                    f"Embedding batch of {len(batch)} failed ({err}); "
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


def verify_embedding_model(model: str, base_url: str) -> bool:
    """Return whether a model exists on the Ollama server and can embed.

    Guards against silently ingesting with a stale or mistyped embedding model.
    Checking the live server here is authoritative: the session-state model list
    can go stale after an endpoint change or a model removal.
    """
    try:
        client = ollama.Client(host=base_url, timeout=30)
        if model not in _ollama_model_names(client):
            return False
        details = client.show(model)
        capabilities = getattr(details, "capabilities", None) or details.get("capabilities", [])
        return "embedding" in capabilities
    except Exception:
        return False


# Note: NOT cached - LlamaIndex Settings are global and must always be updated
# to reflect the current model/backend/chunk settings from session state.
def setup_embedding_model(
    model: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    backend: str = "Ollama",
    ollama_endpoint: Optional[str] = None,
    api_key: str = "",
):
    """
    Sets up the embedding model used for document ingestion.

    Args:
        model (str): The name of the embedding model to use.
        chunk_size (int, optional): Chunk size in tokens.
        chunk_overlap (int, optional): Chunk overlap in tokens.
        backend (str): "Ollama" (local Ollama server) or "OpenAI"
            (any OpenAI-compatible embeddings endpoint).
        ollama_endpoint (str, optional): Server base URL.
        api_key (str, optional): API key for OpenAI-compatible backends.

    Returns:
        An instance of the configured embedding model.

    Raises:
        ValueError: If the specified model is not a valid embedding model.
    """
    try:
        if not ollama_endpoint:
            raise ValueError("A server endpoint is required for embeddings")

        # Lazy import: utils.ollama imports TEXT_QA_TEMPLATE from this module,
        # so a top-level import here would be circular.
        from utils.ollama import _embed_batch_size

        if backend == "OpenAI":
            from llama_index.embeddings.openai import OpenAIEmbedding

            Settings.embed_model = OpenAIEmbedding(
                model_name=model,
                api_key=api_key or "sk-docmind-local",
                api_base=ollama_endpoint,
                embed_batch_size=_embed_batch_size(),
            )
            logs.log.info(
                f"Using OpenAI-compatible model {model} to generate embeddings "
                f"(batched, {ollama_endpoint})"
            )
        else:
            if not verify_embedding_model(model, ollama_endpoint):
                raise ValueError(
                    f"Embedding model '{model}' is not available on the Ollama server "
                    f"at {ollama_endpoint}. Pull it first with: ollama pull {model}"
                )
            Settings.embed_model = OllamaEmbedding(
                model_name=model,
                base_url=ollama_endpoint,
                embed_batch_size=_embed_batch_size(),
            )
            logs.log.info(f"Using Ollama model {model} to generate embeddings (batched)")

        if chunk_size is not None:
            Settings.chunk_size = chunk_size
        if chunk_overlap is not None:
            Settings.chunk_overlap = chunk_overlap

        logs.log.info(f"Embedding model created successfully")

        return
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


def load_documents(data_dir: str, input_files: list = None):
    """
    Loads documents from a directory of files with binary and archive exclusions.

    Args:
        data_dir (str): The path to the directory containing the documents to be loaded.
        input_files (list, optional): An explicit list of files to load. When provided,
            only these files are read, ignoring anything else in the directory.

    Returns:
        A list of documents, where each document is a string representing the content of the corresponding file.

    Raises:
        Exception: If there is an error creating the data index.
    """
    try:
        if input_files:
            files = SimpleDirectoryReader(
                input_files=input_files,
                exclude=EXCLUDED_FILE_PATTERNS,
            )
        else:
            files = SimpleDirectoryReader(
                input_dir=data_dir,
                recursive=True,
                exclude=EXCLUDED_FILE_PATTERNS,
            )
        documents = files.load_data()
        logs.log.info(f"Loaded {len(documents):,} documents from files")
        return documents
    except Exception as err:
        logs.log.error(f"Error creating data index: {err}")
        raise Exception(f"Error creating data index: {err}")


###################################
#
# Create Document Index
#
###################################


def create_index(documents, progress_callback=None):
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
        nodes = run_transformations(
            documents,
            Settings.transformations,
            show_progress=True,
        )

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
        else:
            embed_model = Settings.embed_model

        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            show_progress=False,
        )

        logs.log.info("Index created from loaded documents successfully")

        return index
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


def index_cache_key(documents) -> str:
    """Return a stable cache key for a document set + embedding settings."""
    hasher = hashlib.sha256()
    hasher.update(str(INDEX_CACHE_VERSION).encode("utf-8"))
    embed_model = getattr(Settings.embed_model, "model_name", "unknown")
    hasher.update(str(embed_model).encode("utf-8"))
    hasher.update(str(Settings.chunk_size).encode("utf-8"))
    hasher.update(str(Settings.chunk_overlap).encode("utf-8"))
    texts = sorted(_document_text(document) for document in documents)
    for text in texts:
        hasher.update(text.encode("utf-8", errors="ignore"))
    return hasher.hexdigest()[:20]


def index_cache_dir(documents) -> Optional[str]:
    """Return the cache directory for these documents, or None on failure."""
    try:
        return os.path.join(INDEX_CACHE_DIR, index_cache_key(documents))
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
    """Persist an index to disk and prune old caches."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=cache_dir)
        logs.log.info(f"Index persisted to cache: {cache_dir}")
        _prune_index_cache()
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


# @st.cache_resource(show_spinner=False)
def create_query_engine(documents, progress_callback=None):
    """
    Creates a query engine from the provided documents and service context.

    Args:
        documents (list[str]): A list of strings representing the content of the documents to be indexed.

    Returns:
        An instance of `QueryEngine`, containing the indexed data and allowing for querying of the data using a variety of parameters.

    Raises:
        Exception: If there is an error creating the query engine.

    Notes:
        The `documents` parameter should be a list of strings representing the content of the documents to be indexed.

        This function uses the `create_index` function to create an index from the provided documents and service context, and then creates a query engine from the resulting index. The `query_engine` parameter is used to specify the parameters of the query engine, including the number of top-ranked items to return (`similarity_top_k`) and the response mode (`response_mode`).
    """
    try:
        # Reuse a persisted index when the same documents + settings have
        # already been embedded (no repeated GPU work / heat).
        cache_dir = index_cache_dir(documents)
        index = load_index_from_cache(cache_dir) if cache_dir else None

        if index is None:
            index = create_index(documents, progress_callback=progress_callback)
            if cache_dir:
                persist_index_to_cache(index, cache_dir)

        # top_k of 0 would break retrieval; clamp to a sane minimum.
        similarity_top_k = max(int(st.session_state.get("top_k", 3)), 1)

        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact",
            streaming=True,
        )

        # Replace the generic llama-index QA prompt with the explicit
        # document-grounded template above.
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": TEXT_QA_TEMPLATE}
        )

        # Keep a dedicated retriever in session state so doc-mode chat can
        # retrieve and stream token-by-token instead of buffering the whole
        # response through the (non-streaming) compact synthesizer. Vector +
        # BM25 hybrid catches keyword matches that pure vector search misses.
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        st.session_state["retriever"] = build_hybrid_retriever(
            vector_retriever, index, similarity_top_k,
            st.session_state.get("similarity_cutoff", 0.3)
        )

        st.session_state["query_engine"] = query_engine

        logs.log.info("Query Engine created successfully")

        return query_engine
    except Exception as e:
        logs.log.error(f"Error when creating Query Engine: {e}")
        raise Exception(f"Error when creating Query Engine: {e}")
