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

# Chunks whose best vector similarity falls below this are treated as
# irrelevant: dropped from the context (and if ALL chunks are weak, the app
# refuses to answer instead of hallucinating). nomic-embed relevant matches
# typically score 0.4+; keep the cutoff conservative to avoid false misses.
SIMILARITY_CUTOFF = 0.35

# Rough token budget for the retrieved context sent to the LLM (chars/4).
# Excess chunks are trimmed from the bottom of the ranking, keeping the
# prompt comfortably inside the 2048-token window.
CONTEXT_CHAR_BUDGET = 4800

# Bump when retrieval/ingestion logic changes so stale caches rebuild once.
INDEX_CACHE_VERSION = 2


def _bm25_tokens(text):
    """Lowercase word tokens without punctuation for BM25 matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


# Filler words that carry no retrieval value: removing them helps BM25
# weight the real terms and slightly improves vector matching too.
_QUERY_FILLER_WORDS = {
    "please", "tell", "me", "about", "can", "you", "could", "would",
    "i", "want", "to", "know", "what", "is", "are", "was", "were",
    "the", "a", "an", "of", "for", "with", "and", "or", "do", "does",
}


def _rewrite_query(query):
    """Light, rule-based query cleanup: no LLM calls, no GPU cost."""
    text = " ".join(_bm25_tokens(query))
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

    def _tokenize(self, node_id):
        node = self.docstore.get_node(node_id)
        return _bm25_tokens(node.get_content())

    def retrieve(self, query: str):
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

        # BM25 keyword ranks (exact term matches)
        tokens = _bm25_tokens(rewritten)
        if tokens:
            bm25_ids = self._bm25.get_top_n(tokens, self.corpus, n=len(self.corpus))
            for rank, node_id in enumerate(bm25_ids):
                rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (60 + rank)

        # Drop weak matches: below the similarity cutoff they are more likely
        # to cause hallucinated answers than to help.
        filtered = {
            node_id: rrf
            for node_id, rrf in rrf_scores.items()
            if vector_scores.get(node_id, 0.0) >= self.similarity_cutoff
        }
        if not filtered:
            return []

        # Keep the top chunks within a token budget that fits the model's
        # context window (light context compression).
        ranked = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
        selected = []
        used_chars = 0

        # Deduplicate: skip chunks whose content is nearly identical to an
        # already-selected chunk (cosine similarity > 0.95 on embeddings would
        # be ideal, but we approximate with simple text overlap ratio to avoid
        # extra embedding calls — fast, zero GPU, good enough for near-dups).
        def _overlap_ratio(a, b):
            wa = set(_bm25_tokens(a))
            wb = set(_bm25_tokens(b))
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / min(len(wa), len(wb))

        for node_id, rrf_score in ranked:
            content = self.docstore.get_node(node_id).get_content()
            # Duplicate check against already-selected chunks
            if any(_overlap_ratio(content, self.docstore.get_node(sid).get_content()) > 0.95 for sid, _ in selected):
                continue
            if selected and used_chars + len(content) > CONTEXT_CHAR_BUDGET:
                break
            used_chars += len(content)
            selected.append((node_id, rrf_score))

        return [
            NodeWithScore(
                node=self.docstore.get_node(node_id),
                score=vector_scores.get(node_id, rrf_score),
            )
            for node_id, rrf_score in selected[: self.top_k]
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
        """Send chunks in batches to Ollama for much faster embedding."""
        result = []
        client = self._client_inst()
        for start in range(0, len(texts), self.embed_batch_size):
            batch = texts[start : start + self.embed_batch_size]
            response = client.embed(model=self.model_name, input=batch)
            result.extend(response.embeddings)
        return result


###################################
#
# Setup Embedding Model
#
###################################


# Note: NOT cached - LlamaIndex Settings are global and must always be updated
# to reflect the current model/backend/chunk settings from session state.
def setup_embedding_model(
    model: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    backend: str = "Local Hugging Face",
    ollama_endpoint: Optional[str] = None,
):
    """
    Sets up an embedding model using the Hugging Face library.

    Args:
        model (str): The name of the embedding model to use.

    Returns:
        An instance of the HuggingFaceEmbedding class, configured with the specified model and device.

    Raises:
        ValueError: If the specified model is not a valid embedding model.

    Notes:
        The `device` parameter can be set to 'cpu' or 'cuda' to specify the device to use for the embedding computations. If 'cuda' is used and CUDA is available, the embedding model will be run on the GPU. Otherwise, it will be run on the CPU.
    """
    try:
        if backend == "Ollama":
            if not ollama_endpoint:
                raise ValueError("Ollama endpoint is required for Ollama embeddings")
            Settings.embed_model = OllamaEmbedding(
                model_name=model,
                base_url=ollama_endpoint,
                embed_batch_size=16,
            )
            logs.log.info(f"Using Ollama model {model} to generate embeddings (batched)")
        else:
            try:
                from torch import cuda
                device = "cpu" if not cuda.is_available() else "cuda"
            except Exception:
                device = "cpu"
            if device == "cpu":
                logs.log.warning(
                    "CUDA is not available. Local HuggingFace embeddings will run on "
                    "the CPU, which is slow and generates significant heat. Prefer the "
                    "Ollama embedding backend, which runs on your GPU."
                )
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logs.log.info(f"Using {device} to generate embeddings")
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=model,
                device=device,
            )

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


def load_documents(data_dir: str):
    """
    Loads documents from a directory of files with binary and archive exclusions.

    Args:
        data_dir (str): The path to the directory containing the documents to be loaded.

    Returns:
        A list of documents, where each document is a string representing the content of the corresponding file.

    Raises:
        Exception: If there is an error creating the data index.
    """
    try:
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
            response_mode=st.session_state["chat_mode"],
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
