# RAG Pipeline

DocMind maintains one active local index at a time. A successful new local ingestion replaces the previous active local query engine.

## Local Ingestion Order

1. Read the active provider and chat-model settings.
2. Create and validate the selected LLM.
3. Validate chunk size and overlap.
4. Configure and validate the embedding model.
5. Load source documents.
6. Enforce post-load limits:
   - 300 loaded document objects
   - 4 MiB of extracted Python characters
7. Attempt to load a compatible persisted index.
8. If no compatible cache exists, transform, deduplicate, embed, and build an in-memory `VectorStoreIndex`.
9. Persist the index cache when the filesystem is writable.
10. Create the query engine and hybrid retriever.
11. Remove transient files under `data/` after successful indexing.

Failed local ingestion may leave transient files because several failure paths stop the Streamlit run before success-only cleanup.

## Source Loading

- Local files are saved under `data/` and loaded from an explicit safe file list.
- GitHub repositories are shallow-cloned and loaded from the checkout directory.
- Websites are fetched first and passed to the pipeline as LlamaIndex `Document` objects.

The loader rejects paths outside the selected source root, file symlinks, hidden paths, and configured excluded patterns. It retries unreadable files individually so one parser failure does not necessarily stop the complete batch.

The ingestion limits run after loading. They do not bound clone size, parser CPU/memory use, decompression, or PDF page count before parsing.

## Text Processing

Local files and GitHub content pass through:

1. Basic table verbalization for selected CSV, TSV, and JSON documents
2. LlamaIndex sentence/paragraph-aware chunking
3. Simple odd-fence code-block repair
4. Removal of chunks with fewer than 15 non-whitespace characters
5. Stemmed-token overlap deduplication
6. Title enrichment for file-backed chunks
7. Embedding and vector-index construction

JSONL is normally parsed as one JSON document; ordinary multi-line JSONL often bypasses table verbalization. Website documents do not pass through the local table verbalizer because they enter the pipeline as already-created `Document` objects.

## Embedding

Ollama embedding batches begin at:

- 16 chunks normally
- 4 chunks in Eco Mode

On any batch exception, the current Ollama wrapper halves the batch down to one. This helps with out-of-memory failures but also retries non-batch errors such as timeouts.

OpenAI-compatible embeddings use the installed LlamaIndex OpenAI adapter and do not use the same project-level batch-shrinking loop.

## Index Cache

The cache retains up to five index directories. Cache identity includes:

- Cache version
- Embedding adapter class
- Embedding model name
- Embedding endpoint
- Chunk size and overlap
- Extracted text
- Source identity metadata such as filename or URL

The cache is unencrypted local storage. It avoids re-embedding matching inputs but does not cache generated answers.

## Query Flow

1. Normalize basic number words, hyphens, ASCII tokens, and selected filler words.
2. Retrieve vector candidates.
3. Build a BM25 ranking over the current index corpus.
4. Fuse vector and BM25 ranks with Reciprocal Rank Fusion.
5. Apply the similarity cutoff.
6. For a positive cutoff, require positive BM25 evidence when vector score is below 0.5.
7. Add a quoted-phrase boost when applicable.
8. Remove duplicate selected chunks.
9. Apply the 4,800-character context budget, or 3,200 in Eco Mode.
10. Add up to two introduction chunks for recognized document-level questions.
11. Number the selected context blocks.
12. Stream the model response.

BM25 searches the full corpus, but under a positive cutoff a keyword-only node absent from the vector candidate list has vector score zero and is rejected. BM25 therefore reinforces eligible vector candidates rather than acting as a completely independent retrieval path at the default cutoff.

## Generation

Local RAG adds the system-style message and approximately 500 tokens of recent RAG history, or 300 in Eco Mode. Token counts use `len(text) // 4` and are estimates.

The final user message contains the grounded template, numbered context, and current question. The model response is streamed. If the expected citation substring is missing, `(from [1])` is appended as a formatting fallback; citation numbers and factual support are not automatically verified.

## R2R Path

R2R currently bypasses the local pipeline only for local-file uploads. It calls the configured R2R server, stores returned document IDs in session state, and sends future R2R chat requests to that server.

GitHub and website ingestion continue to use the local pipeline. R2R chat is synchronous, does not receive the application's answer-style/history settings, and does not populate the local source-chip state.
