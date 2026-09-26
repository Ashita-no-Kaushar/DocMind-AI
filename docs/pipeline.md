# RAG Pipeline

DocMind keeps one active local source/index at a time. The pipeline is transactional: a candidate source is prepared and validated, then session state is committed only after the query bundle is complete. A failed replacement restores the previously committed local state when one exists.

## Local ingestion order

1. Read and validate provider, embedding, and chunk settings.
2. Calculate a content signature, source ID, settings signature, and next index generation.
3. Create an operation-owned work directory for uploads or source preparation.
4. Acquire the cross-process ingestion lifecycle lock.
5. Initialize and validate the selected chat provider.
6. Initialize and validate the embedding provider.
7. Load source files or prebuilt website `Document` objects.
8. Apply parser and document budgets and create a per-file extraction report.
9. Transform, filter, deduplicate, title-enrich, embed, and construct a candidate index.
10. Validate and atomically persist a cache entry when cache storage is writable.
11. Create the query engine, vector retriever, and hybrid retriever.
12. Commit documents, retriever, query engine, source state, generation, and extraction report together.
13. Remove the operation-owned work directory when safe.

A failed ingestion can leave a work directory behind if Windows file locks or filesystem errors prevent deletion. Reset Project retries cleanup for owned work paths. The cache is best-effort: a read-only or unavailable cache does not prevent an in-memory index from being used.

The default ingestion limits are 300 loaded document records and 4 MiB extracted Python characters. Individual parsers apply source-byte, JSON, MIME, table, PDF, and archive limits before or during extraction.

## Source loading

### Local files

Uploads are saved to a unique operation directory under `data/work/`, validated for safe names and size, and loaded from the explicit current upload list. The loader excludes hidden paths, configured binary/archive/environment patterns, symlinks/reparse points, special files, and paths outside the selected root.

### GitHub

GitHub input is normalized to exactly `owner/repo`. A metadata preflight can reject a known private or oversized repository. The clone uses `git clone --depth 1 --single-branch --no-tags` without a shell, with a 120-second timeout, bounded output, and a disabled terminal prompt. The checkout is inspected while cloning and after completion, then atomically replaces the prior operation-owned checkout.

The checkout is rejected for unsafe/reparse paths, special files, too many files, an oversized individual file, excessive unpacked bytes, or excessive download bytes. GitHub files are then passed through the same local path/parser checks as other file-backed sources.

### Websites

Website URLs are validated before the fetch. Each hostname is resolved and all A/AAAA answers are checked. The request is made to a validated numeric address with the original hostname used for SNI and certificate verification. Redirects are resolved and checked again. HTML is converted to text, and the resulting `Document` objects enter the local indexing pipeline.

Website documents already exist when the local pipeline starts, so the local file table-verbalizer and file-title enrichment path does not apply to them. Their source metadata is the normalized URL.

## Parser and format behavior

The registered 25-extension matrix has three categories:

| Category | Extensions | Behavior |
|---|---|---|
| Verified local parser path | `.csv`, `.docx`, `.eml`, `.epub`, `.htm`, `.html`, `.ipynb`, `.json`, `.jsonl`, `.markdown`, `.mbox`, `.md`, `.mhtml`, `.msg`, `.odt`, `.pptx`, `.rtf`, `.tsv`, `.txt`, `.xls`, `.xlsx`, `.xml` | Uses the corresponding local reader and metadata. |
| Optional dependency | `.pdf` | Uses `pypdf` for text-layer pages; blank/image-only pages produce an OCR/conversion warning. |
| Unsupported without conversion | `.doc`, `.ppt` | `LegacyFormatReader` returns conversion guidance rather than pretending to parse the file. |

The parser registry limits:

- Source input: 25 MiB.
- Loaded records: 300.
- Extracted characters: 4 MiB.
- JSON nesting: 100.
- MIME parts/depth: 2,048/64.
- Delimited rows/columns: 100,000/16,384.
- PDF pages: 5,000.
- ZIP members: 4,096.
- ZIP compressed input: 25 MiB.
- ZIP total uncompressed data: 128 MiB.
- ZIP member uncompressed data: 32 MiB.
- ZIP member compression ratio: 500:1.

The ZIP preflight rejects absolute or traversing member paths, duplicate paths, symlink/special members, and encrypted members. Format tests generate valid DOCX, EPUB, XLSX, PPTX, ODT, and text-layer PDF fixtures plus malformed/archive/OCR-adjacent cases. They do not generate valid legacy XLS or real MSG fixtures. Encrypted Office/ZIP content and OCR ground truth remain untested.

The extraction report is content-free. It records filename, status, document count, extracted character count, and safe warning/error text. A report entry is tagged with the source ID and index generation and is rendered only when it still belongs to the active source.

## Text processing

File-backed documents and GitHub content pass through:

1. Table/record verbalization for selected CSV, TSV, JSON, and JSONL documents.
2. LlamaIndex sentence/paragraph-aware chunking.
3. Simple odd-fence code-block repair.
4. Removal of chunks with fewer than 15 non-whitespace characters.
5. Stemmed-token near-duplicate filtering.
6. File-title enrichment for file-backed chunks.
7. Embedding and vector-index construction.

A parser failure is isolated per file where possible. Valid files in a mixed batch can still be indexed while malformed, unsupported, or empty files are reported.

## Embedding

Ollama embedding batches start at 16 chunks, or 4 in Eco Mode. Capacity/OOM errors can halve the batch and retry. Other errors are not silently converted into fake embeddings. OpenAI-compatible embeddings use the installed LlamaIndex OpenAI adapter and have a different request path.

The selected embedding provider/model/endpoint and credential fingerprint are part of the indexing settings signature. A provider or model change makes the previous index stale.

## Index cache

The cache is stored under `.index_cache/` by default and retains up to five entries, with a default 512 MiB total limit. Cache identity includes:

- Cache and parser versions.
- Source identity and extracted document text.
- Source metadata such as filename or URL.
- Chunk size, overlap, and overlap percentage.
- Embedding adapter/provider, model, endpoint, and credential fingerprint.

Cache writes use a candidate directory, validate that the candidate can be loaded, atomically replace the target, and restore the prior target if commit fails. Pruning enforces count and byte limits. Cache paths reject symlinks/reparse points. Generated answers are not cached.

The cache is unencrypted and contains source-derived data. It is process/filesystem state, not a per-browser secret or isolation boundary.

## Query flow

1. Detect likely follow-up wording and construct a standalone query plus a history-expanded candidate.
2. Retrieve the vector candidate pool.
3. Build the BM25 candidate pool over the current corpus.
4. Fuse independent ranks with Reciprocal Rank Fusion using `1 / (60 + rank)`.
5. Apply strong lexical or vector evidence rules and the similarity cutoff.
6. Apply exact-phrase boosts and remove duplicate/near-duplicate chunks.
7. Add up to two introduction chunks for recognized document-level questions.
8. Build per-message evidence with source, score, and bounded excerpt.
9. Fit system guidance, history, query, and selected evidence into the tokenizer-aware total RAG input budget.
10. Stream the configured model response and sanitize citation references.

BM25 and vector retrieval are independent candidate pools. At a positive cutoff, a weak vector candidate needs strong lexical evidence or a sufficiently strong vector score; a keyword-only candidate can still be accepted when it has strong BM25 evidence.

The RAG planner derives its total input budget from the model context window, output reserve, safety reserve, message overhead, and the model tokenizer. Eco Mode lowers the configured output reserve and candidate limits. No fixed context-only character budget is presented as the complete prompt limit.

## Map routing stage

When agentic map retrieval is enabled, an additional stage runs between query construction and evidence selection.

After a successful ingest, the builder groups corpus nodes by source, walks each contiguous run, and emits sections of at most `retrieval_map_chunks_per_section` nodes. Each section gets a stable content-derived identifier, a document-scoped source identifier, a bounded title, keywords, a short summary, and its member node identifiers. Documents and sections become map nodes, and `contains`/`next` edges record structure. Node, section, keyword, and summary counts are hard-bounded; truncation is recorded rather than hidden.

Per question the agent:

1. Observes by scoring each section card against the standalone and follow-up queries.
2. Plans by selecting sections. The deterministic planner makes no model call. The optional LLM planner receives only the compact cards, and invalid, oversized, or failed output falls back to the deterministic plan.
3. Acts by retrieving with `allowed_node_ids` restricted to the routed sections' member nodes, expanding to adjacent same-source sections when neighbour expansion is enabled.
4. Reflects once. Weak routed evidence either expands to further positively scored sections or falls back to a single unrestricted retrieval, and the result is re-checked before it is accepted.

Each step is recorded as a bounded `RouteAction`, and the trace stores the considered, planned, and selected section identifiers, the step count, the planner mode, and the measured evidence tokens. With baseline measurement enabled, one unrestricted retrieval is also tokenized so the prompt-context difference is a measured comparison rather than an estimate.

Failure handling: a map build failure after ingest is logged and does not roll back the committed index; a routing failure falls back to unrestricted retrieval for that turn; weak evidence is dropped rather than returned as a weak citation. Route traces are stored per message and per session, and a trace whose `map_id` no longer matches the current map is rendered as stale without claiming metrics.

Routing changes which candidates are scored, not the total number of embedding queries. The offline study in `research/REPORT.md` measures the resulting prompt-context and candidate-scoring change together with the retrieval hits it costs.

## Generation and citations

The local prompt treats retrieved text as untrusted reference data, delimits it, neutralizes attempts to inject the delimiters, and asks the model to ignore instructions inside the context. The model is asked to cite supporting evidence, but no citation is forced onto an answer. Citation numbers outside the selected evidence are removed or bounded.

RAG answers are grounded by the prompt and retrieval filter, not by an automatic factual-entailment checker. A source label or citation marker is not proof that a generated claim is supported.

When no credible local nodes are selected, the pipeline returns:

```text
I could not find this information in the documents.
```

The UI can then offer a one-time direct-chat fallback.

## R2R path

R2R bypasses the local pipeline only for local-file uploads when enabled. The client:

1. Validates the configured HTTP/HTTPS endpoint and optional credential.
2. Performs an R2R v3 health check.
3. Saves the current upload batch to an owned work directory.
4. Uploads documents with deterministic document IDs.
5. Polls status until success, failure, cancellation, or timeout.
6. Deletes replaced owned documents only after new documents are ready.
7. Atomically updates the ownership registry and active document set.
8. Rolls back newly uploaded documents if a transaction fails.

R2R chat is synchronous, does not send application history, and stores bounded citation/search metadata on the assistant message. R2R source labels can be displayed from returned search results. The current R2R server was unavailable, so live compatibility is not claimed.

## State and isolation

Source state includes source kind, source ID, content/settings signatures, index generation, status, cache key, and errors. Reset and replacement operations clear or restore the relevant session state under the lifecycle lock. Extraction/status UI checks also compare source ID and generation.

The lock prevents overlapping sensitive filesystem/index operations across processes. LlamaIndex global settings, adapter caches, logging, and some paths remain process-wide; session-owned bundles reduce stale-state races but do not provide true per-browser index isolation.

See [Usage](usage.md) for user-facing behavior and [Troubleshooting](troubleshooting.md) for categorized failures.
