# DocMind AI

DocMind AI is a Streamlit document-chat application built with LlamaIndex and model providers such as Ollama. It can build a local retrieval-augmented generation (RAG) index from files, a public GitHub repository, or public HTTPS webpages, then answer questions using retrieved context.

The primary verified path is a single-user installation using a local Ollama server. OpenAI-compatible providers and R2R are optional integrations with different privacy and maturity characteristics.

## Current Verification Status

Latest local verification for this snapshot was run on Windows on 2026-09-24 with Python 3.12.10, Streamlit 1.64.0, and LlamaIndex 0.14.25:

| Check | Result |
|---|---:|
| Streamlit health endpoint | Passed in the earlier 2026-09-23 run; current-snapshot rerun pending |
| Ollama health and model discovery | Passed in the earlier 2026-09-23 run; current-snapshot rerun pending |
| Unit tests | 235 passed, 1 skipped (236 total) |
| Real evaluation harness | 43/43 in the earlier 2026-09-23 snapshot; current-snapshot rerun pending |
| Mock evaluation harness | 42/42 scored checks passed, 1 real-LLM check skipped |
| Python compile checks | Passed |
| `pip check` | No broken installed requirements |
| CI Ruff fatal checks | Passed |
| Full Ruff ruleset | Not rerun for this snapshot |
| Black check | Not rerun for this snapshot |
| Docker build/runtime | Not tested; Docker CLI is unavailable in the verification environment |

The earlier real evaluation used a five-document fixture. Its live LLM check ran the same refund question three times and required the expected fact in at least two answers. In that run, the expected `30 days` fact appeared in 2/3 answers. This is useful historical evidence, not a claim about the current snapshot or a guarantee for every model and question.

The project does not commit a dependency lockfile. Package versions can therefore differ between installations.

## Implemented Features

- Streamlit chat interface with direct chat, local RAG, and optional R2R routing
- Local file uploads with validation and content-based duplicate detection
- Public GitHub repository ingestion using a shallow clone
- Public HTTPS webpage ingestion with network and response-size guardrails
- LlamaIndex sentence/paragraph-aware chunking
- CSV, TSV, and JSON table verbalization on a best-effort basis
- Code-fence repair, minimum-content filtering, near-duplicate filtering, and file-title enrichment
- Ollama embeddings with adaptive batch shrinking
- Independent vector and BM25 candidate pools fused through Reciprocal Rank Fusion
- Conversation-aware follow-up retrieval and a tokenizer-aware total RAG input budget
- Configurable similarity cutoff, evidence floor, top-k, temperature, and Eco Mode
- Grounded prompt with numbered context, per-turn evidence, and validated citation references
- Fixed no-match response when local retrieval returns zero nodes
- DOCX chat export
- Browser `localStorage` persistence for a selected subset of non-secret settings
- Optional synchronous R2R upload and chat client for local files

## Data Sources

### Local Files

The uploader accepts 25 filename extensions:

```text
.csv .doc .docx .eml .epub .htm .html .ipynb .json .jsonl
.markdown .mbox .md .mhtml .msg .odt .pdf .ppt .pptx .rtf
.tsv .txt .xls .xlsx .xml
```

Upload limits:

- 10 files per batch
- 25 MiB per file
- 100 MiB total per batch

An accepted extension does not guarantee a dedicated parser. Text extraction quality depends on the installed LlamaIndex readers and optional parser dependencies. The committed evaluation directly exercises TXT, Markdown, CSV, and DOCX. Scanned PDFs and unsupported legacy/container formats may produce no usable text; OCR is not included.

### GitHub

Accepted inputs:

```text
owner/repo
https://github.com/owner/repo
```

DocMind validates the two-segment GitHub repository identifier, checks repository availability, and runs a shallow clone with a 120-second timeout. Only public repositories are supported. Private-repository authentication is not implemented.

### Websites

DocMind accepts up to six public HTTPS webpages per batch. It rejects credentials in URLs, blocked hostnames, private/loopback/link-local/multicast/reserved addresses, unsupported content types, excessive redirects, and response bodies larger than 5 MiB. Website fetches and indexing share a bounded deadline, and failures are categorized for the UI.

Every hostname is resolved and all returned A and AAAA addresses are checked before each request. HTTPS connections use the validated numeric address while retaining the original hostname for SNI and certificate verification, and every redirect is revalidated before it is followed.

## Modes and Providers

| Integration | Current status | Important limitations |
|---|---|---|
| Ollama | Primary verified path | Requires a reachable Ollama server with chat and embedding models |
| OpenAI | Implemented adapter path | Not live-tested here; model identifiers must be accepted by both the installed LlamaIndex adapter and the remote server |
| LM Studio (Local AI) | Routed through the OpenAI-compatible path | Same adapter/model compatibility limitation; no live LM Studio test was available |
| TabbyAPI | Routed through the OpenAI-compatible path | Same adapter/model compatibility limitation; no live TabbyAPI test was available |
| R2R | Partial, experimental integration | Local uploads only; non-streaming chat; no application history/style/source-chip handling; remote documents are not automatically deleted by Reset Project |

### Chat Routing

Every prompt follows this order:

1. R2R, when R2R is enabled and document IDs exist
2. Local RAG, when a local query engine exists
3. Direct model chat, when no document index is active

Routing is based on session state, not on semantic classification. Once a local index exists, later prompts remain in local RAG mode unless the index is cleared or the user chooses the one-time **Ask without documents** fallback after a no-match result.

## Local Setup

The declared Python version is 3.13. The current local verification environment uses Python 3.12.10 successfully, but Python 3.13 remains the project contract used by `Pipfile`, Docker, and CI.

### Ollama Models

Example models used by the current verification environment:

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
ollama list
```

The application prefers `gemma4:latest`, then `llama3:8b`, then `llama2:7b`, then the first discovered chat-capable model. `qwen2.5:0.5b` is an example model, not a hardcoded universal default.

### Pipenv

```bash
python -m pip install pipenv
pipenv install
pipenv run streamlit run main.py
```

Open `http://127.0.0.1:8501`.

No `Pipfile.lock` is currently committed, so `pipenv install` may resolve newer package versions than those used in the recorded verification.

### Existing Virtual Environment

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py `
  --server.port=8501 `
  --server.address=127.0.0.1
```

### Windows Launcher

```powershell
.\run.ps1
```

`run.ps1` clears Python bytecode caches, attempts to start Ollama, stops a matching stale Streamlit process on port 8501, and launches the app. Its Ollama tuning environment variables affect an Ollama process started by the script; they do not reconfigure an Ollama service that is already running.

## Docker

The Compose files build the local source image, expose port 8501, mount writable data and index-cache volumes, and store the application log under the data volume. The container runs without a GPU reservation because model execution is delegated to a separate Ollama service.

```bash
docker compose up --build
```

If Ollama runs on the host, configure the Ollama endpoint as `http://host.docker.internal:11434` in the application settings.

The `docker-compose.yml-rocm` file configures the Streamlit container, not an Ollama ROCm runtime. AMD acceleration must be configured in the separately installed Ollama service.

Docker configuration was statically reviewed but could not be built or run because Docker is not installed in the current verification environment.

## Configuration

| Setting | Default | Behavior |
|---|---:|---|
| Ollama endpoint | `http://localhost:11434` | Used by Ollama mode |
| Top K | 3 | Applied to the next local retrieval query |
| Similarity cutoff | 0.30 | Applied to the next local retrieval query; zero also disables the evidence-floor branch |
| Temperature | 0.4 | Included in the LLM cache key after resolution |
| Chunk size | 256 tokens | Used on the next ingestion |
| Initial chunk overlap | 32 tokens | Opening Advanced Settings recalculates overlap from the percentage slider; 12% of 256 becomes 30 |
| Context budget | 4,800 characters | 3,200 in Eco Mode |
| RAG history estimate | 500 tokens | 300 in Eco Mode; this is a character-based estimate, not tokenizer-exact accounting |
| Direct-chat history estimate | 1,200 tokens | Character-based estimate |
| Normal embedding batch | 16 | 4 in Eco Mode |
| Normal output cap | 512 tokens | 256 in Eco Mode |

Changing provider, embedding model, chunk size, or overlap changes the local upload processing signature and causes the same still-selected upload batch to be ingested again.

## Retrieval and Grounding

For a local RAG query, DocMind:

1. Normalizes basic query tokens and selected filler words.
2. Retrieves vector candidates.
3. Calculates a BM25 ranking.
4. Fuses rankings with `1 / (60 + rank)`.
5. Applies the similarity cutoff.
6. Requires positive BM25 evidence for vector scores below 0.5 when the cutoff is positive.
7. Applies an exact-phrase boost for quoted queries.
8. Removes duplicate selected chunks and applies the context-character budget.
9. Adds up to two introduction chunks for some document-level questions.
10. Sends numbered context to the model.

The evidence rule is a retrieval filter, not a hallucination guarantee. If one weak but accepted chunk is sent to the model, the generated answer is not automatically checked for factual entailment. Citation formatting is also not proof that a claim is supported by the cited text.

## Index Cache

DocMind can persist up to five LlamaIndex index directories under `.index_cache/`.

The cache key includes:

- Cache version
- Embedding adapter class
- Embedding model name
- Embedding endpoint
- Chunk size and overlap
- Extracted document text
- Source identity metadata such as filename or URL

The cache avoids rebuilding embeddings for matching inputs. It does not cache model answers, and its contents are not encrypted.

## Privacy and Storage

With Ollama running on the same machine:

- Model inference can remain local.
- Extracted index/cache data and logs remain on the local filesystem unless the user configures another storage location.
- GitHub and website ingestion still require outbound network access.

OpenAI-compatible mode sends requests to the configured server. R2R mode uploads local files to the configured R2R server. API keys and chat history are not stored in browser `localStorage`, but non-secret settings, local index caches, and logs remain local files.

## Testing

### Unit Tests

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite covers model helpers, settings, browser persistence, format extraction, source/index transactions, provider profiles, website SSRF controls, upload signatures, R2R HTTP behavior, conversational retrieval, evidence and citation rules, cache identity, and import boundaries.

### Compile Checks

```bash
.\.venv\Scripts\python.exe -m py_compile `
  main.py `
  components/page_state.py `
  components/tabs/settings.py `
  utils/browser_settings.py `
  utils/ollama.py
```

### Evaluation Harness

```bash
# Requires Ollama embeddings; the live LLM check is skipped if no chat model is found
.\.venv\Scripts\python.exe eval_harness.py

# Uses stable hash embeddings and skips the live LLM check
.\.venv\Scripts\python.exe eval_harness.py --mock

# Writes results to a separate directory
.\.venv\Scripts\python.exe eval_harness.py --out .\eval-output
```

The harness now:

- Uses a stable hash function for mock embeddings
- Distinguishes passed, failed, and skipped checks
- Excludes skipped checks from the raw score denominator
- Returns a nonzero process status when a scored check fails
- Refuses to silently replace unavailable real embeddings with hash embeddings
- Runs three real LLM trials for its one generation check

The harness is a small-fixture evaluation, not a browser automation suite, load test, security audit, or proof of factual correctness for arbitrary documents.

## Known Limitations

- OpenAI-compatible model identifiers must be supported by the installed LlamaIndex adapter; arbitrary local-server model names may require an `OpenAILike`-style adapter.
- R2R support is limited to local-file upload/chat and is not a complete replacement for local RAG.
- Only one local index is active at a time. A new local ingestion replaces the previous active index.
- Source-specific completion labels are session state and are not a durable source registry.
- Conversation history is not persisted across sessions.
- API keys are session-only in the Streamlit process.
- The application has no authentication and is designed for local/single-user use.
- `data/`, `.index_cache/`, LlamaIndex global settings, and logging are process/filesystem-wide rather than isolated per browser session.
- Operating-system DNS resolution and provider callbacks cannot be forcibly interrupted once entered; website deadlines are checked around those operations and request timeouts bound normal waits.
- GitHub clone size, file count, and parser resource use are not bounded before parsing.
- Temperature zero reduces sampling randomness but does not guarantee byte-identical output.
- Docker remains unverified in the current environment.
- Dependency versions are not locked.

## Project Structure

```text
.
├── main.py
├── components/
│   ├── chatbox.py
│   ├── header.py
│   ├── page_config.py
│   ├── page_state.py
│   ├── sidebar.py
│   ├── ingestion_prerequisites.py
│   └── tabs/
│       ├── local_files.py
│       ├── github_repo.py
│       ├── website.py
│       ├── settings.py
│       └── sources.py
├── utils/
│   ├── browser_settings.py
│   ├── helpers.py
│   ├── llama_index.py
│   ├── logs.py
│   ├── ollama.py
│   ├── r2r.py
│   └── rag_pipeline.py
├── tests/
├── docs/
├── eval_harness.py
├── eval_report.md
├── eval_results.json
├── Pipfile
├── Dockerfile
├── docker-compose.yml
├── docker-compose.yml-rocm
└── run.ps1
```

## License

GPL-3.0. See [LICENSE](LICENSE).
