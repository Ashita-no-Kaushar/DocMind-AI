# Using DocMind

## Start the application

1. Start Ollama or configure another supported provider.
2. Open `http://127.0.0.1:8501`.
3. Confirm the chat and embedding endpoints and model selections in **Settings**.
4. Add one source: local files, a public GitHub repository, or public HTTPS webpages.
5. Wait for indexing and review the extraction report if files were skipped.
6. Ask a question.

A successful local ingestion replaces the active local source/index. Sources are not continuously merged into one growing corpus. Session source metadata and index generations prevent a completed report or status from being reused for a replacement source.

## Settings

A selected subset of non-secret settings is stored in browser `localStorage`. API keys, chat history, source indexes, evidence, answer-style state, and R2R document IDs are not stored there. Browser persistence is a convenience feature, not a secret store.

### Provider settings

| Setting | Default | Notes |
|---|---:|---|
| Ollama endpoint | `http://localhost:11434` | Blank values restore the default. |
| Ollama chat model | First valid discovered model | Preference order is `gemma4:latest`, `llama3:8b`, `llama2:7b`, then first discovered. |
| Ollama embedding model | `nomic-embed-text:latest` initially | Discovery prefers `embeddinggemma:latest`, then `nomic-embed-text:latest`, then `mxbai-embed-large:latest`. |
| Top K | 3 | Number of selected local evidence chunks before the total prompt budget is applied. |
| Candidate depth | 10 | Upper bound for each independent retrieval pool. |
| Vector candidates | 10 | Vector-pool depth. |
| BM25 candidates | 10 | BM25-pool depth. |
| Similarity cutoff | 0.30 | Filters weak evidence; zero changes the evidence-floor behavior. |
| Temperature | 0.4 | Lower values reduce sampling randomness but do not guarantee deterministic output. |

OpenAI, LM Studio, TabbyAPI, and OpenAI-compatible providers have separate endpoint, model, and credential fields. Non-loopback endpoints require HTTPS. The official OpenAI profile requires an explicit key and is restricted to its official host.

### Chunking and Eco Mode

| Setting | Initial value | Notes |
|---|---:|---|
| Chunk size | 256 tokens | Applied on the next local ingestion. |
| Chunk overlap | 32 initially | Advanced settings calculate overlap from the percentage; 12% of 256 is 30. |
| Minimum retained chunk | 15 characters | Source constant, not a user setting. |
| Eco Mode | Off | Lowers configured output reserve, candidate limits, and embedding batch size. It does not guarantee a speed, heat, or temperature change. |

Changing provider, embedding model, chunk size, or overlap changes the upload/indexing signature. A still-selected local batch is processed again when the active source is stale.

### Agentic map retrieval

These controls live under Settings → Advanced → Agentic Map Retrieval and apply to the active local index.

| Setting | Default | Notes |
|---|---:|---|
| Enable map retrieval | On | Turns routing on for local RAG chat. |
| Planner | Deterministic | `Deterministic` makes no extra model call. `LLM` plans from compact section cards and falls back to the deterministic plan on invalid or failed output. |
| Measure global baseline | Off | Adds one unrestricted retrieval per question so the prompt-context token difference is measured instead of estimated. |
| Chunks per section | 6 | Section size in the built map. Applied when the map is next built. |
| Max selected sections | 4 | Upper bound on routed sections per question. |
| Initial sections | 2 | Sections the planner starts from before neighbour expansion. |
| Neighbour sections | 1 | Adjacent same-source sections added around each planned section. |
| Max results | 8 | Result cap for the routed retrieval. |

After an answer, the chat panel shows the map for the question with the observed, planned, and selected sections, the route steps, and — when baseline measurement is on — the prompt-context token comparison. The Data Sources tab shows the map overview for the current source, and earlier answers keep their own recorded route. If the index was rebuilt since an answer, that route is labelled stale and its metrics are not presented as current.

Routing reduces the retrieved context, not the embedding query. It can also miss evidence that unrestricted top-k would have found.

## Chat modes

- **Direct Chat**: no active document source.
- **Local RAG**: a current local query engine is ready.
- **R2R**: R2R is enabled and its active documents are ready.

R2R takes priority when active. R2R is currently used for local-file uploads; GitHub and website ingestion continue through the local pipeline. R2R chat is synchronous and does not send the application's conversation history to the R2R server. Local RAG uses recent user turns for follow-up retrieval and a tokenizer-aware total input budget.

## Local files

DocMind accepts exactly 25 filename extensions:

```text
.csv .doc .docx .eml .epub .htm .html .ipynb .json .jsonl
.markdown .mbox .md .mhtml .msg .odt .pdf .ppt .pptx .rtf
.tsv .txt .xls .xlsx .xml
```

### Capability matrix

| Category | Extensions | What to expect |
|---|---|---|
| Verified local parser path | `.csv`, `.docx`, `.eml`, `.epub`, `.htm`, `.html`, `.ipynb`, `.json`, `.jsonl`, `.markdown`, `.mbox`, `.md`, `.mhtml`, `.msg`, `.odt`, `.pptx`, `.rtf`, `.tsv`, `.txt`, `.xls`, `.xlsx`, `.xml` | Text, records, tables, and document structure are extracted according to the selected parser. |
| Optional dependency | `.pdf` | Text-layer PDFs work when `pypdf` is installed. Image-only/blank PDFs need OCR or conversion. |
| Unsupported without conversion | `.doc`, `.ppt` | Convert to `.docx` or `.pptx` before local ingestion. |

`Verified` means the registered parser path and its tests; it is not a promise for every real-world file. The current format tests generate valid DOCX, EPUB, XLSX, PPTX, ODT, and text-layer PDF fixtures, plus malformed, archive, and OCR-adjacent cases. Valid legacy XLS and real MSG fixtures were not generated. Encrypted Office/ZIP content and OCR ground truth remain untested.

Upload limits are 10 files per batch, 25 MiB per file, and 100 MiB total. The parser also applies source-byte, document-record, extracted-character, JSON, MIME, table, PDF-page, and archive limits. The parser layer rejects unsafe archive paths, duplicate ZIP members, symlink/special members, and encrypted ZIP members.

### Extraction report

After a local, GitHub, or website source is processed, inspect **Extraction report** when it appears. Each entry contains:

- Source filename.
- `loaded`, `skipped`, or `unsupported` status.
- Loaded document-record count and extracted character count.
- A safe warning or error category.

The report does not copy document content. It is tied to the active source ID and index generation, so a report from a replaced source is not displayed as current. A mixed batch can retain valid files while reporting malformed or unsupported files individually.

DOCX, PPTX, and PDF files containing only images need an external OCR engine or conversion. No OCR is bundled. Email attachments, embedded images, and binary attachments are not indexed as text.

## GitHub repositories

Accepted values are:

```text
owner/repo
https://github.com/owner/repo
```

Use a public repository with a reachable `git` executable. Extra path segments, private authentication, query strings, fragments, credentials, ports, and dot-segment names are not supported.

The repository is shallow-cloned and validated before an atomic checkout replacement. Defaults are a 120-second clone timeout, 100 MiB repository download cap, 10,000 checkout-file cap, 10 MiB per checkout file, 250 MiB unpacked checkout cap, and 64 KiB clone-output cap. Hidden paths, configured binary/archive/environment patterns, symlinks/reparse points, special files, and source-root escapes are rejected.

If a repository is too large, unavailable, private, timed out, or fails checkout/parser validation, use the reported category to decide whether to retry, choose a smaller repository, or convert/clean the source. A bounded public clone and validation of `Ashita-no-Kaushar/DocMind-AI` passed in the current verification run using a temporary directory.

## Websites

Add up to six public HTTPS webpages and select **Process**. The fetcher:

- Requires HTTPS and rejects credentials in URLs.
- Blocks local, metadata, private, loopback, link-local, multicast, reserved, unspecified, and other non-global destinations.
- Checks every returned A and AAAA address, then connects to a validated numeric address while retaining the original hostname for certificate verification.
- Revalidates every redirect and allows at most three redirects.
- Accepts HTML and plain text only, converts HTML to readable text, and caps each response at 5 MiB.
- Applies bounded request timeouts and a 180-second total ingestion deadline.
- Reports invalid URL, blocked destination, DNS, HTTP, anti-bot, timeout, redirect, empty page, unsupported content, and response-size failures.

The current public fetch of `https://docs.python.org/3/` passed. JavaScript rendering, authentication, and anti-bot bypass are not guaranteed. DNS and network behavior can still change between validation and use; the application pins the validated address for its HTTPS request and revalidates redirects.

## Local RAG evidence and citations

For a local answer, the UI can show a **Sources** line derived from the evidence selected for that turn. Evidence includes the source label, score, and a bounded excerpt in session state. R2R search results can also produce source labels, and the R2R response metadata is stored on the assistant message.

The grounded prompt asks the model to use only the supplied context and to cite supporting evidence, but DocMind does not force a citation and does not automatically verify claim entailment. Invalid citation numbers are removed or bounded to available evidence. A citation marker is a reference format, not proof that a claim is true.

When retrieval returns no credible nodes, the response is exactly:

```text
I could not find this information in the documents.
```

The one-time **Ask without documents** action sends that question through direct chat. It is a retrieval fallback, not a factual verifier.

## Reset and export

**Clear Chat** removes only the conversation. **Reset Project** clears the current source/index/session state and owned work directories, but retains browser settings, API keys in the current session, shared caches, logs, other sessions, and the R2R ownership registry.

The reset dialog defaults to deleting remote R2R documents owned by the current workspace and endpoint/credential identity. It reports partial failures and retains the registry for retry. The local-only option leaves remote documents untouched.

**Settings > Export Chat History** downloads the current session messages as a DOCX file. Evidence and source metadata are not separate exported message fields, but the transcript content can still contain sensitive document-derived text. Review exports before sharing.

## Limits and safety notes

- The application has no built-in authentication and defaults to loopback.
- Direct chat can use general model knowledge; only local RAG is document-grounded.
- One local source/index is active at a time.
- Session history and source state do not form a durable multi-source registry.
- Local caches, work files, logs, and exports are not encrypted by DocMind.
- See [Setup](setup.md), [Pipeline](pipeline.md), and [Troubleshooting](troubleshooting.md) for setup, implementation, and error details.
