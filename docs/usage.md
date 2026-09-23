# Using DocMind

## Start the Application

1. Start Ollama or configure another supported provider.
2. Open DocMind at `http://127.0.0.1:8501`.
3. Confirm the endpoint and model settings.
4. Import one source: local files, a public GitHub repository, or public HTTPS webpages.
5. Ask a question after local indexing completes.

A successful local ingestion replaces the currently active local index. Sources are not continuously merged into one growing corpus.

## Settings

A selected subset of non-secret settings is stored in browser `localStorage`. API keys, chat history, answer-style state, source indexes, and R2R document IDs are not stored there.

### Ollama

| Setting | Default | Notes |
|---|---:|---|
| Endpoint | `http://localhost:11434` | Empty values return to the default |
| Chat Model | First valid discovered model | The preference order is `gemma4:latest`, `llama3:8b`, `llama2:7b`, then first discovered |
| Embedding Model | First valid discovered model | `embeddinggemma:latest` is preferred when available; fresh state starts with `nomic-embed-text:latest` |
| Top K | 3 | Applied to the next local query; document-level questions can add up to two introduction chunks |
| Similarity Cutoff | 0.30 | Zero also disables the current evidence-floor branch |
| Temperature | 0.4 | Lower values reduce sampling randomness but do not guarantee deterministic output |

### Chunking

| Setting | Initial value | Notes |
|---|---:|---|
| Chunk Size | 256 | Applied on the next ingestion |
| Chunk Overlap | 32 initially | Opening Advanced Settings recalculates it from the percentage; 12% of 256 becomes 30 |
| Minimum retained chunk | 15 characters | Source constant, not a user setting |

Changing the provider, embedding model, chunk size, or overlap changes the upload processing signature and causes a still-selected local upload batch to be processed again.

## Local Files

DocMind accepts 25 filename extensions:

```text
.csv .doc .docx .eml .epub .htm .html .ipynb .json .jsonl
.markdown .mbox .md .mhtml .msg .odt .pdf .ppt .pptx .rtf
.tsv .txt .xls .xlsx .xml
```

Limits:

- 10 files per batch
- 25 MiB per file
- 100 MiB total

Filenames cannot contain path separators or unsupported control characters. The saved destination must remain under `data/`.

An accepted extension is not a guarantee of successful semantic extraction. The committed evaluation directly tests TXT, Markdown, CSV, and DOCX. Scanned PDFs and legacy/container formats may require tools that are not installed. OCR is not included.

## GitHub

Accepted values:

```text
owner/repo
https://github.com/owner/repo
```

Only public repositories are supported. A URL must contain exactly the owner and repository segments. The clone uses `--depth 1` and has a 120-second timeout. Git must be installed and available to the application process.

## Websites

Up to six public HTTPS webpages can be processed in one batch.

Controls:

- HTTPS only
- No credentials in the URL
- `localhost` and known metadata hostnames blocked
- Every returned A and AAAA address checked against private, loopback, link-local, multicast, reserved, unspecified, and other non-global ranges
- HTTPS connections pinned to a validated numeric address with original-hostname certificate verification
- Up to three redirects, with each target revalidated before use
- Per-request timeouts and a 180-second total ingestion deadline
- Categorized invalid URL, blocked destination, DNS, HTTP, anti-bot, timeout, and empty-page errors
- HTML or plain text only
- 5 MiB maximum response body per URL

## Chat Modes

- **Direct Chat**: no active document index
- **Local RAG**: a local query engine is active
- **R2R**: R2R is enabled and document IDs exist

R2R takes priority when active. Local GitHub and website ingestion clears stale R2R document IDs after a successful local index build.

## No-Match Behavior

If local retrieval returns no credible nodes, DocMind returns:

```text
I could not find this information in the documents.
```

The one-time **Ask without documents** action sends the same question through direct chat. The next normal prompt returns to the active index route.

This is a retrieval fallback, not a factual verifier for non-empty answers.

## Export

Use **Settings > Export Chat History** to download the current session messages as a DOCX file. Source captions are not stored as separate message metadata and are not included in the export.
