# 🧠 DocMind AI — Private Offline RAG Assistant

> **Import files, GitHub repos, or websites — then chat with grounded answers. Everything runs locally. No data ever leaves your device.**

[![Quality](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/quality.yml/badge.svg)](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/quality.yml)
[![Docker Build](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/main.yaml/badge.svg)](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/main.yaml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

1. [Problem](#problem)
2. [Solution & Approach](#solution--approach)
3. [System Architecture & Workflow](#system-architecture--workflow)
4. [Tech Stack](#tech-stack)
5. [Features — Everything the App Does](#features--everything-the-app-does)
6. [Supported Documents & Sources](#supported-documents--sources)
7. [Modes & Options](#modes--options)
8. [Quick Start](#quick-start)
9. [Configuration](#configuration)
10. [Evaluation](#evaluation)
11. [Expected Outcomes](#expected-outcomes)
12. [Project Structure](#project-structure)
13. [Roadmap](#roadmap)

---

## Problem

Teams and students who work with sensitive documents face a dilemma:

* **Cloud RAG is risky** — uploading contracts, research papers, or internal reports to a third-party API leaks private data and violates compliance.
* **Naïve local RAG is brittle** — plain vector search misses exact keywords (“30-day”), hallucinates when no evidence exists, breaks on Hinglish queries, and heats weak laptops during ingestion.
* **Tool sprawl** — useful knowledge lives in PDFs, Word docs, scattered CSVs, GitHub repos, and documentation sites, but most chatbots only accept one file type at a time.

**Goal:** one lightweight, fully offline assistant that ingests *any* of those sources, retrieves the *right* chunks reliably, and answers *only* from evidence.

---

## Solution & Approach

DocMind is a **streamlit + LlamaIndex + Ollama** system with a **hybrid retrieval layer** designed for small, local LLMs (e.g. `qwen2.5:0.5b`). The core insight is *novelty in system design, not model size*:

| Design Choice | Why it matters (`code` ref) |
|---|---|
| **Hybrid BM25 + vector with RRF** | BM25 catches exact terms/numbers that dense vectors miss; RRF fusion costs only CPU (`utils/llama_index.py:260`) |
| **Evidence floor 0.5** | Weak vector scores (<0.5) need a BM25 keyword hit, otherwise the query is correctly *rejected* instead of hallucinated (`utils/llama_index.py:75`) |
| **Curated synonym expansion** | Short queries like “money back rules” widen to `refund` synonyms for BM25 only — vector query stays clean (`utils/llama_index.py:201`) |
| **Hyphen / Hinglish hygiene** | `30-day → 30 day`, single-letter tokens dropped, Hinglish fillers (`batao/kya/hai`) stripped (`utils/llama_index.py:184`) |
| **Title-aware chunks** | Every chunk is prefixed with its document title so title queries match all chunks and the LLM knows provenance (`utils/llama_index.py:141`) |
| **Near-duplicate filter** | Jaccard on stemmed tokens drops repeated headers/footers before embedding — less GPU work (`utils/llama_index.py:108`) |
| **Grounded prompt + citations** | “Answer ONLY from context, quote numbers, cite `[n]`” + compact numbered context (`utils/llama_index.py:43`) |
| **Cache & Eco Mode** | On-disk `.index_cache` reuses embeddings; Eco trims batches/context for cool & fast weak-machine answers (`utils/llama_index.py:834` / `166`) |

No hallucination fallback: if retrieval returns 0 nodes the assistant says *“I could not find this information in the documents.”* and offers **Ask without documents** (`utils/ollama.py:528`, `components/chatbox.py:125`).

---

## System Architecture & Workflow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Sources    │     │  Ingestion   │     │    Retrieval     │
│  Local Files│────▶│  Validate    │────▶│  Vector (Ollama) │
│  GitHub Repo│     │  Load docs   │     │  + BM25 (rank-   │
│  Website    │     │  Chunk 256/32│     │    bm25) → RRF   │
│             │     │  Title+dedup │     │  Evidence floor  │
│             │     │  Embed batch │     │  Context budget  │
└─────────────┘     │  Index(cache)│     └────────┬─────────┘
                    └──────────────┘              │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│    UI       │◀────│  Generation  │◀────│    Query         │
│  Chat Box   │     │  Stream Chat │     │  Rewrite →       │
│  Suggestions│     │  Grounded    │     │  Multi-turn      │
│  Sources    │     │  Citations   │     │  History         │
└─────────────┘     │  No-halluc.  │     └──────────────────┘
                    └──────────────┘
```

**Ingestion flow** (`docs/pipeline.md`): validate model → load docs (LlamaIndex `SimpleDirectoryReader` + SSRF-guarded website fetch) → validate limits (≤1000 docs, ≤10 MB) → split → title/dedup → batched embed with progress & OOM shrink → `VectorStoreIndex` → streaming query engine + hybrid retriever → persist to `.index_cache`.

**Query flow** (`components/chatbox.py:56`, `utils/ollama.py:492`): rewrite query → vector+BM25 retrieve → filter by cutoff & evidence → build numbered context → prepend recent chat history (multi-turn) → `stream_chat` → render tokens + source chips.

**State:** `components/page_state.py:117` seeds session state; `sidebar.py:37` shows mode badges (RAG / R2R / Chat); `utils/browser_settings.py` persists preferences in `localStorage`.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| **UI** | Streamlit 1.62 (dark theme `#0E1117`) | Fast Python UI, no JS build |
| **RAG** | LlamaIndex 0.14 + `llama-index-readers-file/web` | Pluggable loaders, `VectorStoreIndex` |
| **LLM / Embed** | Ollama (`qwen2.5:0.5b` / `nomic-embed-text:latest`) + OpenAI-compatible (LM Studio, TabbyAPI, vLLM) | Offline by default, swappable via `components/tabs/settings.py:40` |
| **Retrieval** | `rank-bm25` + `nltk` Porter stemmer | Pure-Python, no torch, no extra model download |
| **Docs** | `python-docx`, `ebooklib`, `html2text`, `fsspec`, `pypdf` | 26 file types |
| **Infra** | Python 3.13, Pipenv, Docker (compose + ROCm variant), GitHub Actions (quality + Docker build) | Reproducible dev & deploy |
| **Observability** | `utils/logs.py`, 112 unit tests, eval harness | Verified via CI |

No `torch`/`transformers` at runtime — removed for lightness (`Pipfile:6`).

---

## Features — Everything the App Does

### Ingestion Sources (sidebar **Data Sources**)

* **Local Files** — drag & drop, 25 MB/file, CSV/TSV/JSON/JSONL/XML/XLS/XLSX/PDF/DOC/DOCX/PPT/PPTX/ODT/RTF/EPUB/EML/MBOX/MSG/TXT/MD/Markdown/HTML/HTM/MHTML/IPYNB (`utils/helpers.py:19`)
* **GitHub Repo** — `owner/repo` or `https://github.com/owner/repo`, SHA-clone `--depth 1`, stale-checkout handling (`utils/helpers.py:254`)
* **Website** — up to 5 URLs, HTTPS-only, redirect + size + SSRF guards, HTML→text via `html2text` (`utils/helpers.py:108`)
* Exclusions: binaries/archives/images (`*.png, *.zip, *.exe, *.mp4, node_modules …`) (`utils/llama_index.py:689`)
* Stages shown live: *validated → cloned/fetched → loaded → embedded → ready* (`components/tabs/sources.py:17`)

### Retrieval & Chat

* **Two modes auto-routed** — RAG (with `query_engine`) vs direct LLM (`components/chatbox.py:58`)
* **Hybrid retriever** with top-k, similarity cutoff, context budget (`utils/llama_index.py:260`)
* **Grounded answers** with citations `[n]` and source chips (file + score) under each answer
* **No-hallucination gate** + “Ask without documents” fallback
* **Multi-turn memory** — last RAG context includes recent history (`utils/ollama.py:115`)
* **Streaming** token-by-token via `write_stream`

### Settings (`Settings` tab — `components/tabs/settings.py:119`)

| Group | Controls |
|---|---|
| **Chat** | Provider (Ollama / OpenAI / LM Studio / TabbyAPI) → Chat Model dropdown + Refresh; Server URL & API key when non-Ollama |
| **Document Search** | Embedding Model dropdown + Refresh |
| **Answer Style** | 6 presets: Concise / Balanced / Detailed / Bulleted / Technical / Simple-ELI5 → prompt preview |
| **Preferences** | Eco Mode (batch 4, 256 tokens, ≤3 chunks, 3200-char budget) + Show advanced controls |
| **Advanced** (when on) | Sources per answer (1–10), Relevance threshold (0–1), Creativity/temperature (0–1.5), Chunk Size/Overlap + live token count |
| **External RAG (R2R)** | Optional `utils/r2r.py` server (`http://localhost:7272`), health check, doc IDs |
| **Export** | Download chat as `.docx` (`chat_history_docx`) |

Sidebar extras: **mode badge** (Chat / RAG / R2R), **Clear Chat & Reset** expander, browser-settings persistence.

### Performance & Safety

* **Eco Mode** for hot/weak machines (`components/tabs/settings.py:218`)
* **Index cache** (`.index_cache`, keep 5, keyed by docs+settings) — repeat uploads instant (`utils/llama_index.py:832`)
* **OOM-safe batching** — halves batch on CUDA OOM and retries (`utils/llama_index.py:540`)
* **Security** — GitHub URL normalize & reject non-github, website SSRF/IP block, upload name/size limits, `SAFE_UPLOAD_NAME_PATTERN` (`utils/helpers.py`)

---

## Supported Documents & Sources

**26 extensions:** `.csv .doc .docx .eml .epub .htm .html .ipynb .json .jsonl .markdown .mbox .md .mhtml .msg .odt .pdf .ppt .pptx .rtf .tsv .txt .xls .xlsx .xml`

Plus: any GitHub repo (public, shallow clone) and up to 5 HTTPS websites at once.

Upload limits: 10 files, 25 MB/file, 100 MB total; ingestion limits: 1000 docs, 10 MB text — enforced with clear errors.

---

## Modes & Options

| Mode | When | What happens |
|---|---|---|
| **Chat** | No index | Direct LLM (`utils/ollama.py:426`) with system prompt + chat history |
| **RAG** | After ingesting files/repo/sites | Hybrid retrieval → grounded prompt → cited answer |
| **R2R** | `Enable R2R` on | Files go to external R2R server (`utils/r2r.py`) — less local RAM/heat |

Other toggles: **Answer tone** quick selector in chat (`components/chatbox.py:58`) mirrors Settings; **Directly import** caption reminds users how grounding works.

---

## Quick Start

**Prereqs:** Ollama running, Python 3.13, at least one chat + embedding model:

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
ollama list
```

**Local (Pipenv):**

```bash
pip install pipenv
pipenv install
pipenv run streamlit run main.py
# open http://localhost:8501
```

**Docker:**

```bash
docker compose up -d
# app at http://localhost:8501
# point Ollama endpoint to host.docker.internal:11434 if Ollama is on host
```

Change Ollama endpoint in **Settings → Chat** if needed. First ingestion builds the index (~5 s for 5 small docs); repeat is instant from cache.

---

## Configuration

* **Ollama endpoint** default `http://localhost:11434`, editable in Settings and persisted in `localStorage`.
* **Chunking** 256 / 32 (12 %) — tweak in Advanced; next ingestion uses new values.
* **Top K / cutoff / temperature** — live, next query uses new values (no re-ingest).
* **Theme** dark (`#0E1117` / `#161B26` / violet `#8B5CF6`) in `.streamlit/config.toml:7`.

---

## Evaluation

All claims are measured, not asserted.

**Harness:** `eval_harness.py:1` — 6 suites, 43 tests, mock-friendly:

```bash
python eval_harness.py              # real Ollama + LLM
python eval_harness.py --mock       # deterministic hash embeddings, no server (CI)
python eval_harness.py --suite retrieval  # single suite
```

**Outputs:** `eval_results.json` (machine-readable) + `eval_report.md` (paste-ready for thesis/README).

**Latest real run (2026-08-23, `nomic-embed-text:latest`): 43/43 — 100%**

| Suite | Score | Covers |
|---|---:|---|
| ingestion | 100% | File types, chunking, dedup, title-aware, cache, exclusions |
| retrieval | 100% | Hybrid vs plain, synonyms, hyphen/Hinglish, evidence floor |
| generation | 100% | No-hallucination fallback, citations, tone presets, multi-turn, e2e |
| performance | 100% | Ingest speed, retrieval latency, eco-mode, cache hit, OOM resilience |
| robustness | 100% | Empty/binary, GitHub/URL validation, special chars |
| architecture | 100% | Backend presets, export, settings persistence, import boundaries |
| **Overall** | **100%** | |

**Headline Retrieval result (most viva-relevant):**

| Metric | Plain RAG | **DocMind Hybrid** |
|---|---:|---:|
| Factual hit rate (n=8) | 100% | **100%** |
| Correct rejection (n=2) | 0% | **100%** |
| Avg latency | 23 ms | 24 ms |

Plain RAG *always* hallucinates on no-match queries (pulls `python_guide.txt` at 0.38); DocMind returns **0 nodes** and the UI shows the fallback — zero hallucination. Per-query table and code refs are in `eval_report.md:17`.

Full unit suite: `pipenv run python -m unittest discover -s tests` — **112 tests, OK** on Windows & Linux CI.

---

## Expected Outcomes

* **Privacy:** zero egress — all embeddings, chat, and storage are local (or to your chosen compatible server).
* **Reliability:** no invented answers when evidence is absent; citations tie every fact to its chunk.
* **Usability:** non-technical flow — add docs → chat; 3 steps in the welcome card (`components/page_state.py:18`).
* **Efficiency:** repeat queries are instantaneous (cache); Eco Mode keeps a hot laptop cool (batch 4, trimmed context).
* **Breadth:** 26 formats + repos + sites in one index — no tool switching.

For a final-year project the *novelty is system design* (hybrid + evidence gating on a tiny 0.5B model), not model size — an honest, defensible story.

---

## Project Structure

```
.
├── main.py                     # bootstrap: header → messages → sidebar → chatbox
├── components/
│   ├── header.py               # branded title + tagline
│   ├── page_config.py          # dark theme, centered layout, chrome hiding
│   ├── page_state.py           # WELCOME_MESSAGE, session-state seeding
│   ├── sidebar.py              # Data Sources / Settings tabs + mode badge + Clear & Reset
│   ├── chatbox.py              # streaming, citations, tone pills, suggestions
│   └── tabs/
│       ├── sources.py / local_files.py / github_repo.py / website.py
│       └── settings.py         # providers, models, style, eco, advanced, R2R, export
├── utils/
│   ├── llama_index.py          # index, hybrid retriever, embeddings, cache, dedup
│   ├── ollama.py               # LLM factories, chat/context_chat, history trimming
│   ├── helpers.py              # upload/URL/GitHub validation, file handling
│   ├── rag_pipeline.py         # ingestion pipeline orchestration
│   ├── browser_settings.py     # localStorage persistence
│   └── r2r.py                  # optional RAG-to-Riches backend
├── tests/                      # 112 tests (ingestion, security, settings, etc.)
├── docs/                       # pipeline / setup / usage / contributing / troubleshooting
├── eval_harness.py             # comprehensive 6-suite eval (43 tests)
├── eval_report.md              # generated report (this section)
├── eval_results.json           # generated JSON
├── Pipfile / Pipfile.lock      # deps (no torch)
├── .streamlit/config.toml      # dark theme + upload limit
├── Dockerfile + docker-compose.yml
└── .github/workflows/quality.yml + main.yaml  # tests + Docker build
```

---

## Roadmap

* [ ] Rerank cross-encoder (optional, still no torch by default)
* [ ] More export formats (markdown, PDF)
* [ ] Collaborative sharing of cached indexes
* [ ] Optional OCR for scanned PDFs (pluggable, off by default)

---

## License

MIT — see `LICENSE` (if present) or the repository header.

> Built with Streamlit · LlamaIndex · Ollama · rank-bm25 · NLTK · python-docx.

