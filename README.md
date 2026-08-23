# 🧠 DocMind AI — Private Offline RAG Assistant

> **Import files, GitHub repos, or websites — then chat with grounded answers. Everything runs locally. No data ever leaves your device.**

[![Quality](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/quality.yml/badge.svg)](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/quality.yml)
[![Docker Build](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/main.yaml/badge.svg)](https://github.com/Ashita-no-Kaushar/DocMind-AI/actions/workflows/main.yaml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Try it in 3 clicks:** `1` Upload files / paste GitHub or website → `2` Wait ~5 s (index built, cached) → `3` Ask anything — answers are cited from *your* docs. No account, no upload to cloud.

---

## Table of Contents

1. [Problem](#problem)
2. [Solution & Approach](#solution--approach)
3. [Novelty — What Makes This Different](#novelty--what-makes-this-different)
4. [System Architecture & Workflow](#system-architecture--workflow)
5. [Behind the Scenes — What Happens When You Add Content](#behind-the-scenes--what-happens-when-you-add-content)
6. [Plain RAG vs DocMind — Feature Matrix](#plain-rag-vs-docmind--feature-matrix)
7. [Tech Stack](#tech-stack)
6. [Features — Everything the App Does](#features--everything-the-app-does)
7. [Supported Documents & Sources](#supported-documents--sources)
8. [Modes & Options](#modes--options)
9. [Quick Start](#quick-start)
10. [Configuration](#configuration)
11. [Evaluation](#evaluation)
12. [Expected Outcomes](#expected-outcomes)
13. [Project Structure](#project-structure)
14. [Troubleshooting & Security](#troubleshooting--security)
15. [Roadmap](#roadmap)

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
| **Numeric word → digit** | `thirty days` ≠ `30 days` for BM25 | `thirty → 30` before stemming so `thirty days` matches `30 days` (`utils/llama_index.py:239`) |
| **Exact-phrase boost** | `"refund policy"` is bag-of-words | Quoted phrase gets +0.5 RRF if found verbatim (`utils/llama_index.py:358`) |
| **Tabular verbalization** | `id,info / 1,HR leave` is not a sentence | CSV/JSON rows → `Row 1: info is HR leave…` making tables retrievable (`utils/llama_index.py:175`) |
| **Code fence aware** | `requests.get(` split across chunks | ` ``` ` odd-count chunks merged before embed (`utils/llama_index.py:808`) |

No hallucination fallback: if retrieval returns 0 nodes the assistant says *“I could not find this information in the documents.”* and offers **Ask without documents** (`utils/ollama.py:528`, `components/chatbox.py:125`).

---

## Novelty — What Makes This Different

Plain RAG (vector search + LLM) is the baseline every student builds. DocMind keeps the same tiny local model (`qwen2.5:0.5b`) but adds **system-level innovations** that are measured in `eval_harness.py:1` (43 tests, **100%** real run). No new model was trained — the contribution is engineering for **accuracy, privacy, and weak hardware**.

### At a glance

| # | Novelty | What plain RAG does | What DocMind does | Where | Eval proof |
|---|---|---|---|---|---|
| 1 | **Hybrid BM25 + Vector (RRF)** | Vector only — misses exact codes/numbers | Fuses BM25 keyword hits (exact) with vector semantics via Reciprocal Rank Fusion — pure CPU | `utils/llama_index.py:260` | Retrieval suite 12/12; e.g. `30-day` still hits |
| 2 | **Evidence Floor (0.5)** | Weak scores (0.3–0.5) still return a chunk → hallucination | Score < 0.5 needs a BM25 hit, otherwise **0 nodes → correct rejection** | `utils/llama_index.py:75` | Correct rejection **0% → 100%** (`eval_report.md:17`) |
| 3 | **Synonym expansion (BM25-only)** | `money back rules` ≠ `refund policy` | Short queries expand via curated map (`refund → return/money/back`) for BM25; vector query stays clean | `utils/llama_index.py:201` | `synonym` 100% |
| 4 | **Hyphen & Hinglish hygiene** | `30-day` and `batao/kya/hai` are tokens | `30-day → 30 day`, single letters dropped, Hinglish fillers stripped | `utils/llama_index.py:184` | `hyphen` + `hinglish` 100% |
| 5 | **Title-aware chunks** | Chunks are anonymous | Every chunk prefixed with `Annual Report.` etc. — title queries match all chunks + provenance visible | `utils/llama_index.py:141` | `title-aware` 100% |
| 6 | **Near-duplicate & cache** | Re-embeds same headers every run | Jaccard dedup before embed + disk `.index_cache` (5 entries, content-hashed) → less heat, instant reload | `utils/llama_index.py:108`/`832` | `dedupe` + `cache` 100% |
| 7 | **Eco Mode (weak-machine)** | One-size compute | Batch 4 / 256 tokens / ≤3 chunks / 3200-char budget when hot | `utils/llama_index.py:166` `utils/ollama.py:100` | `eco_mode_trims` 100% |
| 8 | **No-hallucination UX** | Silent hallucination | Grounded template (`Answer ONLY from context, cite [n]`) + `I could not find…` + **Ask without documents** button | `utils/llama_index.py:43` `utils/ollama.py:528` | `no_hallucination` 100%, `generation` 100% |
| 9 | **No heavy ML at runtime** | Needs `torch`/`transformers` | `rank-bm25` + `nltk` stemmer only — **~500 MB saved**, no torch import (`tests/test_import_boundaries.py`) | `Pipfile:6` | `no_torch_at_import` 100% |
| 10 | **Security & validation** | Trusts any path/URL | GitHub URL normalize, SSRF/IP block, upload limits, excluded `*.png/*.zip` | `utils/helpers.py:19`/`254`/`80` | `robustness` 6/6 |
| 11 | **Numeric word → digit** | `thirty days` missed | `thirty → 30`, `twenty → 20` before stemming | `utils/llama_index.py:239` | `numeric` 100% |
| 12 | **Exact-phrase boost** | Quoted search ignored | `"refund policy"` +0.5 RRF if verbatim in chunk | `utils/llama_index.py:358` | `phrase` 100% |
| 13 | **Tabular verbalization** | CSV/JSON not sentence-like | `Row 1: info is HR leave…` making tables answerable | `utils/llama_index.py:175` | `ingestion` 7/7 |
| 14 | **Code fence aware** | Code split mid-` ``` ` | Odd-fence chunks merged before embed | `utils/llama_index.py:808` | `ingestion` 7/7 |

### Why this is defensible

* **Same model, better system** — you didn't claim a new LLM; you proved a better *pipeline* on the same `nomic-embed-text` + `qwen2.5:0.5b`. Reviewers can rerun `python eval_harness.py` vs `python eval_harness.py --mock` themselves.
* **Reproducible numbers** — every novelty maps to a test in `eval_report.md:5` and a code line; overall **43/43 — 100%** (`eval_results.json:2`).
* **Practical impact** — fully offline, runs cool on a weak laptop, handles 26 formats + Hinglish + GitHub/sites in one index — exactly the gap for colleges / small orgs that can't use cloud RAG.

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

## Behind the Scenes — What Happens When You Add Content

This is the “explain to a non-technical examiner” section — exactly what runs after you drag a file, paste a `owner/repo`, or add a website URL. The UI shows 4–5 stage chips; underneath this is what really happens.

### 1) Local Files — `components/tabs/local_files.py` → `utils/helpers.py` → `utils/llama_index.py` → `utils/rag_pipeline.py`

**You do:** sidebar → **Data Sources → Local Files → Upload** (up to 10 files, 25 MB each).

**System does:**

1. **Save safely** — each `uploaded_file` is checked by `safe_uploaded_filename()` (`helpers.py:174`): no `/` or `\`, matches `^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$`, extension in `ALLOWED_UPLOAD_EXTENSIONS` (26 types). Then `upload_destination()` resolves the path and asserts it stays inside `data/` (prevents `../../etc/passwd`). Total size checked (`MAX_TOTAL_UPLOAD_BYTES=100 MB`, `helpers.py:52`). File is written via `save_uploaded_file()` and the UI shows *files uploaded*.
2. **Load** — `load_documents(data_dir)` uses LlamaIndex `SimpleDirectoryReader` with `EXCLUDED_FILE_PATTERNS` (`llama_index.py:689`): `*.png, *.zip, *.exe, *.mp4, node_modules, .git …` (34 patterns) are never read — even if a user zips a repo, the zip itself is skipped.
3. **Validate** — `validate_ingested_documents()` (`rag_pipeline.py:22`) enforces **≤1000 docs and ≤10 MB text**. Too much → clear error “Too many documents” instead of OOM.
4. **Chunk** — LlamaIndex splitter with `Settings.chunk_size=256` tokens (~1024 chars) and `chunk_overlap=32` (12 %). Small chunks = precise retrieval; overlap = no fact split across a boundary. Both are editable in Advanced and take effect on *next* ingestion.
5. **Enrich** — for each chunk: `MIN_CHUNK_CHARS=50` drops empty fragments, `_dedupe_near_duplicate_nodes()` (`llama_index.py:108`) drops boilerplate repeats via stemmed Jaccard >0.95, `_prepend_document_title()` (`:141`) prefixes `Annual Report.\n\n…` so title queries match *every* chunk.
6. **Embed (the hot part)** — `OllamaEmbedding.get_text_embedding_batch()` (`llama_index.py:530`) sends `embed_batch_size=16` (Eco: 4) chunks to `http://localhost:11434/api/embed` with a 300 s timeout. A `ProgressReportingEmbedding` wrapper calls the progress callback for the progress bar. If Ollama returns `CUDA out of memory`, the batch is halved (`16 → 8 → 4 → 2 → 1`) and retried — ingestion finishes instead of crashing.
7. **Index + cache** — `VectorStoreIndex(nodes=…, embed_model=…)` builds the in-memory index. `index_cache_dir()` hashes `INDEX_CACHE_VERSION + model + chunk settings + sorted doc texts` into `.index_cache/<20-char-key>` and `persist_index_to_cache()` saves it. Next time you add *the same files with same settings* the index loads from disk in ~0.03 s (`eval_report.md:70`). Old caches pruned (keep 5).
8. **Ready** — `create_query_engine()` creates a streaming `RetrieverQueryEngine` (`top_k` from slider) with `TEXT_QA_TEMPLATE` (grounded prompt) and a **hybrid retriever** (`build_hybrid_retriever()`). Temp files under `data/` are deleted.

> **Seen as:** *files uploaded → documents loaded → embeddings generated → index ready* (kept in `st.session_state["file_ingestion_stages"]` so reruns don’t re-trigger).

### 2) GitHub Repo — `components/tabs/github_repo.py` → `utils/helpers.py:254`

**You do:** `Ashita-no-Kaushar/DocMind-AI` or `https://github.com/Ashita-no-Kaushar/DocMind-AI`.

**System does:**

1. **Normalize & validate** — `normalize_github_repo()` strips whitespace, parses URL, requires `https` + `github.com`, needs exactly `owner/repo` (2 path parts), strips trailing `.git`, then regex `^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`. `https://gitlab.com/…` or `…/extra/path` → error. This is the same check `eval_harness.py:698` tests.
2. **Clone** — `clone_github_repo()` ensures `data/` exists, deletes a stale checkout (with `remove_dir_retry` for Windows locks, fallback to `data/clone_<ts>/owner__repo`), then `git clone --depth 1 -q https://github.com/owner/repo.git data/owner/repo` (120 s timeout). No `git` binary → graceful error.
3. **Load** — the cloned folder is then processed exactly like Local Files (steps 2–8 above) — respecting `EXCLUDED_FILE_PATTERNS`, so `.git/` and `node_modules/` are never embedded.

> **Seen as:** *repository validated → repository cloned → files loaded → embeddings generated → index ready*.

### 3) Website — `components/tabs/website.py` → `utils/helpers.py:80`

**You do:** paste `https://docs.python.org/3/library/os.html` (up to 5 at once) → **+** → **Process**.

**System does:**

1. **Validate** — `validate_website_urls()` → `_validate_public_http_url()` per URL: scheme must be `https`, no `user:pass@`, hostname not in `BLOCKED_HOSTNAMES` (`localhost`, `metadata.google.internal`), **DNS → IP** via `getaddrinfo` and reject if IP `is_private / is_loopback / is_link_local` (prevents SSRF to `169.254.169.254` or `127.0.0.1`). Exceed 5 URLs → error.
2. **Fetch** — `load_website_documents()` opens a `requests.Session` with `User-Agent: docmind/website-ingestion`, follows at most `MAX_WEBSITE_REDIRECTS=3` (re-validating each redirect), checks `Content-Type` contains `text/html` or `text/plain`, streams in 64 KB chunks and aborts if `>5 MB` (`MAX_WEBSITE_RESPONSE_BYTES`). Timeout `(5, 20)` s.
3. **Convert** — `html2text.html2text(html)` → clean markdown text → `Document(text=…, metadata={"source": url})`.
4. **Index** — same chunk/title/dedup/embed pipeline as above.

> **Seen as:** *websites fetched → content loaded → embeddings generated → index ready*.

### After indexing — how a question is answered

1. **Rewrite** — `_rewrite_query()` (`llama_index.py:248`): Porter-stem, split hyphens, drop filler words (`the/a/and`, plus Hinglish `batao/kya/hai`), produce `refunded purchases → refund purchas`.
2. **Retrieve** — `HybridRetriever.retrieve()` (`:338`): vector search (rewritten query) → RRF rank + BM25 keyword search (expanded tokens `_bm25_expanded_tokens()` for short queries only) → fuse via `1/(60+rank)`.
3. **Filter** — `_is_credible()`: drop `< similarity_cutoff` (default 0.3); then **evidence floor**: if `vector < 0.5` need a BM25 `score>0` hit or the chunk is discarded — this is why no-match queries correctly return **0 nodes**.
4. **Budget** — keep top chunks within `CONTEXT_CHAR_BUDGET=4800` (Eco: 3200), dedup within selection, and for doc-level queries (“summarize…”) prepend intro chunks.
5. **Generate** — `context_chat()` (`ollama.py:492`): numbers chunks `[1]…`, builds `TEXT_QA_TEMPLATE` (`:43`) with `{context_str}` + `{query_str}`, prepends recent chat history (multi-turn, `CHAT_HISTORY_TOKEN_BUDGET=1200` / Eco 500), `llm.stream_chat()` → UI `write_stream()` → source chips `[file (score%)]`.

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

## Plain RAG vs DocMind — Feature Matrix

Beyond raw scores — what a plain student RAG (vector search + LLM) *lacks* vs what DocMind ships. Use this in viva when asked “what’s different?”

| Feature | Plain RAG (baseline) | **DocMind** | Eval / Code |
|---|:---:|:---:|---|
| **Retrieval** | Vector only | **Hybrid BM25 + vector (RRF)** + evidence floor 0.5 | `llama_index.py:260`/`:75` — rejection 0% → **100%** |
| **Synonyms** | `money back` ≠ `refund` | **Curated synonym expansion** (short queries, BM25-only) | `llama_index.py:201` — 100% |
| **Hyphen** | `30-day` is one token | **`30-day → 30 day`** split | `llama_index.py:184` — 100% |
| **Hinglish** | `batao/kya/hai` are keywords | **Filler filtered** (`batao/kya/hai` removed, `please/tell` too) | `llama_index.py:239` — 100% |
| **Stemming** | `refunded ≠ refund` | **Porter stemmer** symmetric on docs + query | `llama_index.py:184` — 100% |
| **Title-aware** | Chunks anonymous | **Every chunk prefixed with title** | `llama_index.py:141` — 100% |
| **Dedup** | Embeds every duplicate | **Jaccard dedup** before embed | `llama_index.py:108` — 100% |
| **Cache** | Re-embeds every run | **Disk ` .index_cache`** (5, content-hashed) | `llama_index.py:832` — instant reload |
| **Eco / Heat** | One-size compute | **Eco Mode**: batch 4, 256 tokens, ≤3 chunks, 3200-char budget | `llama_index.py:166` — cooler |
| **No-hallucination** | Hallucinates on no-match | **Grounded prompt + `I could not find…` + Ask without docs** | `llama_index.py:43` / `ollama.py:528` — 100% |
| **Multi-turn** | Forgets history | **RAG history budget** (1200 / Eco 500) in prompt | `ollama.py:115` — 100% |
| **Answer style** | Fixed tone | **6 presets** (Concise/Balanced/Detailed/Bulleted/Technical/ELI5) | `chatbox.py:6` |
| **Privacy** | Cloud API | **100% offline** (Ollama); OpenAI-compatible optional | `Pipfile:6` — no torch |
| **File types** | 1–2 (txt/pdf) | **26 types** + GitHub + 5 websites | `helpers.py:19` |
| **Upload safety** | No checks | **Name/size/SSRF/IP/host validation**, 34 excluded patterns | `helpers.py:19`/`:80` |
| **Export** | Copy-paste | **Download chat `.docx`** | `settings.py:287` |
| **Numeric words** | `thirty` ≠ `30` | **`thirty → 30`** before stemming | `llama_index.py:239` — 100% |
| **Exact phrase** | `"refund policy"` bag-of-words | **Quoted phrase +0.5 RRF** if verbatim | `llama_index.py:358` — 100% |
| **Tabular** | CSV/JSON not sentence | **`Row 1: info is …`** verbalization | `llama_index.py:175` — 100% |
| **Code blocks** | Split mid-` ``` ` | **Odd-fence merge** before embed | `llama_index.py:808` — 100% |
| **Heavy deps** | Needs `torch` ~500 MB | **No torch at import** | `test_import_boundaries.py` — 0 torch |
| **Tests** | Manual | **112 unit + 43 eval** tests, CI green | `tests/` + `eval_harness.py:1` |

> Plain hallucinates on every no-match; DocMind is **measured** 100% correct rejection on the same `nomic-embed-text` model.

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
| **Answer Style** | 6 presets: Concise / Balanced / Detailed / Bulleted / Technical / Simple-ELI5 → prompt preview (collapsed) |
| **Preferences** | Eco Mode (batch 4, 256 tokens, ≤3 chunks, 3200-char budget) + Show advanced controls |
| **Advanced** (when on) | Sources per answer, Relevance threshold, Creativity, Chunk Size/Overlap + live token count — **see drill-down below** |
| **External RAG (R2R)** | Optional `utils/r2r.py` server (`http://localhost:7272`), health check, doc IDs |
| **Export** | Download chat as `.docx` (`chat_history_docx`) |

Sidebar extras: **mode badge** (Chat / RAG / R2R), **Clear Chat & Reset** expander, browser-settings persistence.

#### Advanced Settings — In Detail (hidden until “Show advanced controls” is on)

> All of these are **live** — change them and the *next* query uses the new value (chunk settings need a re-ingest). They are deliberately hidden by default so a new user sees only 3 cards.

| Control | Key (`page_state.py`) | Default | Range | What it does — in plain words | When to tweak |
|---|---|---:|---|---|---|
| **Sources per answer** | `top_k` | `3` | 1–10 slider | How many document chunks are glued into the prompt. More = broader context but more noise/hallucination risk. Eco caps at 3. | Raise to 5–7 for long reports where facts are scattered; lower to 1–2 for invoices/Q&A where one chunk holds the answer. (`utils/llama_index.py:319`) |
| **Relevance threshold** | `similarity_cutoff` | `0.30` | 0.00–1.00 (0.05 step) | Minimum vector score to keep a chunk. `0` = no filter. Higher = stricter. After that, the **evidence floor** (`0.5`) still requires a BM25 keyword hit for weak scores. | Raise to 0.4–0.5 if you see irrelevant sources; lower to 0.15 if good chunks are being dropped (but watch hallucinations). (`utils/llama_index.py:376`) |
| **Creativity** | `temperature` | `0.4` | 0.0–1.5 (0.05) | Sampling randomness. 0 = deterministic/focused, 1.5 = very creative/unpredictable. | Keep 0.3–0.5 for factual RAG; raise to 0.8–1.0 for brainstorming/drafting. (`utils/ollama.py:343`) |
| **Chunk Size** | `chunk_size` | `256` tokens | free text (int) | Tokens per piece before embedding (~4 chars/token). Smaller = more precise, more vectors & more GPU work. | Drop to 128–192 for highly structured tables; raise to 384–512 for long narrative docs. Needs **re-ingest**. (`utils/llama_index.py:669`) |
| **Chunk Overlap** | `chunk_overlap_pct` → `chunk_overlap` | `12%` → `32` tokens | 0–50% slider | Overlap between neighbours to keep sentence continuity. Computed live: `overlap = chunk_size * pct // 100`. | Raise to 20% if facts are split across boundaries; lower to 0–5% to shrink index on clean docs. Needs re-ingest. |
| **Eco Mode** | `eco_mode` | `off` | toggle | Shrinks `embed_batch 16→4`, `num_predict 512→256`, `top_k ≤3`, `context 4800→3200`. Cooler & faster, slightly less context. | Turn **on** when laptop is hot, fan loud, or answers lag >3 s. (`utils/llama_index.py:166` `utils/ollama.py:100`) |
| **R2R Base URL / API Key** | `r2r_base_url` / `r2r_api_key` | `http://localhost:7272` / `` | inside collapsed *External RAG* expander | When on, uploads go to an external R2R server instead of local LlamaIndex — less local RAM/disk. | Only if you run the R2R stack separately. (`utils/r2r.py`) |

All advanced values are persisted in `localStorage` (`utils/browser_settings.py`) and survive reloads. Reset via **Clear Chat & Reset → Reset Project** (`components/sidebar.py` / `page_state.py:34`).

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

## Troubleshooting & Security

**Common issues → docs:**

* **Ollama not running / model not found** → `docs/troubleshooting.md` — check `http://localhost:11434`, `ollama list`, `ollama pull qwen2.5:0.5b` / `nomic-embed-text:latest`, and Settings → Connection.
* **No usable content / empty index** → file may be scanned PDF or binary — try `.txt`/`.md` export; check `MIN_CHUNK_CHARS=50` (`utils/llama_index.py:62`).
* **CUDA OOM during embed** → Eco Mode on, or lower `Chunk Size`; batch auto-halves (`utils/llama_index.py:540`).
* **Website fails / SSRF block** → only `https`, no `localhost/metadata`, ≤5 URLs, ≤5 MB, HTML/plain only (`utils/helpers.py:80`). Use a public URL.
* **GitHub clone fails** → use `owner/repo` or `https://github.com/owner/repo`, check `git` is installed, ensure repo is public (`utils/helpers.py:254`).
* **Slow answers / hot laptop** → turn on **Eco Mode** (`Settings → Preferences`), split 300-page PDFs into chapters.

**Security model → `SECURITY.md`:**

* Uploads validated by `SAFE_UPLOAD_NAME_PATTERN` + extension allow-list + size caps (10 files / 25 MB / 100 MB total).
* Websites: SSRF-guarded via DNS→IP check (`_is_blocked_ip`), blocked hosts, redirect & size limits.
* GitHub: strict `owner/repo` regex, only `github.com` over `https`.
* No torch at runtime, no egress unless you point to an OpenAI-compatible server.

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

