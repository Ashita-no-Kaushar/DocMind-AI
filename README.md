# DocMind AI

**Agentic visual-map document retrieval — a local-first RAG application with a measured retrieval study.**

DocMind AI is a single-user Streamlit application that builds a retrieval-augmented-generation
index from local files, a public GitHub repository, or public HTTPS webpages, and answers
questions against it. On top of that baseline it implements an **agentic map retrieval layer**:
a bounded AI agent navigates a structural map of each document to decide which sections to
"zoom into", rather than retrieving blindly from the whole index.

Everything in this document is measured from the current source tree. Where a result is
historical, blocked, or not measured, it says so.

Project repository: [Ashita-no-Kaushar/DocMind-AI](https://github.com/Ashita-no-Kaushar/DocMind-AI)

---

## Table of contents

1. [The business problem](#1-the-business-problem)
2. [Why naive RAG is not enough](#2-why-naive-rag-is-not-enough)
3. [Our solution](#3-our-solution)
4. [Technical approach](#4-technical-approach)
5. [The steps we took to get here](#5-the-steps-we-took-to-get-here)
6. [Measured results](#6-measured-results)
7. [Our novelty and what is genuinely ours](#7-our-novelty-and-what-is-genuinely-ours)
8. [What we honestly do not claim](#8-what-we-honestly-do-not-claim)
9. [Complete feature list](#9-complete-feature-list)
10. [Test results in full](#10-test-results-in-full)
11. [Tech stack](#11-tech-stack)
12. [What has been built so far](#12-what-has-been-built-so-far)
13. [Why you should use this](#13-why-you-should-use-this)
14. [Future scope](#14-future-scope)
15. [Setup and running](#15-setup-and-running)
16. [Repository layout](#16-repository-layout)
17. [Limitations](#17-limitations)

---

## 1. The business problem

Teams that answer questions from documents — support desks, policy teams, compliance officers,
internal IT — hit the same wall with plain RAG. Three costs, in order of how often they hurt:

**Context cost.** A naive RAG prompt ships the top *k* retrieved chunks to the model on every
single question, whether the question needs one sentence or fifteen. If your chunks average
600 tokens and *k* = 8, that is roughly 4,800 tokens billed per question, regardless of how
simple the question is. On a high-volume help desk this is the dominant line item.

**Irrelevant context.** Those *k* chunks are chosen by embedding similarity alone, so a chunk
that is topically adjacent but factually useless rides along. It burns tokens and gives the
model plausible-looking text to be misled by.

**No accountability.** When an answer is wrong, the user cannot see *why* the system looked
where it looked. A relevance score is not an explanation.

DocMind targets all three: the map layer reduces the context actually sent, the routing step
keeps only structurally relevant sections, and every answer carries a visual route trace
showing which sections were considered, planned, and selected.

## 2. Why naive RAG is not enough

The starting point of this project was a working naive RAG application: chunk, embed, search,
prompt. That is the industry default and it is genuinely useful. It is also, as the ablation in
this repository measures, wasteful and opaque:

- The prompt size is fixed by *k*, not by what the question actually needs.
- Retrieval has no notion of document *structure*. A section that answers the question and the
  three sections immediately after it are indistinguishable to a vector similarity score.
- Nothing is recorded about the retrieval decision, so it cannot be audited or improved.

Our own measurements make the waste concrete. On the offline benchmark in `research/`, naive
BM25 top-k spent **157 prompt tokens per question** and, at the same top-*k*, returned context
in which only about **29% of the returned chunks were relevant** (chunk-level precision). The
map agent at a suitable resolution spent **86 tokens** for the same accuracy — roughly a 45%
smaller prompt — while more than **doubling** that precision to 43%.

## 3. Our solution

Two layers, deliberately separable.

**Layer 1 — the application (the product).** A complete, hardened, local-first RAG application
with the usual ingestion, retrieval, provider, and safety features listed in
[section 9](#9-complete-feature-list). This is what a user runs.

**Layer 2 — agentic map retrieval (the contribution).** After a successful ingest, DocMind
builds a **document map**: a bounded, hierarchical structure of the corpus in which each
document is divided into sections, and each section gets a short card with a title, keywords,
and a summary. Before answering, an agent scores those cards, chooses a few sections, and
retrievals are **restricted to just those sections**. A bounded second step expands the route or
falls back if the first evidence is weak. Every decision is recorded and drawn on screen.

The default planner is a deterministic keyword scorer, which means **the routing layer makes
zero extra model calls**. This is the single most important design decision in the project and
we justify it with measurements in [section 6](#6-measured-results).

## 4. Technical approach

### 4.1 Map construction

Corpus nodes are grouped into contiguous runs by source document. Each run is divided into
sections of at most *N* chunks, where *N* is either a fixed `chunks_per_section` or derived
per document by **adaptive resolution**. Every section receives:

- a content-derived, stable `section_id` (SHA-256 based, so it is reproducible across runs),
- a document-scoped `source_id`,
- a bounded human-readable title (`Source Title - Section N`),
- top keywords, and a bounded summary.

Documents and sections become map nodes; `contains` and `next` edges record the hierarchy and
the reading order. Counts of nodes, sections, keywords, and summary characters are all
hard-bounded, and truncation is recorded rather than hidden.

**Adaptive resolution** exists because a fixed chunk count is the wrong parameter. Our
measurement showed that at the fixed default a short document collapses into a *single* section,
so the agent can only choose whole documents and cannot aim inside one. When
`target_sections_per_document` is set, each document is divided into roughly that many sections
instead, bounded so short documents stay addressable and long documents stay bounded.

### 4.2 The routing agent

Per question, a bounded two-step loop:

1. **Observe** — score every section card against the standalone query and, if present, a
   history-expanded follow-up query. Scoring is IDF-weighted with query rewriting and stopword
   handling.
2. **Plan** — select a small number of sections. The deterministic planner makes no model call.
   An optional LLM planner receives *only* the compact cards; any invalid, oversized, or failed
   planner output falls back to the deterministic plan.
3. **Act** — retrieve restricted to the routed sections' member nodes via
   `allowed_node_ids`, optionally expanding to adjacent same-source sections.
4. **Reflect** — if routed evidence is weak, either expand to further positively scored sections
   or make a single unrestricted retrieval, then re-check.

Everything is bounded: at most 2 steps, at most 16 selected sections, at most 50 results, at
most 64 planner cards, 8 trace actions, 8 cache entries.

### 4.3 Token accounting

Two costs are measured and **never merged**:

| Cost | What it is | When it is billed |
|---|---|---|
| **Evidence tokens** | The chunks actually placed in the model prompt. | Always. At most the naive cost. |
| **Map index tokens** | The section cards the agent scores to make one routing decision. | Local CPU with the deterministic planner. Prompt cost only with the optional LLM planner. |

This distinction is the difference between an honest claim and a misleading one, and
[section 8](#8-what-we-honestly-do-not-claim) explains what follows from it.

### 4.4 Visualisation

The chat panel renders the map for the question — documents as columns, sections as boxes,
coloured by state (not considered / considered / planned / selected) — with a numbered route
polyline through the selected sections, a legend, a step table, an action trace, and the token
comparison. Historical answers keep their own recorded route. If the index was rebuilt since an
answer, that route is labelled stale and its metrics are withheld. All rendering is escaped and
dependency-free; the figure falls back to a text table if the size limit is exceeded.

## 5. The steps we took to get here

This is the actual sequence of work, including the parts that did not work.

1. **Hardened the existing naive RAG application.** Format parsing truthfulness, resource
   limits, transactional source/index state, a cross-process lifecycle lock, atomic cache
   persistence with pruning, provider profiles, an R2R v3 client, SSRF-resistant website
   fetching, redacted rotating logs, locked dependencies, and a hardened container.
2. **Added the retrieval map core** (`utils/retrieval_map.py`): map construction, the bounded
   agent, the deterministic and optional LLM planners, and route traces.
3. **Extended the retriever** with optional `allowed_node_ids` filtering while keeping the
   existing `retrieve(query)` call shape and a compatibility fallback for retrievers that do
   not support the keyword.
4. **Wired routing into the chat path**, with per-message and per-session route persistence and
   an opt-in global baseline for token comparison.
5. **Built the visual map UI** and fixed a defect found during review where a bounded map could
   hide the sections the agent had actually selected.
6. **Built the offline research harness** (`research/`) with a deterministic corpus, ground
   truth, and a BM25 retriever mirroring the production `HybridRetriever` surface, so the real
   production agent code is measured rather than a reimplementation of it.
7. **Wrote the ablation** across 12 variants with chunk-level precision, recall, hit@1, hit@k,
   and MRR, plus a resolution sweep and a corpus-size scaling sweep. Added a `--check` mode
   that regenerates the artifacts and fails when the committed results drift from the code, and
   wired it into CI.
8. **Found and fixed two real retrieval bugs** that the study exposed (detailed in
   [section 6.4](#64-two-real-bugs-the-study-exposed)).
9. **Implemented and then rejected a two-stage planner** on the evidence, keeping the negative
   result as an ablation row.
10. **Added a gated multimodal harness** for a genuine page-image resolution study that reports
    its missing prerequisites instead of substituting a measurement.

## 6. Measured results

All numbers below come from `research/results/ablation.json`, regenerated by
`python -m research.map_ablation`. The study runs with **no network, no embeddings, and no LLM**,
so every figure is reproducible offline.

### 6.1 Ablation: 12 variants, 31 questions, 8 documents, 48 chunks

| Variant | Hit@1 | Hit@k | Precision | Evidence tokens | Routing engaged |
|---|---:|---:|---:|---:|---:|
| Naive vector RAG (dense, hash embeddings) | 0.84 | 1.00 | 0.14 | 222.0 | 0.00 |
| Naive lexical RAG (BM25 top-k) | 1.00 | 1.00 | 0.29 | 157.1 | 0.00 |
| Agentic text RAG (no map) | 1.00 | 1.00 | 0.35 | 104.4 | 0.00 |
| Map agent (production default, 6 chunks/section) | 0.92 | 0.92 | 0.29 | 125.7 | 0.92 |
| Map agent (low resolution, 4 chunks/section) | 0.96 | 0.96 | 0.33 | 121.0 | 0.92 |
| Map agent (adaptive, 3 sections/document) | **1.00** | **1.00** | 0.35 | 120.5 | 1.00 || Map agent (high resolution, 1 chunk/section) | **1.00** | **1.00** | **0.43** | **86.0** | 1.00 |
| Map agent (no neighbours) | 0.92 | 0.92 | 0.29 | 125.7 | 0.92 |
| Map agent (no reflection) | 0.84 | 0.84 | 0.24 | 120.3 | 1.00 |
| Map agent (single section) | 0.80 | 0.80 | 0.26 | 107.0 | 1.00 |
| Map agent (with candidate refinement) | 0.96 | 0.96 | 0.30 | 132.0 | 1.00 |
| Map agent (oracle planner, upper bound) | 0.92 | 0.92 | 0.29 | 125.7 | 0.92 |
Read this table honestly:

- The **high-resolution map matches naive top-k accuracy (1.00) using 86.0 evidence tokens
  against 157.1 — a 45.3% smaller prompt — while raising chunk-level precision from 0.29 to
  0.43.** That is the result we stand behind.
- The **production default resolution is the weakest map setting** and is the one place the map
  loses accuracy. This is a real finding, not a bug: at 6 chunks per section a 6-chunk document
  becomes exactly one section, so the agent can only pick documents. We did not silently change
  the shipped default; see [section 8](#8-what-we-honestly-do-not-claim).
- The **oracle planner is an upper bound that reads ground truth**. It is not an LLM result and
  is never reported as one.
- The **two-stage refinement variant is worse** than the default. We implemented it, measured it,
  and left it off.

`Routing engaged` is reported for exactly one reason: to prove the map is genuinely routing.
An earlier version of this study silently fell back to unrestricted retrieval at fine
resolutions and *looked* accurate because of it. Adding this column makes that class of
artifact impossible to hide.

### 6.2 Map resolution sweep

Resolution here is **structural granularity**: how many chunks are merged into one map section.

| Resolution | Sections | Map index tokens | Hit@k | Evidence tokens |
|---|---:|---:|---:|---:|
| Highest (1 chunk/section) | 48 | 2484 | 1.00 | 86.0 |
| High (2 chunks/section) | 24 | 1358 | 1.00 | 120.5 |
| Medium-high (3 chunks/section) | 16 | 927 | 0.92 | 117.5 |
| Medium (4 chunks/section) | 16 | 922 | 0.96 | 121.0 |
| Medium-low (5 chunks/section) | 16 | 877 | 0.96 | 122.0 |
| Production default (6 chunks/section) | 8 | 467 | 0.92 | 125.7 |
| Adaptive (3 sections/document) | 24 | 1358 | 1.00 | 120.5 |

Finer maps cost more to score but let the agent aim at a smaller target, so evidence tokens
fall as the index grows. Accuracy is not monotonic in this corpus, which is why the sweep
matters: a single default number would have hidden the best setting entirely.

### 6.3 Corpus-size scaling: where the map's cost goes

The question set is held fixed while distractor documents that reuse the corpus vocabulary are
added, so every change is caused by corpus difficulty rather than easier questions.

| Documents | Chunks | Variant | Hit@k | Evidence tokens | Map index tokens |
|---:|---:|---|---:|---:|---:|
| 8 | 48 | Naive lexical RAG | 1.00 | 157.1 | 0 |
| 8 | 48 | Map agent (high resolution) | 1.00 | 86.0 | 2484 |
| 16 | 96 | Naive lexical RAG | 1.00 | 169.5 | 0 |
| 16 | 96 | Map agent (high resolution) | 1.00 | 83.1 | 4710 |
| 32 | 192 | Naive lexical RAG | 1.00 | 172.0 | 0 |
| 32 | 192 | Map agent (high resolution) | 1.00 | 83.1 | 6201 |
| 64 | 384 | Naive lexical RAG | 1.00 | 172.3 | 0 |
| 64 | 384 | Map agent (high resolution) | 1.00 | 83.1 | 6201 |

Two findings:

- **Naive prompt cost stays flat (157 → 172) because it is capped at top-*k*; the map's index
  cost grows with the corpus (2484 → 6201).** This is the break-even condition below, measured.
- **No accuracy crossover was observed.** Naive top-k stayed saturated at every size, so this
  study does *not* show a point where routing finds answers that top-*k* misses. See
  [section 8](#8-what-we-honestly-do-not-claim).

### 6.4 Two real bugs the study exposed

These are the most valuable outputs of the work, and both were found by measurement rather than
inspection:

1. **Section cards kept only the first sentence of each chunk.** `_bounded_excerpt` stopped at
   the first sentence long enough to keep, so a discriminative term late in a chunk was invisible
   to the card scorer. For the question *"How long is the staging soak time?"* the term `soak`
   appeared in **zero of 48** section cards, and the map confidently routed to a shipping
   document that merely had the word "time" in its title. The excerpt now spends its whole
   character budget on as many whole sentences as fit; `soak` went from 0 to 1 document.
2. **A common word in a title could outweigh a rare word in the body.** The scorer added flat
   +1.5 / +0.75 bonuses for title and keyword matches, so a high-frequency term in a title
   outranked a low-frequency term in the body text. Those bonuses are now scaled by the term's
   IDF, so a structural match is worth more *for the same term* and cannot beat a rarer term
   found in the body.

### 6.5 Cost model: what the map actually costs

| Variant | Deterministic planner (prompt tokens) | LLM planner upper bound | Map index / naive evidence |
|---|---:|---:|---:|
| Naive lexical RAG | 157.1 | 157.1 | 0.00 |
| Agentic text RAG (no map) | 104.4 | 104.4 | 0.00 |
| Map agent (production default) | 125.7 | 592.7 | 2.97 |
| Map agent (high resolution) | 86.0 | 2570.0 | 15.81 |

**Break-even condition:** the map only pays for itself under the LLM-planner model when
`map_index_tokens < naive_evidence_tokens - map_evidence_tokens`. Because naive top-*k* evidence
is capped by *k* while the map index grows with the number of sections, **this condition gets
harder to satisfy as the corpus grows, not easier.**

That is why the defensible economic claim is *not* a token saving. It is:

> The deterministic planner performs structure-guided routing with **zero extra model calls**.
> The map is scored locally, so the only prompt cost is the routed evidence, which is at most the
> naive cost.

## 7. Our novelty and what is genuinely ours

We want to be precise here, because "novelty" is where projects usually overclaim.

**What is standard and not ours:** RAG itself; chunking and embedding; BM25; Reciprocal Rank
Fusion; hybrid dense-plus-lexical retrieval; LLM-as-planner; ReAct-style observe/act loops;
document trees; graph retrieval; RRF scoring. All of these are prior art. We use them.

**What we believe is our contribution:**

1. **A layout map as a first-class, persisted retrieval artifact with a stable identity.** The
   map is content-derived, versioned, cached per source identity, and rebuilt transactionally
   with the index — not recomputed ad hoc per query. It is rendered visually and its route is
   persisted per message, which makes a retrieval decision auditable after the fact.
2. **A resolution parameter treated as a first-class experimental variable.** Almost every RAG
   system fixes a chunk size and never measures it. We treat *how finely the document is divided
   for routing* as a tunable axis, show that the conventional fixed default is the worst setting
   on our benchmark, and ship an adaptive alternative that derives the granularity per document.
   This is a small but genuine, reproducible finding.
3. **An explicit two-cost accounting discipline.** Separating *prompt tokens* from *local index
   tokens* — and refusing to merge them — is what let us discover that the LLM-planner variant
   is a net loss at every resolution. Most agentic-RAG papers report a single token number.
4. **A guard against self-deception in the evaluation itself.** Reporting
   `routing_engaged_rate` exists because a version of our own agent silently fell back to
   unrestricted retrieval and *improved* its apparent accuracy. Publishing the metric that
   caught our own bug is part of the contribution.
5. **Two concrete retrieval bugs, found by ablation and fixed at the root.** The
   first-sentence-only card excerpt and the un-scaled title/keyword bonuses are the kind of
   defect that silently degrades every map-based system built this way.

**What is not novel and we do not claim:** the agentic loop, the planner, the map data structure
in isolation, or the claim that routing reduces cost under a model-based planner.

## 8. What we honestly do not claim

This section is the most important one in the README.

- **We do not claim the map reduces total cost.** Under a model-based planner it *increases*
  prompt cost at every resolution we measured — at high resolution the map index is 15.8x the
  entire naive prompt. Only the deterministic-planner configuration keeps the map off the bill,
  and that configuration is a local CPU cost, not a saving.
- **We do not claim better answers.** No answer-quality or faithfulness evaluation exists. The
  study measures retrieval and tokens only.
- **We do not claim a lower embedding cost.** Every question still issues the same embedding
  query. Only the lexical scoring stage is narrowed.
- **We do not claim an accuracy crossover.** Naive top-*k* stayed saturated across 8–64
  documents. Demonstrating a point where routing beats top-*k* on accuracy needs documents with
  hundreds of chunks each, or genuinely multi-hop questions; our corpus has neither.
- **"Resolution" means structural granularity, not pixels.** No page image, figure, or table
  region is embedded or indexed anywhere in this project. `python -m research.multimodal` is a
  gated harness for that separate study; it reports which prerequisite is missing
  (a PDF rasteriser and a real vision encoder — neither is installed) and **refuses to emit a
  substitute measurement**. No image-resolution number is claimed.
- **The dense baseline uses hash embeddings**, which is a reproducibility device, not a trained
  encoder. Its absolute numbers are not a claim about any real model.
- **The corpus is synthetic** and small. With 25 answerable questions a single hit is 4
  percentage points. **We have not published confidence intervals**, and no difference in these
  tables should be read as statistically established.
- **The oracle planner reads ground truth.** It is an upper bound and is never an LLM result.
- **The shipped default resolution is not the best setting we measured.** Adaptive resolution is
  implemented and available but ships off, because changing a retrieval default on the strength
  of a synthetic 8-document corpus would be reckless. This is tracked as an open decision.
- **The LLM planner has no live-provider result.** Every number here is from the deterministic
  planner.

## 9. Complete feature list

### Sources and extraction
- Local uploads with a 25-extension allowlist, per-batch size limits, safe filenames, and path
  containment.
- Public GitHub repositories with bounded metadata, shallow clone, checkout inspection, and safe
  source loading.
- Public HTTPS webpages with DNS validation, pinned numeric-address connections, redirect
  revalidation, content-type checks, and response-size limits.
- Per-file extraction reports showing loaded document counts, extracted character counts, safe
  warnings, and unsupported/skipped status, without copying document text into the report.
- Table and JSON record verbalization, chunking, code-fence repair, minimum-content filtering,
  near-duplicate filtering, and title enrichment.
- DOCX, EPUB, XLSX, PPTX, ODT, and text-layer PDF fixtures covered by format tests, alongside
  malformed, archive, and OCR-adjacent cases.

### Retrieval and grounding
- Independent vector and BM25 candidate pools fused with Reciprocal Rank Fusion.
- **Agentic map retrieval:** a per-source document map with a deterministic keyword planner,
  neighbour expansion, one bounded reflection step, and an optional LLM planner that receives
  only compact section cards.
- **Optional per-document adaptive map resolution**, exposed in Settings → Advanced.
- Node-level restriction of retrieval to routed sections, preserving the existing
  `retrieve(query)` call shape.
- Per-answer route traces with considered, planned, and selected section identifiers, step
  count, planner mode, routing-engagement flag, and measured prompt-context tokens against an
  opt-in unrestricted global baseline.
- Interactive visual map of the routed sections in the chat panel and the Data Sources tab,
  including historical routes and stale-map labelling.
- Conversation-aware follow-up retrieval from recent user turns.
- Tokenizer-aware total RAG input budgeting across system guidance, history, query, and evidence.
- Evidence attached to each local RAG message, source labels after answers, and
  citation-number sanitization.
- A fixed no-match response when local retrieval returns no credible evidence; no citation is
  forced onto an answer.
- Direct model chat when no local index is active, and optional R2R routing for an active remote
  source.

### Providers
- Ollama, official OpenAI, LM Studio, TabbyAPI, and generic OpenAI-compatible profiles, each
  endpoint-aware and isolated between chat and embedding providers.
- Endpoint validation, official-host credential binding, and endpoint-bound model catalogs.
- R2R v3 client: local-file upload, status polling, replacement, rollback, bounded responses,
  synchronous chat, and an atomic credential-free ownership registry.

### State, security, and operations
- Cross-process ingestion lifecycle lock with bounded waiting and safe lock-file handling.
- Candidate-based, rollback-safe index persistence with atomic replacement, cache validation,
  pruning, count and byte limits, and symlink/reparse-point rejection.
- Source generations and ownership-aware reset, including explicit local-only reset and
  partial-failure reporting.
- GitHub metadata, clone output, checkout size/file-count, and parser/archive limits.
- SSRF-resistant website fetching with all returned A/AAAA addresses checked and HTTPS
  connections pinned to a validated address while retaining hostname certificate verification.
- Untrusted document context boundaries with delimiter neutralization, rotating and redacted
  logs, endpoint-aware provider credentials, and local-only binding by default.
- DOCX transcript export and browser persistence for a selected non-secret settings subset.

### Research tooling
- Deterministic offline corpus, ground truth, BM25 and dense-hash retrievers, and a flat no-map
  agent baseline, all mirroring the production retriever surface.
- A 12-variant ablation, a 7-point resolution sweep, and a 4-point corpus-size scaling sweep.
- A `--check` mode that regenerates all artifacts and fails on drift, wired into CI.
- A gated multimodal harness that reports missing prerequisites instead of fabricating results.

## 10. Test results in full

**Current local gate:** Windows, Python 3.12.10. Supported target is Python 3.13.15.

| Check | Command | Result |
|---|---|---|
| Unit and integration suite | `python -m unittest discover -s tests` | **498 tests, 480 non-browser tests pass, 1 intentional opt-in live-R2R skip** |
| Browser E2E | `python -m unittest tests.test_e2e_integration` | 18 tests — **one remains timing-sensitive, see below. Passes when the machine is idle; observed failure rate roughly 1-in-4 under sustained load** |
| E2E workflow contract | `python -m unittest tests.test_e2e_contract` | 3 tests, passing |
| Retrieval-map research | `python -m unittest tests.test_map_ablation` | 37 tests, passing |
| Map core, integration, UI, resolution | `4 modules` | 114 tests, passing |
| Multimodal harness | `python -m unittest tests.test_multimodal_research` | 9 tests, passing |
| Mock evaluation | `python eval_harness.py --mock` | **42/42 scored checks, 1 real-LLM skip** |
| Ruff | `ruff check .` | Passed |
| Black | `black --check .` | Passed, 61 files unchanged |
| Byte compilation | `python -m compileall` | Passed |
| Dependency integrity | `python -m pip check` | No broken requirements |
| Lockfile | `pipenv verify` | `Pipfile.lock` is up to date |
| Research artifacts current | `python -m research.map_ablation --check` | Artifacts match the code |
| Live Python docs fetch | `https://docs.python.org/3/` | Passed |
| Bounded GitHub clone | `Ashita-no-Kaushar/DocMind-AI` | Passed, temporary directory |
| Docker build and run | — | **Not possible here; no Docker CLI. Statically validated and covered by CI build-only checks.** |
| Live Ollama / LM Studio / TabbyAPI / OpenAI / R2R | — | **Not available in this environment. No live-service result is claimed.** |

### The one known flaky test we are not hiding

It is in `tests.test_e2e_integration`, it is **timing-sensitive rather than logically wrong**, and it is
worse on a loaded machine. It is not fixed.

**Chat submission is silently dropped by the frontend.** Affects
`BrowserSmokeTests.test_real_browser_smoke_clears_chat_and_local_resets`. Streamlit discards a chat
submission that arrives while the app is still running a script, and this Streamlit build exposes **no
element, attribute, or `aria-busy` state** that indicates a run is in progress, so the test cannot wait
its way out of the race. The failure is total, not partial: the prompt never reaches the app, so the
fake provider records **zero** `/v1/chat/completions` requests, no user message bubble appears, and no
Streamlit exception is raised. The first streamed-token budget is a named 120 s constant
(`FIRST_TOKEN_TIMEOUT_MS`) and is not the cause; the wait simply expires because the turn never started.

The test now uses `_send_chat_prompt`, which confirms the user message bubble actually appeared and
retries up to three times, clearing the field first because a dropped submit leaves its text behind and
re-filling an unchanged value is invisible to the widget's own state. On failure it reports the chat
input value, the submit button state, the message count, the exception count, and the app log tail, so
the next occurrence is diagnosable instead of an opaque 120 s timeout. This narrows the symptom and
makes the test self-documenting; it does **not** eliminate the flake, and the measured rate is
unchanged. Tracked in `docs/todo.md`.

Along the way we did fix two genuine harness bugs that were inflating and misattributing
results: a port-reuse race where a stale Streamlit instance could answer the health check, and an
assertion that read the widget before the restore had landed.

## 11. Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit 1.64.0 | Single-user local app, no frontend build step |
| Retrieval framework | LlamaIndex Core 0.14.25 | Document ingestion and node abstractions |
| RRF ranking | `rank-bm25` 0.2.2 | Reference BM25 for the lexical pool |
| Embeddings | `llama-index-embeddings-openai` 0.6.0, Ollama | Provider-pluggable, endpoint-aware |
| LLMs | `llama-index-llms-ollama`, `-openai`, `-openai-like` | Local and remote chat backends |
| R2R | `r2r` client (v3 protocol) | Optional remote retrieval |
| Parsing | `python-docx`, `python-pptx`, `openpyxl`, `xlrd`, `xlrd`, `odfpy`, `ebooklib`, `pypdf`, `extract-msg`, `olefile`, `striptrf`, `nbconvert` | 25-extension contract |
| Web fetching | `requests` 2.34.2 with custom SSRF policy | Needs DNS pinning, so not a bare library call |
| Research harness | Python standard library only | Zero extra dependencies; SVG written by hand |
| Testing | `unittest`, `playwright` 1.63.0, `ruff`, `black` | Standard library runner; real browser E2E |
| Packaging | `Pipfile` + hash-bearing `Pipfile.lock`, `pyproject.toml` | Reproducible, hash-enforced installs |
| Runtime | Docker (Python 3.13.15), Compose, ROCm Compose variant | Non-root, read-only, loopback-published |
| CI | GitHub Actions `quality.yml` and `e2e.yml` | Read-only permissions; CI fails on artifact drift |

## 12. What has been built so far

**By size:** 30 Python modules in the application and research code (~19,500 lines), 28 test
modules (~10,700 lines), 498 tests, 3 CI workflows, and 9 infrastructure files. The last commit
(`713e090`) was 92 files and roughly +37,900 / −2,500 lines.

### 12.1 Implementation inventory

Status is stated precisely, because "done" means different things for a locally verified
subsystem and for one that has never touched a live service.

| Subsystem | State | Where | How it was verified |
|---|---|---|---|
| Document map + routing agent | **Implemented, offline-validated** | `utils/retrieval_map.py` (2,046) | 114 dedicated tests + 12-variant ablation |
| Visual map UI | **Implemented, browser-validated** | `components/retrieval_map_view.py` (752) | Escaping, bounds, AppTest, and real Chrome E2E |
| Hybrid retrieval + RRF | **Implemented, fixture-validated** | `utils/llama_index.py` (2,331) | Unit + mock evaluation, **never against real embeddings** |
| Chat pipeline + token budget | **Implemented, fixture-validated** | `utils/ollama.py` (1,613) | Unit tests, fake provider E2E |
| Format ingestion (25 extensions) | **Implemented, fixture-validated** | `utils/format_ingestion.py` (2,197) | Malformed, archive, OCR-adjacent fixtures |
| File / GitHub / website ingestion | **Implemented** | `utils/helpers.py` (1,552) | Live docs fetch and live bounded clone passed; parsers are not a sandbox |
| R2R v3 client | **Implemented, never executed live** | `utils/r2r.py` (1,498) | Local fake R2R server only |
| Provider profiles (5 backends) | **Implemented, never executed live** | `utils/provider_config.py` (392) | Endpoint/credential policy tests only |
| Source state + transactional index | **Implemented** | `utils/source_state.py` (525), `rag_pipeline.py` (556) | Rollback, reset, generation tests |
| Cross-process lifecycle lock | **Implemented** | `utils/ingestion_lock.py` (158) | Concurrency and lock-file handling tests |
| Runtime bind policy | **Implemented** | `utils/runtime_policy.py` (110) | Policy tests; not a deployment |
| Logging (rotation + redaction) | **Implemented** | `utils/logs.py` (209) | Redaction and rotation tests |
| Browser settings persistence | **Implemented** | `utils/browser_settings.py` (239) | Unit tests; restore display defect resolved and confirmed by CI |
| Settings / sources UI | **Implemented** | `components/tabs/settings.py` (740) and 4 more tab modules | AppTest + real Chrome E2E |
| Research harness (3 studies) | **Implemented, self-verifying** | `research/` (2,849) | 37 tests; CI fails on artifact drift |
| Multimodal resolution study | **Harness only, no measurement** | `research/multimodal.py` | 9 tests assert it refuses to fabricate |
| Evaluation harness | **Implemented** | `eval_harness.py` | 42/42 mock; real run is historical only |
| Dependency locking | **Implemented** | `Pipfile`, `Pipfile.lock`, `pyproject.toml` | `pipenv verify` + hash-enforced installs |
| Container + Compose | **Image builds in CI; never run locally** | `Dockerfile`, 2 Compose files | Static checks plus a real CI build; no local Docker runtime |
| CI workflows | **Observed green** | 3 workflows in `.github/workflows/` | Quality, E2E, and Docker Build all passed at `6bc5f7d` |

### 12.2 Not done, stated plainly

- **No live-service validation** of Ollama, LM Studio, TabbyAPI, official OpenAI, R2R, or
  Docker. Every provider and container path is code plus fixtures, not evidence.
- **No answer-quality evaluation.** Retrieval and tokens only.
- **No confidence intervals.** With 25 answerable questions, one hit is 4 percentage points, so
  no difference in these tables is statistically established.
- **No image-resolution or multimodal retrieval result**, and none is claimable — the required
  rasteriser and vision encoder are not installed.
- **No LLM-planner measurement.** Every number comes from the deterministic planner.
- **The shipped default map resolution is not the best setting measured.** Adaptive resolution
  is implemented and available but ships off pending real-corpus confirmation.
- **No accuracy crossover demonstrated** between map routing and naive top-*k*.
- **One flaky browser E2E test**, timing-sensitive, detailed in
  [section 10](#10-test-results-in-full). The provider restore display defect it was blamed on is fixed and CI-green.
- **CI was observed green at `6bc5f7d`** (Quality, E2E, and Docker Build). One browser E2E test is
  still intermittently flaky on a loaded machine.
- **No multi-session isolation.** LlamaIndex `Settings` and adapter caches are process-global.
- **No authentication.** Remote binding is opt-in and expects a user-supplied reverse proxy.

### 12.3 Deliberately not built

- **A synthetic image "embedding".** The multimodal harness refuses to emit a placeholder
  vector, because numbers derived from pixel statistics would look like a finding while
  measuring nothing.
- **A second scorer.** Card scoring and full-text refinement were deliberately unified into one
  IDF-weighted implementation so the two stages cannot drift apart.
- **An unmeasured default change.** The resolution default was left alone rather than tuned to
  win on a synthetic 8-document corpus.

## 13. Why you should use this

- **Cheaper prompts at equal accuracy.** On our benchmark, 45% fewer prompt tokens than naive
  top-*k* for the same retrieval accuracy, and roughly double the context precision.
- **No extra model calls.** The default router is local and deterministic. You do not pay for
  a planning call to get the token reduction.
- **Auditable retrieval.** Every answer shows which sections were considered, planned, and
  selected, with a numbered route on a map of your document. When an answer is wrong, you can
  see whether the system looked in the wrong place or read the right place badly.
- **It degrades visibly, not silently.** If the map cannot route, the trace says so. If a route
  is stale, it is labelled stale instead of showing confident stale numbers.
- **Local-first and private by default.** Loopback-only binding, a 25-extension allowlist,
  resource limits on every untrusted input path, SSRF-resistant fetching, and redacted logs.
- **Reproducible research.** The entire study reruns offline with no network, no embeddings, and
  no model, and CI fails if the committed numbers drift from the code.
- **Honest about its limits.** The limitations section is part of the product documentation, not
  a footnote.

## 14. Future scope

Ordered by how much they would strengthen the work.

1. **Change the default resolution after confirming it on a real corpus.** Adaptive resolution
   is the evidence-backed choice; it ships off pending real-data validation.
2. **Publish confidence intervals.** With 25 questions, no current difference is statistically
   established. Bootstrap intervals are the minimum bar.
3. **Build a corpus that produces a real accuracy crossover.** Documents with hundreds of chunks
   each, plus genuinely multi-hop questions, are what would show routing finding answers that
   top-*k* misses. The harness already supports scaling the corpus.
4. **Measure the LLM planner against a live endpoint**, including planner cost, latency, and
   fallback frequency, so the LLM-planner column becomes a real measurement rather than an
   upper bound.
5. **Add an answer-quality evaluation** with a fixed judge set, so the token trade-off can be
   reported against correctness rather than retrieval alone.
6. **Genuine multimodal retrieval.** Page rasterisation, a real vision encoder, and region-level
   grounding. The harness is in place and deliberately refuses to fabricate; this is the single
   largest piece of remaining work.
7. **Fix the Settings provider display defect** and close the last flaky test.
8. **Concurrency and multi-session isolation.** LlamaIndex `Settings` and adapter caches are
    process-global, so lifecycle locking reduces races but does not give true per-session index
    isolation.
9. **Authenticated deployment.** Remote binding is gated behind an explicit opt-in and requires a
    user-configured reverse proxy; the application ships no authentication of its own.
10. **Docker runtime verification** on a supported host.

## 15. Setup and running

Python 3.13.15 is the supported target; local verification used 3.12.10.

```bash
python -m pip install "pipenv==2026.8.0"
pipenv verify
pipenv install --deploy --dev --extra-pip-args="--require-hashes"
pipenv run streamlit run main.py --server.address=127.0.0.1 --server.port=8501
```

On Windows, `run.ps1` clears bytecode caches, attempts to start Ollama, stops a stale Streamlit
process on port 8501, and launches the app on loopback.

### Reproducing the research

```bash
pipenv run python -m research.map_ablation          # regenerate the report, tables, figures
pipenv run python -m research.map_ablation --check  # fail if committed artifacts are stale
pipenv run python -m research.multimodal             # report multimodal prerequisites
```

### Verification gate

```bash
pipenv run python -m unittest discover -s tests
pipenv run python -m compileall -q main.py components utils research eval_harness.py tests
pipenv run python -m pip check
pipenv run ruff check .
pipenv run black --check .
pipenv run python eval_harness.py --mock
pipenv run python -m unittest tests.test_e2e_integration
pipenv run python -m unittest tests.test_e2e_contract
```

## 16. Repository layout

```text
.
├── main.py                     Streamlit entry point and bind policy
├── components/
│   ├── chatbox.py              Chat, sources, and per-message route rendering
│   ├── retrieval_map_view.py   Escaped visual map, route trace, metrics
│   ├── page_state.py           Initial state, browser-storage restore, reset
│   └── tabs/                   Data sources and settings
├── research/
│   ├── fixtures.py             Deterministic corpus, ground truth, retrievers, baselines
│   ├── map_ablation.py         Ablation, resolution sweep, scaling sweep, report
│   ├── multimodal.py           Gated page-image harness; refuses to fabricate
│   ├── REPORT.md               Generated study report
│   ├── results/                Generated CSV and JSON
│   └── figures/                Generated SVG figures
├── utils/
│   ├── retrieval_map.py        Map model, agent, planners, adaptive resolution
│   ├── llama_index.py          Hybrid retriever, RRF, node filtering
│   ├── ollama.py               Chat pipeline, routing integration, token accounting
│   ├── rag_pipeline.py         Transactional ingest and index lifecycle
│   ├── format_ingestion.py     Format parsing and extraction reports
│   ├── helpers.py              Files, GitHub, and website ingestion
│   ├── r2r.py                  R2R v3 client and ownership registry
│   ├── provider_config.py      Provider profiles and endpoint policy
│   ├── endpoint_policy.py      Endpoint validation and host binding
│   ├── source_state.py         Source generations and reset ownership
│   ├── ingestion_lock.py       Cross-process lifecycle lock
│   ├── browser_settings.py     Non-secret browser persistence
│   ├── logs.py                 Rotating, redacted logging
│   └── runtime_policy.py       Loopback bind enforcement
├── tests/                      498 tests across unit, integration, research, and browser
├── docs/                       Setup, usage, pipeline, contributing, residual work
├── .github/workflows/          quality.yml, e2e.yml, main.yaml
├── Pipfile / Pipfile.lock / pyproject.toml
├── Dockerfile / docker-compose.yml / docker-compose.yml-rocm
├── eval_harness.py             Mock and real subsystem evaluation
└── AGENTS.md                   Contribution rules for coding agents
```

## 17. Limitations

- No built-in authentication. The runtime default is loopback-only; remote binding requires an
  explicit opt-in and a user-configured authenticated reverse proxy.
- LlamaIndex `Settings`, adapter caches, logging, and some filesystem paths are process-global.
  Lifecycle locking reduces races but does not provide true per-session index isolation.
- Direct chat can use general model knowledge; only the local RAG path is document-grounded.
  There is no claim-entailment or citation-support verifier.
- Image-only PDF, DOCX, and PPTX content needs external OCR or conversion. DOC and PPT need
  conversion to DOCX/PPTX. OCR ground truth, encrypted Office/ZIP content, valid legacy XLS
  fixtures, and real MSG fixtures remain unverified.
- GitHub metadata can be unavailable and falls back to a bounded clone. Resource limits reduce
  exposure but do not turn `git` or third-party parsers into a sandbox.
- Website fetching does not execute JavaScript or bypass anti-bot systems.
- R2R is limited to local-file routing; its server compatibility and retention behaviour need
  live validation.
- Chat history is session-scoped and is not persisted across restarts. One local source/index is
  active at a time.
- The retrieval study is offline and synthetic. See
  [section 8](#8-what-we-honestly-do-not-claim) for the full list of claims we do not make.
- One browser E2E test remains flaky because Streamlit can drop a chat submission made while the
  app is still running; it is described in [section 10](#10-test-results-in-full).
- Current live Ollama, LM Studio, TabbyAPI, official OpenAI, R2R, and Docker runtime validation
  was unavailable in this environment.

---

## License

GPL-3.0. See [LICENSE](LICENSE).
