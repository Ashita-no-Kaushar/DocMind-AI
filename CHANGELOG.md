# Changelog

## Versioning

Release tags use semantic versioning, for example `v1.5.0`. This file records changes visible in the current source tree.

## Unreleased - 2026-09-25

### Interface

- Redesign the Streamlit shell around a wide `1240px` content column, a `DocMind` wordmark with an
  agentic-map subtitle, and a theme-aware status strip that reports chat mode, provider, active source,
  and map state. The status strip uses `var(--text-color, ...)` fallbacks so it stays legible under both
  the dark and light themes, and the sidebar repeats the same state in a card above the tabs.
- Split the sidebar's destructive control from routine chat management. `Clear Chat` is now a plain
  action; `Reset Project (destructive)` is isolated inside its own expander, so an accidental expand
  cannot wipe the project.
- Add a compact per-answer route summary line (sections routed, steps, prompt-context tokens,
  reflection) so an agentic answer states what the map did without opening the map view.
- Add `components/status.py` as the single source of truth for mode, provider, source, and map state,
  with `_map_summary` tolerating a missing or malformed map instead of raising during a render.

### Verification

- Fix a regression the redesign would have introduced: the E2E reset helper still searched for the old
  `Clear Chat & Reset` expander label and now follows the renamed `Reset Project` control.
- Add `tests/test_ui_status.py` with 14 tests covering mode resolution, provider and source labelling,
  map summaries, clipping of long values, status-strip and mode-card rendering, and route-summary
  formatting including hostile route shapes.
- Harden `_route_summary` so a non-list `selected_section_ids` or a non-mapping `metrics` can no longer
  raise mid-render.
- Give the browser chat submission a real oracle. `_send_chat_prompt` waits for the user message bubble
  instead of assuming the key press landed, clears the field before retrying because a dropped submit
  leaves its text behind, and reports the input value, submit-button state, message count, exception
  count, and app log tail on failure.
- Record the current gate: 498 tests, 480 non-browser tests pass with one intentional opt-in live-R2R
  skip, Black checked 61 files unchanged, Ruff and `compileall` passed.
- One browser E2E test is still intermittently flaky: Streamlit discards a chat submission made while
  the app is running a script, and this build exposes no indicator to wait on. The symptom is measured
  and documented rather than hidden, and the failure rate is unchanged by the work above.

### Current verification

- Record the Windows local gate on Python 3.12.10: configured Ruff passed, Black checked 56 files unchanged, `compileall` passed, `pip check` passed, and `pipenv verify` passed.
- Record 441 unit/integration tests with 440 passes and one intentional opt-in live-R2R skip, reproduced across three consecutive full runs.
- Record 42/42 mock evaluation checks with one real-LLM skip.
- Record the passing local real-Chrome E2E run: `tests.test_e2e_integration` contains 18 integration/browser tests and `tests.test_e2e_contract` contains three workflow contract tests.
- Record the passing current public fetch of `https://docs.python.org/3/` and bounded temporary-directory clone/validation of `Ashita-no-Kaushar/DocMind-AI`.
- Confirm that no Streamlit child process remained after verification.

### Dependencies and runtime

- Pin direct dependencies in `Pipfile`, commit the hash-bearing Python 3.13 `Pipfile.lock`, and verify it with `pipenv verify`.
- Use locked, hash-enforced installs in CI and use default-only locked dependencies in the runtime Docker image.
- Target Python 3.13.15 in CI and Docker while retaining local Python 3.12.10 verification evidence.
- Add the dedicated read-only E2E workflow with locked dependencies and Playwright Chromium installation.
- Keep Docker/Compose build-only in CI; do not publish an image or claim local runtime verification.
- Configure the non-root, read-only Compose runtime, loopback host publication, resource limits, health check, writable data/cache/log volumes, and application-only ROCm Compose contract.

### Providers and R2R

- Keep Ollama, official OpenAI, LM Studio, TabbyAPI, and OpenAI-compatible profiles endpoint-aware and isolated between chat and embedding providers.
- Add the versioned R2R v3 client for local-file upload, status polling, replacement, rollback, bounded responses, and synchronous chat.
- Add an atomic, credential-free R2R ownership registry scoped to project/workspace and endpoint/credential identity.
- Add ownership-aware remote reset, explicit local-only reset, partial-failure reporting, and retention of failed ownership records for retry.
- Apply provider endpoint validation, official-host binding, endpoint-bound model catalogs, and browser-storage exclusion for secrets.
- Do not claim current live compatibility: Ollama, LM Studio, TabbyAPI, official OpenAI, R2R, and Docker were unavailable in the current environment.

### Retrieval and grounding

- Add independent vector and BM25 candidate pools with Reciprocal Rank Fusion and bounded candidate depths.
- Add follow-up retrieval from recent user turns, strong evidence filtering, exact-phrase boosting, near-duplicate removal, and document-level introduction chunks.
- Replace a context-only character budget with tokenizer-aware total RAG input budgeting across system guidance, history, query, and evidence.
- Attach per-message evidence and source labels, sanitize citation references, and preserve the option to answer without a forced citation.
- Preserve the fixed no-match response and one-time direct-chat fallback without treating retrieval as factual verification.
- Add untrusted-context delimiters and marker neutralization for local RAG prompts.

### Agentic map retrieval

- Add a bounded per-source document map with sections, nodes, edges, source identity, and a versioned cache signature.
- Add `MapRetrievalAgent` with a deterministic keyword planner, an optional LLM planner that receives only compact section cards, neighbour expansion, and one reflection step, all under hard limits.
- Add `allowed_node_ids` filtering to `HybridRetriever.retrieve()` while preserving the existing `retrieve(query)` call shape and a compatibility fallback for retrievers without the keyword.
- Route local RAG chat through the map, persist the route trace per message and per session, and measure evidence tokens against an opt-in unrestricted global baseline.
- Build the map after a successful ingest and keep a map build failure from rolling back a committed index.
- Add the escaped, dependency-free visual map with route trace, legend, section table, historical route rendering, and the Data Sources overview, plus Settings → Advanced controls.
- Add `research/fixtures.py` and `research/map_ablation.py` for an offline study over a naive dense baseline, a naive lexical baseline, a flat agentic no-map baseline, the map agent at low/adaptive/production-default/high resolution, component ablations, an oracle upper bound, a map-resolution sweep, and a corpus-size scaling sweep.
- Report chunk-level precision, recall, hit@1, hit@k, and MRR, and keep evidence tokens separate from map index tokens so a local navigation cost is never billed as a prompt saving.
- Add a cost-model section that reports both the deterministic-planner and LLM-planner prompt costs and states the break-even condition, including the finding that charging the map to the prompt costs more than naive top-k at every resolution and that the gap widens with corpus size.
- Add `results/{resolution,scaling}.csv`, a resolution trade-off figure, and a `--check` mode that fails when the committed artifacts drift from the code; wire it into CI.
- Add `routing_engaged_rate` to every row and aggregate so a silent fallback to unrestricted retrieval cannot be mistaken for routing quality.
- Fix a map summary bug where section cards kept only the first sentence of a chunk, hiding discriminative terms from the card scorer, and scale the title/keyword ranking bonuses by term rarity so a common word in a title cannot outweigh a rare word in the body.
- Add optional per-document adaptive map resolution, exposed in Settings → Advanced, so short documents are not collapsed into a single section.
- Implement a two-stage planner that re-scores the shortlist on full section text, measure it, and leave it off by default because it did not pay for itself; the ablation row is retained as the negative result.
- Add distractor, near-duplicate, and multi-hop fixtures, and generalize precision and recall so a multi-hop question is not credited for finding one of its two relevant chunks.
- Add `research/multimodal.py`, a gated harness for a real page-image resolution study that reports the missing PDF rasteriser or vision encoder and refuses to emit a substitute measurement.
- State in the report and the documentation that resolution means structural granularity of the layout map, not pixel or image resolution, and that no page image is embedded or indexed.
- Harden the browser E2E harness: hold the port reservation until spawn, confirm the started process still owns the port after the health check, wait for an app-level restore signal, and report the stored payload on failure.
- Record an unresolved low-frequency defect where the Settings tab shows the default `Ollama` provider although the browser-storage restore already applied and persisted another provider. The persisted configuration stays correct; only the rendered selectbox disagrees. It is tracked in `docs/todo.md` and is not claimed as fixed.

### Ingestion, state, and security

- Add a cross-process lifecycle lock for ingestion, checkout replacement, cache operations, R2R transactions, and reset.
- Add source IDs, index generations, extraction-report tagging, stale-source protection, and rollback-safe local source commits.
- Add atomic candidate/backup cache persistence, candidate validation, rollback, pruning, count/byte limits, and cache-path reparse checks.
- Add upload filename, size, batch, path, archive, parser, JSON, MIME, table, and PDF limits.
- Add GitHub metadata, download, checkout file-count/file-size/unpacked-size, clone-output, timeout, subprocess, symlink, special-file, and atomic-checkout controls.
- Add HTTPS-only website validation, all-address DNS screening, metadata-address blocking, numeric-IP TLS pinning, redirect revalidation, content-type checks, response streaming, deadlines, and categorized errors.
- Add rotating logs, credential-shaped redaction, safe exception handling, and local-only runtime binding with an explicit authenticated-proxy remote-bind contract.
- Add DOCX transcript export and selected non-secret browser settings persistence.

### Formats and tests

- Expand the truthful 25-extension capability matrix: registered local parser paths, optional text-layer PDF, and DOC/PPT conversion-required categories.
- Add safe handling for DOCX, EPUB, XLSX, PPTX, ODT, and text-layer PDF fixtures.
- Add malformed, archive, mixed-batch, and OCR-adjacent tests without leaking document content into extraction reports.
- Add parser, website SSRF, GitHub bounds, provider isolation, R2R lifecycle, cache transaction, lock, logging, prompt-boundary, deployment-contract, browser, and E2E tests.
- Keep valid legacy XLS and real MSG fixture generation, encrypted Office/ZIP behavior, and OCR ground truth explicitly unverified.

### Historical evidence and remaining external work

- Preserve the earlier 2026-09-23 real evaluation result of 43/43 as historical evidence only; it is not a current result.
- The current real Ollama/LLM evaluation was not rerun because the live service was unavailable.
- Current live LM Studio, TabbyAPI, official OpenAI, and R2R validation remains outstanding.
- A local Docker build/runtime, read-only-volume exercise, and authenticated reverse-proxy deployment remain outstanding.
- These external validation and maintenance items are tracked in [docs/todo.md](docs/todo.md); completed implementation work is not listed there.

## v1.5.0 - 2026-08-15

- Improve GitHub stale-checkout and Windows file-lock handling.
- Load only the current upload batch during local ingestion.
- Remove the local Hugging Face embedding backend.
- Apply top-k and similarity settings to the custom retriever.
- Add document-introduction retrieval for summary-style questions.
- Add DOCX chat export.
- Add logging and general reliability/security improvements.

The v1.5 Windows launcher clears stale `__pycache__` directories. Normal Streamlit and Docker startup do not perform that purge.
