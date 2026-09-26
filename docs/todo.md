# Residual Work

This list contains only external validation and maintenance items that remain open. Completed implementation work is documented in [README](../README.md), [SECURITY](../SECURITY.md), and [CHANGELOG](../CHANGELOG.md).

## External service validation

- [ ] Rerun the current real Ollama chat/embedding evaluation when a live Ollama service and suitable models are available.
- [ ] Validate LM Studio, TabbyAPI, and the official OpenAI service with current model IDs, endpoint policy, credential binding, and representative documents.
- [ ] Run the opt-in R2R v3 upload, polling, replacement, rollback, ownership-registry, and cleanup tests against the intended live R2R deployment.
- [ ] Recheck external provider and R2R retention/deletion policies when their server versions or deployment configuration change.

## Docker and deployment validation

- [ ] Build the Docker image on a supported Docker host and run the container health check.
- [ ] Exercise the read-only root filesystem, named-volume permissions, non-root user, resource limits, and log/cache persistence.
- [ ] Validate an authenticated reverse-proxy deployment before permitting any non-loopback or wildcard binding.
- [ ] Run concurrent multi-browser/session tests against the process-global LlamaIndex settings and shared cache paths.

## Format and OCR validation

- [ ] Generate a valid legacy XLS fixture and a real MSG fixture, then verify extraction and safe failure behavior.
- [ ] Test encrypted Office and ZIP inputs with representative real files.
- [ ] Establish OCR ground truth for image-only PDF, DOCX, and PPTX workflows or document the external conversion path.
- [ ] Recheck parser behavior against new dependency releases and real-world malformed documents.

## Retrieval-map research

- [ ] **Decide whether to change the shipped default resolution.** The fixed `chunks_per_section` default is the weakest setting measured: it degenerates into whole-document selection on short documents and its hit rate degrades as the corpus grows, while the adaptive setting holds. Adaptive resolution is implemented and exposed in Settings → Advanced but ships off. The evidence comes from a synthetic corpus, so the default should not change until it is confirmed on a real corpus.
- [ ] Report confidence intervals. With 25 answerable questions a single hit is 4 percentage points, so every difference in the current table needs an interval before it can carry a claim.
- [ ] Re-run the ablation against the production hybrid retriever with real embeddings instead of the deterministic fixtures, and record which conclusions survive. The dense row currently uses hash embeddings, which is a reproducibility device and not a claim about any encoder.
- [ ] Build a corpus long enough to show an accuracy crossover. The scaling study found none: naive top-k stayed saturated at every size, so the map's demonstrated benefit is context efficiency at equal accuracy. Proving a crossover needs documents with hundreds of chunks each, or genuinely multi-hop questions the current corpus does not contain.
- [ ] Measure the optional LLM planner against a live OpenAI-compatible endpoint, including planner-call cost, latency, and fallback frequency. No live LLM planner result exists in this snapshot.
- [ ] Measure end-to-end answer quality with a fixed judge set so the token trade-off can be reported against answer correctness.
- [ ] Decide whether to pursue genuine multimodal retrieval. `research/multimodal.py` is a gated harness and reports the missing prerequisites: a PDF rasteriser (`pypdfium2` is a pure wheel and would satisfy this) and a real vision encoder (none is installed; the project's Ollama or OpenAI-compatible providers could supply one). A real image-resolution study is a separate body of work, and no image-resolution number is claimed anywhere in this project.

## Known defect: Settings provider display after restore

- [ ] Fix the low-frequency case where the Settings tab renders the default `Ollama` provider even though the browser-storage restore already applied and re-persisted a different provider.
- Observed roughly 1 run in 10 of `tests.test_e2e_integration.BrowserProviderTests` on this Windows host, including after switching to another tab and back, so it is not only a first-paint effect.
- The persisted state is correct when this happens: `localStorage['docmind:settings']` contains the restored `llm_backend` and provider endpoints, so the configuration is not lost. Only the rendered selectbox disagrees.
- Evidence collected in the failing run: widget value `Ollama`, restored `llm_backend` present in localStorage, no page errors, no Streamlit exception, and a Streamlit child that owned its port for the whole test.
- Next step: instrument the `restore_settings_from_browser_storage` -> `st.selectbox(key="llm_backend")` sequence to find which run renders the stale value, then fix the app rather than the assertion. The browser test now waits for an app-level restore signal before asserting, and reports the stored payload on failure.

See [Troubleshooting](troubleshooting.md) for current failure categories, [Setup](setup.md) for locked environment commands, and [handover](handover.md) for the ordered work list and the invariants that must survive future edits.
