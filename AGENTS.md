# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

DocMind is a Streamlit retrieval-augmented generation application with local Ollama support, optional OpenAI-compatible providers, optional R2R, local files, public GitHub repositories, and public HTTPS websites. The application is local/single-user by default and has no built-in authentication.

Primary entry points:

- `main.py`: Streamlit bootstrap and runtime bind policy.
- `components/`: UI, source tabs, settings, chat, and session reset.
- `utils/`: ingestion, parsers, providers, RAG, security policy, cache, logging, and R2R helpers.
- `tests/`: unit, integration, browser, and deployment-contract tests.
- `docs/`: setup, usage, pipeline, troubleshooting, contributing, resources, and residual work.

## Setup

Use Python 3.13.15 for the supported project target. The current local verification environment is Python 3.12.10. `Pipfile` records Python 3.13; CI and Docker pin 3.13.15.

Use the committed, hash-bearing `Pipfile.lock`:

```bash
python -m pip install "pipenv==2026.8.0"
pipenv verify
pipenv install --deploy --dev --extra-pip-args="--require-hashes"
pipenv run streamlit run main.py --server.address=127.0.0.1 --server.port=8501
```

Do not run `pipenv lock` for routine setup. For an intentional dependency update, regenerate the lockfile, review hashes and the diff, then rerun verification.

On Windows, `run.ps1` clears bytecode caches, attempts to start Ollama, stops a matching stale Streamlit process on port 8501, and launches the app on loopback by default.

## Verification

Run focused tests while iterating, then the full gate:

```bash
pipenv run python -m unittest discover -s tests -v
pipenv run python -m compileall -q main.py components utils research eval_harness.py tests
pipenv run python -m pip check
pipenv run ruff check .
pipenv run black --check .
pipenv run python eval_harness.py --mock
```

The current snapshot result is 441 tests with 440 passes and one intentional opt-in live-R2R skip, reproduced across three consecutive full runs. The mock evaluation is 42/42 scored checks with one real-LLM skip. Black checked 56 files unchanged.

Run browser/integration and workflow contract tests separately:

```bash
pipenv run python -m playwright install chromium
pipenv run python -m unittest tests.test_e2e_integration
pipenv run python -m unittest tests.test_e2e_contract
pipenv run python -m unittest tests.test_map_ablation
```

`tests.test_e2e_integration` contains 18 integration/browser tests. `tests.test_e2e_contract` contains three workflow contract tests. The real Chrome E2E run passed locally. The real evaluation was not rerun in the current snapshot; the 2026-09-23 43/43 result is historical.

## Retrieval-map research

`research/fixtures.py` holds a deterministic offline corpus, ground truth, a BM25 retriever, a dense hash-embedding retriever, and a flat no-map agent baseline, all mirroring the `HybridRetriever` surface. `research/map_ablation.py` runs the production `MapRetrievalAgent` against naive dense/lexical retrieval and a flat agentic baseline, sweeps map resolution, and writes `research/REPORT.md`, `research/results/{routes,ablation,resolution}.csv`, `research/results/ablation.json`, and four SVG figures:

```bash
pipenv run python -m research.map_ablation
pipenv run python -m research.map_ablation --check
```

The study runs with no network, embeddings, or LLM, and CI fails if the committed artifacts drift from the code. Keep the claims narrow: routing reduces prompt-context tokens, not embedding-query cost, total compute, latency, or answer quality. Resolution means structural granularity of the layout map, not pixel or image resolution; no page image is embedded or indexed, so never present a multimodal result. `research/multimodal.py` is a gated harness for that separate study and must keep refusing to emit a substitute measurement. The oracle-planner variant reads ground truth and is an upper bound, never an LLM result. Two findings must not be softened without new data: charging the map to the prompt costs more than naive top-k at every resolution, and the scaling study found no accuracy crossover, only a context win at equal accuracy.

For a health smoke test, start the app on loopback, check `http://127.0.0.1:8520/_stcore/health`, and stop the process afterward.

## Development notes

- Prefer `rg` for search and follow the surrounding Streamlit/Python style.
- Do not revert unrelated local changes.
- Keep UI changes small and consistent with existing patterns.
- Add or update tests for behavior changes, especially validation, persistence, ingestion safety, provider isolation, and model state.
- Do not add comments unless they are required by the task.
- Keep documentation claims tied to the current source and distinguish current results from historical evidence.

## Runtime and browser settings

The runtime bind policy defaults to `127.0.0.1`. Non-loopback or wildcard binding requires `DOCMIND_ALLOW_REMOTE_BIND=true` and an authenticated reverse proxy; the application does not provide authentication. Compose publishes `127.0.0.1:8501:8501` by default.

Browser settings use `localStorage` through `utils/browser_storage_component/index.html`.

- Persisted fields are defined by `PERSISTED_SETTING_TYPES` in `utils/browser_settings.py`.
- API keys, chat history, source indexes, evidence, answer-style state, and R2R document IDs are not persisted in browser storage.
- Empty `ollama_endpoint` values must restore the default `http://localhost:11434`.
- Model lists are endpoint-aware; changing or restoring an endpoint must invalidate stale model catalogs.
- Browser persistence is not a secret store or an authorization boundary.

## Source and security-sensitive areas

Be careful when editing:

- Upload validation, parser/archive limits, and path handling in `utils.helpers` and `utils.format_ingestion`.
- Website URL validation, DNS screening, pinned HTTPS requests, redirects, and response limits in `utils.helpers`.
- GitHub normalization, metadata, subprocess calls, checkout validation, and atomic replacement in `utils.helpers`.
- Provider endpoint and credential binding in `utils.endpoint_policy` and `utils.provider_config`.
- Cross-process lifecycle locking in `utils.ingestion_lock.py`.
- Source generations, rollback-safe commits, and reset behavior in `utils.source_state.py` and `components.page_state`.
- R2R endpoint, registry, polling, rollback, replacement, and ownership-aware deletion in `utils.r2r.py`.
- Cache atomicity/pruning and process-global LlamaIndex settings in `utils.llama_index.py`.
- Untrusted prompt boundaries, total token budgeting, and citation sanitization in `utils.llama_index.py` and `utils.ollama.py`.
- Document-map construction, planner validation, node filtering, and route traces in `utils.retrieval_map.py`, `utils.llama_index.py`, and `utils.rag_pipeline.py`.
- Escaped rendering, view-model bounds, and stale-map handling in `components.retrieval_map_view.py`.
- Log redaction/rotation in `utils.logs.py`.
- Runtime binding in `utils.runtime_policy.py`, `main.py`, `run.ps1`, and Compose files.

## Residual limitations

External provider services, current real Ollama/LLM evaluation, Docker runtime, authenticated remote deployment, OCR ground truth, encrypted Office/ZIP behavior, real MSG, and valid legacy XLS fixtures are not all locally verified. Keep those limitations in documentation rather than presenting fake or historical results as current.

The retrieval-map ablation runs offline on a deterministic BM25 fixture. Keep the claim narrow: routing reduces prompt-context tokens and lexical scoring work, not embedding-query cost, total compute, latency, or answer quality. Never present the oracle-planner variant as an LLM result, and keep the open items in `docs/todo.md` visible until a live planner and a production-retriever run exist.

## License

GPL-3.0. See [LICENSE](LICENSE).
