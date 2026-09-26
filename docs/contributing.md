# Contributing

## Development setup

Use the locked [setup guide](setup.md). The supported target is Python 3.13.15. `Pipfile` records Python 3.13, and `Pipfile.lock` is committed with hashes for the 3.13 dependency contract.

Install the exact environment used by CI:

```bash
python -m pip install "pipenv==2026.8.0"
pipenv verify
pipenv install --deploy --dev --extra-pip-args="--require-hashes"
```

Do not use an unlocked install for verification. For an intentional dependency update, run `pipenv lock`, review the lockfile and hashes, then run `pipenv verify` and the locked install again.

## Code style

- Follow the surrounding Streamlit and Python patterns.
- Keep UI changes small and consistent.
- Do not add provider, ingestion, security, cache, or reset behavior without tests.
- Run Ruff and Black when they are installed.
- Avoid exposing credentials, private documents, sensitive logs, or registry contents in tests and reports.
- Do not add comments unless the task requires them.

The configured commands are:

```bash
pipenv run ruff check .
pipenv run black --check .
```

`pyproject.toml` targets Python 3.13, configures the Ruff rules, and checks Black formatting. There is no separate static type-check command configured in the project; `compileall` is the repository syntax check.

## Tests

Run the full unit/integration suite:

```bash
pipenv run python -m unittest discover -s tests -v
```

The current snapshot result is 320 tests: 319 passed and one intentional opt-in live-R2R test skipped. The suite covers provider profiles, browser persistence, format extraction, parser/archive limits, source/index transactions, cache identity and rollback, R2R lifecycle, website SSRF controls, upload signatures, retrieval, evidence, citations, logging, runtime binding, and deployment contracts.

Run compile and dependency checks:

```bash
pipenv run python -m compileall -q main.py components utils eval_harness.py tests
pipenv run python -m pip check
pipenv verify
```

The current local gate passed all of these checks.

## Browser and integration tests

Install Playwright Chromium and run the dedicated browser/integration module:

```bash
pipenv run python -m playwright install chromium
pipenv run python -m unittest tests.test_e2e_integration
pipenv run python -m unittest tests.test_e2e_contract
```

`tests.test_e2e_integration` contains 18 integration/browser tests, including real Chrome UI flows backed by local bounded fake provider and R2R servers. `tests.test_e2e_contract` contains three tests for the pinned Playwright dependency and the read-only E2E workflow. The real Chrome run passed locally.

The dedicated E2E workflow installs locked dependencies and Playwright Chromium, uses Python 3.13.15, has read-only repository permissions, and runs only `tests.test_e2e_integration`. The quality workflow runs the full unit suite, compileall, `pip check`, Ruff, Black, and mock evaluation.

For a local health smoke test:

```bash
pipenv run streamlit run main.py --server.headless=true --server.port=8520 --server.address=127.0.0.1
```

Check `http://127.0.0.1:8520/_stcore/health`, then stop the process.

## Evaluation harness

Run the deterministic mock evaluation:

```bash
pipenv run python eval_harness.py --mock --out ./eval-output
```

The current result is 42/42 scored checks passed with one real-LLM check skipped. Mock mode still exercises validation paths that can perform DNS resolution, so it is not completely hermetic.

The real evaluation is separate:

```bash
pipenv run python eval_harness.py --out ./eval-output-real
```

The current real Ollama/LLM evaluation was not rerun because the live service was unavailable. The 43/43 result dated 2026-09-23 is historical evidence, not a current result. The harness uses small fixtures and is not a browser load test, security audit, or proof of arbitrary-document factuality.

## Security-sensitive review areas

Review changes carefully in:

- Upload validation, path containment, parser selection, and archive limits.
- Website URL validation, DNS screening, pinned HTTPS transport, redirects, and response limits.
- GitHub normalization, metadata, subprocess calls, checkout inspection, and atomic replacement.
- Ollama and OpenAI-compatible endpoint validation, provider isolation, and credential binding.
- R2R uploads, polling, rollback, replacement, ownership registry, and reset behavior.
- Cross-process lifecycle locking and source/index generation state.
- Cache identity, candidate validation, atomic persistence, pruning, and reparse-point checks.
- Untrusted prompt boundaries, tokenizer-aware budgeting, evidence, and citation sanitization.
- Log redaction/rotation, browser persistence, exports, and runtime binding.
- Docker/Compose loopback publication, read-only filesystem, non-root user, and resource limits.

Tests in `tests/test_security_workstream.py`, `tests/test_security_controls.py`, `tests/test_source_state.py`, `tests/test_r2r.py`, `tests/test_deployment_contract.py`, and the E2E modules cover many of these behaviors. Review the actual code and test contract rather than relying on a test name alone.

## Runtime and reset testing

The application has no built-in authentication. Keep the default bind at `127.0.0.1`. Any non-loopback or wildcard test requires `DOCMIND_ALLOW_REMOTE_BIND=true` and a controlled authenticated reverse proxy. Do not expose a test server directly to an untrusted network.

R2R lifecycle tests use local fake servers and ownership registries. A current live R2R server was unavailable. Reset tests should verify that only owned work paths and owned remote documents are handled, and that failed cleanup remains visible and retryable.

## Dependencies and CI

`Pipfile` directly pins runtime and development dependencies. `Pipfile.lock` is hash-bearing and committed. CI uses Python 3.13.15, `pipenv verify`, `pipenv install --deploy --dev --extra-pip-args="--require-hashes"`, Ruff, Black, compileall, `pip check`, and mock evaluation. The Docker workflow builds `linux/amd64` without pushing or publishing.

When changing dependencies, update the manifest and lockfile together, review hashes and transitive changes, and run the full gate. Do not update a lockfile only to silence a local installation error.

## Pull requests

- Keep changes focused.
- Add or update tests for behavior changes.
- Run unit/integration tests, E2E tests when relevant, compile checks, Ruff, and Black.
- Update documentation when behavior, limits, environment names, or verification status changes.
- Keep current and historical evaluation results clearly separated.
- Do not include credentials, private documents, sensitive logs, or raw external responses.
- Do not commit unless the repository maintainer explicitly requests it.

## License

Contributions are licensed under GPL-3.0. See [LICENSE](../LICENSE).
