# Contributing

## Development Setup

Use the setup guide in [docs/setup.md](setup.md). The declared Python version is 3.13.

The repository uses a `Pipfile`. No `Pipfile.lock` or `requirements.txt` is currently committed. Generate and commit a lockfile when reproducibility is required.

## Code Style

- Follow the surrounding Streamlit and Python patterns.
- Keep UI changes small and consistent.
- Do not add provider, ingestion, or security behavior without tests.
- Run Ruff and Black when they are installed in the development environment.

The CI workflow runs fatal Ruff checks. It does not currently run Black.

## Tests

Run the full unit suite:

```bash
pipenv run python -m unittest discover -s tests -v
```

Run compile checks:

```bash
pipenv run python -m py_compile \
  main.py \
  components/page_state.py \
  components/tabs/settings.py \
  utils/browser_settings.py \
  utils/ollama.py
```

Run the Streamlit health smoke test:

```bash
pipenv run streamlit run main.py \
  --server.headless=true \
  --server.port=8520 \
  --server.address=127.0.0.1
```

Then check:

```bash
curl http://127.0.0.1:8520/_stcore/health
```

Stop the smoke-test process after checking it.

## Evaluation Harness

```bash
pipenv run python eval_harness.py --mock --out ./eval-output
pipenv run python eval_harness.py --out ./eval-output-real
```

The harness is not currently part of CI. Mock mode still executes some live DNS validation and should not be described as fully hermetic.

## Security-Sensitive Areas

Review changes carefully in:

- Upload validation and path containment
- Website URL validation and redirects
- GitHub normalization, cloning, and file-symlink handling
- Ollama and OpenAI-compatible endpoint handling
- R2R uploads and remote document lifecycle
- Cache identity and local index contents
- Logging and browser persistence

## Dependencies

`Pipfile` declares direct runtime and development dependencies. All versions currently use wildcards, so dependency resolution is not reproducible until a lockfile is committed.

Dependabot configuration exists, but update classification and automatic merge behavior also depend on repository settings and the dependency's release semantics.

## Pull Requests

- Keep changes focused
- Add or update tests
- Run unit and compile checks
- Update documentation when behavior or limits change
- Do not include credentials, private documents, or sensitive logs

## License

Contributions are licensed under GPL-3.0. See [LICENSE](../LICENSE).
