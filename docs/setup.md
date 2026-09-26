# Setup

## Requirements

The supported project target is Python 3.13.15. `Pipfile` records Python 3.13, while the CI and Docker definitions pin 3.13.15. The current local verification gate also passed on Windows with Python 3.12.10; use 3.13.15 for normal installation and deployment.

Git is required for GitHub ingestion. A reachable model service is required for local indexing. The application itself does not provide model weights or an Ollama/R2R server.

## Locked Python environment

`Pipfile` contains direct version pins. The committed `Pipfile.lock` is hash-bearing, declares Python 3.13, and is part of the source contract. Use the lock rather than resolving fresh versions during setup:

```bash
python -m pip install "pipenv==2026.8.0"
pipenv verify
pipenv install --deploy --dev --extra-pip-args="--require-hashes"
pipenv run streamlit run main.py --server.address=127.0.0.1 --server.port=8501
```

`pipenv verify` checks that the lockfile corresponds to `Pipfile`. The locked install command is also the command used by the quality and E2E workflows. The runtime Docker image installs default locked dependencies only with `pipenv sync --categories default --extra-pip-args="--require-hashes"`.

To update dependencies intentionally:

```bash
pipenv lock
pipenv verify
pipenv install --deploy --dev --extra-pip-args="--require-hashes"
```

Review the complete lockfile and hash diff before keeping an update. Do not regenerate the lockfile as part of ordinary application work.

## Ollama

Install and start Ollama, then make one chat-capable model and one embedding-capable model available. Example setup commands are:

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
ollama list
```

The default endpoint is `http://localhost:11434`. Configure another endpoint in **Settings** when Ollama runs elsewhere. The application prefers `gemma4:latest`, then `llama3:8b`, then `llama2:7b`, then the first discovered chat-capable model. The Qwen model is an example, not a hardcoded default.

Ollama was unavailable during the current verification environment, so these commands document setup rather than a current live result.

## OpenAI-compatible providers

The Settings tab exposes separate chat and embedding profiles for:

| Profile | Default endpoint | Credential behavior |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | Explicit key required; the official host is enforced. |
| LM Studio (Local AI) | `http://localhost:1234/v1` | Key optional for a local server. |
| TabbyAPI | `http://localhost:5000/v1` | Key optional for a local server. |
| OpenAI-compatible | `http://localhost:1234/v1` | Key optional when the server permits anonymous local access. |

Chat/embedding provider endpoints reject URL credentials, query strings, fragments, control characters, and non-loopback HTTP. Use HTTPS for a chat/embedding provider on another host. The R2R client accepts absolute HTTP or HTTPS URLs; use HTTPS and an appropriate network boundary for a remote R2R deployment. Model identifiers must be accepted by both the server and the installed LlamaIndex adapter. Official OpenAI, LM Studio, and TabbyAPI live services were unavailable in the current environment.

Provider API keys are held in Streamlit session state, not browser `localStorage` or the R2R registry. Re-enter them after a process/session restart when required. Do not put credentials in Compose files, screenshots, issues, or logs.

## R2R

R2R is optional and currently used for local-file uploads and chat. Enable it under **Settings > External RAG server (R2R)**, configure the URL and optional key, and use **Test Connection**. The default URL is `http://localhost:7272`. The implemented wire contract is R2R 3.6.5 / API v3.

The current environment had no reachable R2R server. Local fake-server lifecycle tests passed, but that is not a live compatibility claim.

R2R behavior is ownership-aware:

- New files are uploaded and polled before an active replacement is committed.
- New uploads are rolled back when indexing or replacement cleanup fails.
- The local registry stores document IDs, source metadata, status, and endpoint/credential fingerprints without storing the credential itself.
- The default registry root is `~/.docmind/r2r`; set `DOCMIND_R2R_STATE_DIR` to use another local directory. `DOCMIND_R2R_PROJECT_NAME` is passed as the optional R2R project header when configured.
- **Reset Project** defaults to deleting documents owned by the current workspace and current endpoint/credential identity. The registry is retained. A partial deletion is reported and can be retried.
- **Keep R2R documents (local-only reset)** does not contact the server. It clears local project state but leaves remote documents and ownership records intact.

R2R does not replace local GitHub or website ingestion in the current UI; those sources still use the local provider/index path.

## Windows launcher

From the repository root:

```powershell
.\run.ps1
```

The launcher clears `__pycache__` directories, attempts to start Ollama, stops a matching stale Streamlit process on port 8501, and starts the app. It defaults to `127.0.0.1` and disables the Streamlit file watcher for the launched process. Its Ollama tuning variables affect an Ollama process started by the launcher, not an already-running Ollama service.

To use another local address, set `DOCMIND_BIND_ADDRESS` before launching. A non-loopback or wildcard address is refused without `DOCMIND_ALLOW_REMOTE_BIND=true`.

## Docker and Compose

Build and start the local application image:

```bash
docker compose up --build
```

The Compose contract is intentionally conservative:

- The host publishes `127.0.0.1:8501:8501` by default.
- The container runs as UID/GID `10001:10001` with a read-only root filesystem, dropped capabilities, `no-new-privileges`, resource limits, and a health check.
- Writable named volumes hold data, index cache, and logs.
- No GPU reservation is configured. Model execution is delegated to a separate Ollama service.
- `docker-compose.yml-rocm` configures the application only; it is not an Ollama ROCm runtime.

The container sets `DOCMIND_ALLOW_REMOTE_BIND=true` and `DOCMIND_BIND_ADDRESS=0.0.0.0` internally so the loopback host mapping can reach the container process. This does not publish the port beyond loopback. Do not change the host mapping to a wildcard or expose the container directly to an untrusted network.

For a deliberate remote deployment, provide an authenticated reverse proxy and keep the explicit runtime opt-in scoped to that deployment:

```text
DOCMIND_ALLOW_REMOTE_BIND=true
DOCMIND_BIND_ADDRESS=0.0.0.0
```

The opt-in is not authentication. A plain-HTTP chat/embedding provider address on a host gateway is not a way to bypass the endpoint policy; use HTTPS for a non-loopback chat/embedding provider. Protect a remote R2R URL with HTTPS and an appropriate network boundary.

Docker/Compose is statically validated and CI is build-only. A local Docker build, runtime, health check, and volume exercise were not possible in the current environment.

## First run

1. Start the selected provider and verify its endpoint in Settings.
2. Confirm a chat model and an embedding model, or enable R2R for local-file uploads.
3. Open `http://127.0.0.1:8501`.
4. Add one source and wait for the pipeline/extraction report to finish.
5. Ask a question. If local retrieval has no credible evidence, use the one-time **Ask without documents** action only when direct chat is acceptable.

## Troubleshooting setup

- `pipenv verify` fails: restore the committed `Pipfile`/`Pipfile.lock` pair and reinstall with the locked command; do not bypass the hash check.
- Models are unavailable: check the provider endpoint, model discovery, and whether the service is actually running. The current environment did not have the live providers.
- Ingestion controls are disabled: local ingestion needs valid chat and embedding settings. R2R local uploads can proceed when R2R is enabled without local models.
- A remote bind is refused: leave the address at `127.0.0.1`, or deliberately configure `DOCMIND_ALLOW_REMOTE_BIND=true` only behind an authenticated reverse proxy.
- A chat/embedding host provider using plain HTTP is rejected: use loopback where possible or configure HTTPS; do not bypass the endpoint policy. A remote R2R URL may be HTTP in the current client, so protect it with HTTPS and an appropriate network boundary.
- R2R reset reports incomplete cleanup: keep the retained registry, restore the same endpoint/credential identity, and retry the owned-document deletion. Do not manually delete registry records to force a success message.
- See [Troubleshooting](troubleshooting.md) for parser, website, GitHub, RAG, and log issues.
