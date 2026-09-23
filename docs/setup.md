# Setup

The declared runtime is Python 3.13. The current local verification environment also runs successfully on Python 3.12.10, but 3.13 is the version used by `Pipfile`, Docker, and CI.

## Ollama

Install and start Ollama, then make at least one chat-capable model and one embedding-capable model available.

Example commands:

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
ollama list
```

The default endpoint is `http://localhost:11434`. Change it in the application Settings tab when Ollama runs elsewhere.

The application prefers `gemma4:latest`, then `llama3:8b`, then `llama2:7b`, then the first discovered chat model. The example Qwen model is not a hardcoded default.

## Pipenv Setup

```bash
python -m pip install pipenv
pipenv install
pipenv run streamlit run main.py
```

Open `http://127.0.0.1:8501`.

The repository currently does not contain `Pipfile.lock`, so dependency versions are resolved when the environment is installed. Generate and commit a lockfile before treating two installations as reproducible.

## Existing Virtual Environment

On Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py `
  --server.port=8501 `
  --server.address=127.0.0.1
```

The provided launcher performs additional Windows setup:

```powershell
.\run.ps1
```

It clears `__pycache__` directories, attempts to start Ollama, stops a matching stale Streamlit process on port 8501, and launches the app.

## Models and Dependencies

- Ollama mode requires a reachable chat model and embedding model.
- OpenAI-compatible modes require a compatible endpoint and model identifiers supported by both the server and the installed LlamaIndex adapter.
- R2R requires a separate R2R server. It is currently integrated only for local-file uploads and chat.
- GitHub ingestion requires a `git` executable.
- Website and GitHub ingestion require outbound network access.

## Docker

The Compose files now build the local source and mount writable volumes for ingestion data and index cache:

```bash
docker compose up --build
```

If Ollama runs on the host, configure the endpoint in the app as:

```text
http://host.docker.internal:11434
```

The ROCm Compose file configures the Streamlit container only. Ollama's AMD/ROCm support must be configured in the separate Ollama installation.

Docker was not built or run during the latest local verification because Docker is not installed in that environment. Treat the Docker files as statically reviewed but runtime-unverified until a Docker host completes a build and health check.
