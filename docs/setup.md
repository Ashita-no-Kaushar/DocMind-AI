# Setup

Before you get started with DocMind, ensure you have:

- A local [Ollama](https://github.com/ollama/ollama/) instance
- At least one chat-capable model available in Ollama
  - `qwen2.5:0.5b` and `llama3` are tested starter choices when installed locally
- At least one embedding-capable model available in Ollama
  - `nomic-embed-text:latest` is the tested default Ollama embedding model
- Python 3.12-3.13

DocMind is tested on Windows and Linux. Windows Subsystem for Linux (WSL) is not currently tested.

## Local

```bash
pip install pipenv
pipenv install
pipenv run streamlit run main.py
```

The default Ollama endpoint is `http://localhost:11434`. You can change it in the Settings tab. The app refreshes chat and embedding model lists for the configured endpoint.

Useful Ollama commands:

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
ollama list
```

## Docker

```bash
docker compose up -d
```

The default Docker Compose file runs the published `ashita-no-kaushar/docmind` image on port `8501`, with a read-only container filesystem, tmpfs cache directories, resource limits, and an NVIDIA GPU reservation. For AMD/ROCm hosts, see `docker-compose.yml-rocm`.

If Ollama is running on the host rather than inside the container, point the app's Ollama endpoint at a host-reachable address. On Linux Docker, you may need this Compose setting:

```
extra_hosts:
- 'host.docker.internal:host-gateway'
```

Then use `http://host.docker.internal:11434` as the Ollama endpoint.
