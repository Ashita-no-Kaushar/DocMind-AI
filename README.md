# DocMind AI


[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8588/badge)](https://www.bestpractices.dev/projects/8588)
![GitHub Commit Activity](https://img.shields.io/github/commit-activity/t/Ashita-no-Kaushar/DocMind-AI)
![GitHub Last Commit](https://img.shields.io/github/last-commit/Ashita-no-Kaushar/DocMind-AI)
![GitHub License](https://img.shields.io/github/license/Ashita-no-Kaushar/DocMind-AI)

Offline, open-source retrieval augmented generation (RAG).

DocMind AI is a local-first RAG application. It ingests local files, GitHub repositories, and websites, then answers questions about them with local Ollama models. Chat, embeddings, and indexed source content stay entirely on your machine or network.

## Features

### Ingestion Sources

- **Local files** — upload `csv`, `docx`, `epub`, `ipynb`, `json`, `md`, `pdf`, `ppt`, `pptx`, and `txt` files. Up to 10 files, 25 MB each, 100 MB total per upload. Re-uploading the same files reuses the existing index; changed files are reprocessed.
- **GitHub repositories** — clone any public repo from `owner/repo` or a full `https://github.com/owner/repo` URL with a shallow `--depth 1` clone. Only `github.com` repository-root URLs are accepted.
- **Websites** — fetch up to 5 public HTTPS URLs per ingestion and convert them to grounded chat content.

### RAG Pipeline

- Document loading, configurable chunking (chunk size and overlap), and Ollama embeddings with exact progress display.
- In-memory LlamaIndex vector store and a streaming query engine.
- Automatic cleanup of transient ingestion files after indexing.

### Chat

- Streaming, grounded RAG responses through LlamaIndex with conversational memory.
- 6 answer-style presets.
- **Live retrieval controls** — Top K and similarity-threshold sliders that apply to the next query immediately.
- Document-introduction retrieval for summary / "about this document" questions.
- Responses built from a compact context budget for speed and lower resource use.

### Settings & Data

- Browser-local settings persistence (Ollama endpoint, chat/embedding models, retrieval and chunking settings) via `localStorage` — no server-side state.
- Chat history export to Word (`.docx`).

### Security & Guardrails

- Upload size and type validation.
- Website URL validation: HTTPS-only, embedded-credential rejection, private/loopback/metadata IP blocking, redirect limits, and response-size caps.
- GitHub repository validation with subprocess isolation and timeouts.
- Stale checkout / file-lock handling for reliable re-cloning on Windows.
- No secrets or sensitive content persisted beyond browser-local settings.
- Hardened Docker deployment (read-only filesystem, dropped capabilities, resource limits).

### Deployment

- Docker Compose for NVIDIA (CUDA) and AMD (ROCm) hosts.
- Tested on Windows and Linux.
- Fully offline once models are installed.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.12–3.13 |
| Web UI | Streamlit |
| RAG framework | LlamaIndex (core, readers, LLM adapters) |
| Local models | Ollama — chat (e.g. `qwen2.5`) + embeddings (e.g. `nomic-embed-text`) |
| Document parsing | pypdf, python-docx, docx2txt, python-pptx, ebooklib, striprtf, nbconvert, Pillow |
| Web fetching | requests + html2text (with network guardrails) |
| Git integration | Git CLI (shallow clones) |
| Chat export | python-docx |
| State | Streamlit session state + browser `localStorage` |
| Packaging | Pipenv (`Pipfile`) + `requirements.txt` |
| Deployment | Docker (NVIDIA / ROCm), `run.ps1` / shell launcher |
| Dev tools | black, ruff, pytest / unittest |

## System Architecture

```
+------------------------ Browser / Streamlit UI ------------------------+
|  main.py                                                               |
|   ├─ page_config.py      page setup, menu links, branding             |
|   ├─ page_state.py       initial session state                        |
|   ├─ header.py · sidebar.py · chatbox.py                              |
|   └─ components/tabs/    files · github_repo · website · settings · about |
+-----------------------------------+------------------------------------+
                                    |
                                    v
+----------------------------- utils ------------------------------------+
|  helpers.py            guardrails, uploads, GitHub clone, website fetch |
|  rag_pipeline.py       ingestion pipeline (load → chunk → embed → index)|
|  llama_index.py        SimpleDirectoryReader, embeddings, index, engine |
|  ollama.py             chat + embedding model discovery                |
|  browser_settings.py   browser localStorage persistence               |
|  logs.py               docmind.log                                    |
+-----------------------------------+------------------------------------+
                                    |
                                    v
                     +---------------------------------+
                     |          Ollama (11434)         |
                     |   chat models · embeddings      |
                     +---------------------------------+
```

## How It Works

1. **Setup** — Install Ollama and pull at least one chat model and one embedding model (e.g. `qwen2.5:0.5b` and `nomic-embed-text:latest`).
2. **Configure** — In Settings, set the Ollama endpoint and select the chat and embedding models. These choices are saved in the browser and restored on your next visit.
3. **Ingest** — Choose a source: upload local files, point at a GitHub repo, or enter website URLs. Every input is checked against security guardrails before processing.
4. **Index** — Documents are loaded, split into chunks using the configured chunk size/overlap, embedded with Ollama, and stored in an in-memory LlamaIndex vector store.
5. **Chat** — Each question retrieves the Top K most relevant chunks (above the similarity threshold), packs them into a compact context, and streams a grounded answer from the local model. Adjust Top K / similarity live and re-ask without re-indexing.
6. **Export** — Download the conversation as a Word document from Settings.

Transient ingestion files are deleted automatically once indexing completes.

## Project Information

- [Planned Features](docs/todo.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Known Bugs & Issues](docs/todo.md#known-issues--bugs)
- [Resources](docs/resources.md)
- [Contributing](docs/contributing.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Getting Started

- [Setup & Deploy the App](docs/setup.md)
- [Using DocMind](docs/usage.md)
- [RAG Pipeline](docs/pipeline.md)

## License

DocMind AI is free software distributed under the **GNU General Public License v3.0 (GPL-3.0)** — see [LICENSE](LICENSE).

Copyright (C) 2026 Ashita-no-Kaushar.

This project is a derivative work of the GPL-3.0-licensed [local-rag](https://github.com/jonfairbanks/local-rag) project. Under the GPL, any distributed modified version must be released under the same license with source code available, preserving the same freedoms for everyone.
