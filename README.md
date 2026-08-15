# DocMind AI

![docmind-demo](demo.gif)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8588/badge)](https://www.bestpractices.dev/projects/8588)
![GitHub Commit Activity](https://img.shields.io/github/commit-activity/t/Ashita-no-Kaushar/DocMind-AI)
![GitHub Last Commit](https://img.shields.io/github/last-commit/Ashita-no-Kaushar/DocMind-AI)
![GitHub License](https://img.shields.io/github/license/Ashita-no-Kaushar/DocMind-AI)

Offline, open-source retrieval augmented generation (RAG).

DocMind ingests local files, GitHub repositories, and websites for retrieval augmented generation with local Ollama models. Chat, embeddings, and indexed source content stay on your machine or network — nothing leaves your infrastructure.

## Features

- Local Ollama chat models (e.g. `qwen2.5`, `llama3`)
- Ollama embedding models (`nomic-embed-text` recommended)
- Multiple ingestion sources:
  - Local files (PDF, DOCX, TXT, Markdown, and more)
  - GitHub repositories (`owner/repo` or full URL)
  - Websites
- Streaming RAG responses through LlamaIndex
- Live retrieval controls: Top K and similarity threshold sliders applied per query
- 6 answer-style presets
- Chat history export to Word (.docx)
- Browser-local settings persistence (no server-side state)
- Upload, URL, repository, and ingestion guardrails
- Docker deployment with GPU support
- Tested on Windows and Linux

## Getting Started

- [Setup & Deploy the App](docs/setup.md)
- [Using DocMind](docs/usage.md)
- [RAG Pipeline](docs/pipeline.md)

## Project Information

- [Planned Features](docs/todo.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Known Bugs & Issues](docs/todo.md#known-issues--bugs)
- [Resources](docs/resources.md)
- [Contributing](docs/contributing.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## License

DocMind is licensed under the [GPL-3.0 license](LICENSE). Copyright (C) 2026 Ashita-no-Kaushar.
