# To Do

## Verified Implemented

- [x] Streamlit chat interface
- [x] Direct model chat
- [x] Local LlamaIndex RAG pipeline
- [x] Local file uploads
- [x] Public GitHub repository ingestion
- [x] Public HTTPS website ingestion
- [x] Ollama chat and embedding model discovery
- [x] OpenAI-compatible routing for OpenAI, LM Studio, and TabbyAPI provider values
- [x] R2R client for local-file upload and chat
- [x] Hybrid vector and BM25 retrieval
- [x] Evidence filtering and no-match response
- [x] Index persistence with source/model/endpoint-aware cache identity
- [x] Browser settings persistence for a selected non-secret subset
- [x] DOCX transcript export
- [x] Unit tests and evaluation harness
- [x] Windows and Docker launch configuration

## High-Priority Fixes

- [ ] Add a tested `OpenAILike` adapter for arbitrary local model identifiers
- [ ] Add live integration tests for LM Studio and TabbyAPI
- [ ] Add a complete R2R lifecycle: status polling, remote deletion, rollback, and GitHub/website routing
- [ ] Add a durable source registry so old completion labels cannot describe a replaced index
- [ ] Isolate temporary files and cache keys per Streamlit session
- [ ] Pin DNS during website requests or use a transport that verifies the connected peer
- [ ] Bound GitHub clone size and file count before parsing
- [ ] Add real Git clone and website-fetch integration tests
- [ ] Add browser-level end-to-end tests
- [ ] Commit a dependency lockfile

## Reliability

- [ ] Add output verification for generated factual claims and citation entailment
- [ ] Evaluate multiple chat models and larger document corpora
- [ ] Repeat live-generation trials and publish confidence intervals
- [ ] Make tokenizer-exact context and history budgeting
- [ ] Clean temporary local data on failed ingestion
- [ ] Make reset behavior fully explicit for retained settings and remote documents

## Deployment

- [ ] Build and health-check the Docker image on a supported host
- [ ] Test read-only filesystem and named-volume permissions
- [ ] Add authentication before exposing the service beyond localhost
- [ ] Test concurrent sessions
- [ ] Decide whether the ROCm Compose file is needed or should point to an Ollama ROCm deployment guide

## Documentation

- [x] Replace unsupported format, limit, privacy, and performance claims
- [x] Align setup, usage, pipeline, troubleshooting, and security docs with source behavior
- [x] Generate evaluation reports with explicit pass/fail/skip accounting
