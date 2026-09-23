# Changelog

## Versioning

Release tags use semantic versioning, for example `v1.5.0`. This file records only changes visible in the current source tree.

## Unreleased

- Replace unsupported README and documentation claims with source-verified behavior and explicit limitations.
- State that DocMind accepts 25 filename extensions and that extension acceptance does not guarantee a dedicated parser.
- Correct local ingestion limits to 300 loaded documents and 4 MiB of extracted characters.
- Correct minimum retained chunk length to 15 characters.
- Correct RAG history estimates to 500 tokens, or 300 in Eco Mode; direct chat uses 1,200.
- Qualify privacy claims: GitHub and website ingestion use outbound network access, and OpenAI-compatible/R2R modes transmit data to configured servers.
- Correct the repository license to GPL-3.0.
- Route OpenAI, LM Studio, and TabbyAPI provider values through the OpenAI-compatible adapter path.
- Resolve temperature before cached LLM construction so a changed temperature produces a new cache key.
- Make R2R bypass local model requirements only for local-file uploads; GitHub and website ingestion still require local models.
- Clear stale R2R document IDs after a successful local index build.
- Include provider, endpoint, model, source metadata, chunk settings, and cache version in index-cache identity.
- Reject hidden paths, excluded files, paths outside the source root, and file symlinks during document loading.
- Include active indexing settings in the local upload processing signature.
- Reject dot-segment GitHub owner/repository names.
- Avoid modifying keyed Streamlit widget state after widget instantiation in blank GitHub and website actions.
- Make evaluation mock embeddings stable across processes.
- Distinguish passed, failed, and skipped evaluation checks; exclude skips from the score denominator and return nonzero status for scored failures.
- Require real embeddings in non-mock evaluation instead of silently falling back to hash embeddings.
- Run the real LLM evaluation check three times and require the expected fact in a majority of trials.
- Add exact tests for all 25 accepted extensions and the actual ingestion-limit boundaries.
- Add tests for cache identity, processing signatures, R2R prerequisite scope, provider routing, dot segments, and symlink rejection.
- Make file logging fall back to console when the configured path is not writable.
- Add direct runtime dependencies for HTTP, NLTK, and OpenAI-compatible integrations to `Pipfile`.
- Build the local source in Compose, install Git in the runtime image, use a Python health check, mount writable index-cache storage, and remove an unnecessary GPU reservation.
- Add Python virtual environments, caches, tests, and evaluation outputs to `.dockerignore`.

## v1.5.0 - 2026-08-15

- Improve GitHub stale-checkout and Windows file-lock handling.
- Load only the current upload batch during local ingestion.
- Remove the local Hugging Face embedding backend.
- Apply top-k and similarity settings to the custom retriever.
- Add document-introduction retrieval for summary-style questions.
- Add DOCX chat export.
- Add logging and general reliability/security improvements.

The v1.5 Windows launcher clears stale `__pycache__` directories. Normal Streamlit and Docker startup do not perform that purge.
