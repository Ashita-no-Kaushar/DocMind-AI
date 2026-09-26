# Troubleshooting

## Application does not start

Confirm the locked environment is active and the lockfile is valid:

```bash
pipenv verify
pipenv install --deploy --dev --extra-pip-args="--require-hashes"
pipenv run python -m pip check
pipenv run python -m compileall -q main.py components utils eval_harness.py tests
```

Start a loopback smoke-test instance:

```bash
pipenv run streamlit run main.py --server.headless=true --server.port=8520 --server.address=127.0.0.1
```

Check:

```text
http://127.0.0.1:8520/_stcore/health
```

Stop the smoke-test process afterward. The current local gate left no Streamlit child process running.

The supported project target is Python 3.13.15. Python 3.12.10 passed the current local gate, but use the supported target for normal setup. If the lock install fails, restore the committed `Pipfile` and `Pipfile.lock` rather than bypassing hash enforcement.

## Models or providers are unavailable

### Ollama

1. Confirm the Ollama server is reachable.
2. Confirm the endpoint, normally `http://localhost:11434`.
3. Select **Refresh Models**.
4. Select an installed chat-capable model and an embedding-capable model.

Ollama was unavailable in the current environment, so the current result does not include a live Ollama generation.

### OpenAI-compatible providers

Confirm the provider base URL, model identifier, and optional key. LM Studio and TabbyAPI use the OpenAI-compatible path. A model name must be accepted by both the server and the installed LlamaIndex adapter. Non-loopback HTTP chat/embedding endpoints are rejected; use HTTPS for a remote chat/embedding provider. R2R accepts HTTP or HTTPS, so protect a remote R2R deployment with HTTPS and an appropriate network boundary.

The official OpenAI service was unavailable in the current environment. It requires an explicit key and the official OpenAI host. A key bound to one provider/endpoint is cleared or rejected when the endpoint changes.

### R2R

R2R was unavailable in the current environment. Confirm the URL, optional key, and **Test Connection**. The implemented contract is R2R 3.6.5 / API v3. Local fake-server tests do not establish live compatibility.

## Ingestion controls are disabled

Local GitHub and website ingestion require valid local chat and embedding settings. Local file uploads can use R2R without local models when R2R is enabled.

For local ingestion:

- Confirm the provider and model are selected.
- Confirm the embedding provider and model are selected.
- Confirm the endpoint is valid.
- Resolve any provider credential binding error.

A source can remain visible while its index is stale after a provider, embedding, chunk, or endpoint change. Re-select or re-process the source after fixing the setting.

## No usable content

The index builder rejects content that produces no usable chunks. Try a text-based TXT, Markdown, CSV, JSON, DOCX, or converted document first. Image-only PDF, DOCX, and PPTX content requires external OCR or conversion. No OCR engine is bundled.

Inspect the **Extraction report**. It distinguishes loaded, skipped, and unsupported files and includes safe warning/error text without copying document content. The minimum retained chunk length is 15 non-whitespace characters.

## Local upload errors

Common causes are:

- More than 10 files in a batch.
- A file larger than 25 MiB.
- A batch larger than 100 MiB.
- Unsupported filename characters, path separators, control characters, trailing spaces/periods, or Windows reserved names.
- An unsupported extension.
- A parser dependency that is unavailable.
- Malformed, encrypted, empty, or archive-unsafe content.
- A source, JSON, MIME, table, PDF, or archive limit being exceeded.

The accepted 25-extension matrix is listed in [Usage](usage.md#local-files). Extension acceptance does not guarantee a reliable parser. Valid legacy XLS and real MSG fixtures were not generated in the current format tests; encrypted Office/ZIP content and OCR ground truth remain untested.

## Parser and archive failures

- `.doc` and `.ppt` are intentionally unsupported without conversion. Convert them to `.docx` and `.pptx`.
- Image-only PDF, DOCX, and PPTX files are reported as requiring OCR or conversion.
- ZIP-based formats are preflighted for member count, compressed size, total/per-member uncompressed size, compression ratio, traversal, duplicate paths, symlink/special members, and encrypted members.
- A malformed file in a mixed batch should not force valid files to be discarded. Review each report entry and retry the failed file separately.

Supported parser and archive limits are configurable in code through the `DOCMIND_MAX_*` and `DOCMIND_ZIP_*` environment variables. Use the exact names in `utils/format_ingestion.py`; invalid or non-positive values fall back to safe defaults.

## Website import errors

Common causes:

- The URL is not HTTPS or contains embedded credentials.
- The hostname is local, a metadata name, or resolves to a blocked address.
- DNS returns no usable public address or fails.
- A redirect is missing, exceeds the three-redirect limit, or points to a blocked destination.
- The content type is not HTML or plain text.
- The response exceeds 5 MiB.
- The site returns an anti-bot/challenge response.
- The request or total ingestion deadline expires.
- The page has no readable text.

The current public fetch of `https://docs.python.org/3/` passed. A site that requires JavaScript, authentication, or anti-bot approval is not guaranteed to work. DNS and network changes can still produce failures; the application checks all addresses and pins the validated address for its HTTPS request, but it is not a network sandbox.

## GitHub import errors

- Use `owner/repo` or `https://github.com/owner/repo`.
- Do not include credentials, ports, query strings, fragments, issue/pull-request paths, or extra path segments.
- The repository must be public and reachable; private authentication is not implemented.
- `git` must be installed and available to the application process.
- The clone may exceed the 120-second timeout or 64 KiB output bound.
- The repository may exceed the 100 MiB download, 10,000-file, 10 MiB per-file, or 250 MiB unpacked checkout bound.
- The checkout may contain a symlink/reparse point, special file, hidden/excluded path, or unsafe source-root path.
- The repository may be valid but contain only unsupported or image-only files.

Metadata preflight can reject a known private or oversized repository. If metadata is unavailable, the code can attempt a bounded clone. A current bounded clone and validation of `Ashita-no-Kaushar/DocMind-AI` passed in a temporary directory.

## RAG answers are vague or miss the context

Small local models can vary even when retrieval succeeds. Check:

- The source actually contains the requested fact.
- The active source is current and not stale.
- The selected evidence chunks and source labels.
- Top K, candidate depths, and similarity cutoff.
- Whether a follow-up query needs more explicit wording.
- A larger or more capable chat model.
- A lower temperature, understanding that this does not guarantee determinism.

A source label or citation marker does not prove factual support. The evidence rule is not an entailment checker.

## No-match behavior

If local retrieval returns no credible nodes, the response is:

```text
I could not find this information in the documents.
```

The one-time **Ask without documents** action sends the same question through direct chat. It is a deliberate fallback, not a claim that the documents support the answer.

## Citations

DocMind asks the model to cite evidence and sanitizes references outside the selected evidence set. It does not force a citation and does not automatically verify that a cited chunk entails a claim. A citation can therefore be absent or still be insufficient; inspect the source excerpt and answer together.

## Settings do not persist

Only fields defined by `PERSISTED_SETTING_TYPES` are saved to browser `localStorage`. API keys, chat history, source indexes, evidence, answer-style state, and R2R document IDs are not persisted there.

Re-enter API keys after a process or session restart when required. Use the DOCX export for the current transcript, and review it for sensitive content before sharing.

## Reset Project behavior

**Clear Chat** clears only the conversation.

**Reset Project** clears the current local source/index/session state and owned work directories. It retains browser settings, current-session API keys, shared caches, logs, other sessions, and the R2R ownership registry.

The recommended reset attempts deletion of R2R documents owned by the current workspace and current endpoint/credential identity. The registry is retained. If deletion is partial, local state is still cleared but the UI reports incomplete cleanup; restore the same R2R identity and retry. Documents associated with another endpoint/credential are not claimed or deleted.

The local-only reset option does not contact R2R and leaves remote documents tracked by the registry.

## Logs

By default, DocMind writes `docmind.log` in the current working directory. Supported settings are:

- `DOCMIND_LOG_FILE`
- `DOCMIND_LOG_MAX_BYTES`
- `DOCMIND_LOG_BACKUP_COUNT`

Logs rotate, redact credential-shaped values, and fall back to stdout if the configured file is not writable. Redaction is defense in depth. Review logs for document content, prompts, local paths, internal addresses, and any sensitive provider response before sharing.

## Runtime binding refused

The default address is `127.0.0.1`. A non-loopback or wildcard address requires:

```text
DOCMIND_ALLOW_REMOTE_BIND=true
DOCMIND_BIND_ADDRESS=0.0.0.0
```

Use those settings only behind a user-configured authenticated reverse proxy. The application does not provide built-in authentication. Compose publishes `127.0.0.1:8501:8501` by default; do not change it to a wildcard without the same proxy and network controls.

## Docker is unavailable

The current environment had no Docker CLI. Docker/Compose was statically validated and CI performs a build-only check, but no local image build, runtime, health check, or read-only-volume exercise was possible. Treat runtime behavior as unverified until a Docker host completes those checks.

## External validation status

The current live Ollama, LM Studio, TabbyAPI, official OpenAI, and R2R services were unavailable. The current real Ollama/LLM evaluation was not rerun. The 43/43 result dated 2026-09-23 is historical, not current. The current mock evaluation passed 42/42 scored checks with one real-LLM skip, and the local real-Chrome E2E run passed.

See [Setup](setup.md) for locked installation, [Usage](usage.md) for source behavior, and [Pipeline](pipeline.md) for implementation details.
