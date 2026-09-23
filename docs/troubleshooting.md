# Troubleshooting

## Application Does Not Start

Confirm Python dependencies are installed in the active environment:

```bash
pipenv run python -m pip check
pipenv run python -m py_compile main.py
```

Then start the app:

```bash
pipenv run streamlit run main.py --server.headless=true --server.port=8520 --server.address=127.0.0.1
```

Check:

```text
http://127.0.0.1:8520/_stcore/health
```

## Ingestion Is Disabled

Local GitHub and website ingestion require valid local-provider chat and embedding settings. Local file uploads can use R2R without local models when R2R is enabled.

For Ollama:

1. Confirm Ollama is reachable.
2. Confirm the endpoint, usually `http://localhost:11434`.
3. Click **Refresh Models**.
4. Select an installed chat-capable model.
5. Select an installed embedding-capable model.

For OpenAI-compatible providers, configure the base URL, chat model, and embedding model. LM Studio and TabbyAPI use this routing path, but the model identifier must be supported by the server and installed LlamaIndex adapter.

## Model Answers Are Vague or Miss the Context

Small local models can produce variable answers even when retrieval succeeds. The current evaluation observed the expected refund fact in 2/3 repeated live generations.

Try:

- A larger chat model
- A lower temperature
- Balanced or Detailed answer style
- Re-running the same query
- Checking the retrieved source labels
- Verifying the document actually contains the requested fact

A citation marker does not prove that the generated claim is supported by the cited chunk.

## No Usable Content

The index builder rejects content that produces no usable chunks. Try a text-based TXT, Markdown, CSV, JSON, or DOCX export. Scanned PDFs require OCR, which is not included.

The current minimum retained chunk length is 15 non-whitespace characters.

## Local Upload Errors

- More than 10 files
- File larger than 25 MiB
- Total batch larger than 100 MiB
- Unsupported filename characters
- Unsupported extension
- A parser or optional dependency cannot extract the file

Accepted extensions are listed in [Usage](usage.md#local-files). Acceptance does not guarantee reliable parsing for every format.

## Website Import Errors

Common causes:

- URL is not HTTPS
- URL contains credentials
- DNS resolves to a blocked address
- Redirect limit was exceeded
- Content-Type is not HTML or plain text
- Response exceeds 5 MiB

DNS validation and the later request resolve independently, so changing DNS between checks can still create an edge case.

## GitHub Import Errors

- Use `owner/repo` or `https://github.com/owner/repo`
- Do not include issue, pull-request, branch, or extra path segments
- The repository must be public and reachable
- `git` must be installed and available on PATH
- Large repositories may take longer or exceed local parser resources

## Settings Do Not Persist

Only the settings listed in `PERSISTED_SETTING_TYPES` are saved to browser `localStorage`. API keys, chat history, indexes, R2R IDs, and answer-style state are not persisted.

Use the DOCX export for the current transcript. Re-enter API keys after a process/session restart when required.

## Reset Project Does Not Remove Everything

Reset Project deletes local `data/` and `.index_cache/` directories and clears most local index state. It intentionally retains some settings and session-only API keys. It also does not delete documents already uploaded to an external R2R server.

## Logs

By default, DocMind writes `docmind.log` in the current working directory. Set `DOCMIND_LOG_FILE` to another writable path when needed.

The logger always writes to stdout. If the configured file path is not writable, file logging is skipped rather than preventing application import.

Do not share logs without reviewing them for document content, prompts, local paths, internal addresses, and credentials.

## Application State

There is no Application State viewer in Settings. Relevant state is visible indirectly through the mode badge, ingestion stages, source labels, and error messages. Inspect `components/page_state.py` when debugging source code.
