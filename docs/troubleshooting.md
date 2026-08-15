# Troubleshooting

In the event that an error occurs when using DocMind, checking out the current application state and logfile can provide insights into what is happening behind the scenes.

Note: To better understand what is happening under the hood and aid in troubleshooting, check out the [Pipeline documentation](pipeline.md) as well.

## Ingestion Is Disabled

Data import controls are disabled until Settings contains a valid Ollama chat model and a valid Ollama embedding model.

To fix this:

- Confirm Ollama is running.
- Confirm the endpoint in Settings, usually `http://localhost:11434`.
- Pull at least one chat-capable model, such as `qwen2.5:0.5b`.
- Pull at least one embedding-capable model, such as `nomic-embed-text:latest`.
- Use the Refresh Models buttons after changing the endpoint or pulling new models.

## Settings Restore Problems

Settings are restored from browser `localStorage`. If a stale browser setting points to the wrong Ollama endpoint or a model you no longer have installed, update it in Settings and refresh the model lists. Empty Ollama endpoint values are ignored and the default endpoint is restored.

## Application State

Each stage of the RAG pipeline stores its data in the application state. 

In order for a successful RAG conversation to take place the following state values must NOT be null:
- `documents` - if null, there was an error processing your documents
- `llm` - if null, there was an error creating the Ollama LLM instance
- `query_engine` - if null, errors occurred when creating embeddings for your your documents

To view the current application state:
- Navigate to the Settings panel
- Toggle Advanced Settings
- State is now visible at the bottom of Settings
- Verify that the above state values are valid

## Import Errors

Common local file import errors:

- Unsupported file extension. Supported extensions are listed in [Usage](usage.md#local-files).
- Too many files, too large a single file, or too large a total upload.
- Filename contains unsupported characters or path separators.

Common GitHub import errors:

- Repository input is not `owner/repo` or `https://github.com/owner/repo`.
- URL is not on `github.com`.
- URL points to an issue, pull request, branch, or another extra path instead of the repository root.
- The repository cannot be validated or cloned.

Common website import errors:

- URL is not HTTPS.
- URL resolves to a blocked local, private, metadata, or otherwise unsafe network address.
- The response is not HTML or plain text.
- The response is larger than the website ingestion limit.
- The URL redirects too many times.

## Log File

By default, DocMind will create a `docmind.log` file in the root application folder.

Each step of the RAG process is logged into this file whether the required step was successful or encountered an error. 

Reviewing this log can give you insights into what took place when processing your documents.
