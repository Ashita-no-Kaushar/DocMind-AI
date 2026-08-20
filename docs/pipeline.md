# RAG Pipeline

DocMind builds an in-memory LlamaIndex query engine from one source at a time: local file uploads, a GitHub repository clone, or fetched website documents.

## Ingestion Flow

1. Validate the current Ollama chat model and embedding settings.
2. Initialize the selected Ollama chat model.
3. Configure the embedding backend:
   - Ollama embeddings through the configured Ollama endpoint
4. Load documents:
   - Local files and GitHub repositories are loaded with LlamaIndex `SimpleDirectoryReader`.
   - Websites are fetched with request size, redirect, content type, and network guardrails, then converted to text.
5. Validate ingestion limits:
   - At most 1,000 loaded documents
   - At most 10 MB of loaded source text
6. Split documents into chunks using the configured chunk size and chunk overlap.
7. Generate embeddings and display exact progress while indexing.
8. Create a streaming LlamaIndex query engine with the configured `top_k` and response mode.
9. Remove transient on-disk ingestion files from `data/`.

## Source-Specific Stages

The UI stores completed ingestion stages in Streamlit session state so reruns can show the current status without reprocessing unchanged inputs.

- Local files: files uploaded, documents loaded, embeddings generated, index ready
- GitHub repositories: repository validated, repository cloned, repository files loaded, embeddings generated, index ready
- Websites: websites fetched, website content loaded, embeddings generated, index ready

## Key Parameters

Users can adjust these advanced settings:

1. **`top_k`**: Number of similar chunks retrieved for each query. Higher values provide more context but may add noise. Applies to the next query immediately.
2. **`similarity_cutoff`**: Minimum vector similarity score for a chunk to be used. Higher = only strong matches (less hallucination), lower = more recall. `0` disables the filter. Applies to the next query immediately.
3. **`chunk_size`**: Maximum size of each text chunk before embedding. Smaller chunks can improve precision but increase embedding work. Applies to the next ingestion.
4. **`chunk_overlap`**: Overlap between consecutive chunks. This must be greater than or equal to `0` and less than `chunk_size`. Applies to the next ingestion.

## Runtime State

A successful RAG conversation requires:

- `llm`: the initialized Ollama LLM
- `documents`: loaded source documents
- `query_engine`: the LlamaIndex query engine

If any of these are missing, ingestion did not complete and chat is blocked until data is imported successfully.
