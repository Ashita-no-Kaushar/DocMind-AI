import os
import shutil
import time
from pathlib import Path

import streamlit as st

import utils.logs as logs
import utils.r2r as r2r
from utils.browser_settings import (
    ensure_ollama_endpoint,
    restore_settings_from_browser_storage,
    should_refresh_models_for_endpoint,
)
from utils.llama_index import INDEX_CACHE_DIR
from utils.ollama import default_embedding_model, get_embedding_models, get_models
from utils.provider_config import (
    OLLAMA,
    get_embedding_profile,
    initialize_provider_state,
)
from utils.source_state import (
    effective_indexing_settings,
    ensure_active_source,
    initial_source_state,
    mark_active_index_stale_if_needed,
    normalize_chunk_settings,
    reset_active_source,
)

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "Hi! I'm **DocMind AI** — your document assistant. 👋\n\n"
        "**How to use — 3 simple steps:**\n"
        "1. Add your files, GitHub repo, or a website from the left sidebar\n"
        "2. Wait a moment while it reads them\n"
        "3. Ask anything below — answers come from your documents\n\n"
        "_No documents? Just chat — ask me anything._"
    ),
}


def _remove_dir_retry(path, attempts=5, delay=0.5):
    """Delete one owned directory, retrying transient Windows file locks."""
    if not os.path.exists(path):
        return True
    if not os.path.isdir(path):
        return False
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            time.sleep(delay)
    return False


def _owned_work_paths(state, explicit_paths=None):
    candidates = []
    if explicit_paths:
        candidates.extend(Path(path) for path in explicit_paths)
    candidates.extend(
        Path(path)
        for path in state.get("_session_work_dirs", [])
        if path
    )
    for key in ("ingestion_work_dir", "source_work_dir"):
        if state.get(key):
            candidates.append(Path(state[key]))
    active_source = state.get("active_source") or {}
    for key in ("work_dir", "operation_dir"):
        if isinstance(active_source, dict) and active_source.get(key):
            candidates.append(Path(active_source[key]))

    work_root = (Path(os.getcwd()) / "data" / "work").resolve()
    data_root = (Path(os.getcwd()) / "data").resolve()
    cache_root = Path(INDEX_CACHE_DIR).resolve()
    result = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == data_root or (cache_root is not None and resolved == cache_root):
            continue
        try:
            resolved.relative_to(data_root)
            inside_data = True
        except ValueError:
            inside_data = False
        if inside_data:
            try:
                resolved.relative_to(work_root)
            except ValueError:
                continue
            if resolved == work_root:
                continue
        elif not explicit_paths:
            continue
        result.append(resolved)
    return list(dict.fromkeys(result))


def perform_project_reset(state, owned_dirs=None, delete_remote=True):
    """Reset session sources and, by default, owned remote R2R documents."""
    remote_result = None
    if delete_remote:
        try:
            remote_result = r2r.delete_owned_remote_documents(state)
        except (r2r.R2RLifecycleError, OSError, ValueError) as err:
            remote_result = r2r.R2RRemoteResetResult(
                r2r.R2RDeleteResult(failed=((None, r2r.safe_error_category(err)),)),
                attempted=True,
            )
            logs.log.error(
                "R2R reset deletion failed: category=%s",
                r2r.safe_error_category(err),
            )
    else:
        remote_result = r2r.R2RRemoteResetResult(
            r2r.R2RDeleteResult(),
            attempted=False,
        )

    deleted = []
    failed = []
    for path in _owned_work_paths(state, owned_dirs):
        if not os.path.exists(path):
            continue
        try:
            removed = _remove_dir_retry(path)
        except OSError:
            removed = False
        if removed:
            deleted.append(str(path))
        else:
            failed.append(str(path))

    reset_active_source(state)
    state["index_generation"] = 0
    state["query_engine"] = None
    state["retriever"] = None
    state["llm"] = None
    state["documents"] = None
    state["file_list"] = []
    state["file_selection_signature"] = None
    state["failed_upload_selection_signature"] = None
    state["processed_file_signature"] = None
    state["processing_file_signature"] = None
    state["processed_github_repo"] = None
    state["processed_website_urls"] = None
    state["github_repo"] = ""
    state["websites"] = []
    state["new_website"] = ""
    state["website_input_error"] = None
    state["github_ingestion_stages"] = []
    state["github_ingestion_source_id"] = None
    state["github_ingestion_generation"] = None
    state["website_ingestion_stages"] = []
    state["website_ingestion_source_id"] = None
    state["website_ingestion_generation"] = None
    state["file_ingestion_stages"] = []
    state["file_extraction_report"] = []
    state["extraction_report"] = []
    state["r2r_ingestion_stages"] = []
    state["r2r_document_ids"] = []
    state["r2r_document_ids_signature"] = None
    state["r2r_current_source_signature"] = None
    state["r2r_connection_ok"] = None
    state["last_r2r_metadata"] = None
    state["last_doc_sources"] = []
    state["last_rag_evidence"] = []
    state["last_rag_no_result"] = False
    state["last_rag_question"] = None
    state["ask_without_docs"] = False
    state["last_ingestion_error"] = None
    state["indexing_settings_error"] = None
    state["messages"] = [dict(WELCOME_MESSAGE)]
    state["confirm_project_reset"] = False
    state["_session_work_dirs"] = failed
    result = {
        "deleted": deleted,
        "failed": failed,
        "remote": remote_result.as_dict(),
        "remote_delete_success": remote_result.success if delete_remote else None,
    }
    state["reset_result"] = result
    if remote_result.attempted and not remote_result.success:
        logs.log.warning("Project reset could not delete every owned R2R document")
    if failed:
        logs.log.warning("Project reset could not delete: %s", ", ".join(failed))
    elif not delete_remote or remote_result.success:
        logs.log.info("Project reset cleared the current session state")
    return result


def default_chat_model(models):
    """Return the preferred default chat model from discovered Ollama models."""
    preferred_models = (
        "gemma4:latest",
        "llama3:8b",
        "llama2:7b",
    )

    for model in preferred_models:
        if model in models:
            return model

    if models:
        return models[0]

    return None


def ensure_valid_model_selections(state):
    """Keep selected model values consistent with discovered model lists."""
    chat_models = state.get("ollama_models", [])
    if chat_models:
        if state.get("selected_model") not in chat_models:
            state["selected_model"] = default_chat_model(chat_models)
    elif "selected_model" in state:
        state["selected_model"] = None

    embedding_models = state.get("ollama_embedding_models", [])
    if embedding_models:
        if state.get("ollama_embedding_model") not in embedding_models:
            state["ollama_embedding_model"] = default_embedding_model(embedding_models)
    elif "ollama_embedding_model" in state:
        state["ollama_embedding_model"] = None


def set_initial_state():
    restore_settings_from_browser_storage()

    # A pending project reset (requested from the sidebar button) must be
    # executed here: widgets are not instantiated yet, so resetting
    # widget-backed keys is still legal.
    if st.session_state.get("reset_requested"):
        delete_remote = bool(st.session_state.pop("reset_delete_remote", True))
        perform_project_reset(
            st.session_state,
            delete_remote=delete_remote,
        )
        st.session_state["reset_requested"] = False

    ###########
    # General #
    ###########

    if "sidebar_state" not in st.session_state:
        st.session_state["sidebar_state"] = "expanded"

    try:
        ensure_ollama_endpoint(st.session_state)
    except ValueError as err:
        st.session_state["ollama_endpoint"] = "http://localhost:11434"
        st.session_state["provider_endpoint_error"] = str(err)
    initialize_provider_state(st.session_state)

    if "ollama_embedding_model" not in st.session_state:
        st.session_state["ollama_embedding_model"] = "nomic-embed-text:latest"

    if should_refresh_models_for_endpoint(st.session_state, "ollama_models"):
        try:
            models = get_models()
            st.session_state["ollama_models"] = models
        except Exception:
            st.session_state["ollama_models"] = []
            pass
        st.session_state["ollama_models_endpoint"] = st.session_state["ollama_endpoint"]

    embedding_profile = get_embedding_profile(st.session_state)
    embedding_endpoint = (
        embedding_profile["base_url"]
        if embedding_profile["provider_kind"] == OLLAMA
        else None
    )
    if should_refresh_models_for_endpoint(
        st.session_state,
        "ollama_embedding_models",
        endpoint=embedding_endpoint,
    ):
        try:
            models = get_embedding_models(embedding_endpoint)
            st.session_state["ollama_embedding_models"] = models
        except Exception:
            st.session_state["ollama_embedding_models"] = []
            pass
        st.session_state["ollama_embedding_models_endpoint"] = (
            embedding_endpoint or st.session_state["ollama_endpoint"]
        )

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = default_chat_model(
            st.session_state.get("ollama_models", [])
        )

    ensure_valid_model_selections(st.session_state)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [dict(WELCOME_MESSAGE)]

    if "last_doc_sources" not in st.session_state:
        st.session_state["last_doc_sources"] = []

    if "last_rag_evidence" not in st.session_state:
        st.session_state["last_rag_evidence"] = []

    if "last_rag_no_result" not in st.session_state:
        st.session_state["last_rag_no_result"] = False

    if "last_rag_question" not in st.session_state:
        st.session_state["last_rag_question"] = None

    ################################
    #  Files, Documents & Websites #
    ################################

    if "file_list" not in st.session_state:
        st.session_state["file_list"] = []

    if "file_selection_signature" not in st.session_state:
        st.session_state["file_selection_signature"] = None

    if "failed_upload_selection_signature" not in st.session_state:
        st.session_state["failed_upload_selection_signature"] = None

    if "_session_work_dirs" not in st.session_state:
        st.session_state["_session_work_dirs"] = []

    if "active_source" not in st.session_state:
        st.session_state["active_source"] = initial_source_state()

    if "index_generation" not in st.session_state:
        st.session_state["index_generation"] = 0

    if "reset_result" not in st.session_state:
        st.session_state["reset_result"] = None

    if "indexing_settings_error" not in st.session_state:
        st.session_state["indexing_settings_error"] = None

    if "processed_file_signature" not in st.session_state:
        st.session_state["processed_file_signature"] = None

    if "processing_file_signature" not in st.session_state:
        st.session_state["processing_file_signature"] = None

    if "file_ingestion_stages" not in st.session_state:
        st.session_state["file_ingestion_stages"] = []

    if "file_extraction_report" not in st.session_state:
        st.session_state["file_extraction_report"] = []

    if "extraction_report" not in st.session_state:
        st.session_state["extraction_report"] = []

    if "r2r_ingestion_stages" not in st.session_state:
        st.session_state["r2r_ingestion_stages"] = []

    if "r2r_document_ids" not in st.session_state:
        st.session_state["r2r_document_ids"] = []

    if "r2r_document_ids_signature" not in st.session_state:
        st.session_state["r2r_document_ids_signature"] = None

    if "r2r_connection_ok" not in st.session_state:
        st.session_state["r2r_connection_ok"] = None

    if "github_ingestion_stages" not in st.session_state:
        st.session_state["github_ingestion_stages"] = []

    if "website_ingestion_stages" not in st.session_state:
        st.session_state["website_ingestion_stages"] = []

    if "github_repo" not in st.session_state:
        st.session_state["github_repo"] = ""
    elif st.session_state["github_repo"] is None:
        st.session_state["github_repo"] = ""

    if "processed_github_repo" not in st.session_state:
        st.session_state["processed_github_repo"] = None

    if "processed_website_urls" not in st.session_state:
        st.session_state["processed_website_urls"] = None

    if "github_ingestion_source_id" not in st.session_state:
        st.session_state["github_ingestion_source_id"] = None

    if "github_ingestion_generation" not in st.session_state:
        st.session_state["github_ingestion_generation"] = None

    if "website_ingestion_source_id" not in st.session_state:
        st.session_state["website_ingestion_source_id"] = None

    if "website_ingestion_generation" not in st.session_state:
        st.session_state["website_ingestion_generation"] = None

    if "websites" not in st.session_state:
        st.session_state["websites"] = []

    if "new_website" not in st.session_state:
        st.session_state["new_website"] = ""

    if "website_input_error" not in st.session_state:
        st.session_state["website_input_error"] = None

    ###############
    # Llama-Index #
    ###############

    if "llm" not in st.session_state:
        st.session_state["llm"] = None

    if "documents" not in st.session_state:
        st.session_state["documents"] = None

    if "query_engine" not in st.session_state:
        st.session_state["query_engine"] = None

    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    #####################
    # Advanced Settings #
    #####################

    if "advanced" not in st.session_state:
        st.session_state["advanced"] = False

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = (
            "You are DocMind AI, a helpful and accurate virtual assistant. "
            "When document context is provided, answer strictly from that context "
            "and do not invent information. If you are unsure, say so directly. "
            "Otherwise answer from your general knowledge. "
            "Be concise, factual, and conversational."
        )

    if "top_k" not in st.session_state:
        st.session_state["top_k"] = 3

    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = 256

    if "chunk_overlap" not in st.session_state:
        st.session_state["chunk_overlap"] = 32

    if "chunk_overlap_pct" not in st.session_state:
        st.session_state["chunk_overlap_pct"] = 12

    if "similarity_cutoff" not in st.session_state:
        st.session_state["similarity_cutoff"] = 0.3

    if "candidate_depth" not in st.session_state:
        st.session_state["candidate_depth"] = 10

    if "vector_candidate_depth" not in st.session_state:
        st.session_state["vector_candidate_depth"] = 10

    if "bm25_candidate_depth" not in st.session_state:
        st.session_state["bm25_candidate_depth"] = 10

    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.4

    if "eco_mode" not in st.session_state:
        st.session_state["eco_mode"] = False

    if "quick_answer_style" not in st.session_state:
        st.session_state["quick_answer_style"] = "Balanced (default)"

    if "answer_style" not in st.session_state:
        st.session_state["answer_style"] = st.session_state.get("quick_answer_style", "Balanced (default)")

    ##################
    # LLM Backends   #
    ##################

    if "llm_backend" not in st.session_state:
        st.session_state["llm_backend"] = "Ollama"

    if "openai_base_url" not in st.session_state:
        st.session_state["openai_base_url"] = "http://localhost:1234/v1"

    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = ""

    if "openai_embedding_model" not in st.session_state:
        st.session_state["openai_embedding_model"] = "text-embedding-3-small"

    if "openai_models" not in st.session_state:
        st.session_state["openai_models"] = []

    ###########
    # R2R     #
    ###########

    if "r2r_enabled" not in st.session_state:
        st.session_state["r2r_enabled"] = False

    if "r2r_base_url" not in st.session_state:
        st.session_state["r2r_base_url"] = "http://localhost:7272"

    if "r2r_api_key" not in st.session_state:
        st.session_state["r2r_api_key"] = ""

    r2r.initialize_r2r_state(st.session_state)

    try:
        normalize_chunk_settings(st.session_state)
        effective_indexing_settings(st.session_state)
        st.session_state["indexing_settings_error"] = None
    except ValueError as err:
        st.session_state["indexing_settings_error"] = str(err)
        normalize_chunk_settings(st.session_state, strict=False)

    source = ensure_active_source(st.session_state)
    st.session_state["index_generation"] = max(
        int(st.session_state.get("index_generation") or 0),
        int(source.get("index_generation") or 0),
    )
    mark_active_index_stale_if_needed(st.session_state)
