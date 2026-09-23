import copy
import hashlib
import os
import time

import streamlit as st

import utils.helpers as func
import utils.llama_index as llama_index
import utils.logs as logs
import utils.ollama as ollama
from utils.provider_config import (
    get_chat_profile,
    get_embedding_profile,
    initialize_provider_state,
)
from utils.source_state import (
    effective_indexing_settings,
    ensure_active_source,
    initial_source_state,
    make_source_state,
    mark_source_pending,
    mark_source_ready,
    normalize_chunk_settings,
    report_matches_active_source,
    source_identity,
    stable_digest,
    tag_report,
)

MAX_INGESTED_DOCUMENTS = 300
MAX_INGESTED_TEXT_CHARS = 4 * 1024 * 1024


def _check_ingestion_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Website ingestion deadline exceeded.")


_SOURCE_STATE_KEYS = (
    "active_source",
    "index_generation",
    "documents",
    "retriever",
    "query_engine",
    "r2r_document_ids",
    "r2r_document_ids_signature",
    "llm",
    "file_extraction_report",
    "extraction_report",
    "file_ingestion_stages",
    "github_ingestion_stages",
    "website_ingestion_stages",
    "r2r_ingestion_stages",
    "github_ingestion_source_id",
    "github_ingestion_generation",
    "website_ingestion_source_id",
    "website_ingestion_generation",
    "last_ingestion_error",
    "_session_work_dirs",
)


def _document_text(document):
    if hasattr(document, "get_content"):
        return document.get_content() or ""
    if hasattr(document, "text"):
        return document.text or ""
    return str(document)


def validate_ingested_documents(documents):
    if len(documents) > MAX_INGESTED_DOCUMENTS:
        raise ValueError(
            f"Too many documents were loaded. Limit: {MAX_INGESTED_DOCUMENTS}."
        )

    total_chars = sum(len(_document_text(document)) for document in documents)
    if total_chars > MAX_INGESTED_TEXT_CHARS:
        raise ValueError("Loaded documents exceed the ingestion text limit.")


def render_pipeline_status(status_container, completed_stages, active_stage=None):
    """Render truthful ingestion progress for the currently running pipeline."""
    if status_container is None:
        return

    status_container.empty()
    with status_container.container():
        for stage in completed_stages:
            st.caption(f"✔️ {stage}")
        if active_stage is not None:
            st.caption(f"⏳ {active_stage}")


def render_embedding_progress(status_container, completed_stages, completed, total):
    """Render exact embedding progress for the active indexing stage."""
    if status_container is None:
        return

    progress = 0 if total == 0 else min(completed / total, 1)
    progress_label = f"Generating Embeddings — {progress:.0%}"

    status_container.empty()
    with status_container.container():
        for stage in completed_stages:
            st.caption(f"✔️ {stage}")
        st.caption("⏳ Generating Embeddings")
        st.progress(progress, text=progress_label)
        st.caption(f"{completed:,} / {total:,} chunks embedded")


def render_completed_ingestion_status(status_container, completed_stages):
    """Render final ingestion status without leaving stale progress widgets behind."""
    if status_container is None:
        return

    status_container.empty()
    with status_container.container():
        for stage in completed_stages:
            st.caption(f"✔️ {stage}")
        st.empty()
        st.empty()


def render_extraction_report(
    status_container=None,
    source_id: str | None = None,
    index_generation: int | None = None,
) -> None:
    """Render parser outcomes only for the active source and generation."""
    try:
        report = st.session_state.get("file_extraction_report", [])
    except Exception:
        return
    if not report or not report_matches_active_source(
        report,
        st.session_state,
        source_id=source_id,
        index_generation=index_generation,
    ):
        return
    problem_count = sum(
        entry.get("status") in {"skipped", "unsupported"} for entry in report
    )
    with st.expander("Extraction report", expanded=problem_count > 0):
        for entry in report:
            filename = entry.get("filename") or "Unnamed file"
            status = entry.get("status", "skipped")
            detail = entry.get("error") or entry.get("warning") or ""
            if status == "loaded":
                summary = (
                    f"{filename}: loaded "
                    f"({entry.get('document_count', 0)} documents, "
                    f"{entry.get('extracted_characters', 0)} characters)"
                )
                if detail:
                    summary += f" — {detail}"
                st.caption(summary)
            elif status == "unsupported":
                st.error(f"{filename}: unsupported — {detail}")
            else:
                st.warning(f"{filename}: skipped — {detail}")
    if problem_count:
        st.warning(
            f"{problem_count} file(s) were skipped or unsupported; see the extraction report."
        )


def _snapshot_state(state):
    return {
        key: copy.deepcopy(state.get(key))
        for key in _SOURCE_STATE_KEYS
        if key in state
    }


def _restore_state(state, snapshot):
    for key in _SOURCE_STATE_KEYS:
        if key in snapshot:
            state[key] = copy.deepcopy(snapshot[key])
        elif key in {"active_source", "index_generation"}:
            state.pop(key, None)
    if "active_source" not in state:
        state["active_source"] = initial_source_state()
    if "index_generation" not in state:
        state["index_generation"] = 0


def _uploaded_content_signature(uploaded_files):
    records = []
    for uploaded_file in uploaded_files or []:
        payload = uploaded_file.getvalue()
        records.append(
            {
                "name": str(uploaded_file.name),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    records.sort(key=lambda record: (record["name"].casefold(), record["sha256"]))
    return stable_digest(records)


def _documents_content_signature(documents):
    return stable_digest(
        sorted(
            [_document_text(document) for document in documents or []],
        )
    )


def _default_source_name(source_kind, uploaded_files, documents, source_uri):
    if source_kind == "github":
        return str(source_uri or "GitHub repository").rstrip("/").split("/")[-1]
    if source_kind == "website":
        return "Website source"
    if source_kind == "r2r":
        return "Uploaded files (R2R)"
    if uploaded_files:
        names = sorted((str(item.name) for item in uploaded_files), key=str.casefold)
        return ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
    if documents:
        return "Website source" if source_kind == "website" else "Document source"
    return "Document source"


def _record_work_dir(state, path):
    if not path:
        return
    work_dirs = list(state.get("_session_work_dirs", []))
    if path not in work_dirs:
        work_dirs.append(path)
    state["_session_work_dirs"] = work_dirs


def _unrecord_work_dir(state, path):
    state["_session_work_dirs"] = [
        item for item in state.get("_session_work_dirs", []) if item != path
    ]


def _show_ingestion_error(error):
    message = str(error)
    if "is not available on the Ollama server" in message:
        warning = getattr(st, "warning", None)
        if callable(warning):
            warning(
                "⚠️ **Model unavailable.**\n\n"
                f"{message}\n\nCheck the active provider settings and model list."
            )
    elif getattr(error, "category", None):
        error_renderer = getattr(st, "error", None)
        if callable(error_renderer):
            error_renderer(func.website_error_message(error))
    else:
        exception = getattr(st, "exception", None)
        if callable(exception):
            exception(error)
    stop = getattr(st, "stop", None)
    if callable(stop):
        stop()


def rag_pipeline(
    uploaded_files: list = None,
    documents: list = None,
    data_dir: str | None = None,
    status_container=None,
    initial_stages: list[str] | None = None,
    status_state_key: str = "file_ingestion_stages",
    documents_loaded_stage: str = "Documents Loaded",
    source_kind: str = "local",
    source_id: str | None = None,
    display_name: str | None = None,
    source_uri: str | None = None,
    content_signature: str | None = None,
    deadline: float | None = None,
    ingestion_report: list[dict] | None = None,
):
    """Build one source index transactionally and commit it only on success."""
    state = st.session_state
    snapshot = _snapshot_state(state)
    previous_source = ensure_active_source(state)
    previous_committed = (
        previous_source.get("status") == "ready"
        and (
            (
                previous_source.get("kind") in {None, "local", "github", "website"}
                and state.get("query_engine") is not None
            )
            or (
                previous_source.get("kind") == "r2r"
                and bool(state.get("r2r_document_ids"))
            )
        )
    )
    owned_work_dir = None
    candidate = None
    error = None
    completed_stages = list(initial_stages or [])
    report = []

    def record_completed_stages():
        state[status_state_key] = list(completed_stages)

    try:
        _check_ingestion_deadline(deadline)
        chunk_settings = normalize_chunk_settings(state)
        indexing_settings = effective_indexing_settings(state)
        if content_signature is None:
            if uploaded_files is not None:
                content_signature = _uploaded_content_signature(uploaded_files)
            elif documents is not None:
                content_signature = _documents_content_signature(documents)
            else:
                content_signature = stable_digest(
                    {"data_dir": str(data_dir or ""), "source_kind": source_kind}
                )
        content_signature = str(content_signature)
        if source_id is None:
            source_id = source_identity(source_kind, content_signature)
        if display_name is None:
            display_name = _default_source_name(
                source_kind, uploaded_files, documents, source_uri
            )
        generation = max(
            int(state.get("index_generation") or 0),
            int(previous_source.get("index_generation") or 0),
        ) + 1
        candidate = make_source_state(
            source_kind,
            source_id=source_id,
            display_name=display_name,
            source_uri=source_uri,
            content_signature=content_signature,
            settings_signature=stable_digest(indexing_settings),
            index_generation=generation,
            status="pending",
        )
        mark_source_pending(state, candidate)
        state["file_extraction_report"] = []
        state["extraction_report"] = []

        if uploaded_files is not None:
            func.validate_uploaded_files(uploaded_files)
            if not uploaded_files:
                raise ValueError("No files were selected for ingestion.")
            if uploaded_files:
                owned_work_dir = func.create_ingestion_work_dir("upload")
                _record_work_dir(state, owned_work_dir)
                ingest_dir = owned_work_dir
                uploaded_paths = []
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        func.save_uploaded_file(uploaded_file, ingest_dir)
                        uploaded_paths.append(
                            os.path.join(ingest_dir, uploaded_file.name)
                        )
                completed_stages.append("Files Uploaded")
                record_completed_stages()
                render_pipeline_status(status_container, completed_stages)
            else:
                ingest_dir = func.create_ingestion_work_dir("upload")
                _record_work_dir(state, ingest_dir)
                uploaded_paths = []
        elif data_dir is not None:
            ingest_dir = data_dir
            uploaded_paths = None
        else:
            ingest_dir = func.create_ingestion_work_dir("scan")
            _record_work_dir(state, ingest_dir)
            uploaded_paths = None

        with llama_index.ingestion_lock():
            backend = state.get("llm_backend", "Ollama")
            _check_ingestion_deadline(deadline)
            initialize_provider_state(state)
            chat_profile = get_chat_profile(state, backend)
            if chat_profile["provider_kind"] == "ollama":
                selected_model = chat_profile["model"]
                if not ollama.verify_chat_model(
                    selected_model, chat_profile["base_url"]
                ):
                    raise ValueError(
                        f"Chat model '{selected_model}' is not available at "
                        f"{chat_profile['base_url']}. "
                        f"Pull it first with: ollama pull {selected_model}"
                    )
            _check_ingestion_deadline(deadline)
            llm = ollama.create_llm(
                chat_profile["model"],
                chat_profile["base_url"],
                chat_profile["api_key"],
                system_prompt=state.get("system_prompt"),
                backend=backend,
            )
            _check_ingestion_deadline(deadline)
            completed_stages.append("LLM Initialized")
            record_completed_stages()
            render_pipeline_status(status_container, completed_stages)

            embedding_profile = get_embedding_profile(state)
            embedding_kwargs = {
                "chunk_size": chunk_settings["chunk_size"],
                "chunk_overlap": chunk_settings["chunk_overlap"],
                "backend": embedding_profile["backend"],
                "base_url": embedding_profile["base_url"],
                "api_key": embedding_profile["api_key"],
                "provider_kind": embedding_profile["provider_kind"],
            }
            if deadline is not None:
                embedding_kwargs["deadline"] = deadline
            llama_index.setup_embedding_model(
                embedding_profile["model"],
                **embedding_kwargs,
            )
            _check_ingestion_deadline(deadline)
            completed_stages.append("Embedding Model Ready")
            record_completed_stages()
            render_pipeline_status(status_container, completed_stages)

            if documents is not None:
                _check_ingestion_deadline(deadline)
                if len(documents) == 0:
                    raise ValueError("No documents were loaded from the selected source.")
                validate_ingested_documents(documents)
                report = [dict(entry) for entry in (ingestion_report or [])]
            else:
                _check_ingestion_deadline(deadline)
                documents, report = llama_index.load_documents(
                    ingest_dir,
                    input_files=uploaded_paths,
                    return_report=True,
                    store_report=False,
                    source_id=candidate["id"],
                    index_generation=candidate["index_generation"],
                )
                if len(documents) == 0:
                    raise ValueError("No files were found to process.")
                validate_ingested_documents(documents)
            _check_ingestion_deadline(deadline)
            completed_stages.append(documents_loaded_stage)
            record_completed_stages()
            render_pipeline_status(status_container, completed_stages)

            def update_embedding_progress(completed, total):
                if total is None or total == 0:
                    render_pipeline_status(
                        status_container, completed_stages, "Generating Embeddings"
                    )
                    return
                render_embedding_progress(
                    status_container, completed_stages, completed, total
                )

            render_pipeline_status(
                status_container, completed_stages, "Generating Embeddings"
            )
            _check_ingestion_deadline(deadline)
            query_engine_kwargs = {
                "progress_callback": update_embedding_progress,
                "settings": indexing_settings,
                "source_identity": content_signature,
            }
            if deadline is not None:
                query_engine_kwargs["deadline"] = deadline
            bundle = llama_index.create_query_engine(
                documents,
                **query_engine_kwargs,
            )
            if not isinstance(bundle, dict) and all(
                hasattr(bundle, key)
                for key in ("query_engine", "retriever", "index", "cache_key")
            ):
                bundle = {
                    key: getattr(bundle, key)
                    for key in ("query_engine", "retriever", "index", "cache_key")
                }
            if not isinstance(bundle, dict) or not all(
                key in bundle for key in ("query_engine", "retriever", "index", "cache_key")
            ):
                raise ValueError("Index builder did not return a complete candidate bundle.")
            _check_ingestion_deadline(deadline)
            completed_stages.append("Embeddings Generated")
            completed_stages.append("Index Ready")

            tagged_report = tag_report(
                report, candidate["id"], candidate["index_generation"]
            )
            ready_source = copy.deepcopy(candidate)
            ready_source["cache_key"] = bundle["cache_key"]
            ready_source["error"] = None
            state["documents"] = documents
            state["retriever"] = bundle["retriever"]
            state["query_engine"] = bundle["query_engine"]
            state["r2r_document_ids"] = []
            state["r2r_document_ids_signature"] = None
            state["index_generation"] = candidate["index_generation"]
            state["llm"] = llm
            state["file_extraction_report"] = tagged_report
            state["extraction_report"] = [dict(entry) for entry in tagged_report]
            state["last_ingestion_error"] = None
            if status_state_key == "github_ingestion_stages":
                state["github_ingestion_source_id"] = candidate["id"]
                state["github_ingestion_generation"] = candidate["index_generation"]
            elif status_state_key == "website_ingestion_stages":
                state["website_ingestion_source_id"] = candidate["id"]
                state["website_ingestion_generation"] = candidate["index_generation"]
            mark_source_ready(state, ready_source)
            record_completed_stages()
            render_completed_ingestion_status(status_container, completed_stages)
    except Exception as err:
        if source_kind == "website" and (
            isinstance(err, TimeoutError)
            or (deadline is not None and time.monotonic() >= deadline)
        ):
            err = func.WebsiteIngestionError(
                "timeout",
                "Website ingestion exceeded its total deadline before indexing completed.",
            )
        error = err
        logs.log.error(f"Source ingestion failed: {type(err).__name__}: {err}")
        _restore_state(state, snapshot)
        error_text = (
            func.website_error_message(err)
            if getattr(err, "category", None)
            else str(err)
        )
        state["last_ingestion_error"] = error_text
        if not previous_committed:
            if candidate is not None:
                failed_source = copy.deepcopy(candidate)
                failed_source["status"] = "failed"
                failed_source["error"] = error_text
            else:
                failed_source = initial_source_state()
                failed_source["status"] = "failed"
                failed_source["error"] = error_text
            state["active_source"] = failed_source
            state["index_generation"] = 0
            state["documents"] = None
            state["retriever"] = None
            state["query_engine"] = None
            state["r2r_document_ids"] = []
            state["r2r_document_ids_signature"] = None
        _show_ingestion_error(err)
    finally:
        if owned_work_dir:
            if func.cleanup_ingestion_work_dir(owned_work_dir):
                _unrecord_work_dir(state, owned_work_dir)
            else:
                _record_work_dir(state, owned_work_dir)
                logs.log.warning(
                    f"Unable to delete ingestion work directory: {owned_work_dir}"
                )

    return error
