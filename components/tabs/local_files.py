import hashlib

import streamlit as st

import utils.rag_pipeline as rag
import utils.r2r as r2r
import utils.helpers as func
from components.ingestion_prerequisites import (
    ingestion_is_configured,
)
from utils.format_ingestion import SUPPORTED_EXTENSIONS
from utils.source_state import (
    effective_indexing_settings,
    ensure_active_source,
    make_source_state,
    mark_active_index_stale_if_needed,
    mark_source_ready,
    source_identity,
    stable_digest,
)

supported_files = SUPPORTED_EXTENSIONS


def _bytes_to_mb(size_in_bytes):
    return size_in_bytes // (1024 * 1024)


def upload_limit_help_text():
    return (
        f"Up to {func.MAX_UPLOAD_FILES} files. "
        f"{_bytes_to_mb(func.MAX_UPLOAD_FILE_BYTES)}MB per file, "
        f"{_bytes_to_mb(func.MAX_TOTAL_UPLOAD_BYTES)}MB total."
    )


def uploaded_files_signature(uploaded_files):
    """Return a stable signature for the current uploader contents."""
    return tuple(
        sorted(
            (
                (
                    uploaded_file.name,
                    uploaded_file.size,
                    uploaded_file.type,
                    hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
                )
                for uploaded_file in uploaded_files
            ),
            key=lambda value: (value[0].casefold(), value[3]),
        )
    )


def upload_processing_signature(uploaded_files):
    """Include source identity and active indexing settings only."""
    files_signature = uploaded_files_signature(uploaded_files)
    settings = effective_indexing_settings(st.session_state)
    settings_signature = stable_digest(settings)
    if st.session_state.get("r2r_enabled"):
        return (
            "r2r",
            files_signature,
            settings_signature,
            st.session_state.get("r2r_base_url", "http://localhost:7272"),
            settings.get("r2r_credential_fingerprint"),
        )
    return (
        "local",
        files_signature,
        settings_signature,
    )


def should_process_uploads(
    current_signature,
    processed_signature,
    processing_signature,
    query_engine,
    active_source=None,
    content_signature=None,
):
    """Return whether uploaded files need ingestion for the current app state."""
    if current_signature == processing_signature:
        return False
    if st.session_state.get("r2r_enabled"):
        source = active_source or st.session_state.get("active_source")
        if not isinstance(source, dict):
            return (
                current_signature != processed_signature
                or not st.session_state.get("r2r_document_ids")
                or not isinstance(source, dict)
            )
        return (
            current_signature != processed_signature
            or source.get("kind") != "r2r"
            or source.get("status") != "ready"
            or not st.session_state.get("r2r_document_ids")
            or (
                st.session_state.get("r2r_document_ids_signature") is not None
                and st.session_state.get("r2r_document_ids_signature")
                != stable_digest(
                    {
                        "contract": r2r.R2R_CONTRACT,
                        "processing_signature": current_signature,
                    }
                )
            )
            or (
                content_signature is not None
                and source.get("content_signature") != content_signature
            )
        )
    if isinstance(active_source, dict) and active_source.get("kind") not in {
        None,
        "local",
    }:
        return (
            content_signature is not None
            and active_source.get("content_signature") != content_signature
        )
    return current_signature != processed_signature or query_engine is None


def _upload_content_signature(uploaded_files):
    return stable_digest(
        [
            {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "sha256": hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
            }
            for uploaded_file in sorted(
                uploaded_files, key=lambda item: item.name.casefold()
            )
        ]
    )


def local_files():
    try:
        effective_indexing_settings(st.session_state)
        mark_active_index_stale_if_needed(st.session_state)
    except ValueError as err:
        st.error(str(err))
        return

    if ingestion_is_configured(allow_r2r=True):
        uploaded_files = st.file_uploader(
            "Select Files",
            accept_multiple_files=True,
            type=supported_files,
            help=upload_limit_help_text(),
        )
    else:
        file_upload_container = st.container(border=True)
        with file_upload_container:
            uploaded_files = st.file_uploader(
                "Select Files",
                accept_multiple_files=True,
                type=supported_files,
                disabled=True,
                help=upload_limit_help_text(),
            )
    uploaded_files = uploaded_files or []
    if not uploaded_files:
        if (
            st.session_state.get("r2r_enabled")
            and st.session_state.get("file_selection_signature") is not None
        ):
            r2r.invalidate_source_selection(st.session_state)
        return

    try:
        func.validate_uploaded_files(uploaded_files)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    large_files = [
        f.name
        for f in uploaded_files
        if getattr(f, "size", 0) > 8 * 1024 * 1024
    ]
    if large_files:
        st.info(
            "💡 **Large file(s):** "
            + ", ".join(large_files)
            + ". Big files take minutes to embed and heat up your laptop. "
            + "Consider splitting them into 10-20 page PDFs or smaller documents."
        )

    selection_signature = uploaded_files_signature(uploaded_files)
    previous_selection = st.session_state.get("file_selection_signature")
    active_source = ensure_active_source(st.session_state)
    retained_file_selection = previous_selection == selection_signature
    failed_selection = st.session_state.get("failed_upload_selection_signature")
    if failed_selection == selection_signature:
        retained_file_selection = False
    if previous_selection is None and st.session_state.get("file_list"):
        try:
            retained_file_selection = uploaded_files_signature(
                st.session_state["file_list"]
            ) == selection_signature
        except Exception:
            retained_file_selection = False
    retained_for_other_source = (
        active_source.get("kind") not in {None, "local", "r2r"}
        and retained_file_selection
    )
    if retained_for_other_source:
        return

    st.session_state["file_selection_signature"] = selection_signature
    st.session_state["file_list"] = uploaded_files
    content_signature = _upload_content_signature(uploaded_files)
    current_upload_signature = upload_processing_signature(uploaded_files)
    r2r_source_id = source_identity("r2r", content_signature)
    r2r_source_signature = stable_digest(
        {
            "contract": r2r.R2R_CONTRACT,
            "processing_signature": current_upload_signature,
        }
    )
    if st.session_state.get("r2r_enabled"):
        st.session_state["r2r_current_source_signature"] = r2r_source_signature
    needs_processing = should_process_uploads(
        current_upload_signature,
        st.session_state.get("processed_file_signature"),
        st.session_state.get("processing_file_signature"),
        st.session_state.get("query_engine"),
        active_source=st.session_state.get("active_source"),
        content_signature=content_signature,
    )
    status_container = st.empty()

    if needs_processing:
        with st.spinner("Processing..."):
            st.session_state["processing_file_signature"] = current_upload_signature
            processing_succeeded = False
            try:
                if st.session_state.get("r2r_enabled"):
                    r2r.r2r_ingest_files(
                        uploaded_files,
                        status_container=status_container,
                        content_signature=content_signature,
                        source_id=r2r_source_id,
                        source_signature=r2r_source_signature,
                    )
                    r2r_settings = effective_indexing_settings(st.session_state)
                    r2r_generation = max(
                        int(st.session_state.get("index_generation") or 0),
                        int(active_source.get("index_generation") or 0),
                    ) + 1
                    r2r_source = make_source_state(
                        "r2r",
                        source_id=r2r_source_id,
                        display_name=", ".join(
                            sorted((item.name for item in uploaded_files), key=str.casefold)
                        ),
                        content_signature=content_signature,
                        settings_signature=stable_digest(r2r_settings),
                        index_generation=r2r_generation,
                        status="pending",
                    )
                    ready_r2r = dict(r2r_source)
                    ready_r2r["status"] = "ready"
                    st.session_state["index_generation"] = r2r_generation
                    mark_source_ready(st.session_state, ready_r2r)
                    r2r.registry_for_state(st.session_state).update_active_metadata(
                        {
                            "source_id": r2r_source_id,
                            "content_signature": content_signature,
                            "settings_signature": stable_digest(r2r_settings),
                            "source_signature": r2r_source_signature,
                            "index_generation": r2r_generation,
                            "display_name": r2r_source["display_name"],
                        }
                    )
                    st.session_state["processed_file_signature"] = current_upload_signature
                    st.session_state["r2r_document_ids_signature"] = r2r_source_signature
                    processing_succeeded = True
                else:
                    error = rag.rag_pipeline(
                        uploaded_files,
                        status_container=status_container,
                        source_kind="local",
                        source_id=source_identity("local", content_signature),
                        content_signature=content_signature,
                    )
                    if error is None:
                        st.session_state["processed_file_signature"] = current_upload_signature
                        processing_succeeded = True
                    else:
                        st.session_state["failed_upload_selection_signature"] = selection_signature
            except BaseException:
                st.session_state["failed_upload_selection_signature"] = selection_signature
                raise
            finally:
                st.session_state["processing_file_signature"] = None
                if processing_succeeded:
                    st.session_state["failed_upload_selection_signature"] = None
    else:
        if st.session_state.get("r2r_enabled"):
            status_container.empty()
        else:
            rag.render_pipeline_status(
                status_container,
                st.session_state.get("file_ingestion_stages", []),
            )

    active_source = ensure_active_source(st.session_state)
    if active_source.get("kind") in {"local", "r2r"} and active_source.get(
        "content_signature"
    ) == content_signature:
        if not st.session_state.get("r2r_enabled"):
            rag.render_extraction_report(
                source_id=active_source.get("id"),
                index_generation=active_source.get("index_generation"),
            )
        if st.session_state.get("r2r_enabled") and r2r.r2r_is_ready(
            st.session_state
        ):
            st.write("Your files are ready on the R2R server. Let's chat! 😎")
        elif st.session_state.get("query_engine") is not None:
            st.write("Your files are ready. Let's chat! 😎")
