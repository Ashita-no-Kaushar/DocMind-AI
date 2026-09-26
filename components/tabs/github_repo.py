import streamlit as st

import utils.helpers as func
import utils.rag_pipeline as rag
from components.ingestion_prerequisites import (
    ingestion_is_configured,
)
from utils.ingestion_lock import lifecycle_lock
from utils.source_state import (
    effective_indexing_settings,
    ensure_active_source,
    mark_active_index_stale_if_needed,
    source_identity,
)

GITHUB_DOCUMENTS_LOADED_STAGE = "Repository Files Loaded"


def should_show_github_ingestion_status(
    current_repo,
    processed_repo,
    ingestion_stages,
    query_engine,
    active_source=None,
    ingestion_source_id=None,
    ingestion_generation=None,
):
    """Return whether saved GitHub status belongs to the active generation."""
    try:
        normalized_current_repo = func.normalize_github_repo(current_repo)
    except ValueError:
        return False

    if active_source is not None:
        if not isinstance(active_source, dict):
            return False
        if active_source.get("kind") != "github":
            return False
        if active_source.get("status") != "ready":
            return False
        source_label = str(
            active_source.get("display_name") or active_source.get("source_uri") or ""
        ).rstrip("/")
        if (
            source_label
            not in {
                normalized_current_repo,
                normalized_current_repo.split("/")[-1],
            }
            and normalized_current_repo not in source_label
        ):
            return False
        if ingestion_source_id is None or ingestion_source_id != active_source.get(
            "id"
        ):
            return False
        if ingestion_generation is None or ingestion_generation != active_source.get(
            "index_generation"
        ):
            return False
    return (
        normalized_current_repo == processed_repo
        and len(ingestion_stages) > 0
        and query_engine is not None
    )


def github_repo():
    try:
        effective_indexing_settings(st.session_state)
        mark_active_index_stale_if_needed(st.session_state)
    except ValueError as err:
        st.error(str(err))
        return
    if ingestion_is_configured():
        with st.form("github_repo_form"):
            st.text_input(
                "Select a GitHub.com repo",
                placeholder="Ashita-no-Kaushar/DocMind-AI",
                key="github_repo",
            )
            repo_processed = st.form_submit_button("Process")

        if repo_processed:
            input_repo = (st.session_state.get("github_repo") or "").strip()
            if not input_repo:
                input_repo = "Ashita-no-Kaushar/DocMind-AI"
            status_container = st.empty()
            completed_stages = []
            work_dir = None
            try:
                with lifecycle_lock():
                    repo = func.normalize_github_repo(input_repo)
                    rag.render_pipeline_status(
                        status_container, completed_stages, "Validating Repository"
                    )
                    if not func.validate_github_repo(repo):
                        st.error(
                            "That GitHub repository could not be validated. Use `owner/repo` or a GitHub URL and ensure it exists."
                        )
                        st.stop()
                    completed_stages.append("Repository Validated")
                    rag.render_pipeline_status(status_container, completed_stages)
                    rag.render_pipeline_status(
                        status_container, completed_stages, "Cloning Repository"
                    )
                    work_dir = func.create_ingestion_work_dir("github")
                    st.session_state.setdefault("_session_work_dirs", []).append(
                        work_dir
                    )
                    cloned_repo_dir = func.clone_github_repo(
                        repo, destination_base=work_dir
                    )
                    if not cloned_repo_dir:
                        st.error(
                            "Failed to clone repository. Check the repo value and try again."
                        )
                        st.stop()
                    completed_stages.append("Repository Cloned")
                    rag.render_pipeline_status(status_container, completed_stages)
                    content_signature = source_identity("github", repo)
                    error = rag.rag_pipeline(
                        data_dir=cloned_repo_dir,
                        status_container=status_container,
                        initial_stages=completed_stages,
                        status_state_key="github_ingestion_stages",
                        documents_loaded_stage=GITHUB_DOCUMENTS_LOADED_STAGE,
                        source_kind="github",
                        source_id=content_signature,
                        display_name=repo,
                        source_uri=f"https://github.com/{repo}",
                        content_signature=content_signature,
                    )
                    if error is not None:
                        st.error(
                            "The repository could not be indexed safely. Try again."
                        )
                    else:
                        st.session_state["processed_github_repo"] = repo
                        active_source = ensure_active_source(st.session_state)
                        rag.render_extraction_report(
                            source_id=active_source.get("id"),
                            index_generation=active_source.get("index_generation"),
                        )
                        st.write("Your files are ready. Let's chat! 😎")
            finally:
                if work_dir:
                    if func.cleanup_ingestion_work_dir(work_dir):
                        st.session_state["_session_work_dirs"] = [
                            item
                            for item in st.session_state.get("_session_work_dirs", [])
                            if item != work_dir
                        ]
                    else:
                        st.warning(
                            "The repository work directory could not be deleted."
                        )
        elif should_show_github_ingestion_status(
            st.session_state.get("github_repo", ""),
            st.session_state.get("processed_github_repo"),
            st.session_state.get("github_ingestion_stages", []),
            st.session_state.get("query_engine"),
            active_source=st.session_state.get("active_source"),
            ingestion_source_id=st.session_state.get("github_ingestion_source_id"),
            ingestion_generation=st.session_state.get("github_ingestion_generation"),
        ):
            status_container = st.empty()
            rag.render_pipeline_status(
                status_container,
                st.session_state.get("github_ingestion_stages", []),
            )
            active_source = ensure_active_source(st.session_state)
            rag.render_extraction_report(
                source_id=active_source.get("id"),
                index_generation=active_source.get("index_generation"),
            )
            st.write("Your files are ready. Let's chat! 😎")

    else:
        st.text_input(
            "Select a GitHub.com repo",
            placeholder="Ashita-no-Kaushar/DocMind-AI",
            key="github_repo_disabled",
            disabled=True,
        )
        st.button("Process Repo", disabled=True)
