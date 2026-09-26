import streamlit as st

from components.page_state import WELCOME_MESSAGE
from components.tabs.settings import settings
from components.tabs.sources import sources
from utils.browser_settings import persist_settings_to_browser_storage
from utils.source_state import active_index_matches_settings, ensure_active_source


def reset_project(delete_remote=True):
    """Flag a project reset before Streamlit widgets are instantiated."""
    st.session_state["reset_requested"] = True
    st.session_state["reset_delete_remote"] = bool(delete_remote)


def sidebar():
    with st.sidebar:
        tab1, tab2 = st.sidebar.tabs(["Data Sources", "Settings"])

        with tab1:
            sources()

        with tab2:
            settings()

        st.divider()

        reset_result = st.session_state.get("reset_result")
        if reset_result and reset_result.get("remote_delete_success") is False:
            remote = reset_result.get("remote") or {}
            st.error(
                "Project state was cleared, but remote R2R deletion was incomplete. "
                f"{len(remote.get('failed') or [])} deletion(s) failed and "
                f"{remote.get('other_identity_count', 0)} document(s) belong to another "
                "endpoint or credential. The ownership registry was retained."
            )
            if reset_result.get("failed"):
                st.error(
                    "These owned work directories could not be deleted: "
                    + ", ".join(reset_result["failed"])
                )
        elif reset_result and reset_result.get("remote_delete_success") is None:
            st.info(
                "Local project state was reset. Remote R2R documents were retained "
                "and remain tracked in the ownership registry."
            )
        elif reset_result:
            if reset_result.get("failed"):
                st.error(
                    "Remote R2R cleanup completed, but these work directories could not be deleted: "
                    + ", ".join(reset_result["failed"])
                )
            else:
                st.success(
                    "Project reset completed; owned remote R2R documents were deleted or already absent."
                )

        active_source = ensure_active_source(st.session_state)
        if st.session_state.get("r2r_enabled") and st.session_state.get(
            "r2r_document_ids"
        ):
            if (
                active_source.get("kind") == "r2r"
                and active_source.get("status") == "ready"
            ):
                st.success("R2R Mode: using documents on the external R2R server")
            else:
                st.info("R2R is enabled but its active documents are not ready.")
        elif (
            st.session_state.get("query_engine")
            and active_source.get("kind") in {None, "local", "github", "website"}
            and active_source.get("status") == "ready"
            and active_index_matches_settings(st.session_state)
        ):
            st.success("RAG Mode: using the active local document index")
        else:
            st.info("Chat Mode: direct model conversation without a document index")

        with st.expander("🧹 Clear Chat & Reset", expanded=False):
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state["messages"] = [dict(WELCOME_MESSAGE)]
                st.session_state["last_doc_sources"] = []
                st.session_state["last_rag_evidence"] = []
                st.session_state["last_retrieval_route"] = {}
                st.session_state["last_rag_no_result"] = False
                st.session_state["last_rag_question"] = None
                st.rerun()

            st.markdown(
                "Clears only the conversation above. To wipe everything "
                "(indexes, uploads, settings), use **Reset Project** below."
            )

            st.warning(
                "The safe default deletes documents owned by this workspace from the "
                "configured R2R server, then clears local source state. Shared caches, "
                "logs, other sessions, browser settings, and API keys are retained."
            )
            reset_mode = st.radio(
                "Remote R2R cleanup",
                [
                    "Delete owned R2R documents (recommended)",
                    "Keep R2R documents (local-only reset)",
                ],
                key="project_reset_mode",
                horizontal=False,
            )
            delete_remote = reset_mode.startswith("Delete")
            if st.checkbox(
                "I understand this reset cannot be undone",
                key="confirm_project_reset",
            ):
                button_label = (
                    "Reset and delete owned R2R documents"
                    if delete_remote
                    else "Reset local project only"
                )
                st.button(
                    button_label,
                    use_container_width=True,
                    on_click=reset_project,
                    args=(delete_remote,),
                )

        persist_settings_to_browser_storage()
