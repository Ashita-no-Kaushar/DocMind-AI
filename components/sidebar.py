import streamlit as st

from components.page_state import WELCOME_MESSAGE
from components.status import render_mode_card, resolve_status
from components.tabs.settings import settings
from components.tabs.sources import sources
from utils.browser_settings import persist_settings_to_browser_storage


def reset_project(delete_remote=True):
    """Flag a project reset before Streamlit widgets are instantiated."""
    st.session_state["reset_requested"] = True
    st.session_state["reset_delete_remote"] = bool(delete_remote)


def _render_work_directory_errors(reset_result):
    failed = reset_result.get("failed")
    if failed:
        st.error(
            "These owned work directories could not be deleted: " + ", ".join(failed)
        )


def _render_reset_result():
    reset_result = st.session_state.get("reset_result")
    if not reset_result:
        return
    remote = reset_result.get("remote") or {}
    if reset_result.get("remote_delete_success") is False:
        st.error(
            "Project state was cleared, but remote R2R deletion was incomplete. "
            f"{len(remote.get('failed') or [])} deletion(s) failed and "
            f"{remote.get('other_identity_count', 0)} document(s) belong to another "
            "endpoint or credential. The ownership registry was retained."
        )
        _render_work_directory_errors(reset_result)
    elif reset_result.get("remote_delete_success") is None:
        st.info(
            "Local project state was reset. Remote R2R documents were retained "
            "and remain tracked in the ownership registry."
        )
    elif reset_result.get("failed"):
        st.error(
            "Remote R2R cleanup completed, but these work directories could not "
            "be deleted: " + ", ".join(reset_result["failed"])
        )
    else:
        st.success(
            "Project reset completed; owned remote R2R documents were deleted "
            "or already absent."
        )


def sidebar():
    with st.sidebar:
        render_mode_card(resolve_status())
        st.divider()

        tab1, tab2 = st.sidebar.tabs(["Data Sources", "Settings"])

        with tab1:
            sources()

        with tab2:
            settings()

        st.divider()
        _render_reset_result()

        if st.button("💬 Clear Chat", use_container_width=True):
            st.session_state["messages"] = [dict(WELCOME_MESSAGE)]
            st.session_state["last_doc_sources"] = []
            st.session_state["last_rag_evidence"] = []
            st.session_state["last_retrieval_route"] = {}
            st.session_state["last_rag_no_result"] = False
            st.session_state["last_rag_question"] = None
            st.rerun()

        with st.expander("Reset Project (destructive)", expanded=False):
            st.caption(
                "Clears the conversation, indexes, uploads, and local source state. "
                "Shared caches, logs, other sessions, browser settings, and API keys "
                "are retained."
            )
            st.warning(
                "The safe default deletes documents owned by this workspace from the "
                "configured R2R server. This cannot be undone."
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
