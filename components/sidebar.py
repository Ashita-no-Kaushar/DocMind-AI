import streamlit as st

from components.page_state import WELCOME_MESSAGE
from components.tabs.sources import sources
from components.tabs.settings import settings
from utils.browser_settings import persist_settings_to_browser_storage


def reset_project():
    """Flag a full project reset; performed in set_initial_state before any
    widget is instantiated (widget keys can't be changed after their widget)."""
    st.session_state["reset_requested"] = True


def sidebar():
    with st.sidebar:
        tab1, tab2 = st.sidebar.tabs(["Data Sources", "Settings"])

        with tab1:
            sources()

        with tab2:
            settings()

        st.divider()

        # Status Badge
        if st.session_state.get("r2r_document_ids"):
            st.success("🟢 **R2R Mode**: Grounded via R2R backend")
        elif st.session_state.get("query_engine"):
            st.success("🟢 **RAG Mode**: Grounded in documents")
        else:
            st.info("💬 **Chat Mode**: Direct LLM conversation")

        with st.expander("🧹 Clear Chat & Reset", expanded=False):
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state["messages"] = [dict(WELCOME_MESSAGE)]
                st.session_state["last_doc_sources"] = []
                st.session_state["last_rag_no_result"] = False
                st.session_state["last_rag_question"] = None
                st.rerun()

            st.markdown(
                "Clears only the conversation above. To wipe everything "
                "(indexes, uploads, settings), use **Reset Project** below."
            )

            # Reset Project (danger zone)
            st.warning(
                "Deletes ALL cached indexes and uploaded data, and restores "
                "default settings. The app becomes a fresh project. "
                "Note: after a reset, the first re-upload rebuilds the index "
                "once (normal speed); repeat uploads are instant again."
            )
            if st.checkbox("I understand", key="confirm_project_reset"):
                if st.button("Yes, reset everything", use_container_width=True):
                    reset_project()
                    st.rerun()

        persist_settings_to_browser_storage()
