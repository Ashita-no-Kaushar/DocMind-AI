import streamlit as st

from components.tabs.about import about
from components.tabs.sources import sources
from components.tabs.settings import settings
from utils.browser_settings import persist_settings_to_browser_storage


def clear_index_and_chat():
    st.session_state["query_engine"] = None
    st.session_state["file_list"] = []
    st.session_state["processed_file_signature"] = None
    st.session_state["processed_github_repo"] = None
    st.session_state["processed_website_urls"] = None
    st.session_state["github_ingestion_stages"] = []
    st.session_state["website_ingestion_stages"] = []
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Welcome to DocMind AI! To begin, please either ask a question directly or import some files/repositories to chat with your data.",
        }
    ]


def sidebar():
    with st.sidebar:
        tab1, tab2, tab3 = st.sidebar.tabs(["Data Sources", "Settings", "About"])

        with tab1:
            sources()

        with tab2:
            settings()

        with tab3:
            about()

        st.divider()

        # Status Badge
        if st.session_state.get("query_engine"):
            st.success("🟢 **RAG Mode**: Grounded in documents")
        else:
            st.info("💬 **Chat Mode**: Direct LLM conversation")

        # Clear Index & Reset Button
        if st.button("🔄 Clear Index & Reset Chat", use_container_width=True):
            clear_index_and_chat()
            st.rerun()

        persist_settings_to_browser_storage()
