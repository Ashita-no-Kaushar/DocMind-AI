import streamlit as st

from datetime import datetime


def about():
    st.title("📚 DocMind AI")
    st.caption("LLMOps-Based Document Intelligence System")
    st.write("")

    links_html = """
    <ul style="list-style-type:none; padding-left:0;">
        <li>
            <a href="https://github.com/Ashita-no-Kaushar/Docmind" style="color: grey;">GitHub</a>
        </li>
    </ul>
    """

    resources_html = """
    <ul style="list-style-type:none; padding-left:0;">
        <li>
            <a href="https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/" style="color: grey;">
                What is RAG?
            </a>
        </li>
        <li>
            <a href="https://aws.amazon.com/what-is/embeddings-in-machine-learning/" style="color: grey;">
                What are embeddings?
            </a>
        </li>
    </ul>
    """

    help_html = """
    <ul style="list-style-type:none; padding-left:0;">
        <li>
            <a href="https://github.com/Ashita-no-Kaushar/Docmind/issues" style="color: grey;">
                Bug Reports
            </a>
        </li>
        <li>
            <a href="https://github.com/Ashita-no-Kaushar/Docmind/discussions" style="color: grey;">
                Feature Requests
            </a>
        </li>
    </ul>
    """

    st.subheader("Links")
    st.markdown(links_html, unsafe_allow_html=True)

    st.subheader("Resources")
    st.markdown(resources_html, unsafe_allow_html=True)

    st.subheader("Help")
    st.markdown(help_html, unsafe_allow_html=True)
