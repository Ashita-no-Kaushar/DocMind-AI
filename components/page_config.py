import streamlit as st


def set_page_config():
    if "sidebar_state" not in st.session_state:
        st.session_state["sidebar_state"] = "expanded"

    st.set_page_config(
        page_title="DocMind AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state=st.session_state["sidebar_state"],
        menu_items={
            "Get Help": "https://github.com/Ashita-no-Kaushar/DocMind-AI/discussions",
            "Report a bug": "https://github.com/Ashita-no-Kaushar/DocMind-AI/issues",
        },
    )

    # Remove the Streamlit `Deploy` button from the Header
    st.markdown(
        r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
