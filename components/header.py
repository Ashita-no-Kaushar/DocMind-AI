import streamlit as st


def set_page_header():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
            <span style="font-size:1.65rem;">🧠</span>
            <span style="font-size:1.45rem; font-weight:700; letter-spacing:-0.02em;">DocMind AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Local document chat with Ollama; external providers send data to the server you configure."
    )
