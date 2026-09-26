import streamlit as st

from components.status import render_status_strip, resolve_status


def set_page_header():
    st.markdown(
        """
        <div class="dm-brand">
            <span class="dm-brand-mark">DocMind</span>
            <span class="dm-brand-sub">agentic map retrieval</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_status_strip(resolve_status())
