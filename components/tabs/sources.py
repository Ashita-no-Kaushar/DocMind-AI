import streamlit as st

from components.tabs.local_files import local_files
from components.tabs.github_repo import github_repo
from components.tabs.website import website
from components.ingestion_prerequisites import (
    ingestion_is_configured,
    render_ingestion_settings_warning,
)


def sources():
    st.title("Directly import your data")
    st.caption("Convert your data into embeddings for utilization during chat")
    st.write("")

    if not ingestion_is_configured():
        render_ingestion_settings_warning()
        st.write("")

    with st.expander("💻 &nbsp; **Local Files**", expanded=False):
        local_files()

    with st.expander("🗂️ &nbsp;**GitHub Repo**", expanded=False):
        github_repo()

    with st.expander("🌐 &nbsp; **Website**", expanded=False):
        website()

    with st.expander("💡 &nbsp; **Cooling & Speed Tips**", expanded=False):
        st.markdown(
            "- Ingestion time depends on document size, parser, embedding model, and hardware.\n"
            "- Reuse the active index when the files and indexing settings are unchanged.\n"
            "- Changing the embedding model or chunk settings requires a new ingestion to affect the index.\n"
            "- Split large documents and process only the material you need.\n"
            "- Eco Mode lowers configured embedding batch, output, and context limits; it does not guarantee a speed or temperature change.\n"
            "- Website and GitHub ingestion require outbound network access."
        )
