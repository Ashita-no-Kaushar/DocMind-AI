import streamlit as st

from components.chatbox import chatbox
from components.header import set_page_header
from components.page_config import set_page_config
from components.page_state import set_initial_state
from components.retrieval_map_view import begin_render_pass, render_stored_route
from components.sidebar import sidebar
from utils.runtime_policy import enforce_runtime_bind_policy

try:
    _server_address = st.get_option("server.address")
except Exception:
    _server_address = None
enforce_runtime_bind_policy(_server_address)


### Page Setup
set_page_config()

### Setup Initial State
set_initial_state()

set_page_header()

begin_render_pass()

document_map = st.session_state.get("retrieval_map")
for index, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("role") == "assistant":
            render_stored_route(msg, document_map, key=f"stored-route-{index}")

### Sidebar
sidebar()

### Chat Box
chatbox()
