import streamlit as st

from utils.ollama import chat, context_chat, get_models, get_embedding_models


def chatbox():
    if prompt := st.chat_input("How can I help?"):
        if not st.session_state.get("selected_model"):
            # Try to auto-discover models before giving up
            try:
                models = get_models()
                if models:
                    st.session_state["ollama_models"] = models
                    st.session_state["selected_model"] = models[0]
                    get_embedding_models()
                    st.rerun()
            except Exception:
                pass
            if not st.session_state.get("selected_model"):
                st.warning(
                    "⚠️ No chat model available. Please go to **Settings → Chat** "
                    "and click **Refresh Models**, then select a model.",
                    icon=None,
                )
                return

        # Add the user input to messages state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate stream with user input (Context RAG vs General Chat)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.get("query_engine"):
                    stream = context_chat(
                        prompt=prompt,
                        query_engine=st.session_state["query_engine"],
                    )
                else:
                    stream = chat(prompt=prompt)

                response = st.write_stream(stream)

        # Add the final response to messages state
        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})

