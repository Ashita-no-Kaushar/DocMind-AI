import streamlit as st

from utils.ollama import chat, context_chat


def chatbox():
    if prompt := st.chat_input("How can I help?"):
        if not st.session_state.get("selected_model"):
            st.warning("Please ensure Ollama is running and select a model in Settings.")
            st.stop()

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
