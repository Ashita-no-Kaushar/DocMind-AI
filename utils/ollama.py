import ollama
import os
import time

import streamlit as st

import utils.logs as logs

# OpenAI API Key placeholder required by llama-index
os.environ["OPENAI_API_KEY"] = "sk-abc123"

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from utils.llama_index import TEXT_QA_TEMPLATE

# Model context budget (tokens) reserved for chat history, leaving room for
# the current prompt and the model's reply. Keep it below num_ctx (2048) so
# the KV cache stays small: less GPU compute per token = less heat.
CHAT_HISTORY_TOKEN_BUDGET = 1200


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token) for history trimming."""
    return max(1, len(text or "") // 4)


def _trim_history(messages, budget_tokens: int = CHAT_HISTORY_TOKEN_BUDGET):
    """Keep the most recent messages that fit within a token budget."""
    recent = []
    used = 0
    for message in reversed(messages):
        estimated = _estimate_tokens(message.content)
        if recent and used + estimated > budget_tokens:
            break
        recent.append(message)
        used += estimated
    recent.reverse()
    return recent

###################################
#
# Create Client
#
###################################


def create_client(host: str):
    """
    Creates a client for interacting with the Ollama API.

    Parameters:
        - host (str): The hostname or IP address of the Ollama server.

    Returns:
        - An instance of the Ollama client.

    Raises:
        - Exception: If there is an error creating the client.

    Notes:
        This function creates a client for interacting with the Ollama API using the `ollama` library. It takes a single parameter, `host`, which should be the hostname or IP address of the Ollama server. The function returns an instance of the Ollama client, or raises an exception if there is an error creating the client.
    """
    try:
        client = ollama.Client(host=host)
        logs.log.info("Ollama chat client created successfully")
        return client
    except Exception as err:
        logs.log.error(f"Failed to create Ollama client: {err}")
        return False


###################################
#
# Get Models
#
###################################


def _get_installed_model_names(chat_client):
    data = chat_client.list()
    models = []
    for model in data["models"]:
        try:
            model_name = model.get("model") or model.get("name")
        except AttributeError:
            model_name = getattr(model, "model", None) or getattr(model, "name", None)

        if model_name:
            models.append(model_name)
    return models


def default_embedding_model(models):
    """Return the preferred default embedding model from discovered Ollama models."""
    preferred_models = ("embeddinggemma:latest",)

    for model in preferred_models:
        if model in models:
            return model

    if models:
        return models[0]

    return None


def get_models():
    """Return installed Ollama models that declare completion capability."""
    try:
        chat_client = create_client(st.session_state["ollama_endpoint"])
        models = []
        for model_name in _get_installed_model_names(chat_client):
            details = chat_client.show(model_name)
            capabilities = getattr(details, "capabilities", None) or details.get("capabilities", [])
            if "completion" in capabilities:
                models.append(model_name)

        st.session_state["ollama_models"] = models

        if len(models) > 0:
            logs.log.info("Ollama chat models loaded successfully")
        else:
            logs.log.warning("Ollama did not return any chat-capable models")

        return models
    except Exception as err:
        logs.log.error(f"Failed to retrieve Ollama model list: {err}")
        st.session_state["ollama_models"] = []
        return []


def get_embedding_models():
    """Return installed Ollama models that declare embedding capability."""
    try:
        chat_client = create_client(st.session_state["ollama_endpoint"])
        embedding_models = []

        for model_name in _get_installed_model_names(chat_client):
            details = chat_client.show(model_name)
            capabilities = getattr(details, "capabilities", None) or details.get("capabilities", [])
            if "embedding" in capabilities:
                embedding_models.append(model_name)

        st.session_state["ollama_embedding_models"] = embedding_models

        if embedding_models:
            if st.session_state.get("ollama_embedding_model") not in embedding_models:
                st.session_state["ollama_embedding_model"] = default_embedding_model(
                    embedding_models
                )
            logs.log.info("Ollama embedding models loaded successfully")
        else:
            logs.log.warning("Ollama did not return any embedding-capable models")

        return embedding_models
    except Exception as err:
        logs.log.error(f"Failed to retrieve Ollama embedding model list: {err}")
        st.session_state["ollama_embedding_models"] = []
        return []


###################################
#
# Create Ollama LLM instance
#
###################################


@st.cache_resource(show_spinner=False)
def create_ollama_llm(model: str, base_url: str, system_prompt: str = None, request_timeout: int = 300) -> Ollama:
    """
    Create an instance of the Ollama language model.

    Parameters:
        - model (str): The name of the model to use for language processing.
        - base_url (str): The base URL for making API requests.
        - system_prompt (str, optional): Kept for call-site compatibility. The
            installed Ollama wrapper does not support a system_prompt field, so
            the system message is injected into the chat history instead.
        - request_timeout (int, optional): The timeout for API requests in seconds. Defaults to 300.

    Returns:
        - llm: An instance of the Ollama language model with the specified configuration.
    """
    try:
        Settings.llm = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout,
            # Keep num_ctx modest: RAG context + chat history + answer all fit
            # in 2048 tokens. Smaller KV cache = less GPU compute = less heat.
            context_window=2048,
            # Moderate temperature: focused answers without being robotic.
            temperature=0.4,
            # Unload models after 2 minutes idle: no wasted VRAM/heat when the
            # app sits unused.
            keep_alive="2m",
            # Cap output length: answers stop at ~512 tokens (~400 words),
            # which keeps the model from rambling on and generating heat.
            additional_kwargs={"num_predict": 512},
        )
        logs.log.info("Ollama LLM instance created successfully")
        return Settings.llm
    except Exception as e:
        logs.log.error(f"Error creating Ollama language model: {e}")
        raise


###################################
#
# Chat (no context)
#
###################################


def chat(prompt: str):
    """
    Initiates a chat with the Ollama language model using multi-turn conversational history.

    Parameters:
        - prompt (str): The starting prompt for the conversation.

    Yields:
        - str: Successive chunks of conversation from the Ollama model.
    """
    try:
        llm = create_ollama_llm(
            st.session_state["selected_model"],
            st.session_state["ollama_endpoint"],
        )

        chat_messages = []
        system_prompt = st.session_state.get("system_prompt")
        if system_prompt:
            chat_messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )
        for msg in st.session_state.get("messages", []):
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            if role_str == "assistant":
                chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
            elif role_str == "user":
                chat_messages.append(ChatMessage(role=MessageRole.USER, content=content))
            elif role_str == "system":
                chat_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=content))

        # Trim the conversational history (not the system message) so it fits
        # comfortably inside the model's context window. Oversized histories
        # silently truncate and degrade response quality.
        history = _trim_history(chat_messages[1:]) if system_prompt else _trim_history(chat_messages)
        if not history or history[-1].content != prompt:
            history.append(ChatMessage(role=MessageRole.USER, content=prompt))

        recent_messages = ([chat_messages[0]] + history) if system_prompt else history
        stream = llm.stream_chat(recent_messages)
        for chunk in stream:
            yield chunk.delta
    except Exception as err:
        logs.log.error(f"Ollama chat stream error: {err}")
        yield f"⚠️ **Error during chat:** {err}. Please ensure Ollama is running and model '{st.session_state.get('selected_model')}' is installed."
        return


###################################
#
# Document Chat (with context)
#
###################################


def context_chat(prompt: str, query_engine: RetrieverQueryEngine):
    """
    Initiates a chat with context using the Llama-Index query_engine.

    Retrieves the most relevant document chunks, then streams the grounded
    answer token-by-token. llama-index's compact synthesizer buffers the full
    response (no real streaming), so we bypass it: retrieve + stream_chat.

    Parameters:
        - prompt (str): The starting prompt for the conversation.
        - query_engine (RetrieverQueryEngine): The Llama-Index query engine.

    Yields:
        - str: Successive chunks of conversation from the Llama-Index model with context.

    Raises:
        - Exception: If there is an error retrieving answers from the Llama-Index model.
    """

    try:
        llm = create_ollama_llm(
            st.session_state["selected_model"],
            st.session_state["ollama_endpoint"],
        )

        retriever = st.session_state.get("retriever")
        if retriever is None:
            retriever = getattr(query_engine, "_retriever", None)
        if retriever is None:
            yield "⚠️ **No retriever available.** Please re-ingest your documents."
            return

        t0 = time.time()
        nodes = retriever.retrieve(prompt)
        if not nodes:
            st.session_state["last_doc_sources"] = []
            yield "I could not find this information in the documents."
            return

        # Number the context chunks [1], [2], ... and remember their source
        # files so the UI can show citations under the answer.
        numbered_context = []
        sources = []
        for index, node_score in enumerate(nodes, start=1):
            metadata = node_score.node.metadata or {}
            file_name = metadata.get(
                "file_name", metadata.get("source", "document")
            )
            numbered_context.append(
                f"[{index}]:\n{node_score.node.get_content()}"
            )
            sources.append((file_name, node_score.score))
        st.session_state["last_doc_sources"] = sources

        context = "\n\n".join(numbered_context)

        # System message first, then the grounded QA template with the
        # retrieved context. Using stream_chat yields deltas as they arrive.
        messages = []
        system_prompt = st.session_state.get("system_prompt")
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=TEXT_QA_TEMPLATE.format(context_str=context, query_str=prompt),
            )
        )

        logs.log.info(
            f"Doc query: {len(nodes)} chunks | top score {nodes[0].score:.3f}"
        )

        stream = llm.stream_chat(messages)
        for chunk in stream:
            yield chunk.delta
        logs.log.info(f"Doc query answered in {time.time() - t0:.1f}s")
    except Exception as err:
        logs.log.error(f"Ollama chat stream error: {err}")
        yield f"⚠️ **Error generating response:** {err}. If the model is taking longer to respond on CPU, please try again."
        return
