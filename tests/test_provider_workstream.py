import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage, MessageRole

from utils import ollama as ollama_module
from utils.endpoint_policy import (
    endpoint_identity,
    model_catalog_signature,
    normalize_provider_endpoint,
    validate_credential_endpoint,
)
from utils.llama_index import OllamaEmbedding, index_cache_key, setup_embedding_model
from utils.provider_config import (
    OLLAMA,
    OPENAI_COMPATIBLE,
    OPENAI_OFFICIAL,
    get_embedding_profile,
    initialize_provider_state,
    normalize_provider_kind,
)
from utils.source_state import effective_indexing_settings


class EndpointPolicyTests(unittest.TestCase):
    def test_normalizes_loopback_and_https_endpoints(self):
        self.assertEqual(
            normalize_provider_endpoint("http://localhost:1234/v1/"),
            "http://localhost:1234/v1",
        )
        self.assertEqual(
            normalize_provider_endpoint("https://models.example.test/v1"),
            "https://models.example.test/v1",
        )

    def test_rejects_unsafe_or_remote_http_endpoints(self):
        for value in (
            "file:///tmp/model",
            "javascript:alert(1)",
            "http://models.example.test/v1",
            "https://user:password@models.example.test/v1",
            "https://models.example.test/v1?api_key=secret",
            "https://models.example.test/v1#secret",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_provider_endpoint(value)

    def test_official_credentials_are_bound_to_official_host(self):
        with self.assertRaises(ValueError):
            validate_credential_endpoint(
                "official-key",
                "https://proxy.example.test/v1",
                OPENAI_OFFICIAL,
            )
        self.assertEqual(
            validate_credential_endpoint(
                "official-key",
                "https://api.openai.com/v1",
                OPENAI_OFFICIAL,
            ),
            "https://api.openai.com/v1",
        )

    def test_catalog_signature_changes_for_accounts(self):
        first = model_catalog_signature(
            "http://localhost:1234/v1", "account-a", OPENAI_COMPATIBLE
        )
        second = model_catalog_signature(
            "http://localhost:1234/v1", "account-b", OPENAI_COMPATIBLE
        )
        self.assertNotEqual(first, second)
        self.assertEqual(
            endpoint_identity("http://localhost:1234/v1"),
            endpoint_identity("http://localhost:1234/v1/"),
        )


class ProviderConfigurationTests(unittest.TestCase):
    def test_backend_normalization_has_three_provider_kinds(self):
        self.assertEqual(normalize_provider_kind("OpenAI"), OPENAI_OFFICIAL)
        self.assertEqual(
            normalize_provider_kind("LM Studio (Local AI)"), OPENAI_COMPATIBLE
        )
        self.assertEqual(normalize_provider_kind("TabbyAPI"), OPENAI_COMPATIBLE)
        self.assertEqual(
            normalize_provider_kind("my-openai-compatible-server"),
            OPENAI_COMPATIBLE,
        )
        self.assertEqual(normalize_provider_kind("Ollama"), OLLAMA)

    def test_local_provider_does_not_receive_official_key_during_migration(self):
        state = {
            "llm_backend": "LM Studio (Local AI)",
            "openai_base_url": "http://localhost:1234/v1",
            "openai_model": "legacy-chat",
            "openai_api_key": "official-secret",
        }
        initialize_provider_state(state)
        self.assertEqual(state["lm_studio_base_url"], "http://localhost:1234/v1")
        self.assertEqual(state["lm_studio_model"], "legacy-chat")
        self.assertEqual(state["lm_studio_api_key"], "")
        self.assertEqual(state["openai_api_key"], "official-secret")
        self.assertEqual(state["embedding_api_key"], "")

    def test_embedding_key_is_not_reused_for_another_provider(self):
        state = {
            "llm_backend": "LM Studio (Local AI)",
            "embedding_backend": "TabbyAPI",
            "embedding_base_url": "http://localhost:5000/v1",
            "embedding_model": "embedding-model",
            "embedding_api_key": "official-key",
            "embedding_api_key_provider": OPENAI_OFFICIAL,
        }
        initialize_provider_state(state)
        self.assertEqual(state["embedding_api_key"], "")

    def test_catalog_is_cleared_when_profile_endpoint_changes(self):
        state = {
            "llm_backend": "LM Studio (Local AI)",
            "lm_studio_base_url": "http://localhost:1235/v1",
            "lm_studio_api_key": "",
            "lm_studio_models": ["old-model"],
            "lm_studio_models_endpoint": "http://localhost:1234/v1",
            "lm_studio_models_provider": OPENAI_COMPATIBLE,
        }
        initialize_provider_state(state)
        self.assertEqual(state["lm_studio_models"], [])

    def test_official_endpoint_is_not_migrated_into_local_profile(self):
        state = {
            "llm_backend": "LM Studio (Local AI)",
            "openai_base_url": "https://api.openai.com/v1",
        }
        initialize_provider_state(state)
        self.assertEqual(state["lm_studio_base_url"], "http://localhost:1234/v1")

    def test_embedding_profile_is_independent_after_first_migration(self):
        state = {
            "llm_backend": "LM Studio (Local AI)",
            "lm_studio_base_url": "http://localhost:1234/v1",
            "lm_studio_model": "chat-model",
        }
        initialize_provider_state(state)
        state.update(
            {
                "embedding_backend": "TabbyAPI",
                "embedding_base_url": "http://localhost:5000/v1",
                "embedding_model": "embedding-model",
                "embedding_api_key": "embedding-key",
            }
        )
        state["lm_studio_model"] = "new-chat-model"
        initialize_provider_state(state)
        profile = get_embedding_profile(state)
        self.assertEqual(profile["base_url"], "http://localhost:5000/v1")
        self.assertEqual(profile["model"], "embedding-model")
        self.assertEqual(profile["api_key"], "embedding-key")

    def test_import_does_not_create_a_global_openai_key(self):
        script = (
            "import os; import utils.ollama; "
            "assert os.environ.get('OPENAI_API_KEY') is None"
        )
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env={
                key: value
                for key, value in os.environ.items()
                if key != "OPENAI_API_KEY"
            },
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_official_adapter_requires_explicit_key_even_with_environment(self):
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "environment-key"},
            ),
            self.assertRaisesRegex(ValueError, "explicit API key"),
        ):
            ollama_module.create_openai_llm(
                "gpt-4o-mini",
                "https://api.openai.com/v1",
                api_key="",
                backend="OpenAI",
            )

    def test_index_settings_include_embedding_profile_and_key_fingerprint(self):
        state = {
            "llm_backend": "LM Studio (Local AI)",
            "embedding_backend": "LM Studio (Local AI)",
            "embedding_base_url": "http://localhost:1234/v1",
            "embedding_model": "embedding-model",
            "embedding_api_key": "embedding-key",
            "chunk_size": 256,
            "chunk_overlap_pct": 12,
        }
        settings = effective_indexing_settings(state)
        self.assertEqual(settings["embedding_provider"], OPENAI_COMPATIBLE)
        self.assertEqual(settings["embedding_backend"], "LM Studio (Local AI)")
        self.assertEqual(settings["embedding_base_url"], "http://localhost:1234/v1")
        self.assertTrue(settings["embedding_credential_fingerprint"])
        self.assertNotIn("embedding-key", json.dumps(settings))

    def test_compatible_embedding_uses_configured_endpoint_and_model(self):
        from llama_index.core import Settings

        original = getattr(Settings, "_embed_model", None)
        try:
            with patch(
                "llama_index.embeddings.openai.OpenAIEmbedding",
                _EmbeddingSpy,
            ):
                setup_embedding_model(
                    "local-embedding-model",
                    backend="LM Studio (Local AI)",
                    base_url="http://localhost:1234/v1",
                    api_key="",
                )
            embedding = Settings.embed_model
            self.assertEqual(embedding.kwargs["model_name"], "local-embedding-model")
            self.assertEqual(embedding.kwargs["api_base"], "http://localhost:1234/v1")
            self.assertEqual(embedding.kwargs["api_key"], "local")
        finally:
            Settings.embed_model = original

    def test_browser_settings_never_persist_provider_keys(self):
        from utils.browser_settings import serialize_persisted_settings

        payload = serialize_persisted_settings(
            {
                "openai_api_key": "official",
                "lm_studio_api_key": "lm",
                "tabby_api_key": "tabby",
                "embedding_api_key": "embedding",
                "llm_backend": "OpenAI",
            }
        )
        self.assertEqual(payload, {"llm_backend": "OpenAI"})

    def test_chat_endpoint_change_clears_key_and_models(self):
        from components.tabs import settings as settings_tab

        state = {
            "llm_backend": "LM Studio (Local AI)",
            "lm_studio_base_url": "http://localhost:1235/v1",
            "lm_studio_api_key": "local-secret",
            "lm_studio_api_key_endpoint": "http://localhost:1234/v1",
            "lm_studio_models": ["old-model"],
        }
        with patch.object(settings_tab, "st", SimpleNamespace(session_state=state)):
            settings_tab._on_chat_url_change()
        self.assertEqual(state["lm_studio_api_key"], "")
        self.assertEqual(state["lm_studio_models"], [])
        self.assertEqual(state["lm_studio_base_url"], "http://localhost:1235/v1")


class _EmbeddingSpy(BaseEmbedding):
    def __init__(self, **kwargs):
        super().__init__()
        object.__setattr__(self, "kwargs", kwargs)

    def _get_query_embedding(self, query):
        return [0.0]

    async def _aget_query_embedding(self, query):
        return [0.0]

    def _get_text_embedding(self, text):
        return [0.0]


class _CompatibleHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def log_message(self, format_string, *args):
        return

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.__class__.requests.append(
            {"path": self.path, "authorization": self.headers.get("Authorization")}
        )
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "data": [
                        {"id": "chat-model"},
                        {"id": "text-embedding-3-small", "type": "embedding"},
                    ]
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.__class__.requests.append(
            {
                "path": self.path,
                "body": body,
                "authorization": self.headers.get("Authorization"),
            }
        )
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body.get("model"),
            "choices": [
                {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
            ],
        }
        data = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class OpenAICompatibleServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _CompatibleHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        _CompatibleHandler.requests = []

    def test_models_endpoint_is_fetched_once_and_classified(self):
        catalog = ollama_module.get_openai_model_catalog(
            self.base_url,
            "account-key",
            provider_kind=OPENAI_COMPATIBLE,
        )
        self.assertEqual(catalog["all"], ["chat-model", "text-embedding-3-small"])
        self.assertEqual(catalog["chat"], ["chat-model"])
        self.assertEqual(catalog["embedding"], ["text-embedding-3-small"])
        self.assertEqual(
            _CompatibleHandler.requests[0]["authorization"], "Bearer account-key"
        )

    def test_openai_like_streams_against_compatible_server(self):
        ollama_module._create_openai_llm_cached.clear()
        llm = ollama_module.create_openai_llm(
            "chat-model",
            self.base_url,
            temperature=0.2,
            backend="LM Studio (Local AI)",
        )
        responses = list(
            llm.stream_chat([ChatMessage(role=MessageRole.USER, content="hello")])
        )
        self.assertTrue(responses)
        self.assertEqual(_CompatibleHandler.requests[0]["path"], "/v1/chat/completions")
        body = _CompatibleHandler.requests[0]["body"]
        self.assertEqual(body["model"], "chat-model")
        self.assertEqual(body["temperature"], 0.2)
        self.assertTrue(body["stream"])


class _OllamaClient:
    instances: ClassVar[list["_OllamaClient"]] = []

    def __init__(self, host, timeout=300):
        self.host = host
        self.timeout = timeout
        self.list_calls = 0
        self.show_calls = []
        self.__class__.instances.append(self)

    def list(self):
        self.list_calls += 1
        return {
            "models": [
                {"model": "llama3:8b"},
                {"model": "broken-model"},
                {"model": "nomic-embed-text:latest"},
            ]
        }

    def show(self, model):
        self.show_calls.append(model)
        if model == "broken-model":
            raise RuntimeError("metadata unavailable")
        if model == "nomic-embed-text:latest":
            return {"capabilities": ["embedding"]}
        return {"capabilities": ["completion"]}


class OllamaDiscoveryTests(unittest.TestCase):
    def setUp(self):
        _OllamaClient.instances = []

    def test_discovery_reuses_list_and_isolates_metadata_failure(self):
        state = {"ollama_endpoint": "http://localhost:11434"}
        with (
            patch.object(ollama_module, "st", SimpleNamespace(session_state=state)),
            patch.object(
                ollama_module,
                "create_client",
                side_effect=lambda endpoint: _OllamaClient(endpoint),
            ),
        ):
            chat_models = ollama_module.get_models()
            embedding_models = ollama_module.get_embedding_models()
        client = _OllamaClient.instances[0]
        self.assertEqual(client.list_calls, 1)
        self.assertIn("llama3:8b", chat_models)
        self.assertIn("broken-model", chat_models)
        self.assertEqual(embedding_models, ["nomic-embed-text:latest"])

    def test_ollama_client_uses_explicit_timeout(self):
        with patch.object(
            ollama_module.ollama, "Client", return_value=object()
        ) as client:
            ollama_module.create_client("http://localhost:11434")
        client.assert_called_once_with(host="http://localhost:11434", timeout=300)

    def test_default_chat_model_is_used_for_recovery_order(self):
        state = {"ollama_endpoint": "http://localhost:11434"}
        with (
            patch.object(ollama_module, "st", SimpleNamespace(session_state=state)),
            patch.object(
                ollama_module,
                "create_client",
                side_effect=lambda endpoint: _OllamaClient(endpoint),
            ),
        ):
            models = ollama_module.get_models()
        self.assertEqual(models[0], "llama3:8b")


class _BatchClient:
    def __init__(self, error):
        self.error = error
        self.calls = []

    def embed(self, model, input):
        self.calls.append(len(input))
        raise RuntimeError(self.error)


class EmbeddingBatchTests(unittest.TestCase):
    def _embedding(self, client):
        embedding = OllamaEmbedding(
            model_name="embedding-model",
            base_url="http://localhost:11434",
        )
        embedding.embed_batch_size = 8
        embedding._client = client
        return embedding

    def test_auth_failure_is_not_retried_with_smaller_batches(self):
        client = _BatchClient("401 unauthorized")
        with self.assertRaises(RuntimeError):
            self._embedding(client).get_text_embedding_batch(["a", "b"])
        self.assertEqual(client.calls, [2])

    def test_timeout_is_not_retried_with_smaller_batches(self):
        client = _BatchClient("read timed out")
        with self.assertRaises(RuntimeError):
            self._embedding(client).get_text_embedding_batch(["a", "b"])
        self.assertEqual(client.calls, [2])


class CacheKeyTests(unittest.TestCase):
    def test_llm_cache_signature_keeps_temperature_but_not_system_prompt(self):
        import inspect

        parameters = inspect.signature(
            ollama_module._create_openai_llm_cached
        ).parameters
        self.assertIn("temperature", parameters)
        self.assertNotIn("system_prompt", parameters)

    def test_llm_cache_reuses_temperature_and_separates_values(self):
        ollama_module._create_openai_llm_cached.clear()
        first = ollama_module.create_openai_compatible_llm(
            "local-model",
            "http://localhost:1234/v1",
            temperature=0.2,
            backend="LM Studio (Local AI)",
        )
        same = ollama_module.create_openai_compatible_llm(
            "local-model",
            "http://localhost:1234/v1",
            temperature=0.2,
            backend="LM Studio (Local AI)",
        )
        different = ollama_module.create_openai_compatible_llm(
            "local-model",
            "http://localhost:1234/v1",
            temperature=0.8,
            backend="LM Studio (Local AI)",
        )
        self.assertIs(first, same)
        self.assertIsNot(first, different)
        ollama_module._create_openai_llm_cached.clear()
        ollama_module._create_ollama_llm_cached.clear()
        first_ollama = ollama_module.create_ollama_llm(
            "qwen2.5:0.5b",
            "http://localhost:11434",
            system_prompt="first style",
            temperature=0.2,
        )
        second_ollama = ollama_module.create_ollama_llm(
            "qwen2.5:0.5b",
            "http://localhost:11434",
            system_prompt="second style",
            temperature=0.2,
        )
        self.assertIs(first_ollama, second_ollama)
        ollama_module._create_ollama_llm_cached.clear()

    def test_style_and_temperature_session_values_do_not_change_explicit_index_key(
        self,
    ):
        settings = {
            "chunk_size": 256,
            "chunk_overlap": 30,
            "embedding_provider": OPENAI_COMPATIBLE,
            "embedding_model": "embedding-model",
            "embedding_endpoint": "http://localhost:1234/v1",
        }
        state = {"system_prompt": "first", "temperature": 0.2}
        with patch("utils.llama_index.st", SimpleNamespace(session_state=state)):
            first = index_cache_key(
                [SimpleNamespace(metadata={}, text="document text")], settings=settings
            )
            state.update(system_prompt="second", temperature=0.8)
            second = index_cache_key(
                [SimpleNamespace(metadata={}, text="document text")], settings=settings
            )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
