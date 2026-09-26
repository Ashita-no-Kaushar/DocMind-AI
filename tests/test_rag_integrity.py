import re
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode

from components import chatbox
from utils import ollama as ollama_module
from utils.llama_index import (
    HybridRetriever,
    build_retrieval_query,
    retrieval_query_candidates,
)


class _Docstore:
    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}

    def get_node(self, node_id):
        return self.nodes[node_id]


class _VectorRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.results)


class ConversationQueryTests(unittest.TestCase):
    def test_followup_uses_recent_user_turns_without_assistant_text(self):
        history = [
            {"role": "user", "content": "What is the refund policy?"},
            {"role": "assistant", "content": "ASSISTANT_HALLUCINATION"},
            {"role": "user", "content": "What about electronics?"},
        ]

        expanded = build_retrieval_query("What about electronics?", history)

        self.assertIn("refund policy", expanded)
        self.assertIn("electronics", expanded)
        self.assertNotIn("ASSISTANT_HALLUCINATION", expanded)
        self.assertEqual(
            retrieval_query_candidates("What about electronics?", history),
            ["What about electronics?", expanded],
        )

    def test_standalone_query_is_not_expanded(self):
        history = [{"role": "user", "content": "Tell me about the old document"}]
        query = "What is the capital of France?"

        self.assertEqual(build_retrieval_query(query, history), query)
        self.assertEqual(retrieval_query_candidates(query, history), [query])
        self.assertEqual(
            build_retrieval_query("Who is Ada?", history),
            "Who is Ada?",
        )


class HybridPoolTests(unittest.TestCase):
    def _retriever(self, nodes, vector_results, **kwargs):
        return HybridRetriever(
            _VectorRetriever(vector_results),
            _Docstore(nodes),
            [node.node_id for node in nodes],
            top_k=kwargs.pop("top_k", 3),
            similarity_cutoff=kwargs.pop("similarity_cutoff", 0.3),
            **kwargs,
        )

    def test_strong_bm25_only_candidate_is_kept(self):
        target = TextNode(
            text="The release code is ZX-9981 and the owner is Ada.",
            metadata={"file_name": "release.txt"},
        )
        unrelated = TextNode(
            text="Quarterly operating notes for the finance team.",
            metadata={"file_name": "finance.txt"},
        )
        retriever = self._retriever(
            [target, unrelated],
            [NodeWithScore(node=unrelated, score=0.2)],
            vector_candidate_depth=1,
            bm25_candidate_depth=2,
        )

        with patch("utils.llama_index.st", SimpleNamespace(session_state={})):
            results = retriever.retrieve("ZX-9981")

        self.assertEqual([result.node.node_id for result in results], [target.node_id])

    def test_weak_and_unmatched_candidates_are_filtered(self):
        first = TextNode(
            text="A short unrelated sentence.", metadata={"file_name": "a.txt"}
        )
        second = TextNode(
            text="Another unrelated sentence.", metadata={"file_name": "b.txt"}
        )
        retriever = self._retriever(
            [first, second],
            [
                NodeWithScore(node=first, score=0.46),
                NodeWithScore(node=second, score=0.44),
            ],
        )

        with patch("utils.llama_index.st", SimpleNamespace(session_state={})):
            self.assertEqual(retriever.retrieve("quantum physics equations"), [])

    def test_candidate_depth_is_bounded_and_separate(self):
        nodes = [
            TextNode(
                text=f"keyword record {index} unique{index} evidence value {index * 11}",
                metadata={"file_name": f"{index}.txt"},
            )
            for index in range(6)
        ]
        vector = _VectorRetriever(
            [NodeWithScore(node=node, score=0.8) for node in nodes]
        )
        retriever = HybridRetriever(
            vector,
            _Docstore(nodes),
            [node.node_id for node in nodes],
            top_k=6,
            similarity_cutoff=0.3,
            vector_candidate_depth=2,
            bm25_candidate_depth=1,
        )

        with patch("utils.llama_index.st", SimpleNamespace(session_state={})):
            results = retriever.retrieve("keyword record 5 unique5 evidence value 55")

        self.assertEqual(retriever.vector_candidate_depth, 2)
        self.assertEqual(retriever.bm25_candidate_depth, 1)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(vector.queries), 1)
        self.assertIn("keyword", vector.queries[0])

    def test_candidate_depth_is_clamped(self):
        node = TextNode(text="bounded evidence", metadata={"file_name": "a.txt"})
        retriever = self._retriever(
            [node], [NodeWithScore(node=node, score=0.8)], vector_candidate_depth=999
        )
        with patch("utils.llama_index.st", SimpleNamespace(session_state={})):
            retriever.retrieve("bounded evidence")
        self.assertLessEqual(retriever.vector_candidate_depth, 50)


class RagBudgetAndCitationTests(unittest.TestCase):
    def test_unified_budget_counts_all_rag_input_parts(self):
        def tokenizer(value):
            return re.findall(r"\w+|[^\w\s]", value)

        history = [
            ChatMessage(role=MessageRole.USER, content="old question " * 100),
            ChatMessage(role=MessageRole.ASSISTANT, content="old answer " * 100),
        ]
        chunks = [f"whole evidence chunk {index} " * 30 for index in range(5)]

        plan = ollama_module.plan_rag_prompt(
            "current question",
            chunks,
            history,
            "system instruction " * 30,
            output_tokens=256,
            safety_tokens=32,
            tokenizer=tokenizer,
        )

        self.assertLessEqual(plan["input_tokens"], plan["input_budget"])
        self.assertTrue(plan["selected_chunks"])
        self.assertTrue(all(chunk in chunks for chunk in plan["selected_chunks"]))

    def test_invalid_citations_are_removed_and_valid_ones_remain(self):
        text = "Supported (from [1]); invalid (from [3]); malformed [x]; mixed [1, 9]."

        cleaned = ollama_module.sanitize_citations(text, 1)

        self.assertIn("(from [1])", cleaned)
        self.assertNotIn("[3]", cleaned)
        self.assertNotIn("[x]", cleaned)
        self.assertIn("[1]", cleaned)

    def test_no_citation_is_added_when_sources_exist(self):
        node = TextNode(
            text="A fact from the document.", metadata={"file_name": "a.txt"}
        )
        retriever = SimpleNamespace(
            retrieve=lambda query: [NodeWithScore(node=node, score=0.9)]
        )
        state = {
            "messages": [{"role": "user", "content": "What is the fact?"}],
            "retriever": retriever,
            "query_engine": SimpleNamespace(_retriever=retriever),
            "system_prompt": "system",
            "selected_model": "model",
            "ollama_endpoint": "http://localhost:11434",
            "llm_backend": "Ollama",
            "top_k": 3,
            "similarity_cutoff": 0.3,
            "eco_mode": False,
        }
        llm = SimpleNamespace(
            stream_chat=lambda messages: iter(
                [SimpleNamespace(delta="A grounded answer.")]
            )
        )

        with (
            patch.object(ollama_module.st, "session_state", state),
            patch.object(ollama_module, "create_llm", return_value=llm),
            patch.object(ollama_module, "_active_chat_model", return_value="model"),
            patch.object(
                ollama_module, "_active_base_url", return_value="http://localhost:11434"
            ),
            patch.object(ollama_module, "_active_api_key", return_value=""),
        ):
            answer = "".join(
                ollama_module.context_chat("What is the fact?", state["query_engine"])
            )

        self.assertEqual(answer, "A grounded answer.")
        self.assertNotIn("[1]", answer)
        self.assertEqual(state["last_rag_evidence"][0]["citation_index"], 1)
        self.assertEqual(state["last_rag_evidence"][0]["source"], "a.txt")

    def test_context_retries_expanded_followup_without_assistant_text(self):
        node = TextNode(
            text="Electronics are covered for 15 days.",
            metadata={"file_name": "policy.txt"},
        )

        class FollowupRetriever:
            def __init__(self):
                self.queries = []

            def retrieve(self, query):
                self.queries.append(query)
                if len(self.queries) == 1:
                    return []
                return [NodeWithScore(node=node, score=0.8)]

        retriever = FollowupRetriever()
        state = {
            "messages": [
                {"role": "user", "content": "What is the refund policy?"},
                {"role": "assistant", "content": "UNTRUSTED_ASSISTANT_TEXT"},
                {"role": "user", "content": "What about electronics?"},
            ],
            "retriever": retriever,
            "query_engine": SimpleNamespace(_retriever=retriever),
            "system_prompt": "system",
            "selected_model": "model",
            "ollama_endpoint": "http://localhost:11434",
            "llm_backend": "Ollama",
            "top_k": 3,
            "similarity_cutoff": 0.3,
            "eco_mode": False,
        }
        llm = SimpleNamespace(
            stream_chat=lambda messages: iter([SimpleNamespace(delta="ok")])
        )

        with (
            patch.object(ollama_module.st, "session_state", state),
            patch.object(ollama_module, "create_llm", return_value=llm),
            patch.object(ollama_module, "_active_chat_model", return_value="model"),
            patch.object(
                ollama_module, "_active_base_url", return_value="http://localhost:11434"
            ),
            patch.object(ollama_module, "_active_api_key", return_value=""),
        ):
            list(
                ollama_module.context_chat(
                    "What about electronics?", state["query_engine"]
                )
            )

        self.assertEqual(len(retriever.queries), 2)
        self.assertIn("refund policy", retriever.queries[1])
        self.assertNotIn("UNTRUSTED_ASSISTANT_TEXT", retriever.queries[1])

    def test_no_result_clears_stale_evidence(self):
        retriever = SimpleNamespace(retrieve=lambda query: [])
        state = {
            "messages": [],
            "retriever": retriever,
            "query_engine": SimpleNamespace(_retriever=retriever),
            "system_prompt": "",
            "last_rag_evidence": [{"stale": True}],
            "last_doc_sources": [("stale.txt", 0.9)],
            "selected_model": "model",
            "ollama_endpoint": "http://localhost:11434",
            "llm_backend": "Ollama",
            "top_k": 3,
            "similarity_cutoff": 0.3,
            "eco_mode": False,
        }
        llm = SimpleNamespace(stream_chat=lambda messages: iter(()))
        with (
            patch.object(ollama_module.st, "session_state", state),
            patch.object(ollama_module, "create_llm", return_value=llm),
            patch.object(ollama_module, "_active_chat_model", return_value="model"),
            patch.object(
                ollama_module, "_active_base_url", return_value="http://localhost:11434"
            ),
            patch.object(ollama_module, "_active_api_key", return_value=""),
        ):
            answer = "".join(
                ollama_module.context_chat("unknown", state["query_engine"])
            )

        self.assertIn("could not find", answer.lower())
        self.assertEqual(state["last_rag_evidence"], [])
        self.assertEqual(state["last_doc_sources"], [])


class _ChatStreamlit:
    def __init__(self, state):
        self.session_state = state
        self.captions = []

    def chat_message(self, *args, **kwargs):
        return nullcontext()

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def markdown(self, *args, **kwargs):
        pass

    def caption(self, value, *args, **kwargs):
        self.captions.append(str(value))

    def write_stream(self, stream):
        return "".join(str(item) for item in stream)

    def warning(self, *args, **kwargs):
        pass

    def button(self, *args, **kwargs):
        return False


class MessageEvidenceTests(unittest.TestCase):
    def test_local_turn_attaches_evidence_and_renders_from_snapshot(self):
        state = {
            "messages": [],
            "selected_model": "model",
            "llm_backend": "Ollama",
            "query_engine": object(),
        }
        streamlit = _ChatStreamlit(state)
        evidence = [
            {
                "citation_index": 1,
                "source": "notes.txt",
                "score": 0.8,
                "excerpt": "exact excerpt",
            }
        ]

        def fake_context_chat(
            prompt, query_engine, evidence_sink=None, route_sink=None
        ):
            evidence_sink.extend(evidence)
            if isinstance(route_sink, dict):
                route_sink.update({"planner_mode": "deterministic"})
            yield "answer"

        with (
            patch.object(chatbox, "st", streamlit),
            patch.object(chatbox.r2r, "r2r_is_ready", return_value=False),
            patch.object(chatbox, "is_openai_compatible_backend", return_value=False),
            patch.object(chatbox, "_local_index_ready", return_value=True),
            patch.object(chatbox, "context_chat", side_effect=fake_context_chat),
        ):
            chatbox._process_prompt("question")

        assistant = state["messages"][-1]
        self.assertEqual(assistant["evidence"], evidence)
        self.assertTrue(any("notes.txt" in caption for caption in streamlit.captions))


if __name__ == "__main__":
    unittest.main()
