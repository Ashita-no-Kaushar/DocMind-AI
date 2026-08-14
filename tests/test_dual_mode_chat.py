import unittest
from unittest.mock import MagicMock, patch

from utils.llama_index import EXCLUDED_FILE_PATTERNS, load_documents
from components.sidebar import clear_index_and_chat


class DualModeChatTests(unittest.TestCase):
    def test_excluded_file_patterns_contain_common_binaries(self):
        self.assertIn("*.png", EXCLUDED_FILE_PATTERNS)
        self.assertIn("*.exe", EXCLUDED_FILE_PATTERNS)
        self.assertIn("*.zip", EXCLUDED_FILE_PATTERNS)
        self.assertIn("*.pyc", EXCLUDED_FILE_PATTERNS)
        self.assertIn("*.mp4", EXCLUDED_FILE_PATTERNS)

    def test_clear_index_and_chat_resets_session_state(self):
        state = {
            "query_engine": MagicMock(),
            "file_list": ["doc1.pdf", "doc2.txt"],
            "processed_file_signature": "sig123",
            "processed_github_repo": "owner/repo",
            "processed_website_urls": ["https://example.com"],
            "github_ingestion_stages": ["Stage 1"],
            "website_ingestion_stages": ["Stage 1"],
            "messages": [{"role": "user", "content": "hello"}],
        }
        with patch("components.sidebar.st.session_state", state):
            clear_index_and_chat()
            self.assertIsNone(state["query_engine"])
            self.assertEqual(state["file_list"], [])
            self.assertIsNone(state["processed_file_signature"])
            self.assertIsNone(state["processed_github_repo"])
            self.assertIsNone(state["processed_website_urls"])
            self.assertEqual(state["github_ingestion_stages"], [])
            self.assertEqual(len(state["messages"]), 1)
            self.assertEqual(state["messages"][0]["role"], "assistant")
            self.assertIn("DocMind AI", state["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
