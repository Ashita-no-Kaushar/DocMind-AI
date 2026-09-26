import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from components.tabs.local_files import supported_files
from utils import helpers
from utils.llama_index import _safe_input_files
from utils.rag_pipeline import (
    MAX_INGESTED_DOCUMENTS,
    MAX_INGESTED_TEXT_CHARS,
    validate_ingested_documents,
)


class FakeUpload:
    def __init__(self, name, payload):
        self.name = name
        self.size = len(payload)
        self._payload = payload

    def getbuffer(self):
        return self._payload


class UploadSafetyTests(unittest.TestCase):
    def test_safe_upload_destination_stays_inside_save_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = helpers.upload_destination(tmpdir, "notes.txt")

        self.assertEqual(destination.name, "notes.txt")

    def test_upload_extension_whitelist_accepts_new_document_formats(self):
        for name in (
            "report.odt",
            "page.html",
            "data.xlsx",
            "notes.rtf",
            "mail.eml",
            "book.epub",
            "archive.mbox",
            "sheet.tsv",
        ):
            with self.subTest(name=name):
                self.assertEqual(helpers.safe_uploaded_filename(name), name)

    def test_documented_upload_format_set_has_25_unique_extensions(self):
        expected = {
            "csv",
            "doc",
            "docx",
            "eml",
            "epub",
            "htm",
            "html",
            "ipynb",
            "json",
            "jsonl",
            "markdown",
            "mbox",
            "md",
            "mhtml",
            "msg",
            "odt",
            "pdf",
            "ppt",
            "pptx",
            "rtf",
            "tsv",
            "txt",
            "xls",
            "xlsx",
            "xml",
        }

        self.assertEqual(set(supported_files), expected)
        self.assertEqual(len(supported_files), 25)
        self.assertEqual(
            helpers.ALLOWED_UPLOAD_EXTENSIONS, {f".{ext}" for ext in expected}
        )

    def test_upload_extension_whitelist_still_blocks_executables(self):
        for name in ("payload.exe", "virus.bat", "lib.so", "archive.zip"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                helpers.safe_uploaded_filename(name)

    def test_upload_names_reject_case_insensitive_duplicates(self):
        uploads = [FakeUpload("Report.txt", b"a"), FakeUpload("report.TXT", b"b")]
        with self.assertRaisesRegex(ValueError, "unique"):
            helpers.validate_uploaded_files(uploads)

    def test_upload_names_reject_windows_device_basenames(self):
        for name in ("CON.txt", "nul.json", "COM1.md", "lpt9.csv"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "reserved"),
            ):
                helpers.safe_uploaded_filename(name)

    def test_upload_destination_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaises(ValueError):
            helpers.upload_destination(tmpdir, "../notes.txt")

    def test_upload_destination_rejects_backslash_separator(self):
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaises(ValueError):
            helpers.upload_destination(tmpdir, "nested\\notes.txt")

    def test_validate_uploaded_files_enforces_count_limit(self):
        uploads = [
            FakeUpload(f"file-{index}.txt", b"x")
            for index in range(helpers.MAX_UPLOAD_FILES + 1)
        ]

        with self.assertRaises(ValueError):
            helpers.validate_uploaded_files(uploads)

    def test_validate_uploaded_files_enforces_size_limit(self):
        upload = FakeUpload("large.txt", b"x" * (helpers.MAX_UPLOAD_FILE_BYTES + 1))

        with self.assertRaises(ValueError):
            helpers.validate_uploaded_files([upload])

    def test_empty_input_selection_does_not_scan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "should-not-be-loaded.txt").write_text("sentinel", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No files were selected"):
                from utils.llama_index import load_documents

                load_documents(str(root), input_files=[])

    def test_directory_loader_rejects_symlinks_and_hidden_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_file = root / "safe.txt"
            linked_file = root / "linked.txt"
            hidden_file = root / ".env"
            for path in (safe_file, linked_file, hidden_file):
                path.write_text("fixture", encoding="utf-8")

            original_is_symlink = Path.is_symlink

            def is_symlink(path):
                return path.name == "linked.txt" or original_is_symlink(path)

            with patch.object(Path, "is_symlink", is_symlink):
                selected = _safe_input_files(tmpdir)

        self.assertEqual(selected, [str(safe_file.resolve())])


class WebsiteValidationTests(unittest.TestCase):
    def test_validate_website_urls_requires_https(self):
        with self.assertRaises(ValueError):
            helpers.validate_website_urls(["http://example.com"])

    def test_validate_website_urls_blocks_loopback(self):
        with self.assertRaises(ValueError):
            helpers.validate_website_urls(["https://127.0.0.1"])

    def test_validate_website_urls_blocks_metadata_hostname(self):
        with self.assertRaises(ValueError):
            helpers.validate_website_urls(["https://metadata.google.internal"])

    def test_validate_website_urls_enforces_count_limit(self):
        urls = [
            f"https://example-{index}.com"
            for index in range(helpers.MAX_WEBSITE_URLS + 1)
        ]

        with self.assertRaises(ValueError):
            helpers.validate_website_urls(urls)


class GitHubRepoValidationTests(unittest.TestCase):
    def test_normalize_github_repo_accepts_owner_repo(self):
        self.assertEqual(
            helpers.normalize_github_repo("TNTwise/REAL-Video-Enhancer"),
            "TNTwise/REAL-Video-Enhancer",
        )

    def test_normalize_github_repo_accepts_full_github_url(self):
        self.assertEqual(
            helpers.normalize_github_repo(
                "https://github.com/TNTwise/REAL-Video-Enhancer"
            ),
            "TNTwise/REAL-Video-Enhancer",
        )

    def test_normalize_github_repo_strips_git_suffix(self):
        self.assertEqual(
            helpers.normalize_github_repo(
                "https://github.com/TNTwise/REAL-Video-Enhancer.git"
            ),
            "TNTwise/REAL-Video-Enhancer",
        )

    def test_normalize_github_repo_rejects_non_github_url(self):
        with self.assertRaises(ValueError):
            helpers.normalize_github_repo("https://example.com/TNTwise/repo")

    def test_normalize_github_repo_rejects_extra_path_segments(self):
        with self.assertRaises(ValueError):
            helpers.normalize_github_repo(
                "https://github.com/TNTwise/REAL-Video-Enhancer/issues"
            )

    def test_normalize_github_repo_rejects_dot_segments(self):
        for value in ("../repo", "owner/..", "https://github.com/../repo"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                helpers.normalize_github_repo(value)

    def test_clone_github_repo_accepts_isolated_destination_base(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(helpers.os, "getcwd", return_value=tmpdir),
            patch.object(helpers.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = Mock(returncode=0)
            destination = helpers.clone_github_repo(
                "owner/repo",
                destination_base=Path(tmpdir) / "work" / "operation",
            )
        self.assertEqual(
            destination,
            str(Path(tmpdir) / "work" / "operation" / "owner" / "repo"),
        )

    def test_clone_github_repo_returns_scoped_repo_directory(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(helpers.os, "getcwd", return_value=tmpdir),
            patch.object(helpers.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = Mock(returncode=0)

            destination = helpers.clone_github_repo("owner/repo")

        self.assertEqual(destination, str(Path(tmpdir) / "data" / "owner" / "repo"))


class IngestionLimitTests(unittest.TestCase):
    def test_validate_ingested_documents_enforces_actual_document_limit(self):
        documents = ["x"] * (MAX_INGESTED_DOCUMENTS + 1)

        with self.assertRaises(ValueError):
            validate_ingested_documents(documents)

    def test_validate_ingested_documents_accepts_document_limit(self):
        documents = ["x"] * MAX_INGESTED_DOCUMENTS

        self.assertIsNone(validate_ingested_documents(documents))

    def test_validate_ingested_documents_enforces_actual_text_limit(self):
        documents = ["x" * (MAX_INGESTED_TEXT_CHARS + 1)]

        with self.assertRaises(ValueError):
            validate_ingested_documents(documents)

    def test_validate_ingested_documents_accepts_text_limit(self):
        documents = ["x" * MAX_INGESTED_TEXT_CHARS]

        self.assertIsNone(validate_ingested_documents(documents))


if __name__ == "__main__":
    unittest.main()
