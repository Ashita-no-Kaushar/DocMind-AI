import contextlib
import csv
import io
import json
import ntpath
import os
import stat
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

SUPPORTED_EXTENSIONS = (
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
)

CAPABILITY_CATEGORIES = frozenset(
    {"verified", "optional_dependency", "unsupported_without_converter"}
)
ZIP_FORMATS = frozenset({"docx", "epub", "odt", "pptx", "xlsx"})
MAX_ARCHIVE_COMPRESSED_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_ENTRY_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 500.0
MAX_SOURCE_FILE_BYTES = 25 * 1024 * 1024
MAX_INGESTED_DOCUMENTS = 300
MAX_INGESTED_TEXT_CHARS = 4 * 1024 * 1024
MAX_EXTRACTED_CHARS_PER_FILE = MAX_INGESTED_TEXT_CHARS
MAX_EXTRACTED_TEXT_CHARS = MAX_INGESTED_TEXT_CHARS
MAX_RECORDS_PER_FILE = MAX_INGESTED_DOCUMENTS
MAX_JSON_DEPTH = 100
MAX_MIME_PARTS = 2048
MAX_MIME_DEPTH = 64
MAX_TABLE_ROWS = 100_000
MAX_TABLE_COLUMNS = 16_384
MAX_PDF_PAGES = 5000
MAX_ZIP_MEMBERS = MAX_ARCHIVE_MEMBERS
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES
MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES = MAX_ARCHIVE_ENTRY_UNCOMPRESSED_BYTES
MAX_ZIP_COMPRESSION_RATIO = MAX_ARCHIVE_COMPRESSION_RATIO
MAX_ZIP_COMPRESSED_BYTES = MAX_ARCHIVE_COMPRESSED_BYTES
MAX_ZIP_TOTAL_BYTES = MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES
MAX_ZIP_ENTRY_BYTES = MAX_ARCHIVE_ENTRY_UNCOMPRESSED_BYTES
MAX_ZIP_RATIO = MAX_ARCHIVE_COMPRESSION_RATIO


@dataclass(frozen=True)
class ParserCapability:
    category: str
    parser: str
    dependency: str
    note: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def status(self) -> str:
        return self.category

    def __getitem__(self, key: str) -> str:
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()


PARSER_CAPABILITIES = {
    "csv": ParserCapability(
        "verified",
        "DelimitedTextReader",
        "Python csv",
        "Header and every data row are verbalized while the original table is retained.",
    ),
    "doc": ParserCapability(
        "unsupported_without_converter",
        "ExternalConverter",
        "Office converter",
        "Convert the legacy binary file to DOCX before local ingestion.",
    ),
    "docx": ParserCapability(
        "verified",
        "DocxReader",
        "python-docx",
        "Text and tables in DOCX packages are extracted locally; image-only content has no indexable text and requires external OCR or conversion.",
    ),
    "eml": ParserCapability(
        "verified",
        "EmailReader",
        "Python email",
        "Text MIME parts are extracted; attachments are not indexed.",
    ),
    "epub": ParserCapability(
        "verified",
        "EpubReader",
        "ebooklib and html2text",
        "EPUB document items are converted to text locally in spine order.",
    ),
    "htm": ParserCapability(
        "verified",
        "WholeDocumentHTMLReader",
        "Python HTMLParser",
        "The complete document is read and script/style content is excluded.",
    ),
    "html": ParserCapability(
        "verified",
        "WholeDocumentHTMLReader",
        "Python HTMLParser",
        "The complete document is read and script/style content is excluded.",
    ),
    "ipynb": ParserCapability(
        "verified",
        "IPYNBReader",
        "nbconvert and nbformat",
        "Notebook cells are exported to text locally.",
    ),
    "json": ParserCapability(
        "verified",
        "JSONReader",
        "Python json",
        "The original JSON is validated and retained for record verbalization.",
    ),
    "jsonl": ParserCapability(
        "verified",
        "JSONLReader",
        "Python json",
        "Each non-empty line is parsed independently and retained as a record.",
    ),
    "markdown": ParserCapability(
        "verified",
        "MarkdownTextReader",
        "Python text decoding",
        "Markdown sections and their text are retained locally.",
    ),
    "mbox": ParserCapability(
        "verified",
        "MboxReader",
        "Python mailbox",
        "Mailbox messages are parsed locally; attachments and unsupported message parts are not indexed.",
    ),
    "md": ParserCapability(
        "verified",
        "MarkdownTextReader",
        "Python text decoding",
        "Markdown sections and their text are retained locally.",
    ),
    "mhtml": ParserCapability(
        "verified",
        "MHTMLReader",
        "Python email and HTMLParser",
        "MIME text and HTML parts are extracted from the saved archive.",
    ),
    "msg": ParserCapability(
        "verified",
        "MSGReader",
        "extract-msg and olefile",
        "Message headers and body text are extracted; attachments are not indexed.",
    ),
    "odt": ParserCapability(
        "verified",
        "ODTReader",
        "odfpy",
        "ODF paragraphs and headings are extracted locally; image-only content is not OCRed.",
    ),
    "pdf": ParserCapability(
        "optional_dependency",
        "PDFReader",
        "pypdf",
        "Text PDFs are supported; image-only PDFs have no indexable text and require an external OCR engine or conversion. No OCR is bundled.",
    ),
    "ppt": ParserCapability(
        "unsupported_without_converter",
        "ExternalConverter",
        "Office converter",
        "Convert the legacy binary file to PPTX before local ingestion.",
    ),
    "pptx": ParserCapability(
        "verified",
        "PptxReader",
        "python-pptx",
        "Slide text, tables, and speaker notes are extracted locally; image-only slides have no indexable text and require external OCR or conversion.",
    ),
    "rtf": ParserCapability(
        "verified",
        "RTFReader",
        "striprtf",
        "RTF control words are removed and visible text is extracted; embedded images are not OCRed.",
    ),
    "tsv": ParserCapability(
        "verified",
        "DelimitedTextReader",
        "Python csv",
        "Header and every data row are verbalized while the original table is retained.",
    ),
    "txt": ParserCapability(
        "verified",
        "UTF8TextReader",
        "Python text decoding",
        "UTF-8 text is retained with replacement for invalid byte sequences.",
    ),
    "xls": ParserCapability(
        "verified",
        "XLSReader",
        "xlrd",
        "Every worksheet row is converted with its column header; image-only content is not OCRed.",
    ),
    "xlsx": ParserCapability(
        "verified",
        "XLSXReader",
        "openpyxl",
        "Every worksheet row is converted with its column header; image-only content is not OCRed.",
    ),
    "xml": ParserCapability(
        "verified",
        "XMLTextReader",
        "defusedxml",
        "XML is parsed safely and element text is retained.",
    ),
}


class _CapabilityRegistry(dict):
    @staticmethod
    def _key(extension: str) -> str:
        return str(extension).lower().lstrip(".")

    def __contains__(self, extension: object) -> bool:
        if not isinstance(extension, str):
            return False
        return super().__contains__(self._key(extension))

    def __getitem__(self, extension: str) -> ParserCapability:
        return super().__getitem__(self._key(extension))

    def get(self, extension: str, default=None):
        return super().get(self._key(extension), default)


PARSER_CAPABILITIES = _CapabilityRegistry(PARSER_CAPABILITIES)
PARSER_CAPABILITIES_BY_EXTENSION = {
    f".{extension}": capability for extension, capability in PARSER_CAPABILITIES.items()
}
CAPABILITY_REGISTRY = PARSER_CAPABILITIES_BY_EXTENSION
PARSER_REGISTRY = PARSER_CAPABILITIES_BY_EXTENSION
PARSER_CAPABILITIES_WITH_DOTS = PARSER_CAPABILITIES_BY_EXTENSION
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_EXTENSIONS


class FormatIngestionError(ValueError):
    def __init__(self, message: str, category: str = "format_error") -> None:
        self.safe_message = str(message)
        self.category = str(category)
        super().__init__(self.safe_message)


class IngestionResourceError(FormatIngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "resource_limit")


class UnsafeArchiveError(FormatIngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "unsafe_archive")


class MalformedFormatError(FormatIngestionError):
    def __init__(
        self, message: str = "The document could not be parsed safely."
    ) -> None:
        super().__init__(message, "malformed_format")


class EmptyFormatError(FormatIngestionError):
    def __init__(
        self, message: str = "The document contained no extractable text."
    ) -> None:
        super().__init__(message, "empty_document")


class LegacyFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = MAX_ARCHIVE_MEMBERS
    max_total_uncompressed_bytes: int = MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES
    max_entry_uncompressed_bytes: int = MAX_ARCHIVE_ENTRY_UNCOMPRESSED_BYTES
    max_compression_ratio: float = MAX_ARCHIVE_COMPRESSION_RATIO
    max_compressed_bytes: int = MAX_ARCHIVE_COMPRESSED_BYTES

    @property
    def member_count(self) -> int:
        return self.max_members

    @property
    def total_bytes(self) -> int:
        return self.max_total_uncompressed_bytes

    @property
    def entry_bytes(self) -> int:
        return self.max_entry_uncompressed_bytes


@dataclass(frozen=True)
class TextLimits:
    max_documents: int = MAX_INGESTED_DOCUMENTS
    max_characters: int = MAX_INGESTED_TEXT_CHARS
    max_source_bytes: int = MAX_SOURCE_FILE_BYTES
    max_json_depth: int = MAX_JSON_DEPTH
    max_mime_parts: int = MAX_MIME_PARTS
    max_mime_depth: int = MAX_MIME_DEPTH
    max_table_rows: int = MAX_TABLE_ROWS
    max_table_columns: int = MAX_TABLE_COLUMNS
    max_pdf_pages: int = MAX_PDF_PAGES

    @property
    def max_text_chars(self) -> int:
        return self.max_characters

    @property
    def max_records(self) -> int:
        return self.max_documents


DEFAULT_ARCHIVE_LIMITS = ArchiveLimits()
DEFAULT_TEXT_LIMITS = TextLimits()


def _environment_value(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
    return None


def _environment_int(names: tuple[str, ...], default: int) -> int:
    value = _environment_value(names)
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _environment_float(names: tuple[str, ...], default: float) -> float:
    value = _environment_value(names)
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def _limit_override(overrides: dict[str, Any] | None, *names: str) -> Any:
    if not overrides:
        return None
    for name in names:
        if name in overrides and overrides[name] is not None:
            return overrides[name]
    return None


def _positive_limit(value: Any, default: int, label: str) -> int:
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid {label} limit.") from err
    if parsed <= 0:
        raise ValueError(f"Invalid {label} limit.")
    return parsed


def _positive_float_limit(value: Any, default: float, label: str) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid {label} limit.") from err
    if parsed <= 0:
        raise ValueError(f"Invalid {label} limit.")
    return parsed


def archive_limits(**overrides: Any) -> ArchiveLimits:
    defaults = ArchiveLimits(
        max_members=MAX_ARCHIVE_MEMBERS,
        max_total_uncompressed_bytes=MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES,
        max_entry_uncompressed_bytes=MAX_ARCHIVE_ENTRY_UNCOMPRESSED_BYTES,
        max_compression_ratio=MAX_ARCHIVE_COMPRESSION_RATIO,
        max_compressed_bytes=MAX_ARCHIVE_COMPRESSED_BYTES,
    )
    return ArchiveLimits(
        max_members=_positive_limit(
            _limit_override(
                overrides,
                "max_members",
                "members",
                "member_count",
            )
            or _environment_int(
                (
                    "DOCMIND_ZIP_MAX_MEMBERS",
                    "DOCMIND_MAX_ZIP_MEMBERS",
                    "DOCMIND_ARCHIVE_MAX_MEMBERS",
                    "DOCMIND_MAX_ARCHIVE_MEMBERS",
                ),
                defaults.max_members,
            ),
            defaults.max_members,
            "archive member",
        ),
        max_total_uncompressed_bytes=_positive_limit(
            _limit_override(
                overrides,
                "max_total_uncompressed_bytes",
                "total_bytes",
                "uncompressed_bytes",
            )
            or _environment_int(
                (
                    "DOCMIND_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES",
                    "DOCMIND_ZIP_MAX_TOTAL_BYTES",
                    "DOCMIND_MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES",
                    "DOCMIND_MAX_ZIP_UNCOMPRESSED_BYTES",
                    "DOCMIND_MAX_ZIP_TOTAL_BYTES",
                    "DOCMIND_ZIP_MAX_TOTAL_SIZE",
                    "DOCMIND_ARCHIVE_MAX_TOTAL_UNCOMPRESSED_BYTES",
                    "DOCMIND_ARCHIVE_MAX_TOTAL_BYTES",
                    "DOCMIND_ARCHIVE_MAX_TOTAL_SIZE",
                ),
                defaults.max_total_uncompressed_bytes,
            ),
            defaults.max_total_uncompressed_bytes,
            "archive total byte",
        ),
        max_entry_uncompressed_bytes=_positive_limit(
            _limit_override(
                overrides,
                "max_entry_uncompressed_bytes",
                "entry_bytes",
                "max_entry_bytes",
            )
            or _environment_int(
                (
                    "DOCMIND_ZIP_MAX_ENTRY_UNCOMPRESSED_BYTES",
                    "DOCMIND_ZIP_MAX_ENTRY_BYTES",
                    "DOCMIND_MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES",
                    "DOCMIND_MAX_ZIP_UNCOMPRESSED_ENTRY_BYTES",
                    "DOCMIND_MAX_ZIP_ENTRY_BYTES",
                    "DOCMIND_ZIP_MAX_ENTRY_SIZE",
                    "DOCMIND_ARCHIVE_MAX_ENTRY_UNCOMPRESSED_BYTES",
                    "DOCMIND_ARCHIVE_MAX_ENTRY_BYTES",
                    "DOCMIND_ARCHIVE_MAX_ENTRY_SIZE",
                ),
                defaults.max_entry_uncompressed_bytes,
            ),
            defaults.max_entry_uncompressed_bytes,
            "archive entry byte",
        ),
        max_compression_ratio=_positive_float_limit(
            _limit_override(
                overrides,
                "max_compression_ratio",
                "compression_ratio",
            )
            or _environment_float(
                (
                    "DOCMIND_ZIP_MAX_COMPRESSION_RATIO",
                    "DOCMIND_ZIP_MAX_RATIO",
                    "DOCMIND_MAX_ZIP_COMPRESSION_RATIO",
                    "DOCMIND_ARCHIVE_MAX_COMPRESSION_RATIO",
                ),
                defaults.max_compression_ratio,
            ),
            defaults.max_compression_ratio,
            "archive compression ratio",
        ),
        max_compressed_bytes=_positive_limit(
            _limit_override(
                overrides,
                "max_compressed_bytes",
                "compressed_bytes",
            )
            or _environment_int(
                (
                    "DOCMIND_ZIP_MAX_COMPRESSED_BYTES",
                    "DOCMIND_MAX_ZIP_COMPRESSED_BYTES",
                    "DOCMIND_ARCHIVE_MAX_COMPRESSED_BYTES",
                ),
                defaults.max_compressed_bytes,
            ),
            defaults.max_compressed_bytes,
            "archive compressed byte",
        ),
    )


def text_limits(**overrides: Any) -> TextLimits:
    defaults = TextLimits(
        max_documents=MAX_INGESTED_DOCUMENTS,
        max_characters=MAX_INGESTED_TEXT_CHARS,
        max_source_bytes=MAX_SOURCE_FILE_BYTES,
        max_json_depth=MAX_JSON_DEPTH,
        max_mime_parts=MAX_MIME_PARTS,
        max_mime_depth=MAX_MIME_DEPTH,
        max_table_rows=MAX_TABLE_ROWS,
        max_table_columns=MAX_TABLE_COLUMNS,
        max_pdf_pages=MAX_PDF_PAGES,
    )
    return TextLimits(
        max_documents=_positive_limit(
            _limit_override(overrides, "max_documents", "documents", "records")
            or _environment_int(
                ("DOCMIND_MAX_INGESTED_DOCUMENTS", "DOCMIND_MAX_RECORDS_PER_FILE"),
                defaults.max_documents,
            ),
            defaults.max_documents,
            "document",
        ),
        max_characters=_positive_limit(
            _limit_override(
                overrides,
                "max_characters",
                "max_text_chars",
                "characters",
            )
            or _environment_int(
                (
                    "DOCMIND_MAX_INGESTED_TEXT_CHARS",
                    "DOCMIND_MAX_EXTRACTED_CHARS_PER_FILE",
                    "DOCMIND_MAX_EXTRACTED_TEXT_CHARS_PER_FILE",
                ),
                defaults.max_characters,
            ),
            defaults.max_characters,
            "text character",
        ),
        max_source_bytes=_positive_limit(
            _limit_override(overrides, "max_source_bytes", "source_bytes")
            or _environment_int(
                ("DOCMIND_MAX_SOURCE_FILE_BYTES",),
                defaults.max_source_bytes,
            ),
            defaults.max_source_bytes,
            "source byte",
        ),
        max_json_depth=_positive_limit(
            _limit_override(overrides, "max_json_depth", "json_depth")
            or _environment_int(("DOCMIND_MAX_JSON_DEPTH",), defaults.max_json_depth),
            defaults.max_json_depth,
            "JSON nesting",
        ),
        max_mime_parts=_positive_limit(
            _limit_override(overrides, "max_mime_parts", "mime_parts")
            or _environment_int(("DOCMIND_MAX_MIME_PARTS",), defaults.max_mime_parts),
            defaults.max_mime_parts,
            "MIME part",
        ),
        max_mime_depth=_positive_limit(
            _limit_override(overrides, "max_mime_depth", "mime_depth")
            or _environment_int(("DOCMIND_MAX_MIME_DEPTH",), defaults.max_mime_depth),
            defaults.max_mime_depth,
            "MIME nesting",
        ),
        max_table_rows=_positive_limit(
            _limit_override(overrides, "max_table_rows", "table_rows")
            or _environment_int(("DOCMIND_MAX_TABLE_ROWS",), defaults.max_table_rows),
            defaults.max_table_rows,
            "table row",
        ),
        max_table_columns=_positive_limit(
            _limit_override(overrides, "max_table_columns", "table_columns")
            or _environment_int(
                ("DOCMIND_MAX_TABLE_COLUMNS",), defaults.max_table_columns
            ),
            defaults.max_table_columns,
            "table column",
        ),
        max_pdf_pages=_positive_limit(
            _limit_override(overrides, "max_pdf_pages", "pdf_pages")
            or _environment_int(("DOCMIND_MAX_PDF_PAGES",), defaults.max_pdf_pages),
            defaults.max_pdf_pages,
            "PDF page",
        ),
    )


get_archive_limits = archive_limits
get_text_limits = text_limits


def _limit_error(message: str) -> IngestionResourceError:
    return IngestionResourceError(message)


def _source_size(path: Path) -> int:
    try:
        size = path.stat().st_size
    except OSError as err:
        raise MalformedFormatError("The input file could not be inspected.") from err
    if size < 0:
        raise MalformedFormatError("The input file could not be inspected.")
    return int(size)


def _read_source_bytes(path: Path, limits: TextLimits | None = None) -> bytes:
    selected = limits or text_limits()
    size = _source_size(path)
    if size > selected.max_source_bytes:
        raise _limit_error("The input file exceeds the configured source-size limit.")
    try:
        with path.open("rb") as handle:
            data = handle.read(selected.max_source_bytes + 1)
    except OSError as err:
        raise MalformedFormatError("The input file could not be read.") from err
    if len(data) > selected.max_source_bytes:
        raise _limit_error("The input file exceeds the configured source-size limit.")
    return data


def _read_utf8(file: Path) -> str:
    return _read_source_bytes(Path(file)).decode("utf-8-sig", errors="replace")


def _metadata(file: Path, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "file_name": Path(file).name,
        "file_path": str(Path(file)),
    }
    if isinstance(extra_info, dict):
        metadata.update(extra_info)
    return metadata


def _clean_text(text: str, max_characters: int | None = None) -> str:
    value = "\n".join(
        " ".join(line.split())
        for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    )
    if max_characters is not None and len(value) > max_characters:
        raise _limit_error(
            "Extracted text exceeds the configured per-file character limit."
        )
    return value


def _append_bounded(parts: list[str], value: str, limit: int) -> None:
    if not value:
        return
    current = sum(len(part) for part in parts)
    if current + len(value) > limit:
        raise _limit_error(
            "Extracted text exceeds the configured per-file character limit."
        )
    parts.append(value)


class _ExtractionBudget:
    def __init__(
        self,
        path: Path,
        extra_info: dict[str, Any] | None = None,
        limits: TextLimits | None = None,
    ) -> None:
        self.path = Path(path)
        self.extra_info = extra_info
        self.limits = limits or text_limits()
        self.documents: list[Document] = []
        self.characters = 0

    def add(
        self,
        text: Any,
        metadata: dict[str, Any] | None = None,
        parser_warning: str = "",
    ) -> Document:
        value = str(text or "")
        if len(self.documents) >= self.limits.max_documents:
            raise _limit_error(
                "The document record count exceeds the configured limit."
            )
        if self.characters + len(value) > self.limits.max_characters:
            raise _limit_error(
                "Extracted text exceeds the configured per-file character limit."
            )
        document_metadata = self._metadata(metadata)
        if parser_warning:
            document_metadata["parser_warning"] = parser_warning
        document = Document(text=value, metadata=document_metadata)
        self.documents.append(document)
        self.characters += len(value)
        return document

    def _metadata(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        metadata = _metadata(self.path, self.extra_info)
        if isinstance(extra, dict):
            metadata.update(extra)
        return metadata

    def result(self) -> list[Document]:
        return list(self.documents)


def _json_depth(value: str, max_depth: int | None = None) -> int:
    selected = max_depth if max_depth is not None else text_limits().max_json_depth
    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > selected:
                raise _limit_error("JSON nesting exceeds the configured limit.")
        elif character in "]}":
            depth = max(0, depth - 1)
    return depth


def _json_record_count(value: str, max_records: int) -> int:
    stripped = value.lstrip()
    if not stripped.startswith("["):
        return 1
    decoder = json.JSONDecoder()
    index = 1
    count = 0
    length = len(stripped)
    while index < length:
        while index < length and stripped[index].isspace():
            index += 1
        if index >= length or stripped[index] == "]":
            return count
        if stripped[index] == ",":
            index += 1
            continue
        try:
            _, index = decoder.raw_decode(stripped, index)
        except (json.JSONDecodeError, RecursionError) as err:
            raise MalformedFormatError(
                "The JSON document could not be parsed safely."
            ) from err
        count += 1
        if count > max_records:
            raise _limit_error("JSON record count exceeds the configured limit.")
    return count


def parse_json_text(
    value: str,
    *,
    max_records: int | None = None,
    max_depth: int | None = None,
) -> Any:
    limits = text_limits()
    selected_records = limits.max_documents if max_records is None else max_records
    selected_depth = limits.max_json_depth if max_depth is None else max_depth
    text = str(value)
    if not text.strip():
        raise EmptyFormatError("The JSON document contained no extractable text.")
    _json_depth(text, selected_depth)
    stripped = text.lstrip()
    if stripped.startswith("["):
        _json_record_count(stripped, selected_records)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError) as err:
        raise MalformedFormatError(
            "The JSON document could not be parsed safely."
        ) from err


def _check_json_line(value: str, max_depth: int | None = None) -> None:
    _json_depth(value, max_depth)


class _HTMLTextParser(HTMLParser):
    _BLOCK_TAGS: ClassVar[set[str]] = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    _SKIP_TAGS: ClassVar[set[str]] = {"script", "style", "noscript", "template"}

    def __init__(self, max_characters: int | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._max_characters = max_characters
        self._characters = 0

    def _append(self, value: str) -> None:
        if not value:
            return
        if (
            self._max_characters is not None
            and self._characters + len(value) > self._max_characters
        ):
            raise _limit_error(
                "Extracted HTML text exceeds the configured per-file character limit."
            )
        self._characters += len(value)
        self._parts.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() not in self._SKIP_TAGS and not self._skip_depth:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._append(data)

    def text(self) -> str:
        return _clean_text("".join(self._parts), self._max_characters)


def html_to_text(
    value: str,
    *,
    max_characters: int | None = None,
    max_source_characters: int | None = None,
) -> str:
    text = str(value)
    if max_source_characters is not None and len(text) > max_source_characters:
        raise _limit_error(
            "The input document exceeds the configured source-size limit."
        )
    parser = _HTMLTextParser(max_characters)
    parser.feed(text)
    parser.close()
    return parser.text()


def _safe_archive_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise UnsafeArchiveError("ZIP package contains an invalid member path.")
    normalized = name.replace("\\", "/")
    if (
        normalized.startswith(("/", "\\"))
        or ntpath.isabs(name)
        or ntpath.splitdrive(name)[0]
    ):
        raise UnsafeArchiveError("ZIP package contains an absolute member path.")
    if ".." in PurePosixPath(normalized).parts:
        raise UnsafeArchiveError("ZIP package contains a path-traversal member.")
    return normalized


def validate_zip_archive(
    path: Path,
    limits: ArchiveLimits | None = None,
    **overrides: Any,
) -> None:
    selected = limits or archive_limits(**overrides)
    archive_path = Path(path)
    try:
        compressed_size = _source_size(archive_path)
        if compressed_size > selected.max_compressed_bytes:
            raise _limit_error(
                "ZIP package exceeds the configured compressed-size limit."
            )
        if not zipfile.is_zipfile(archive_path):
            raise UnsafeArchiveError("ZIP package structure is invalid.")
        with zipfile.ZipFile(archive_path, "r") as archive:
            entries = archive.infolist()
            if not entries:
                raise UnsafeArchiveError("ZIP package contains no members.")
            if len(entries) > selected.max_members:
                raise _limit_error(
                    "ZIP package member count exceeds the configured limit."
                )
            total_uncompressed = 0
            names = set()
            for entry in entries:
                normalized = _safe_archive_member_name(entry.filename)
                duplicate_key = normalized.casefold()
                if duplicate_key in names:
                    raise UnsafeArchiveError(
                        "ZIP package contains duplicate member paths."
                    )
                names.add(duplicate_key)
                if entry.flag_bits & 0x1:
                    raise UnsafeArchiveError(
                        "Encrypted ZIP package members are not supported."
                    )
                file_size = int(entry.file_size)
                compressed_member_size = int(entry.compress_size)
                if file_size < 0 or compressed_member_size < 0:
                    raise UnsafeArchiveError(
                        "ZIP package member size metadata is invalid."
                    )
                mode = (entry.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}:
                    raise UnsafeArchiveError(
                        "ZIP package contains a symlink or special member."
                    )
                if file_size > selected.max_entry_uncompressed_bytes:
                    raise _limit_error(
                        "ZIP package member size exceeds the configured limit."
                    )
                total_uncompressed += file_size
                if total_uncompressed > selected.max_total_uncompressed_bytes:
                    raise _limit_error(
                        "ZIP package uncompressed size exceeds the configured limit."
                    )
                if file_size and compressed_member_size == 0:
                    raise _limit_error(
                        "ZIP package member compression ratio exceeds the configured limit."
                    )
                if file_size and compressed_member_size:
                    ratio = file_size / compressed_member_size
                    if ratio > selected.max_compression_ratio:
                        raise _limit_error(
                            "ZIP package member compression ratio exceeds the configured limit."
                        )
    except UnsafeArchiveError:
        raise
    except IngestionResourceError:
        raise
    except (OSError, RuntimeError, EOFError, zipfile.BadZipFile, ValueError) as err:
        raise UnsafeArchiveError("ZIP package structure is invalid.") from err


def preflight_input_file(path: Path) -> None:
    selected_path = Path(path)
    _read_source_bytes(selected_path)
    if selected_path.suffix.lower().lstrip(".") in ZIP_FORMATS:
        validate_zip_archive(selected_path)


def _format_delimited_text(
    text: str,
    delimiter: str,
    *,
    max_characters: int | None = None,
    max_rows: int | None = None,
) -> str:
    limits = text_limits()
    selected_characters = (
        limits.max_characters if max_characters is None else max_characters
    )
    selected_rows = limits.max_table_rows if max_rows is None else max_rows
    rows = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    first_row = next(rows, None)
    if first_row is None:
        return text
    if len(first_row) > limits.max_table_columns:
        raise _limit_error("Delimited table column count exceeds the configured limit.")
    headers = [
        str(value).strip() or f"Column {index + 1}"
        for index, value in enumerate(first_row)
    ]
    lines = ["Columns: " + ", ".join(headers)]
    raw_lines: list[str] = []
    row_number = 1
    for row in rows:
        row_number += 1
        if row_number > selected_rows:
            raise _limit_error(
                "Delimited table row count exceeds the configured limit."
            )
        if len(row) > limits.max_table_columns:
            raise _limit_error(
                "Delimited table column count exceeds the configured limit."
            )
        values = []
        raw_values = []
        for index, value in enumerate(row):
            value_text = str(value).strip()
            raw_values.append(value_text)
            if not value_text:
                continue
            header = (
                headers[index] if index < len(headers) else f"Extra column {index + 1}"
            )
            values.append(f"{header} is {value_text}")
        lines.append(
            f"Row {row_number}: " + (", ".join(values) if values else "(empty row)")
        )
        raw_lines.append(" | ".join(raw_values))
    result = "\n".join(lines) + "\n\nOriginal table:\n" + text
    if len(result) > selected_characters:
        raise _limit_error(
            "Extracted text exceeds the configured per-file character limit."
        )
    return result


class WholeDocumentHTMLReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        path = Path(file)
        limits = text_limits()
        text = _read_utf8(path)
        if not text.strip():
            raise EmptyFormatError("The HTML document contained no extractable text.")
        extracted = html_to_text(
            text,
            max_characters=limits.max_characters,
            max_source_characters=limits.max_source_bytes,
        )
        if not extracted:
            raise EmptyFormatError("The HTML document contained no extractable text.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(extracted)
        return budget.result()


class DelimitedTextReader(BaseReader):
    def __init__(self, delimiter: str) -> None:
        self.delimiter = delimiter

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        path = Path(file)
        limits = text_limits()
        text = _read_utf8(path)
        if not text.strip():
            raise EmptyFormatError("The delimited file contained no extractable text.")
        try:
            content = _format_delimited_text(
                text,
                self.delimiter,
                max_characters=limits.max_characters,
                max_rows=limits.max_table_rows,
            )
        except csv.Error as err:
            raise MalformedFormatError(
                "The delimited file could not be parsed safely."
            ) from err
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(content, {"tabular_verbalized": True})
        return budget.result()


class UTF8TextReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        path = Path(file)
        limits = text_limits()
        text = _read_utf8(path)
        if not text.strip():
            raise EmptyFormatError("The text file contained no extractable text.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text)
        return budget.result()


class MarkdownTextReader(UTF8TextReader):
    pass


class JSONReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        path = Path(file)
        limits = text_limits()
        text = _read_utf8(path)
        parse_json_text(
            text, max_records=limits.max_documents, max_depth=limits.max_json_depth
        )
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text)
        return budget.result()


class JSONLReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        path = Path(file)
        limits = text_limits()
        if _source_size(path) > limits.max_source_bytes:
            raise _limit_error(
                "The input file exceeds the configured source-size limit."
            )
        budget = _ExtractionBudget(path, extra_info, limits)
        try:
            handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
        except OSError as err:
            raise MalformedFormatError("The JSONL file could not be read.") from err
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                if len(budget.documents) >= limits.max_documents:
                    raise _limit_error(
                        "JSONL record count exceeds the configured limit."
                    )
                if budget.characters + len(line) > limits.max_characters:
                    raise _limit_error(
                        "Extracted text exceeds the configured per-file character limit."
                    )
                _check_json_line(line, limits.max_json_depth)
                try:
                    json.loads(line)
                except (json.JSONDecodeError, RecursionError):
                    budget.add(
                        line,
                        {
                            "record_index": line_number,
                            "parser_warning": (
                                f"Invalid JSONL record on line {line_number}; raw line retained."
                            ),
                        },
                    )
                else:
                    budget.add(line, {"record_index": line_number})
        if not budget.documents:
            raise EmptyFormatError("The JSONL file contained no records.")
        return budget.result()


class IPYNBReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            import nbformat
            from nbconvert.exporters import ScriptExporter
        except ImportError as err:
            raise ImportError(
                "nbconvert and nbformat are required to read IPYNB files."
            ) from err
        path = Path(file)
        limits = text_limits()
        _read_source_bytes(path, limits)
        try:
            notebook = nbformat.read(str(path), as_version=4)
        except (OSError, ValueError, TypeError, RecursionError) as err:
            raise MalformedFormatError(
                "The notebook document could not be parsed safely."
            ) from err
        try:
            text, _ = ScriptExporter().from_notebook_node(notebook)
        except Exception:
            parts = []
            for cell in notebook.get("cells", []):
                source = str(cell.get("source", ""))
                if source.strip():
                    parts.append(source)
            text = "\n\n".join(parts)
        if not str(text).strip():
            raise EmptyFormatError(
                "The notebook document contained no extractable cells."
            )
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text)
        return budget.result()


class XMLTextReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            from defusedxml import ElementTree
        except ImportError as err:
            raise ImportError("defusedxml is required to read XML files.") from err
        path = Path(file)
        limits = text_limits()
        source = io.BytesIO(_read_source_bytes(path, limits))
        parts: list[str] = []
        current_length = 0
        depth = 0
        context = None
        try:
            context = ElementTree.iterparse(source, events=("start", "end"))
            for event, element in context:
                if event == "start":
                    depth += 1
                    if depth > limits.max_json_depth:
                        raise _limit_error("XML nesting exceeds the configured limit.")
                    continue
                for value in (element.text, element.tail):
                    if value and value.strip():
                        current_length += len(value)
                        if current_length > limits.max_characters:
                            raise _limit_error(
                                "Extracted text exceeds the configured per-file character limit."
                            )
                        parts.append(value)
                element.clear()
                depth = max(0, depth - 1)
        except (IngestionResourceError, UnsafeArchiveError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The XML document could not be parsed safely."
            ) from err
        finally:
            close = getattr(context, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            source.close()
        text = _clean_text(" ".join(parts), limits.max_characters)
        if not text:
            raise EmptyFormatError("The XML document contained no extractable text.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text)
        return budget.result()


def _decode_email_part(part: Any, limits: TextLimits) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        payload = part.get_payload()
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        if len(payload) > limits.max_source_bytes:
            raise _limit_error(
                "Email part size exceeds the configured source-size limit."
            )
        charset = part.get_content_charset() or "utf-8"
        try:
            value = payload.decode(charset, errors="replace")
        except LookupError:
            value = payload.decode("utf-8", errors="replace")
    else:
        value = str(payload)
    if len(value) > limits.max_characters:
        raise _limit_error(
            "Email text exceeds the configured per-file character limit."
        )
    return value


def _email_text(message: Any, limits: TextLimits | None = None) -> str:
    selected = limits or text_limits()
    header_values: list[str] = []
    header_length = 0
    for label, key in (("Subject", "subject"), ("From", "from"), ("To", "to")):
        value = message.get(key)
        if value:
            value_text = f"{label}: {value}"
            header_length += len(value_text)
            if header_length > selected.max_characters:
                raise _limit_error(
                    "Email text exceeds the configured per-file character limit."
                )
            header_values.append(value_text)
    text_parts: list[str] = []
    html_parts: list[str] = []
    part_count = 0
    stack = [(message, 1)]
    while stack:
        part, depth = stack.pop()
        if depth > selected.max_mime_depth:
            raise _limit_error("Email MIME nesting exceeds the configured limit.")
        part_count += 1
        if part_count > selected.max_mime_parts:
            raise _limit_error("Email MIME part count exceeds the configured limit.")
        if part.is_multipart():
            children = list(part.iter_parts())
            for child in reversed(children):
                stack.append((child, depth + 1))
            continue
        disposition = str(part.get("Content-Disposition", ""))
        if "attachment" in disposition.lower():
            continue
        content = _decode_email_part(part, selected)
        if not content.strip():
            continue
        if part.get_content_type() == "text/plain":
            current_length = sum(len(value) for value in text_parts)
            if header_length + current_length + len(content) > selected.max_characters:
                raise _limit_error(
                    "Email text exceeds the configured per-file character limit."
                )
            text_parts.append(content)
        elif part.get_content_type() == "text/html":
            html_parts.append(content)
    if not text_parts and html_parts:
        body_length = 0
        for content in html_parts:
            remaining = selected.max_characters - header_length - body_length
            if remaining <= 0:
                raise _limit_error(
                    "Email text exceeds the configured per-file character limit."
                )
            converted = html_to_text(
                content,
                max_characters=remaining,
                max_source_characters=selected.max_source_bytes,
            )
            body_length += len(converted) + (1 if text_parts else 0)
            if body_length > selected.max_characters - header_length:
                raise _limit_error(
                    "Email text exceeds the configured per-file character limit."
                )
            text_parts.append(converted)
    body = _clean_text("\n".join(text_parts), selected.max_characters)
    return _clean_text(
        "\n".join(header_values + ([body] if body else [])), selected.max_characters
    )


class EmailReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        path = Path(file)
        limits = text_limits()
        data = _read_source_bytes(path, limits)
        try:
            message = BytesParser(policy=policy.default).parsebytes(data)
            text = _email_text(message, limits)
        except (IngestionResourceError, EmptyFormatError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The email message could not be parsed safely."
            ) from err
        if not text.strip():
            raise EmptyFormatError("The email message contained no text parts.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text)
        return budget.result()


class MHTMLReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        path = Path(file)
        limits = text_limits()
        data = _read_source_bytes(path, limits)
        try:
            message = BytesParser(policy=policy.default).parsebytes(data)
            text = _email_text(message, limits)
        except IngestionResourceError:
            raise
        except EmptyFormatError:
            text = ""
        except Exception as err:
            raise MalformedFormatError(
                "The MHTML archive could not be parsed safely."
            ) from err
        raw = _read_utf8(path)
        if not message.is_multipart() and raw.lstrip().casefold().startswith(
            ("<html", "<!doctype", "<body", "<p")
        ):
            text = html_to_text(
                raw,
                max_characters=limits.max_characters,
                max_source_characters=limits.max_source_bytes,
            )
        if not text:
            raw_lower = raw.lstrip().casefold()
            is_mime_container = bool(
                message.is_multipart()
                or raw_lower.startswith(
                    ("content-type:", "mime-version:", "from:", "to:", "subject:")
                )
            )
            if is_mime_container:
                raise EmptyFormatError("The MHTML archive contained no text parts.")
            text = (
                html_to_text(
                    raw,
                    max_characters=limits.max_characters,
                    max_source_characters=limits.max_source_bytes,
                )
                if "<" in raw
                else raw.strip()
            )
        if not text.strip():
            raise EmptyFormatError("The MHTML archive contained no text parts.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text)
        return budget.result()


class MboxReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        import mailbox

        path = Path(file)
        limits = text_limits()
        _read_source_bytes(path, limits)
        budget = _ExtractionBudget(path, extra_info, limits)
        try:
            mailbox_file = mailbox.mbox(
                str(path), factory=BytesParser(policy=policy.default).parse
            )
        except Exception as err:
            raise MalformedFormatError(
                "The MBOX file could not be opened safely."
            ) from err
        try:
            for message_index, message in enumerate(mailbox_file, start=1):
                if len(budget.documents) >= limits.max_documents:
                    raise _limit_error(
                        "MBOX message count exceeds the configured limit."
                    )
                try:
                    text = _email_text(message, limits)
                except IngestionResourceError:
                    raise
                if not text:
                    text = f"Message {message_index} contained no text parts."
                    warning = f"MBOX message {message_index} had no text parts; record retained."
                else:
                    warning = ""
                budget.add(text, {"message_index": message_index}, warning)
        except (IngestionResourceError, MalformedFormatError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The MBOX file could not be parsed safely."
            ) from err
        finally:
            with contextlib.suppress(Exception):
                mailbox_file.close()
        if not budget.documents:
            raise EmptyFormatError("The MBOX file contained no messages.")
        return budget.result()


def _odt_fallback_text(path: Path, limits: TextLimits | None = None) -> str:
    return ""


class ODTReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            from odf import teletype
            from odf import text as odf_text
            from odf.opendocument import load
        except ImportError as err:
            raise ImportError("odfpy is required to read ODT files.") from err
        path = Path(file)
        limits = text_limits()
        validate_zip_archive(path)
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                document = load(str(path))
        except Exception as err:
            text = _odt_fallback_text(path, limits)
            if not text:
                raise MalformedFormatError(
                    "The ODT document could not be opened safely."
                ) from err
            budget = _ExtractionBudget(path, extra_info, limits)
            budget.add(text)
            return budget.result()
        paragraphs = document.getElementsByType(odf_text.P)
        parts: list[str] = []
        total_length = 0
        for paragraph in paragraphs:
            part = teletype.extractText(paragraph).strip()
            if not part:
                continue
            total_length += len(part) + (1 if parts else 0)
            if total_length > limits.max_characters:
                raise _limit_error(
                    "Extracted text exceeds the configured per-file character limit."
                )
            parts.append(part)
        if not parts:
            try:
                text = teletype.extractText(document).strip()
            except Exception:
                text = ""
        else:
            text = "\n".join(parts)
        if not text:
            text = _odt_fallback_text(path, limits)
        if not text:
            raise EmptyFormatError("The ODT document contained no extractable text.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(_clean_text(text, limits.max_characters))
        return budget.result()


def _message_value(value: Any) -> str:
    if callable(value):
        value = value()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


class MSGReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            import extract_msg
            import olefile
        except ImportError as err:
            raise ImportError(
                "extract-msg and olefile are required to read MSG files."
            ) from err
        path = Path(file)
        limits = text_limits()
        _read_source_bytes(path, limits)
        if not olefile.isOleFile(str(path)):
            raise MalformedFormatError("The MSG file is not a valid OLE compound file.")
        try:
            message = extract_msg.openMsg(str(path))
        except Exception as err:
            raise MalformedFormatError(
                "The MSG message could not be opened safely."
            ) from err
        try:
            try:
                body = _message_value(getattr(message, "body", ""))
                if len(body) > limits.max_characters:
                    raise _limit_error(
                        "MSG text exceeds the configured per-file character limit."
                    )
                if not body.strip():
                    body = html_to_text(
                        _message_value(getattr(message, "htmlBody", "")),
                        max_characters=limits.max_characters,
                        max_source_characters=limits.max_source_bytes,
                    )
                values = [
                    ("Subject", _message_value(getattr(message, "subject", ""))),
                    ("From", _message_value(getattr(message, "sender", ""))),
                    ("To", _message_value(getattr(message, "to", ""))),
                ]
                text = _clean_text(
                    "\n".join(
                        [f"{label}: {value}" for label, value in values if value]
                        + ([body] if body.strip() else [])
                    ),
                    limits.max_characters,
                )
            except IngestionResourceError:
                raise
            except Exception as err:
                raise MalformedFormatError(
                    "The MSG message body could not be read safely."
                ) from err
        finally:
            close = getattr(message, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        if not text.strip():
            raise EmptyFormatError("The MSG message contained no extractable text.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text)
        return budget.result()


def _spreadsheet_text(
    sheet_name: str,
    rows: Iterable[Iterable[Any]],
    *,
    max_characters: int | None = None,
    max_rows: int | None = None,
) -> str:
    limits = text_limits()
    selected_characters = (
        limits.max_characters if max_characters is None else max_characters
    )
    selected_rows = limits.max_table_rows if max_rows is None else max_rows
    lines: list[str] = []
    raw_lines: list[str] = []
    row_count = 0
    first_row = None
    for row in rows:
        row_count += 1
        if row_count > selected_rows:
            raise _limit_error("Spreadsheet row count exceeds the configured limit.")
        values = [_message_value(value) for value in row]
        if len(values) > limits.max_table_columns:
            raise _limit_error("Spreadsheet column count exceeds the configured limit.")
        if first_row is None:
            first_row = values
            headers = [
                str(value).strip() or f"Column {index + 1}"
                for index, value in enumerate(first_row)
            ]
            lines.append(f"Sheet: {sheet_name}")
            lines.append("Columns: " + ", ".join(headers))
            continue
        verbal_values = []
        for index, value in enumerate(values):
            if not value.strip():
                continue
            header = (
                headers[index] if index < len(headers) else f"Extra column {index + 1}"
            )
            verbal_values.append(f"{header} is {value}")
        lines.append(
            f"Row {row_count - 1}: "
            + (", ".join(verbal_values) if verbal_values else "(empty row)")
        )
        raw_lines.append(" | ".join(values))
    if first_row is None:
        return ""
    lines.append("Raw sheet:\n" + "\n".join(raw_lines))
    result = "\n".join(lines)
    if len(result) > selected_characters:
        raise _limit_error(
            "Extracted spreadsheet text exceeds the configured per-file character limit."
        )
    return result


def _limited_rows(rows: Iterable[Iterable[Any]], max_rows: int):
    for index, row in enumerate(rows, start=1):
        if index > max_rows:
            raise _limit_error("Spreadsheet row count exceeds the configured limit.")
        yield row


class XLSXReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            from openpyxl import load_workbook
        except ImportError as err:
            raise ImportError("openpyxl is required to read XLSX files.") from err
        path = Path(file)
        limits = text_limits()
        validate_zip_archive(path)
        try:
            workbook = load_workbook(
                path, read_only=True, data_only=True, keep_links=False
            )
        except Exception as err:
            raise MalformedFormatError(
                "The XLSX workbook could not be opened safely."
            ) from err
        budget = _ExtractionBudget(path, extra_info, limits)
        try:
            worksheets = list(workbook.worksheets)
            if len(worksheets) > limits.max_documents:
                raise _limit_error(
                    "Spreadsheet sheet count exceeds the configured limit."
                )
            for worksheet in worksheets:
                text = _spreadsheet_text(
                    worksheet.title,
                    _limited_rows(
                        worksheet.iter_rows(values_only=True), limits.max_table_rows
                    ),
                    max_characters=limits.max_characters,
                    max_rows=limits.max_table_rows,
                )
                if text:
                    budget.add(text, {"sheet_name": worksheet.title})
        except (IngestionResourceError, MalformedFormatError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The XLSX workbook could not be read safely."
            ) from err
        finally:
            with contextlib.suppress(Exception):
                workbook.close()
        if not budget.documents:
            raise EmptyFormatError("The XLSX workbook contained no extractable rows.")
        return budget.result()


class XLSReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            import xlrd
        except ImportError as err:
            raise ImportError("xlrd is required to read XLS files.") from err
        path = Path(file)
        limits = text_limits()
        _read_source_bytes(path, limits)
        try:
            workbook = xlrd.open_workbook(str(path), on_demand=True)
        except Exception as err:
            raise MalformedFormatError(
                "The XLS workbook could not be opened safely."
            ) from err
        budget = _ExtractionBudget(path, extra_info, limits)
        try:
            sheet_names = list(workbook.sheet_names())
            if len(sheet_names) > limits.max_documents:
                raise _limit_error(
                    "Spreadsheet sheet count exceeds the configured limit."
                )
            for sheet_name in sheet_names:
                worksheet = workbook.sheet_by_name(sheet_name)
                text = _spreadsheet_text(
                    sheet_name,
                    _limited_rows(
                        (
                            [
                                worksheet.cell_value(row_index, column_index)
                                for column_index in range(worksheet.ncols)
                            ]
                            for row_index in range(worksheet.nrows)
                        ),
                        limits.max_table_rows,
                    ),
                    max_characters=limits.max_characters,
                    max_rows=limits.max_table_rows,
                )
                if text:
                    budget.add(text, {"sheet_name": sheet_name})
        except (IngestionResourceError, MalformedFormatError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The XLS workbook could not be read safely."
            ) from err
        finally:
            with contextlib.suppress(Exception):
                workbook.release_resources()
        if not budget.documents:
            raise EmptyFormatError("The XLS workbook contained no extractable rows.")
        return budget.result()


class _ArchiveReader(BaseReader):
    extension = ""

    def validate(self, file: Path) -> None:
        validate_zip_archive(Path(file))

    def make_budget(
        self,
        file: Path,
        extra_info: dict[str, Any] | None,
    ) -> _ExtractionBudget:
        return _ExtractionBudget(Path(file), extra_info)


class DocxReader(_ArchiveReader):
    extension = "docx"

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            from docx import Document as WordDocument
            from docx.oxml.ns import qn
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as err:
            raise ImportError("python-docx is required to read DOCX files.") from err
        path = Path(file)
        self.validate(path)
        try:
            document = WordDocument(str(path))
        except Exception as err:
            raise MalformedFormatError(
                "The DOCX document could not be opened safely."
            ) from err
        parts: list[str] = []
        total_length = 0
        limits = text_limits()
        try:
            for child in document.element.body.iterchildren():
                if child.tag == qn("w:p"):
                    value = Paragraph(child, document._body).text.strip()
                    if value:
                        total_length += len(value) + (1 if parts else 0)
                        if total_length > limits.max_characters:
                            raise _limit_error(
                                "Extracted text exceeds the configured per-file character limit."
                            )
                        parts.append(value)
                elif child.tag == qn("w:tbl"):
                    table = Table(child, document._body)
                    for row in table.rows:
                        values = [cell.text.strip() for cell in row.cells]
                        if any(values):
                            value = " | ".join(values)
                            total_length += len(value) + (1 if parts else 0)
                            if total_length > limits.max_characters:
                                raise _limit_error(
                                    "Extracted text exceeds the configured per-file character limit."
                                )
                            parts.append(value)
        except IngestionResourceError:
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The DOCX document content could not be read safely."
            ) from err
        text = "\n".join(parts)
        budget = self.make_budget(path, extra_info)
        if not text.strip():
            budget.add(
                "",
                parser_warning=(
                    "DOCX contains no extractable text; image-only content requires external OCR or conversion."
                ),
            )
            return budget.result()
        budget.add(text)
        return budget.result()


class PptxReader(_ArchiveReader):
    extension = "pptx"

    def __init__(
        self, raise_on_error: bool = True, num_workers: int = 0, **kwargs: Any
    ) -> None:
        self.raise_on_error = raise_on_error
        self.num_workers = num_workers

    @staticmethod
    def _shape_parts(shape) -> list[str]:
        parts: list[str] = []
        try:
            if getattr(shape, "has_text_frame", False):
                value = "\n".join(
                    paragraph.text.strip()
                    for paragraph in shape.text_frame.paragraphs
                    if paragraph.text.strip()
                )
                if value:
                    parts.append(value)
        except Exception as err:
            raise MalformedFormatError(
                "The PPTX shape text could not be read safely."
            ) from err
        try:
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    values = [cell.text.strip() for cell in row.cells]
                    if any(values):
                        parts.append(" | ".join(values))
        except Exception as err:
            raise MalformedFormatError(
                "The PPTX table text could not be read safely."
            ) from err
        try:
            if getattr(shape, "shape_type", None) == 6:
                for child in shape.shapes:
                    parts.extend(PptxReader._shape_parts(child))
        except Exception as err:
            raise MalformedFormatError(
                "The PPTX grouped shape could not be read safely."
            ) from err
        return parts

    @staticmethod
    def _notes_text(slide) -> str:
        try:
            notes_frame = slide.notes_slide.notes_text_frame
            return "\n".join(
                paragraph.text.strip()
                for paragraph in notes_frame.paragraphs
                if paragraph.text.strip()
            ).strip()
        except Exception:
            return ""

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            from pptx import Presentation
        except ImportError as err:
            raise ImportError("python-pptx is required to read PPTX files.") from err
        path = Path(file)
        self.validate(path)
        try:
            presentation = Presentation(str(path))
        except Exception as err:
            raise MalformedFormatError(
                "The PPTX presentation could not be opened safely."
            ) from err
        budget = self.make_budget(path, extra_info)
        try:
            slides = list(presentation.slides)
            if len(slides) > budget.limits.max_documents:
                raise _limit_error(
                    "Presentation slide count exceeds the configured limit."
                )
            for slide_number, slide in enumerate(slides, start=1):
                parts: list[str] = []
                for shape in slide.shapes:
                    parts.extend(self._shape_parts(shape))
                notes = self._notes_text(slide)
                if notes:
                    parts.append("Speaker notes: " + notes)
                text = "\n".join(parts)
                metadata = {
                    "slide_number": slide_number,
                    "has_text": bool(text.strip()),
                }
                if not text.strip():
                    budget.add(
                        "",
                        metadata,
                        (
                            "PPTX slide contains no extractable text; image-only slides require external OCR or conversion."
                        ),
                    )
                else:
                    budget.add(text, metadata)
        except (IngestionResourceError, MalformedFormatError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The PPTX presentation content could not be read safely."
            ) from err
        if not budget.documents:
            raise EmptyFormatError("The PPTX presentation contained no slides.")
        return budget.result()


class EpubReader(_ArchiveReader):
    extension = "epub"

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            import ebooklib
            import html2text
            from ebooklib import epub
        except ImportError as err:
            raise ImportError(
                "ebooklib and html2text are required to read EPUB files."
            ) from err
        path = Path(file)
        self.validate(path)
        reader = None
        try:
            reader = epub.EpubReader(str(path), options={"ignore_ncx": True})
            book = reader.load()
            reader.process()
        except Exception as err:
            raise MalformedFormatError(
                "The EPUB document could not be opened safely."
            ) from err
        finally:
            archive = getattr(reader, "zf", None)
            close = getattr(archive, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        parts: list[str] = []
        current_length = 0
        limits = text_limits()
        try:
            for spine_index, spine_entry in enumerate(book.spine, start=1):
                if spine_index > limits.max_documents:
                    raise _limit_error(
                        "EPUB document count exceeds the configured limit."
                    )
                item_id = (
                    spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
                )
                item = book.get_item_with_id(item_id)
                if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                    continue
                content = item.get_content()
                if len(content) > archive_limits().max_entry_uncompressed_bytes:
                    raise _limit_error(
                        "EPUB document item exceeds the configured limit."
                    )
                value = html2text.html2text(content.decode("utf-8", errors="replace"))
                if value.strip():
                    value = value.strip()
                    current_length += len(value) + (1 if parts else 0)
                    if current_length > limits.max_characters:
                        raise _limit_error(
                            "Extracted text exceeds the configured per-file character limit."
                        )
                    parts.append(value)
        except (IngestionResourceError, MalformedFormatError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The EPUB document content could not be read safely."
            ) from err
        text = "\n".join(parts)
        budget = self.make_budget(path, extra_info)
        if not text.strip():
            budget.add(
                "",
                parser_warning=(
                    "EPUB contains no extractable text; image-only content requires external OCR or conversion."
                ),
            )
            return budget.result()
        budget.add(text)
        return budget.result()


class PDFReader(_ArchiveReader):
    extension = "pdf"

    def __init__(self, return_full_document: bool = True, **kwargs: Any) -> None:
        self.return_full_document = return_full_document

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            import pypdf
        except ImportError as err:
            raise ImportError("pypdf is required to read PDF files.") from err
        path = Path(file)
        limits = text_limits()
        _read_source_bytes(path, limits)
        try:
            reader = pypdf.PdfReader(str(path), strict=False)
            page_count = len(reader.pages)
            if page_count > limits.max_pdf_pages:
                raise _limit_error("PDF page count exceeds the configured limit.")
            page_text = []
            total_length = 0
            for page in reader.pages:
                value = page.extract_text() or ""
                total_length += len(value) + (1 if page_text else 0)
                if total_length > limits.max_characters:
                    raise _limit_error(
                        "PDF text exceeds the configured per-file character limit."
                    )
                page_text.append(value)
            text = "\n".join(page_text)
        except (IngestionResourceError, EmptyFormatError):
            raise
        except Exception as err:
            raise MalformedFormatError(
                "The PDF document could not be parsed safely."
            ) from err
        budget = _ExtractionBudget(path, extra_info, limits)
        if not text.strip():
            budget.add(
                "",
                parser_warning=(
                    "PDF contains no extractable text; image-only or blank pages require an external OCR engine or conversion. No OCR is bundled."
                ),
            )
        else:
            budget.add(text)
        return budget.result()


class RTFReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as err:
            raise ImportError("striprtf is required to read RTF files.") from err
        path = Path(file)
        limits = text_limits()
        text = rtf_to_text(_read_utf8(path))
        if len(text) > limits.max_characters:
            raise _limit_error(
                "Extracted text exceeds the configured per-file character limit."
            )
        if not text.strip():
            raise EmptyFormatError("The RTF file contained no extractable text.")
        budget = _ExtractionBudget(path, extra_info, limits)
        budget.add(text.strip())
        return budget.result()


class LegacyFormatReader(BaseReader):
    def __init__(self, extension: str) -> None:
        self.extension = extension

    def load_data(
        self,
        file: Path,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        raise LegacyFormatError(legacy_conversion_message(self.extension))


def get_parser_capability(extension: str) -> ParserCapability | None:
    return PARSER_CAPABILITIES.get(str(extension).lower().lstrip("."))


def capability_matrix() -> dict[str, dict[str, str]]:
    return {
        extension: capability.as_dict()
        for extension, capability in PARSER_CAPABILITIES.items()
    }


def legacy_conversion_message(extension: str) -> str:
    suffix = str(extension).lower().lstrip(".")
    target = "DOCX" if suffix == "doc" else "PPTX"
    return (
        f"Legacy .{suffix} files are not parsed by the local pipeline. "
        f"Convert the file to .{target.lower()} with an external office converter "
        f"and upload the .{target.lower()} file instead."
    )


def build_file_extractors() -> dict[str, BaseReader]:
    return {
        ".csv": DelimitedTextReader(","),
        ".doc": LegacyFormatReader("doc"),
        ".docx": DocxReader(),
        ".eml": EmailReader(),
        ".epub": EpubReader(),
        ".htm": WholeDocumentHTMLReader(),
        ".html": WholeDocumentHTMLReader(),
        ".ipynb": IPYNBReader(),
        ".json": JSONReader(),
        ".jsonl": JSONLReader(),
        ".markdown": MarkdownTextReader(),
        ".mbox": MboxReader(),
        ".md": MarkdownTextReader(),
        ".mhtml": MHTMLReader(),
        ".msg": MSGReader(),
        ".odt": ODTReader(),
        ".pdf": PDFReader(return_full_document=True),
        ".ppt": LegacyFormatReader("ppt"),
        ".pptx": PptxReader(raise_on_error=True, num_workers=0),
        ".rtf": RTFReader(),
        ".tsv": DelimitedTextReader("\t"),
        ".txt": UTF8TextReader(),
        ".xls": XLSReader(),
        ".xlsx": XLSXReader(),
        ".xml": XMLTextReader(),
    }


def validate_document_budget(
    documents: Iterable[Any],
    *,
    max_documents: int | None = None,
    max_characters: int | None = None,
) -> None:
    limits = text_limits()
    selected_documents = (
        limits.max_documents if max_documents is None else max_documents
    )
    selected_characters = (
        limits.max_characters if max_characters is None else max_characters
    )
    characters = 0
    for count, document in enumerate(documents, start=1):
        if count > selected_documents:
            raise _limit_error(
                f"Too many documents were loaded. Limit: {selected_documents}."
            )
        if hasattr(document, "get_content"):
            value = document.get_content() or ""
        elif hasattr(document, "text"):
            value = document.text or ""
        else:
            value = str(document)
        characters += len(str(value))
        if characters > selected_characters:
            raise _limit_error(
                "Loaded documents exceed the configured ingestion text limit."
            )


__all__ = [
    "CAPABILITY_CATEGORIES",
    "CAPABILITY_REGISTRY",
    "DEFAULT_ARCHIVE_LIMITS",
    "DEFAULT_TEXT_LIMITS",
    "MAX_ARCHIVE_COMPRESSED_BYTES",
    "MAX_ARCHIVE_COMPRESSION_RATIO",
    "MAX_ARCHIVE_ENTRY_UNCOMPRESSED_BYTES",
    "MAX_ARCHIVE_MEMBERS",
    "MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES",
    "MAX_EXTRACTED_CHARS_PER_FILE",
    "MAX_INGESTED_DOCUMENTS",
    "MAX_INGESTED_TEXT_CHARS",
    "MAX_JSON_DEPTH",
    "MAX_MIME_DEPTH",
    "MAX_MIME_PARTS",
    "MAX_RECORDS_PER_FILE",
    "MAX_SOURCE_FILE_BYTES",
    "MAX_TABLE_ROWS",
    "MAX_ZIP_COMPRESSED_BYTES",
    "MAX_ZIP_COMPRESSION_RATIO",
    "MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES",
    "MAX_ZIP_MEMBERS",
    "MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES",
    "PARSER_CAPABILITIES",
    "PARSER_CAPABILITIES_BY_EXTENSION",
    "PARSER_REGISTRY",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FILE_EXTENSIONS",
    "ZIP_FORMATS",
    "ArchiveLimits",
    "EmptyFormatError",
    "FormatIngestionError",
    "IngestionResourceError",
    "LegacyFormatError",
    "MalformedFormatError",
    "TextLimits",
    "UnsafeArchiveError",
    "archive_limits",
    "build_file_extractors",
    "capability_matrix",
    "get_archive_limits",
    "get_parser_capability",
    "get_text_limits",
    "html_to_text",
    "legacy_conversion_message",
    "parse_json_text",
    "preflight_input_file",
    "text_limits",
    "validate_document_budget",
    "validate_zip_archive",
]
