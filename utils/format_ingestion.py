import contextlib
import csv
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


@dataclass(frozen=True)
class ParserCapability:
    category: str
    parser: str
    dependency: str
    note: str

    def as_dict(self) -> Dict[str, str]:
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
        "docx2txt",
        "Text and tables in DOCX packages are extracted locally.",
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
        "EPUB document items are converted to text locally.",
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
        "MarkdownReader",
        "LlamaIndex MarkdownReader",
        "Markdown sections and their text are extracted.",
    ),
    "mbox": ParserCapability(
        "verified",
        "MboxReader",
        "Python mailbox and beautifulsoup4",
        "Mailbox messages are parsed locally; unsupported message parts are not silently counted as documents.",
    ),
    "md": ParserCapability(
        "verified",
        "MarkdownReader",
        "LlamaIndex MarkdownReader",
        "Markdown sections and their text are extracted.",
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
        "ODF paragraphs and headings are extracted locally.",
    ),
    "pdf": ParserCapability(
        "optional_dependency",
        "PDFReader",
        "pypdf",
        "Text PDFs are supported; image-only PDFs require an external OCR engine.",
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
        "Slide text, tables, and speaker notes are extracted locally; image-only slides require external OCR.",
    ),
    "rtf": ParserCapability(
        "verified",
        "RTFReader",
        "striprtf",
        "RTF control words are removed and visible text is extracted.",
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
        "Every worksheet row is converted with its column header.",
    ),
    "xlsx": ParserCapability(
        "verified",
        "XLSXReader",
        "openpyxl",
        "Every worksheet row is converted with its column header.",
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
    f".{extension}": capability
    for extension, capability in PARSER_CAPABILITIES.items()
}
CAPABILITY_REGISTRY = PARSER_CAPABILITIES_BY_EXTENSION
PARSER_REGISTRY = PARSER_CAPABILITIES_BY_EXTENSION
PARSER_CAPABILITIES_WITH_DOTS = PARSER_CAPABILITIES_BY_EXTENSION
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_EXTENSIONS


class LegacyFormatError(ValueError):
    """Raised when a legacy binary office format needs conversion."""


def get_parser_capability(extension: str) -> Optional[ParserCapability]:
    return PARSER_CAPABILITIES.get(str(extension).lower().lstrip("."))


def capability_matrix() -> Dict[str, Dict[str, str]]:
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


def _read_utf8(file: Path) -> str:
    return file.read_bytes().decode("utf-8-sig", errors="replace")


def _metadata(file: Path, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = {
        "file_name": file.name,
        "file_path": str(file),
    }
    if extra_info:
        metadata.update(extra_info)
    return metadata


def _clean_text(text: str) -> str:
    return "\n".join(
        " ".join(line.split())
        for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    )


class _HTMLTextParser(HTMLParser):
    _BLOCK_TAGS = {
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
    _SKIP_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(
        self, tag: str, attrs: List[tuple[str, Optional[str]]]
    ) -> None:
        if tag.lower() not in self._SKIP_TAGS and not self._skip_depth:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return _clean_text("".join(self._parts))


def html_to_text(value: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(value)
    parser.close()
    return parser.text()


class WholeDocumentHTMLReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        text = html_to_text(_read_utf8(Path(file)))
        if not text:
            raise ValueError("HTML contained no extractable text.")
        return [Document(text=text, metadata=_metadata(Path(file), extra_info))]


def _format_delimited_text(text: str, delimiter: str) -> str:
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if not rows:
        return text
    headers = [
        str(value).strip() or f"Column {index + 1}"
        for index, value in enumerate(rows[0])
    ]
    lines = ["Columns: " + ", ".join(headers)]
    for row_number, row in enumerate(rows[1:], start=1):
        values = []
        for index, value in enumerate(row):
            value = str(value).strip()
            if not value:
                continue
            header = headers[index] if index < len(headers) else f"Extra column {index + 1}"
            values.append(f"{header} is {value}")
        lines.append(
            f"Row {row_number}: " + (", ".join(values) if values else "(empty row)")
        )
    return "\n".join(lines) + "\n\nOriginal table:\n" + text


class DelimitedTextReader(BaseReader):
    def __init__(self, delimiter: str) -> None:
        self.delimiter = delimiter

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        path = Path(file)
        text = _read_utf8(path)
        if not text.strip():
            raise ValueError("Delimited file contained no extractable text.")
        try:
            content = _format_delimited_text(text, self.delimiter)
        except csv.Error:
            content = text
        metadata = _metadata(path, extra_info)
        metadata["tabular_verbalized"] = True
        return [Document(text=content, metadata=metadata)]


class UTF8TextReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        text = _read_utf8(Path(file))
        if not text.strip():
            raise ValueError("Text file contained no extractable text.")
        return [Document(text=text, metadata=_metadata(Path(file), extra_info))]


class JSONReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        path = Path(file)
        text = _read_utf8(path)
        try:
            json.loads(text)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON document at line {err.lineno}.") from err
        if not text.strip():
            raise ValueError("JSON file contained no extractable text.")
        return [Document(text=text, metadata=_metadata(path, extra_info))]


class JSONLReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        path = Path(file)
        text = _read_utf8(path)
        documents: List[Document] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            metadata = _metadata(path, extra_info)
            metadata["record_index"] = line_number
            try:
                json.loads(line)
            except json.JSONDecodeError:
                metadata["parser_warning"] = (
                    f"Invalid JSONL record on line {line_number}; raw line retained."
                )
            documents.append(Document(text=line, metadata=metadata))
        if not documents:
            raise ValueError("JSONL file contained no records.")
        return documents


class IPYNBReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            import nbformat
            from nbconvert.exporters import ScriptExporter
        except ImportError as err:
            raise ImportError(
                "nbconvert and nbformat are required to read IPYNB files."
            ) from err
        path = Path(file)
        try:
            notebook = nbformat.read(str(path), as_version=4)
        except Exception as err:
            raise ValueError("Notebook document could not be parsed.") from err
        try:
            text, _ = ScriptExporter().from_notebook_node(notebook)
        except Exception:
            parts = []
            for cell in notebook.get("cells", []):
                source = str(cell.get("source", ""))
                if source.strip():
                    parts.append(source)
            text = "\n\n".join(parts)
        if not text:
            raise ValueError("Notebook document contained no extractable cells.")
        return [Document(text=text, metadata=_metadata(path, extra_info))]


class XMLTextReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            from defusedxml import ElementTree
        except ImportError as err:
            raise ImportError("defusedxml is required to read XML files.") from err
        path = Path(file)
        try:
            root = ElementTree.parse(path).getroot()
        except Exception as err:
            raise ValueError("XML document could not be parsed safely.") from err
        text = _clean_text(" ".join(part for part in root.itertext() if part))
        if not text:
            raise ValueError("XML document contained no extractable text.")
        return [Document(text=text, metadata=_metadata(path, extra_info))]


def _decode_email_part(part: Any) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        payload = part.get_payload()
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")
    return str(payload)


def _email_text(message: Any) -> str:
    header_values = []
    for label, key in (("Subject", "subject"), ("From", "from"), ("To", "to")):
        value = message.get(key)
        if value:
            header_values.append(f"{label}: {value}")
    text_parts = []
    html_parts = []
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_maintype() == "multipart":
            continue
        disposition = str(part.get("Content-Disposition", ""))
        if "attachment" in disposition.lower():
            continue
        content = _decode_email_part(part)
        if not content.strip():
            continue
        if part.get_content_type() == "text/plain":
            text_parts.append(content)
        elif part.get_content_type() == "text/html":
            html_parts.append(content)
    if not text_parts and html_parts:
        text_parts = [html_to_text(content) for content in html_parts]
    body = _clean_text("\n".join(text_parts))
    return _clean_text("\n".join(header_values + ([body] if body else [])))


class EmailReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        path = Path(file)
        try:
            message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        except Exception as err:
            raise ValueError("Email message could not be parsed.") from err
        text = _email_text(message)
        if not text:
            raise ValueError("Email message contained no text parts.")
        return [Document(text=text, metadata=_metadata(path, extra_info))]


class MHTMLReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        path = Path(file)
        try:
            message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        except Exception as err:
            raise ValueError("MHTML archive could not be parsed.") from err
        text = _email_text(message)
        if not text:
            raw = _read_utf8(path)
            text = html_to_text(raw) if "<" in raw else raw.strip()
        if not text:
            raise ValueError("MHTML archive contained no text parts.")
        return [Document(text=text, metadata=_metadata(path, extra_info))]


class MboxReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        import mailbox
        from email.parser import BytesParser

        path = Path(file)
        mailbox_file = mailbox.mbox(
            str(path), factory=BytesParser(policy=policy.default).parse
        )
        documents = []
        try:
            for message_index, message in enumerate(mailbox_file, start=1):
                text = _email_text(message)
                metadata = _metadata(path, extra_info)
                metadata["message_index"] = message_index
                if not text:
                    text = f"Message {message_index} contained no text parts."
                    metadata["parser_warning"] = (
                        f"MBOX message {message_index} had no text parts; record retained."
                    )
                documents.append(Document(text=text, metadata=metadata))
        finally:
            mailbox_file.close()
        if not documents:
            raise ValueError("MBOX file contained no messages.")
        return documents


def _odt_fallback_text(path: Path) -> str:
    try:
        from defusedxml import ElementTree

        with zipfile.ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("content.xml"))
    except Exception:
        return ""
    return _clean_text(" ".join(part for part in root.itertext() if part))


class ODTReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            from odf import teletype
            from odf import text as odf_text
            from odf.opendocument import load
        except ImportError as err:
            raise ImportError("odfpy is required to read ODT files.") from err
        path = Path(file)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                document = load(str(path))
        except Exception as err:
            text = _odt_fallback_text(path)
            if not text:
                raise ValueError("ODT document could not be opened.") from err
            return [Document(text=text, metadata=_metadata(path, extra_info))]
        paragraphs = document.getElementsByType(odf_text.P)
        parts = [teletype.extractText(paragraph).strip() for paragraph in paragraphs]
        parts = [part for part in parts if part]
        if not parts:
            try:
                text = teletype.extractText(document).strip()
            except Exception:
                text = ""
        else:
            text = "\n".join(parts)
        if not text:
            text = _odt_fallback_text(path)
        if not text:
            raise ValueError("ODT document contained no extractable text.")
        return [Document(text=text, metadata=_metadata(path, extra_info))]


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
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            import extract_msg
            import olefile
        except ImportError as err:
            raise ImportError(
                "extract-msg and olefile are required to read MSG files."
            ) from err
        path = Path(file)
        if not olefile.isOleFile(str(path)):
            raise ValueError("MSG file is not an OLE compound file.")
        try:
            message = extract_msg.openMsg(str(path))
        except Exception as err:
            raise ValueError("MSG message could not be opened.") from err
        try:
            body = _message_value(getattr(message, "body", ""))
            if not body.strip():
                body = html_to_text(_message_value(getattr(message, "htmlBody", "")))
            values = [
                ("Subject", _message_value(getattr(message, "subject", ""))),
                ("From", _message_value(getattr(message, "sender", ""))),
                ("To", _message_value(getattr(message, "to", ""))),
            ]
            text = _clean_text(
                "\n".join(
                    [f"{label}: {value}" for label, value in values if value]
                    + ([body] if body.strip() else [])
                )
            )
        finally:
            close = getattr(message, "close", None)
            if callable(close):
                close()
        if not text:
            raise ValueError("MSG message contained no extractable text.")
        return [Document(text=text, metadata=_metadata(path, extra_info))]


def _spreadsheet_text(sheet_name: str, rows: Iterable[Iterable[Any]]) -> str:
    rows = [list(row) for row in rows]
    if not rows:
        return ""
    headers = [
        str(value).strip() or f"Column {index + 1}"
        for index, value in enumerate(rows[0])
    ]
    lines = [f"Sheet: {sheet_name}", "Columns: " + ", ".join(headers)]
    raw_lines = []
    for row_number, row in enumerate(rows[1:], start=1):
        values = []
        raw_values = []
        for index, value in enumerate(row):
            value_text = _message_value(value)
            raw_values.append(value_text)
            if not value_text.strip():
                continue
            header = headers[index] if index < len(headers) else f"Extra column {index + 1}"
            values.append(f"{header} is {value_text}")
        lines.append(
            f"Row {row_number}: " + (", ".join(values) if values else "(empty row)")
        )
        raw_lines.append(" | ".join(raw_values))
    lines.append("Raw sheet:\n" + "\n".join(raw_lines))
    return "\n".join(lines)


class XLSXReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            from openpyxl import load_workbook
        except ImportError as err:
            raise ImportError("openpyxl is required to read XLSX files.") from err
        path = Path(file)
        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
        except Exception as err:
            raise ValueError("XLSX workbook could not be opened.") from err
        documents = []
        try:
            for worksheet in workbook.worksheets:
                text = _spreadsheet_text(
                    worksheet.title, worksheet.iter_rows(values_only=True)
                )
                if text:
                    metadata = _metadata(path, extra_info)
                    metadata["sheet_name"] = worksheet.title
                    documents.append(Document(text=text, metadata=metadata))
        finally:
            workbook.close()
        if not documents:
            raise ValueError("XLSX workbook contained no extractable rows.")
        return documents


class XLSReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            import xlrd
        except ImportError as err:
            raise ImportError("xlrd is required to read XLS files.") from err
        path = Path(file)
        try:
            workbook = xlrd.open_workbook(str(path), on_demand=True)
        except Exception as err:
            raise ValueError("XLS workbook could not be opened.") from err
        documents = []
        try:
            for sheet_name in workbook.sheet_names():
                worksheet = workbook.sheet_by_name(sheet_name)
                text = _spreadsheet_text(
                    sheet_name,
                    (
                        [worksheet.cell_value(row_index, column_index) for column_index in range(worksheet.ncols)]
                        for row_index in range(worksheet.nrows)
                    ),
                )
                if text:
                    metadata = _metadata(path, extra_info)
                    metadata["sheet_name"] = sheet_name
                    documents.append(Document(text=text, metadata=metadata))
        finally:
            workbook.release_resources()
        if not documents:
            raise ValueError("XLS workbook contained no extractable rows.")
        return documents


class LegacyFormatReader(BaseReader):
    def __init__(self, extension: str) -> None:
        self.extension = extension

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        raise LegacyFormatError(legacy_conversion_message(self.extension))


def build_file_extractors() -> Dict[str, BaseReader]:
    from llama_index.readers.file.docs import DocxReader, PDFReader
    from llama_index.readers.file.epub import EpubReader
    from llama_index.readers.file.markdown import MarkdownReader
    from llama_index.readers.file.rtf import RTFReader
    from llama_index.readers.file.slides import PptxReader

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
        ".markdown": MarkdownReader(),
        ".mbox": MboxReader(),
        ".md": MarkdownReader(),
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
