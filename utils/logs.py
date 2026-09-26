import contextlib
import logging
import logging.handlers
import os
import re
import sys
from urllib.parse import urlsplit

DEFAULT_LOG_MAX_BYTES = 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 3
_REDACTED = "[REDACTED]"
_REDACTED_URL = "[REDACTED_URL]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "account_token",
    "auth",
    "authorization",
    "client_secret",
    "credential",
    "password",
    "passwd",
    "refresh_token",
    "secret",
    "token",
}
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_BEARER_RE = re.compile(r"\bBearer(?:\s|%20)+[^\s,;]+", re.IGNORECASE)
_HEADER_RE = re.compile(
    r"(?i)\b(authorization|proxy-authorization|x-api-key|api-key)\s*[:=]\s*[^\s,;]+"
)
_LABELED_API_KEY_RE = re.compile(
    r"(?i)\b(?:r2r\s+|openai\s+|github\s+)?api\s*key\s*[:=]\s*[^\s,;]+"
)
_KEY_VALUE_RE = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret|secret|token)[\"']?\s*[:=]\s*)([\"']?)([^\s,;\"'}]+)\2"
)
_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret|secret|token|key)=)([^&#\s]*)"
)
_PROVIDER_TOKEN_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"
)


def _redact_url(value: str) -> str:
    raw = str(value)
    trailing = ""
    while raw and raw[-1] in ".,;)]}":
        trailing = raw[-1] + trailing
        raw = raw[:-1]
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return _REDACTED_URL
    query_keys = {
        part.split("=", 1)[0].casefold() for part in parsed.query.split("&") if part
    }
    fragment_keys = {
        part.split("=", 1)[0].casefold() for part in parsed.fragment.split("&") if part
    }
    sensitive = bool(
        parsed.username
        or parsed.password
        or query_keys & _SENSITIVE_KEYS
        or fragment_keys & _SENSITIVE_KEYS
    )
    if sensitive:
        return _REDACTED_URL + trailing
    if parsed.query:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{_REDACTED}" + trailing
    return raw + trailing


def redact_text(value) -> str:
    text = str(value or "")
    text = _URL_RE.sub(lambda match: _redact_url(match.group(0)), text)
    text = _BEARER_RE.sub(f"Bearer {_REDACTED}", text)
    text = _HEADER_RE.sub(lambda match: f"{match.group(1)}: {_REDACTED}", text)
    text = _LABELED_API_KEY_RE.sub(
        lambda match: f"{match.group(0).split(':')[0].split('=')[0]}: {_REDACTED}",
        text,
    )
    text = _QUERY_RE.sub(lambda match: f"{match.group(1)}{_REDACTED}", text)
    text = _KEY_VALUE_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}{match.group(2)}",
        text,
    )
    text = _PROVIDER_TOKEN_RE.sub(_REDACTED, text)
    return text


def redact(value) -> str:
    return redact_text(value)


def safe_log_exception(error: BaseException) -> str:
    text = redact_text(str(error))
    if len(text) > 512:
        text = text[:509] + "..."
    return f"{type(error).__name__}: {text}"


def safe_user_error(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "The operation timed out before it completed."
    category = getattr(error, "category", None)
    if category == "timeout":
        return "The operation timed out before it completed."
    if category in {"invalid_url", "url_limit"}:
        return "The website input is invalid. Check the URL and try again."
    if category in {"blocked_destination", "dns_failure", "http_failure", "anti_bot"}:
        return "The website could not be retrieved safely. Check the URL and try again."
    if category == "response_too_large":
        return "The retrieved content is too large to process."
    return "The operation could not be completed. Check the input and try again."


def redact_exception(error: BaseException) -> str:
    return safe_log_exception(error)


def safe_error_message(error: BaseException) -> str:
    return safe_user_error(error)


class RedactingFilter(logging.Filter):
    def filter(self, record):
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = redact_text(message)[:4000]
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _parse_positive(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _new_file_handler(log_file, max_bytes, backup_count):
    parent = os.path.dirname(os.path.abspath(log_file))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max(1, int(max_bytes)),
        backupCount=max(0, int(backup_count)),
        encoding="utf-8",
    )


def setup_logger(
    log_file: str | None = None,
    level: int | str = logging.INFO,
    *,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    maxBytes: int | None = None,
    backupCount: int | None = None,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False
    configured_file = log_file or os.environ.get(
        "DOCMIND_LOG_FILE", os.path.join(os.getcwd(), "docmind.log")
    )
    configured_max = (
        max_bytes
        or maxBytes
        or _parse_positive(
            os.environ.get("DOCMIND_LOG_MAX_BYTES"), DEFAULT_LOG_MAX_BYTES
        )
    )
    configured_backups = (
        backup_count
        if backup_count is not None
        else (
            backupCount
            if backupCount is not None
            else _parse_positive(
                os.environ.get("DOCMIND_LOG_BACKUP_COUNT"), DEFAULT_LOG_BACKUP_COUNT
            )
        )
    )
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        with contextlib.suppress(OSError):
            handler.close()
    formatter = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    redactor = RedactingFilter()
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redactor)
    logger.addHandler(console_handler)
    try:
        file_handler = _new_file_handler(
            configured_file, configured_max, configured_backups
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        logger.addHandler(file_handler)
    except OSError:
        pass
    return logger


log = setup_logger()


__all__ = [
    "DEFAULT_LOG_BACKUP_COUNT",
    "DEFAULT_LOG_MAX_BYTES",
    "RedactingFilter",
    "log",
    "redact",
    "redact_exception",
    "redact_text",
    "safe_error_message",
    "safe_log_exception",
    "safe_user_error",
    "setup_logger",
]
