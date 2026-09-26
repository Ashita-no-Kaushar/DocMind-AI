import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import html2text
import requests
import urllib3
from llama_index.core import Document
from urllib3.util import Timeout

import utils.logs as logs
from utils.format_ingestion import SUPPORTED_EXTENSIONS
from utils.ingestion_lock import ingestion_lock

GITHUB_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SAFE_UPLOAD_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
ALLOWED_UPLOAD_EXTENSIONS = {f".{extension}" for extension in SUPPORTED_EXTENSIONS}
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "metadata.goog",
        "metadata.azure.internal",
        "metadata.tencentyun.com",
        "metadata.openstack.org",
        "metadata.cloudstack.org",
        "instance-data",
        "instance-data.ec2.internal",
        "kubernetes.default.svc",
        "kubernetes.default.svc.cluster.local",
    }
)
BLOCKED_METADATA_IP_ADDRESSES = frozenset(
    {
        "100.100.100.200",
        "168.63.129.16",
        "169.254.169.254",
        "169.254.170.2",
        "169.254.0.23",
        "169.254.42.42",
        "fd00:ec2::254",
        "fd20:ce::254",
    }
)
MAX_WEBSITE_URLS = 6
MAX_WEBSITE_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_WEBSITE_REDIRECTS = 3
WEBSITE_REQUEST_TIMEOUT = (5, 20)
MAX_WEBSITE_INGESTION_SECONDS = 180
WEBSITE_ERROR_CATEGORIES = frozenset(
    {
        "invalid_url",
        "blocked_destination",
        "dns_failure",
        "http_failure",
        "anti_bot",
        "timeout",
        "empty_page",
        "url_limit",
        "response_too_large",
        "unsupported_content",
    }
)
MAX_UPLOAD_FILES = 10
MAX_UPLOAD_FILE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 100 * 1024 * 1024
GIT_CLONE_TIMEOUT_SECONDS = 120
GITHUB_MAX_REPOSITORY_DOWNLOAD_BYTES = 100 * 1024 * 1024
GITHUB_MAX_CHECKOUT_FILES = 10_000
GITHUB_MAX_FILE_BYTES = 10 * 1024 * 1024
GITHUB_MAX_UNPACKED_BYTES = 250 * 1024 * 1024
GITHUB_MAX_CLONE_OUTPUT_BYTES = 64 * 1024
GITHUB_METADATA_TIMEOUT_SECONDS = 10
GITHUB_MAX_METADATA_RESPONSE_BYTES = 2 * 1024 * 1024
GITHUB_CHECKOUT_POLL_SECONDS = 0.1
GITHUB_API_URL = "https://api.github.com/repos"
GITHUB_CLONE_OUTPUT_CHUNK_BYTES = 64 * 1024
_ORIGINAL_SUBPROCESS_RUN = subprocess.run
MAX_GITHUB_REPOSITORY_DOWNLOAD_BYTES = GITHUB_MAX_REPOSITORY_DOWNLOAD_BYTES
MAX_GITHUB_CHECKOUT_FILES = GITHUB_MAX_CHECKOUT_FILES
MAX_GITHUB_FILE_BYTES = GITHUB_MAX_FILE_BYTES
MAX_GITHUB_UNPACKED_BYTES = GITHUB_MAX_UNPACKED_BYTES
MAX_GITHUB_CLONE_OUTPUT_BYTES = GITHUB_MAX_CLONE_OUTPUT_BYTES
WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "clock$", "conin$", "conout$"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


class GitHubIngestionError(ValueError):
    def __init__(self, category, message):
        self.category = str(category or "clone_failed")
        self.message = str(message or "GitHub repository ingestion failed.")
        super().__init__(self.message)


@dataclass(frozen=True)
class GitHubResourceLimits:
    repository_download_bytes: int = GITHUB_MAX_REPOSITORY_DOWNLOAD_BYTES
    checkout_files: int = GITHUB_MAX_CHECKOUT_FILES
    file_bytes: int = GITHUB_MAX_FILE_BYTES
    unpacked_bytes: int = GITHUB_MAX_UNPACKED_BYTES
    clone_output_bytes: int = GITHUB_MAX_CLONE_OUTPUT_BYTES
    timeout_seconds: float = float(GIT_CLONE_TIMEOUT_SECONDS)
    metadata_timeout_seconds: float = float(GITHUB_METADATA_TIMEOUT_SECONDS)

    @property
    def max_repository_download_bytes(self):
        return self.repository_download_bytes

    @property
    def max_checkout_files(self):
        return self.checkout_files

    @property
    def max_file_bytes(self):
        return self.file_bytes

    @property
    def max_unpacked_bytes(self):
        return self.unpacked_bytes

    @property
    def max_clone_output_bytes(self):
        return self.clone_output_bytes


def _env_limit(name, default):
    names = name if isinstance(name, (tuple, list)) else (name,)
    raw = next(
        (os.environ.get(item) for item in names if os.environ.get(item) is not None),
        None,
    )
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def github_resource_limits(**overrides) -> GitHubResourceLimits:
    defaults = GitHubResourceLimits(
        repository_download_bytes=_env_limit(
            (
                "DOCMIND_GITHUB_MAX_DOWNLOAD_BYTES",
                "DOCMIND_GITHUB_MAX_REPOSITORY_DOWNLOAD_BYTES",
                "DOCMIND_GITHUB_MAX_REPO_BYTES",
            ),
            GITHUB_MAX_REPOSITORY_DOWNLOAD_BYTES,
        ),
        checkout_files=_env_limit(
            (
                "DOCMIND_GITHUB_MAX_CHECKOUT_FILES",
                "DOCMIND_GITHUB_MAX_FILES",
            ),
            GITHUB_MAX_CHECKOUT_FILES,
        ),
        file_bytes=_env_limit("DOCMIND_GITHUB_MAX_FILE_BYTES", GITHUB_MAX_FILE_BYTES),
        unpacked_bytes=_env_limit(
            "DOCMIND_GITHUB_MAX_UNPACKED_BYTES", GITHUB_MAX_UNPACKED_BYTES
        ),
        clone_output_bytes=_env_limit(
            (
                "DOCMIND_GITHUB_MAX_CLONE_OUTPUT_BYTES",
                "DOCMIND_GITHUB_MAX_OUTPUT_BYTES",
            ),
            GITHUB_MAX_CLONE_OUTPUT_BYTES,
        ),
        timeout_seconds=_env_float(
            "DOCMIND_GITHUB_CLONE_TIMEOUT_SECONDS", GIT_CLONE_TIMEOUT_SECONDS
        ),
        metadata_timeout_seconds=_env_float(
            "DOCMIND_GITHUB_METADATA_TIMEOUT_SECONDS",
            GITHUB_METADATA_TIMEOUT_SECONDS,
        ),
    )
    values = {
        "repository_download_bytes": overrides.get(
            "repository_download_bytes",
            overrides.get(
                "max_repository_download_bytes",
                overrides.get("max_download_bytes", overrides.get("download_bytes")),
            ),
        ),
        "checkout_files": overrides.get(
            "checkout_files",
            overrides.get("max_checkout_files", overrides.get("max_files")),
        ),
        "file_bytes": overrides.get(
            "file_bytes",
            overrides.get("max_file_bytes", overrides.get("individual_file_bytes")),
        ),
        "unpacked_bytes": overrides.get(
            "unpacked_bytes",
            overrides.get(
                "max_unpacked_bytes",
                overrides.get("max_total_bytes", overrides.get("max_total_size")),
            ),
        ),
        "clone_output_bytes": overrides.get(
            "clone_output_bytes",
            overrides.get("max_clone_output_bytes", overrides.get("max_output_bytes")),
        ),
        "timeout_seconds": overrides.get("timeout_seconds", overrides.get("timeout")),
        "metadata_timeout_seconds": overrides.get("metadata_timeout_seconds"),
    }
    for name, value in values.items():
        if value is not None:
            try:
                parsed = float(value) if name.endswith("seconds") else int(value)
            except (TypeError, ValueError) as err:
                raise ValueError(f"Invalid GitHub resource limit: {name}.") from err
            if parsed <= 0:
                raise ValueError(f"Invalid GitHub resource limit: {name}.")
            if name.endswith("seconds"):
                values[name] = parsed
            else:
                values[name] = int(parsed)
    return GitHubResourceLimits(
        **{
            **defaults.__dict__,
            **{key: value for key, value in values.items() if value is not None},
        }
    )


def _github_status(response):
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", 0)
    try:
        return int(status)
    except (TypeError, ValueError):
        return 0


def _bounded_github_metadata(response):
    headers = getattr(response, "headers", {}) or {}
    declared = headers.get("content-length") or headers.get("Content-Length")
    if declared:
        try:
            if int(declared) > GITHUB_MAX_METADATA_RESPONSE_BYTES:
                return None
        except (TypeError, ValueError):
            pass
    iterator = getattr(response, "iter_content", None)
    try:
        if callable(iterator):
            body = bytearray()
            for chunk in iterator(chunk_size=64 * 1024):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > GITHUB_MAX_METADATA_RESPONSE_BYTES:
                    return None
            return json.loads(bytes(body))
        content = getattr(response, "content", None)
        if (
            isinstance(content, (bytes, bytearray))
            and len(content) > GITHUB_MAX_METADATA_RESPONSE_BYTES
        ):
            return None
        return response.json()
    except Exception:
        return None
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


def github_metadata_preflight(
    repo: str,
    *,
    limits: GitHubResourceLimits | None = None,
    timeout: float | None = None,
    **overrides,
) -> dict | None:
    normalized = normalize_github_repo(repo)
    selected = limits or github_resource_limits(**overrides)
    request_timeout = min(
        float(selected.metadata_timeout_seconds if timeout is None else timeout),
        float(selected.metadata_timeout_seconds),
    )
    request_timeout = max(0.1, request_timeout)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "DocMind-AI-GitHub-Ingestion",
    }
    try:
        response = requests.get(
            f"{GITHUB_API_URL}/{normalized}",
            headers=headers,
            timeout=(min(5.0, request_timeout), request_timeout),
            allow_redirects=False,
            stream=True,
        )
    except Exception:
        return None
    status = _github_status(response)
    if status in {403, 429} or status != 200:
        with suppress(Exception):
            response.close()
        return None
    payload = _bounded_github_metadata(response)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("private") is True
        or str(payload.get("visibility", "")).lower() == "private"
    ):
        raise GitHubIngestionError(
            "private", "Private GitHub repositories are not supported."
        )
    size_bytes = payload.get("size_bytes")
    if size_bytes is None:
        size_kb = payload.get("size")
        if size_kb is None:
            return None
        try:
            size_bytes = int(size_kb) * 1024
        except (TypeError, ValueError):
            return None
    try:
        size_bytes = int(size_bytes)
    except (TypeError, ValueError):
        return None
    if size_bytes < 0:
        return None
    if size_bytes > selected.repository_download_bytes:
        raise GitHubIngestionError(
            "repository_too_large",
            "The GitHub repository exceeds the configured download limit.",
        )
    return {
        "size_bytes": size_bytes,
        "default_branch": payload.get("default_branch"),
        "full_name": payload.get("full_name"),
    }


class WebsiteIngestionError(ValueError):
    def __init__(self, category, message, url=None, status_code=None):
        if category not in WEBSITE_ERROR_CATEGORIES:
            category = "http_failure"
        self.category = category
        self.url = url
        self.status_code = status_code
        self.message = message
        super().__init__(f"{category}: {message}")

    def diagnostic(self):
        return {
            "category": self.category,
            "url": self.url,
            "status_code": self.status_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class _WebsiteTarget:
    url: str
    hostname: str
    port: int
    request_target: str
    addresses: tuple[str, ...]


def remove_dir_retry(path, attempts=8, delay=0.75):
    """Delete a directory tree, retrying transient file locks (Windows)."""
    if not os.path.exists(path):
        return True
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            time.sleep(delay)
    return False


def create_ingestion_work_dir(
    kind: str = "ingestion", destination_base: str | None = None
):
    """Create and return a unique directory owned by one ingestion operation."""
    base = Path(destination_base or os.path.join(os.getcwd(), "data", "work"))
    base.mkdir(parents=True, exist_ok=True)
    safe_kind = re.sub(r"[^A-Za-z0-9_-]+", "-", str(kind or "ingestion")).strip("-")
    return tempfile.mkdtemp(prefix=f"{safe_kind or 'ingestion'}-", dir=str(base))


def cleanup_ingestion_work_dir(path: str | None) -> bool:
    """Remove one operation-owned work directory without touching its parent."""
    if not path or not os.path.exists(path):
        return True
    if not os.path.isdir(path):
        return False
    return remove_dir_retry(path)


def _is_blocked_ip(ip_address: str) -> bool:
    try:
        ip = ipaddress.ip_address(str(ip_address).split("%", 1)[0])
    except ValueError:
        return True
    if str(ip) in BLOCKED_METADATA_IP_ADDRESSES:
        return True
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or getattr(ip, "is_site_local", False)
        or not ip.is_global
    ):
        return True
    if isinstance(ip, ipaddress.IPv6Address):
        embedded_addresses = (ip.ipv4_mapped, ip.sixtofour)
        if ip.teredo is not None:
            embedded_addresses += (ip.teredo[1],)
        if any(
            embedded is not None and not embedded.is_global
            for embedded in embedded_addresses
        ):
            return True
    return False


def _website_error(category, message, url=None, status_code=None):
    return WebsiteIngestionError(category, message, url=url, status_code=status_code)


def website_error_message(error):
    if isinstance(error, WebsiteIngestionError):
        return str(error)
    return "Website ingestion failed. Check the URL and try again."


def website_ingestion_deadline(seconds=None):
    duration = MAX_WEBSITE_INGESTION_SECONDS if seconds is None else seconds
    try:
        duration = float(duration)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "Website ingestion deadline must be a positive number."
        ) from err
    if duration <= 0:
        raise ValueError("Website ingestion deadline must be a positive number.")
    return time.monotonic() + duration


def _website_deadline_remaining(deadline):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _website_error(
            "timeout",
            "Website ingestion exceeded its total deadline.",
        )
    return remaining


def check_website_deadline(deadline):
    _website_deadline_remaining(deadline)


def _website_url_list(urls):
    if urls is None or isinstance(urls, (str, bytes)):
        raise _website_error(
            "invalid_url", "Provide website URLs as a list of absolute HTTPS URLs."
        )
    try:
        return list(urls)
    except (TypeError, ValueError) as err:
        raise _website_error(
            "invalid_url", "Provide website URLs as a list of absolute HTTPS URLs."
        ) from err


def validate_website_url_limit(urls):
    values = _website_url_list(urls)
    if len(values) > MAX_WEBSITE_URLS:
        raise _website_error(
            "url_limit",
            f"At most {MAX_WEBSITE_URLS} website URLs can be processed at once.",
        )
    return values


def _website_hostname(parsed):
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as err:
        raise _website_error(
            "invalid_url", "The website URL contains an invalid hostname or port."
        ) from err
    if not hostname:
        raise _website_error("invalid_url", "The website URL must include a hostname.")
    hostname = hostname.rstrip(".").lower()
    if not hostname:
        raise _website_error("invalid_url", "The website URL must include a hostname.")
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError) as err:
        raise _website_error(
            "invalid_url", "The website URL contains an invalid hostname."
        ) from err
    if ascii_hostname in BLOCKED_HOSTNAMES or ascii_hostname.endswith(
        (".localhost", ".metadata.google.internal")
    ):
        raise _website_error(
            "blocked_destination",
            "Local and cloud metadata destinations are not allowed.",
        )
    if port is None:
        port = 443
    if not 1 <= port <= 65535:
        raise _website_error("invalid_url", "The website URL contains an invalid port.")
    return ascii_hostname, port


def _resolve_public_addresses(hostname, port, url, deadline):
    check_website_deadline(deadline)
    try:
        resolved = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except TimeoutError as err:
        raise _website_error(
            "timeout",
            f"DNS lookup for {hostname} exceeded the ingestion deadline.",
            url=url,
        ) from err
    except (socket.gaierror, OSError, UnicodeError) as err:
        raise _website_error(
            "dns_failure",
            f"DNS lookup failed for {hostname}. Check the URL and network.",
            url=url,
        ) from err
    check_website_deadline(deadline)
    addresses = []
    for result in resolved or ():
        try:
            raw_address = str(result[4][0])
            address_text = raw_address.split("%", 1)[0]
            address = ipaddress.ip_address(address_text)
        except (IndexError, TypeError, ValueError) as err:
            raise _website_error(
                "dns_failure",
                f"DNS returned an invalid address for {hostname}.",
                url=url,
            ) from err
        if "%" in raw_address or _is_blocked_ip(str(address)):
            raise _website_error(
                "blocked_destination",
                "The website resolves to a blocked network destination.",
                url=url,
            )
        if address_text not in addresses:
            addresses.append(address_text)
    if not addresses:
        raise _website_error(
            "dns_failure",
            f"DNS returned no usable address for {hostname}.",
            url=url,
        )
    return tuple(addresses)


def _website_target(url, deadline=None):
    if not isinstance(url, str) or not url:
        raise _website_error(
            "invalid_url", "The website URL must be a non-empty string."
        )
    if "\\" in url or any(
        ord(character) < 32 or character.isspace() for character in url
    ):
        raise _website_error(
            "invalid_url", "The website URL contains whitespace or control characters."
        )
    try:
        parsed = urlsplit(url)
    except ValueError as err:
        raise _website_error("invalid_url", "The website URL is malformed.") from err
    if parsed.scheme.lower() != "https":
        raise _website_error("invalid_url", "Only HTTPS website URLs are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise _website_error(
            "invalid_url", "URLs with embedded credentials are not allowed."
        )
    hostname, port = _website_hostname(parsed)
    addresses = _resolve_public_addresses(hostname, port, url, deadline)
    host_for_url = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host_for_url if parsed.port is None else f"{host_for_url}:{parsed.port}"
    request_target = parsed.path or "/"
    if not request_target.startswith("/"):
        request_target = "/" + request_target
    if parsed.query:
        request_target += "?" + parsed.query
    normalized_url = urlunsplit(
        ("https", netloc, parsed.path, parsed.query, parsed.fragment)
    )
    return _WebsiteTarget(
        url=normalized_url,
        hostname=hostname,
        port=port,
        request_target=request_target,
        addresses=addresses,
    )


def _validate_public_http_url(url: str, deadline=None) -> str:
    return _website_target(url, deadline=deadline).url


def validate_website_urls(urls: list[str], deadline=None) -> list[str]:
    values = validate_website_url_limit(urls)
    return [_validate_public_http_url(url, deadline=deadline) for url in values]


def _header_value(headers, name):
    if not headers:
        return ""
    wanted = name.casefold()
    try:
        items = headers.items()
    except AttributeError:
        return ""
    for key, value in items:
        if str(key).casefold() == wanted:
            return str(value or "")
    return ""


def _response_status(response):
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "status_code", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return 0


def _is_timeout_exception(error):
    if isinstance(error, (TimeoutError, urllib3.exceptions.TimeoutError)):
        return True
    reason = getattr(error, "reason", None)
    if reason is None or reason is error:
        return False
    return _is_timeout_exception(reason)


def _close_pool(pool):
    with suppress(Exception):
        close = getattr(pool, "close", None)
        if callable(close):
            close()


def _close_response(response, pool):
    with suppress(Exception):
        close = getattr(response, "close", None)
        if callable(close):
            close()
    with suppress(Exception):
        release = getattr(response, "release_conn", None)
        if callable(release):
            release()
    _close_pool(pool)


def _request_pinned_response(target, deadline):
    remaining = _website_deadline_remaining(deadline)
    connect_timeout = WEBSITE_REQUEST_TIMEOUT[0]
    read_timeout = WEBSITE_REQUEST_TIMEOUT[1]
    if remaining is not None:
        connect_timeout = min(connect_timeout, remaining)
        read_timeout = min(read_timeout, remaining)
    timeout = Timeout(connect=connect_timeout, read=read_timeout, total=remaining)
    host_header = f"[{target.hostname}]" if ":" in target.hostname else target.hostname
    if target.port != 443:
        host_header = f"{host_header}:{target.port}"
    headers = {
        "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "Host": host_header,
        "User-Agent": "docmind/website-ingestion",
    }
    last_error = None
    for address in target.addresses:
        check_website_deadline(deadline)
        try:
            pool = urllib3.HTTPSConnectionPool(
                address,
                target.port,
                maxsize=1,
                block=True,
                retries=False,
                cert_reqs="CERT_REQUIRED",
                assert_hostname=target.hostname,
                server_hostname=target.hostname,
            )
        except Exception as err:
            last_error = err
            if _is_timeout_exception(err):
                raise _website_error(
                    "timeout",
                    "The website request exceeded its per-request or total timeout.",
                    url=target.url,
                ) from err
            check_website_deadline(deadline)
            continue
        try:
            response = pool.urlopen(
                "GET",
                target.request_target,
                headers=headers,
                redirect=False,
                preload_content=False,
                decode_content=True,
                timeout=timeout,
            )
            return response, pool
        except Exception as err:
            last_error = err
            _close_pool(pool)
            if _is_timeout_exception(err):
                raise _website_error(
                    "timeout",
                    "The website request exceeded its per-request or total timeout.",
                    url=target.url,
                ) from err
            check_website_deadline(deadline)
    raise _website_error(
        "http_failure",
        "The website server could not be reached over HTTPS.",
        url=target.url,
    ) from last_error


def _response_chunks(response, deadline):
    chunk_size = 64 * 1024
    stream = getattr(response, "stream", None)
    if callable(stream):
        try:
            iterator = stream(amt=chunk_size, decode_content=True)
        except TypeError:
            iterator = stream(amt=chunk_size)
    else:
        iterator = getattr(response, "iter_content", None)
        iterator = iterator(chunk_size=chunk_size) if callable(iterator) else None
    if iterator is not None:
        for chunk in iterator:
            check_website_deadline(deadline)
            if not chunk:
                continue
            yield chunk.encode() if isinstance(chunk, str) else bytes(chunk)
        return
    while True:
        check_website_deadline(deadline)
        read = getattr(response, "read", None)
        if not callable(read):
            raise _website_error(
                "http_failure", "The website response could not be read."
            )
        chunk = read(chunk_size)
        if not chunk:
            return
        yield chunk.encode() if isinstance(chunk, str) else bytes(chunk)


def _response_encoding(response, content_type):
    encoding = getattr(response, "encoding", None)
    if encoding:
        return str(encoding)
    match = re.search(r"charset\s*=\s*([\w-]+)", content_type, re.IGNORECASE)
    return match.group(1) if match else "utf-8"


def _is_challenge_response(headers, text):
    header_text = " ".join(
        f"{key}={value}" for key, value in (headers or {}).items()
    ).casefold()
    if any(
        marker in header_text
        for marker in (
            "cf-mitigated",
            "captcha",
            "challenge",
            "cloudflare",
            "datadome",
        )
    ):
        return True
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            "cf-chl-",
            "just a moment",
            "verify you are human",
            "checking your browser",
            "access denied",
            "complete the captcha",
        )
    )


def _document_from_response(response, target, deadline):
    status = _response_status(response)
    headers = getattr(response, "headers", {}) or {}
    if status in {401, 403, 429}:
        raise _website_error(
            "anti_bot",
            "The website rejected automated access with an anti-bot or challenge response.",
            url=target.url,
            status_code=status,
        )
    if status == 503 and _is_challenge_response(headers, ""):
        raise _website_error(
            "anti_bot",
            "The website returned an anti-bot or challenge response.",
            url=target.url,
            status_code=status,
        )
    if not 200 <= status < 300:
        raise _website_error(
            "http_failure",
            f"The website returned HTTP {status or 'unknown'}.",
            url=target.url,
            status_code=status or None,
        )
    if status in {204, 205}:
        raise _website_error(
            "empty_page",
            "The website returned no readable text. Choose a page with visible content.",
            url=target.url,
            status_code=status,
        )
    content_type = _header_value(headers, "content-type").lower()
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise _website_error(
            "unsupported_content",
            f"The website returned an unsupported content type: {content_type or 'unknown'}.",
            url=target.url,
            status_code=status,
        )
    content_length = _header_value(headers, "content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = None
        if declared_length is not None and declared_length > MAX_WEBSITE_RESPONSE_BYTES:
            raise _website_error(
                "response_too_large",
                "The website response exceeded the 5 MiB ingestion limit.",
                url=target.url,
                status_code=status,
            )
    body = bytearray()
    try:
        for chunk in _response_chunks(response, deadline):
            body.extend(chunk)
            if len(body) > MAX_WEBSITE_RESPONSE_BYTES:
                raise _website_error(
                    "response_too_large",
                    "The website response exceeded the 5 MiB ingestion limit.",
                    url=target.url,
                    status_code=status,
                )
    except WebsiteIngestionError:
        raise
    except Exception as err:
        if _is_timeout_exception(err):
            raise _website_error(
                "timeout",
                "The website response exceeded its per-request or total timeout.",
                url=target.url,
                status_code=status,
            ) from err
        raise _website_error(
            "http_failure",
            "The website response could not be read completely.",
            url=target.url,
            status_code=status,
        ) from err
    encoding = _response_encoding(response, content_type)
    try:
        decoded = bytes(body).decode(encoding, errors="replace")
    except LookupError:
        decoded = bytes(body).decode("utf-8", errors="replace")
    if _is_challenge_response(headers, decoded):
        raise _website_error(
            "anti_bot",
            "The website returned an anti-bot or challenge page instead of content.",
            url=target.url,
            status_code=status,
        )
    try:
        text = html2text.html2text(decoded) if "text/html" in content_type else decoded
    except Exception as err:
        raise _website_error(
            "http_failure",
            "The website HTML could not be converted to readable text.",
            url=target.url,
            status_code=status,
        ) from err
    if not str(text).strip():
        raise _website_error(
            "empty_page",
            "The website returned no readable text. Choose a page with visible content.",
            url=target.url,
            status_code=status,
        )
    return Document(text=str(text), metadata={"source": target.url})


def load_website_documents(
    urls: list[str], deadline=None, return_report: bool = False
) -> list[Document]:
    values = validate_website_url_limit(urls)
    if deadline is None:
        deadline = website_ingestion_deadline()
    validated_urls = validate_website_urls(values, deadline=deadline)
    documents = []
    report = []
    for url in validated_urls:
        target = _website_target(url, deadline=deadline)
        for redirect_count in range(MAX_WEBSITE_REDIRECTS + 1):
            try:
                response, pool = _request_pinned_response(target, deadline)
            except WebsiteIngestionError:
                raise
            except (TimeoutError, requests.exceptions.Timeout) as err:
                raise _website_error(
                    "timeout",
                    "The website request exceeded its per-request or total timeout.",
                    url=target.url,
                ) from err
            except requests.exceptions.RequestException as err:
                raise _website_error(
                    "http_failure",
                    "The website server could not be reached over HTTPS.",
                    url=target.url,
                ) from err
            try:
                check_website_deadline(deadline)
                status = _response_status(response)
                if 300 <= status < 400:
                    location = _header_value(
                        getattr(response, "headers", {}) or {}, "location"
                    )
                    if not location:
                        raise _website_error(
                            "invalid_url",
                            "The website returned a redirect without a destination.",
                            url=target.url,
                            status_code=status,
                        )
                    if redirect_count >= MAX_WEBSITE_REDIRECTS:
                        raise _website_error(
                            "http_failure",
                            f"The website exceeded the {MAX_WEBSITE_REDIRECTS}-redirect limit.",
                            url=target.url,
                            status_code=status,
                        )
                    next_url = urljoin(target.url, location)
                    target = _website_target(next_url, deadline=deadline)
                    continue
                document = _document_from_response(response, target, deadline)
                documents.append(document)
                report.append(
                    {
                        "filename": url,
                        "status": "loaded",
                        "document_count": 1,
                        "extracted_characters": len(document.text),
                        "warning": "",
                        "error": "",
                    }
                )
                break
            finally:
                _close_response(response, pool)
        else:
            raise _website_error(
                "http_failure",
                f"The website exceeded the {MAX_WEBSITE_REDIRECTS}-redirect limit.",
                url=url,
            )
    if return_report:
        return documents, report
    return documents


def safe_uploaded_filename(filename: str) -> str:
    if not filename:
        raise ValueError("Uploaded file must have a filename.")
    if "/" in filename or "\\" in filename:
        raise ValueError("Uploaded filename cannot include path separators.")
    if any(ord(char) < 32 for char in filename):
        raise ValueError("Uploaded filename cannot include control characters.")
    if filename.endswith((" ", ".")):
        raise ValueError("Uploaded filename cannot end with a space or period.")
    if not SAFE_UPLOAD_NAME_PATTERN.fullmatch(filename):
        raise ValueError("Uploaded filename contains unsupported characters.")
    device_name = filename.split(".", 1)[0].casefold()
    if device_name in WINDOWS_RESERVED_DEVICE_NAMES:
        raise ValueError("Uploaded filename uses a reserved Windows device name.")
    if Path(filename).suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError("Uploaded file type is not supported.")
    return filename


def upload_destination(save_dir: str, filename: str) -> Path:
    safe_name = safe_uploaded_filename(filename)
    base_dir = Path(save_dir).resolve()
    destination = (base_dir / safe_name).resolve()
    if not destination.is_relative_to(base_dir):
        raise ValueError("Uploaded filename resolves outside the upload directory.")
    return destination


def validate_uploaded_files(uploaded_files: list) -> None:
    if len(uploaded_files) > MAX_UPLOAD_FILES:
        raise ValueError(f"At most {MAX_UPLOAD_FILES} files can be uploaded at once.")

    total_size = 0
    seen_names = set()
    for uploaded_file in uploaded_files:
        safe_name = safe_uploaded_filename(uploaded_file.name)
        name_key = safe_name.casefold()
        if name_key in seen_names:
            raise ValueError("Uploaded filenames must be unique ignoring case.")
        seen_names.add(name_key)
        size = getattr(uploaded_file, "size", None)
        if size is None:
            size = len(uploaded_file.getbuffer())
        if size > MAX_UPLOAD_FILE_BYTES:
            raise ValueError(f"{uploaded_file.name} exceeds the per-file upload limit.")
        total_size += size

    if total_size > MAX_TOTAL_UPLOAD_BYTES:
        raise ValueError("Uploaded files exceed the total upload limit.")


###################################
#
# Save File Upload to Disk
#
###################################


def save_uploaded_file(uploaded_file: bytes, save_dir: str):
    """
    Saves the uploaded file to the specified directory.

    Args:
        uploaded_file (BytesIO): The uploaded file content.
        save_dir (str): The directory where the file will be saved.

    Returns:
        None

    Raises:
        Exception: If there is an error saving the file to disk.
    """
    try:
        destination = upload_destination(save_dir, uploaded_file.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for existing in destination.parent.iterdir():
            if existing.name.casefold() == destination.name.casefold():
                raise ValueError("An upload with this filename already exists.")
        with destination.open("wb") as f:
            f.write(uploaded_file.getbuffer())
            logs.log.info(f"Upload {uploaded_file.name} saved to disk")
    except Exception as e:
        logs.log.error("Error saving upload to disk: %s", logs.safe_log_exception(e))
        raise


###################################
#
# Confirm a GitHub Repo Exists
#
###################################


def normalize_github_repo(repo: str):
    """Normalize owner/repo or a github.com URL into owner/repo."""
    if repo is None:
        raise ValueError("GitHub repository is required.")
    try:
        repo = str(repo).strip()
    except (TypeError, ValueError) as err:
        raise ValueError("GitHub repository is required.") from err
    if not repo or any(ord(character) < 32 for character in repo):
        raise ValueError("GitHub repository is required.")
    parsed = urlparse(repo)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.lower() != "https":
            raise ValueError("GitHub repository URLs must use https.")
        try:
            port = parsed.port
        except ValueError as err:
            raise ValueError("GitHub repository URL is malformed.") from err
        if parsed.hostname is None or parsed.hostname.lower() != "github.com":
            raise ValueError("Only github.com repository URLs are supported.")
        if (
            parsed.username is not None
            or parsed.password is not None
            or port is not None
        ):
            raise ValueError(
                "GitHub repository URLs must not contain credentials or ports."
            )
        if parsed.query or parsed.fragment:
            raise ValueError(
                "GitHub repository URLs must not contain query strings or fragments."
            )
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) != 2:
            raise ValueError("GitHub repository URL must point to owner/repo.")
        owner, repo_name = path_parts
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        repo = f"{owner}/{repo_name}"

    if not GITHUB_REPO_PATTERN.fullmatch(repo):
        raise ValueError("Use the format owner/repo or a github.com repository URL.")
    owner, repo_name = repo.split("/", 1)
    if owner in {".", ".."} or repo_name in {".", ".."}:
        raise ValueError("GitHub owner and repository names cannot be dot segments.")
    if len(owner) > 100 or len(repo_name) > 100:
        raise ValueError("GitHub owner and repository names are too long.")
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    if not repo_name or repo_name in {".", ".."}:
        raise ValueError("GitHub owner and repository names cannot be dot segments.")
    return f"{owner}/{repo_name}"


def validate_github_repo(repo: str):
    try:
        normalized = normalize_github_repo(repo)
    except ValueError:
        return False
    try:
        metadata = github_metadata_preflight(normalized)
    except GitHubIngestionError:
        return False
    if metadata is not None:
        return True
    repo_endpoint = f"https://github.com/{normalized}.git"
    try:
        headers = {"User-Agent": "DocMind-AI-GitHub-Ingestion"}
        response = requests.head(
            repo_endpoint,
            timeout=(5, 10),
            allow_redirects=False,
            headers=headers,
        )
        status = _github_status(response)
        if status == 200:
            return True
        if status in {301, 302, 403, 405}:
            response = requests.get(
                f"https://github.com/{normalized}",
                timeout=(5, 10),
                headers=headers,
                allow_redirects=False,
            )
            return _github_status(response) == 200
        return False
    except Exception as err:
        logs.log.warning(
            "GitHub repository validation failed: %s", logs.safe_log_exception(err)
        )
        return False


def _is_reparse_path(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attributes & marker)
    except FileNotFoundError:
        return False


def _github_path_chain(path: Path):
    current = Path(path.anchor) if path.anchor else Path()
    for part in path.parts:
        if path.anchor and part == path.anchor:
            continue
        current = current / part
        yield current


def _reject_reparse_chain(path: Path) -> None:
    for component in _github_path_chain(path):
        if _is_reparse_path(component):
            raise GitHubIngestionError(
                "unsafe_checkout", "GitHub paths must not be links or reparse points."
            )


def _reject_reparse_path(path: Path) -> None:
    if _is_reparse_path(path):
        raise GitHubIngestionError(
            "unsafe_checkout",
            "GitHub checkout paths must not be links or reparse points.",
        )


def _github_checkout_stats(
    root: Path,
    limits: GitHubResourceLimits,
) -> tuple[int, int]:
    _reject_reparse_chain(root)
    if _is_reparse_path(root) or not root.is_dir():
        raise GitHubIngestionError(
            "invalid_checkout", "GitHub checkout is unavailable."
        )
    file_count = 0
    total_bytes = 0
    stack = [root]
    while stack:
        current = stack.pop()
        _reject_reparse_path(current)
        try:
            iterator = os.scandir(current)
        except OSError as err:
            raise GitHubIngestionError(
                "invalid_checkout", "GitHub checkout could not be inspected."
            ) from err
        try:
            for entry in iterator:
                path = Path(entry.path)
                _reject_reparse_path(path)
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError as err:
                    raise GitHubIngestionError(
                        "invalid_checkout", "GitHub checkout could not be inspected."
                    ) from err
                if is_directory:
                    stack.append(path)
                    continue
                if not is_file:
                    raise GitHubIngestionError(
                        "special_file", "GitHub checkout contains a special file."
                    )
                file_count += 1
                try:
                    size = int(entry.stat(follow_symlinks=False).st_size)
                except OSError as err:
                    raise GitHubIngestionError(
                        "invalid_checkout",
                        "GitHub checkout file size could not be read.",
                    ) from err
                total_bytes += size
                if size > limits.file_bytes:
                    raise GitHubIngestionError(
                        "file_too_large",
                        "A GitHub checkout file exceeds the configured size limit.",
                    )
                if file_count > limits.checkout_files:
                    raise GitHubIngestionError(
                        "too_many_files",
                        "GitHub checkout exceeds the configured file-count limit.",
                    )
                if total_bytes > limits.unpacked_bytes:
                    raise GitHubIngestionError(
                        "checkout_too_large",
                        "GitHub checkout exceeds the configured unpacked-size limit.",
                    )
        finally:
            iterator.close()
    return file_count, total_bytes


def _github_download_bytes(root: Path) -> int:
    _reject_reparse_chain(root)
    objects = root / ".git" / "objects"
    if not objects.exists():
        return 0
    total = 0
    for current, directories, filenames in os.walk(objects, followlinks=False):
        current_path = Path(current)
        _reject_reparse_path(current_path)
        for directory in list(directories):
            _reject_reparse_path(current_path / directory)
        for filename in filenames:
            path = current_path / filename
            _reject_reparse_path(path)
            try:
                total += path.stat().st_size
            except OSError as err:
                raise GitHubIngestionError(
                    "invalid_checkout", "GitHub download size could not be inspected."
                ) from err
    return total


def validate_github_checkout(
    root,
    limits: GitHubResourceLimits | None = None,
    *,
    download_bytes: int | None = None,
    **overrides,
):
    selected = limits or github_resource_limits(**overrides)
    path = Path(os.path.abspath(os.fspath(root)))
    file_count, total_bytes = _github_checkout_stats(path, selected)
    measured_download = _github_download_bytes(path)
    if download_bytes is None:
        download_bytes = measured_download
    if int(download_bytes) > selected.repository_download_bytes:
        raise GitHubIngestionError(
            "repository_too_large",
            "The GitHub repository exceeds the configured download limit.",
        )
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "download_bytes": int(download_bytes),
    }


def validate_github_repository_size(
    size_bytes,
    limits: GitHubResourceLimits | None = None,
    **overrides,
):
    selected = limits or github_resource_limits(**overrides)
    try:
        size = int(size_bytes)
    except (TypeError, ValueError) as err:
        raise GitHubIngestionError(
            "invalid_metadata", "GitHub repository size is invalid."
        ) from err
    if size < 0:
        raise GitHubIngestionError(
            "invalid_metadata", "GitHub repository size is invalid."
        )
    if size > selected.repository_download_bytes:
        raise GitHubIngestionError(
            "repository_too_large",
            "The GitHub repository exceeds the configured download limit.",
        )
    return size


def _clone_stream_reader(stream, counter, lock, overrun, maximum):
    while True:
        chunk = stream.read(GITHUB_CLONE_OUTPUT_CHUNK_BYTES)
        if not chunk:
            return
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", errors="replace")
        try:
            size = len(chunk)
        except TypeError:
            return
        with lock:
            counter[0] += size
            exceeded = counter[0] > maximum
        if exceeded:
            overrun.set()
            return


def _stop_clone_process(process):
    terminated = False
    process_id = getattr(process, "pid", None)
    if os.name != "nt" and isinstance(process_id, int) and process_id > 0:
        try:
            os.killpg(process_id, signal.SIGTERM)
            terminated = True
        except (AttributeError, OSError, ProcessLookupError):
            pass
    if not terminated:
        try:
            process.terminate()
        except (AttributeError, OSError):
            return
    try:
        process.wait(timeout=2)
    except Exception:
        if os.name != "nt" and isinstance(process_id, int) and process_id > 0:
            with suppress(OSError, ProcessLookupError):
                os.killpg(process_id, signal.SIGKILL)
        try:
            process.kill()
        except (AttributeError, OSError):
            return
        try:
            process.wait(timeout=2)
        except Exception:
            return


def _run_bounded_git_clone(
    command,
    checkout: Path,
    limits: GitHubResourceLimits,
):
    if subprocess.run is not _ORIGINAL_SUBPROCESS_RUN:
        with (
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            result = subprocess.run(
                command,
                check=False,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=limits.timeout_seconds,
            )
            output_size = stdout_file.tell() + stderr_file.tell()
            for name in ("stdout", "stderr"):
                value = getattr(result, name, None)
                if isinstance(value, str):
                    output_size += len(value.encode("utf-8", errors="replace"))
                elif isinstance(value, bytes):
                    output_size += len(value)
        if output_size > limits.clone_output_bytes:
            raise GitHubIngestionError(
                "clone_output_limit",
                "GitHub clone output exceeded the configured limit.",
            )
        if int(getattr(result, "returncode", 1)) != 0:
            raise GitHubIngestionError(
                "clone_failed", "GitHub clone did not complete successfully."
            )
        checkout.mkdir(parents=True, exist_ok=True)
        return
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": environment,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    counter = [0]
    counter_lock = threading.Lock()
    overrun = threading.Event()
    streams = [getattr(process, "stdout", None), getattr(process, "stderr", None)]
    readers = [
        threading.Thread(
            target=_clone_stream_reader,
            args=(stream, counter, counter_lock, overrun, limits.clone_output_bytes),
            daemon=True,
        )
        for stream in streams
        if stream is not None
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + limits.timeout_seconds
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            if overrun.is_set():
                _stop_clone_process(process)
                raise GitHubIngestionError(
                    "clone_output_limit",
                    "GitHub clone output exceeded the configured limit.",
                )
            if time.monotonic() >= deadline:
                _stop_clone_process(process)
                raise GitHubIngestionError(
                    "clone_timeout", "GitHub clone exceeded the configured timeout."
                )
            if checkout.exists():
                try:
                    _github_checkout_stats(checkout, limits)
                    if (
                        _github_download_bytes(checkout)
                        > limits.repository_download_bytes
                    ):
                        raise GitHubIngestionError(
                            "repository_too_large",
                            "The GitHub repository exceeds the configured download limit.",
                        )
                except GitHubIngestionError:
                    _stop_clone_process(process)
                    raise
            time.sleep(GITHUB_CHECKOUT_POLL_SECONDS)
        for reader in readers:
            reader.join(timeout=2)
        if overrun.is_set():
            _stop_clone_process(process)
            raise GitHubIngestionError(
                "clone_output_limit",
                "GitHub clone output exceeded the configured limit.",
            )
        if returncode != 0:
            raise GitHubIngestionError(
                "clone_failed", "GitHub clone did not complete successfully."
            )
    finally:
        for reader in readers:
            if reader.is_alive():
                reader.join(timeout=0.1)


def _git_executable():
    git_bin = shutil.which("git")
    if git_bin:
        return git_bin
    local_git = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe")
    return local_git if os.path.exists(local_git) else "git"


def _remove_owned_operation(path: Path, parent: Path) -> None:
    try:
        path.absolute().relative_to(parent.absolute())
    except ValueError:
        return
    if not (
        path.name.startswith(".github-clone-")
        or ".failed-" in path.name
        or ".backup-" in path.name
    ):
        return
    if _is_reparse_path(path):
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _atomic_replace_github_checkout(candidate: Path, target: Path) -> None:
    if _is_reparse_path(target):
        raise GitHubIngestionError(
            "unsafe_checkout", "The existing GitHub checkout is not a safe directory."
        )
    backup = None
    if target.exists():
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(candidate, target)
    except BaseException:
        if backup is not None and backup.exists():
            try:
                if target.exists() or target.is_symlink():
                    failed = target.parent / f".{target.name}.failed-{uuid.uuid4().hex}"
                    os.replace(target, failed)
                    _remove_owned_operation(failed, target.parent)
                os.replace(backup, target)
            except OSError:
                pass
        raise
    if backup is not None and backup.exists():
        with suppress(OSError):
            shutil.rmtree(backup)


def clone_github_repo(
    repo: str,
    destination_base: str | None = None,
    *,
    limits: GitHubResourceLimits | None = None,
    metadata_preflight: bool = True,
    **overrides,
):
    try:
        normalized = normalize_github_repo(repo)
    except ValueError:
        logs.log.warning("GitHub repository format was invalid")
        return False
    selected = limits or github_resource_limits(**overrides)
    save_dir = Path(
        os.path.abspath(destination_base or os.path.join(os.getcwd(), "data"))
    )
    destination = save_dir.joinpath(*normalized.split("/"))
    try:
        destination.relative_to(save_dir)
    except ValueError:
        logs.log.warning("GitHub destination escaped its operation directory")
        return False
    operation_dir = None
    with ingestion_lock():
        try:
            _reject_reparse_chain(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            _reject_reparse_chain(save_dir)
            _reject_reparse_chain(destination)
            if _is_reparse_path(destination):
                raise GitHubIngestionError(
                    "unsafe_checkout", "The existing GitHub checkout is not safe."
                )
            if metadata_preflight and subprocess.run is _ORIGINAL_SUBPROCESS_RUN:
                github_metadata_preflight(normalized, limits=selected)
            if destination.exists() and not destination.is_dir():
                raise GitHubIngestionError(
                    "unsafe_checkout", "The GitHub destination is not a directory."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            _reject_reparse_chain(destination.parent)
            operation_dir = Path(
                tempfile.mkdtemp(prefix=".github-clone-", dir=str(save_dir))
            )
            checkout = operation_dir / "checkout"
            command = [
                _git_executable(),
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--no-tags",
                "-q",
                f"https://github.com/{normalized}.git",
                str(checkout),
            ]
            _run_bounded_git_clone(command, checkout, selected)
            validate_github_checkout(checkout, selected)
            _atomic_replace_github_checkout(checkout, destination)
            logs.log.info("GitHub repository cloned")
            return str(destination)
        except GitHubIngestionError as err:
            logs.log.warning("GitHub clone rejected: category=%s", err.category)
            return False
        except Exception as err:
            logs.log.warning("GitHub clone failed: %s", logs.safe_log_exception(err))
            return False
        finally:
            if operation_dir is not None:
                try:
                    _remove_owned_operation(operation_dir, save_dir)
                except OSError:
                    logs.log.warning("GitHub operation cleanup failed")


enforce_github_checkout_limits = validate_github_checkout
validate_github_tree = validate_github_checkout
preflight_github_repository = github_metadata_preflight
