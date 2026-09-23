import ipaddress
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
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
WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "clock$", "conin$", "conout$"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


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


def create_ingestion_work_dir(kind: str = "ingestion", destination_base: str | None = None):
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
        raise ValueError("Website ingestion deadline must be a positive number.") from err
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
    if (
        ascii_hostname in BLOCKED_HOSTNAMES
        or ascii_hostname.endswith((".localhost", ".metadata.google.internal"))
    ):
        raise _website_error(
            "blocked_destination", "Local and cloud metadata destinations are not allowed."
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
        raise _website_error("invalid_url", "The website URL must be a non-empty string.")
    if "\\" in url or any(
        ord(character) < 32 or character.isspace() for character in url
    ):
        raise _website_error("invalid_url", "The website URL contains whitespace or control characters.")
    try:
        parsed = urlsplit(url)
    except ValueError as err:
        raise _website_error("invalid_url", "The website URL is malformed.") from err
    if parsed.scheme.lower() != "https":
        raise _website_error("invalid_url", "Only HTTPS website URLs are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise _website_error("invalid_url", "URLs with embedded credentials are not allowed.")
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
        if callable(iterator):
            iterator = iterator(chunk_size=chunk_size)
        else:
            iterator = None
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
        logs.log.error(f"Error saving upload to disk: {e}")
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

    repo = repo.strip()
    parsed = urlparse(repo)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme != "https":
            raise ValueError("GitHub repository URLs must use https.")
        if parsed.netloc.lower() != "github.com":
            raise ValueError("Only github.com repository URLs are supported.")
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

    return repo


def validate_github_repo(repo: str):
    """
    Validates whether a GitHub repository exists.

    Args:
        repo (str): The name of the GitHub repository.

    Returns:
        True if the repository exists, False otherwise.

    Raises:
        Exception: If there is an error validating the repository.
    """
    try:
        repo = normalize_github_repo(repo)
    except ValueError:
        return False

    repo_endpoint = f"https://github.com/{repo}.git"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DocMind/1.0"}
        resp = requests.head(repo_endpoint, timeout=10, allow_redirects=True, headers=headers)
        if resp.status_code == 200:
            return True
        if resp.status_code in (301, 302, 403, 405):
            get_resp = requests.get(f"https://github.com/{repo}", timeout=10, headers=headers)
            return get_resp.status_code == 200
        return False
    except Exception as err:
        logs.log.warning(f"Unable to validate GitHub repository {repo}: {err}")
        return False


###################################
#
# Clone a GitHub Repo
#
###################################


def clone_github_repo(repo: str, destination_base: str | None = None):
    """
    Clones a GitHub repository.

    Args:
        repo (str): The name of the GitHub repository.
        destination_base (str, optional): Parent directory for this checkout.

    Returns:
        The cloned repository directory if successful, False otherwise.

    Raises:
        Exception: If there is an error cloning the repository.
    """
    try:
        repo = normalize_github_repo(repo)
    except ValueError:
        logs.log.error(f"Invalid GitHub repository format: {repo}")
        return False

    repo_endpoint = f"https://github.com/{repo}.git"
    save_dir = os.path.abspath(destination_base or os.path.join(os.getcwd(), "data"))
    destination = os.path.abspath(os.path.join(save_dir, *repo.split("/")))
    try:
        inside_destination = os.path.commonpath([save_dir, destination]) == save_dir
    except ValueError:
        inside_destination = False
    if not inside_destination:
        logs.log.error(f"GitHub destination escaped save directory: {repo}")
        return False

    try:
        os.makedirs(save_dir, exist_ok=True)

        # Ensure retries of the same repo don't fail because a stale checkout exists.
        if os.path.isdir(destination):
            logs.log.info(f"Removing existing repository directory: {destination}")
            if not remove_dir_retry(destination):
                logs.log.warning(
                    "Stale repository checkout is locked; "
                    f"cloning into a timestamped directory instead: {destination}"
                )
                destination = os.path.normpath(
                    os.path.join(
                        save_dir,
                        f"clone_{int(time.time())}_{os.getpid()}",
                        repo.replace("/", "__"),
                    )
                )

        git_bin = shutil.which("git")
        if not git_bin:
            local_git = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe")
            if os.path.exists(local_git):
                git_bin = local_git
            else:
                git_bin = "git"

        result = subprocess.run(
            [git_bin, "clone", "--depth", "1", "-q", repo_endpoint, destination],
            check=False,
            capture_output=True,
            text=True,
            timeout=GIT_CLONE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            logs.log.error(
                f"Error cloning {repo} GitHub repo: {result.stderr.strip() or result.stdout.strip()}"
            )
            return False

        logs.log.info(f"Cloned {repo} repo")
        return destination
    except Exception as err:
        logs.log.error(f"Error cloning {repo} GitHub repo: {err}")
        return False
