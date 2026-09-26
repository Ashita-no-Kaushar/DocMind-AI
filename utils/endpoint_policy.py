import hashlib
import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
ALLOWED_ENDPOINT_SCHEMES = frozenset({"http", "https"})
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")


def _loopback_hostname(hostname: str) -> bool:
    value = hostname.strip().lower().rstrip(".")
    if value == "localhost" or value.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def is_loopback_endpoint(endpoint: str) -> bool:
    try:
        parsed = urlsplit(str(endpoint).strip())
        return bool(parsed.hostname and _loopback_hostname(parsed.hostname))
    except ValueError:
        return False


def _normalized_authority(parsed) -> str:
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Provider endpoint must include a hostname.")
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError as err:
        raise ValueError("Provider endpoint has an invalid port.") from err
    if port is None:
        return hostname.lower()
    return f"{hostname.lower()}:{port}"


def normalize_provider_endpoint(
    value: str | None,
    *,
    default: str | None = None,
    label: str = "Provider endpoint",
) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        if default is None:
            raise ValueError(f"{label} is required.")
        raw = str(default).strip()
    if "\\" in raw or _CONTROL_OR_SPACE.search(raw):
        raise ValueError(f"{label} contains an invalid character.")
    if "?" in raw or "#" in raw:
        raise ValueError(f"{label} must not contain a query string or fragment.")
    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as err:
        raise ValueError(f"{label} is not a valid URL.") from err
    if scheme not in ALLOWED_ENDPOINT_SCHEMES:
        raise ValueError(f"{label} must use HTTP or HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain URL credentials.")
    if not hostname:
        raise ValueError(f"{label} must include a hostname.")
    if port == 0:
        raise ValueError(f"{label} must use a valid port.")
    if scheme == "http" and not _loopback_hostname(hostname):
        raise ValueError(f"{label} must use HTTPS for non-loopback hosts.")
    authority = _normalized_authority(parsed)
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, authority, path, "", ""))


def validate_provider_endpoint(value: str | None, **kwargs) -> str:
    return normalize_provider_endpoint(value, **kwargs)


normalize_endpoint = normalize_provider_endpoint
validate_endpoint_url = validate_provider_endpoint


def normalize_ollama_endpoint(value: str | None) -> str:
    return normalize_provider_endpoint(
        value,
        default=DEFAULT_OLLAMA_ENDPOINT,
        label="Ollama endpoint",
    )


def normalize_openai_endpoint(value: str | None) -> str:
    return normalize_provider_endpoint(
        value,
        default=DEFAULT_OPENAI_BASE_URL,
        label="OpenAI endpoint",
    )


def validate_local_endpoint(value: str | None, **kwargs) -> str:
    kwargs.setdefault("label", "Local endpoint")
    return normalize_provider_endpoint(value, **kwargs)


def endpoint_host(endpoint: str) -> str:
    parsed = urlsplit(normalize_provider_endpoint(endpoint))
    return parsed.hostname.lower()


def endpoint_identity(endpoint: str) -> tuple[str, str, int | None]:
    normalized = normalize_provider_endpoint(endpoint)
    parsed = urlsplit(normalized)
    return parsed.scheme, (parsed.hostname or "").lower(), parsed.port


def validate_credential_endpoint(
    api_key: str | None,
    endpoint: str,
    provider_kind: str,
    bound_endpoint: str | None = None,
) -> str:
    normalized = normalize_provider_endpoint(endpoint)
    if not api_key:
        return normalized
    kind = str(provider_kind or "").strip().lower().replace("-", "_")
    official = kind in {"openai", "openai_api", "openai_official", "official_openai"}
    host = endpoint_host(normalized)
    if official and host != "api.openai.com":
        raise ValueError(
            "OpenAI official credentials may only be sent to https://api.openai.com."
        )
    if not official and host == "api.openai.com":
        raise ValueError(
            "The official OpenAI host requires the official OpenAI provider profile."
        )
    if bound_endpoint:
        try:
            if endpoint_identity(bound_endpoint) != endpoint_identity(normalized):
                raise ValueError(
                    "The provider API key is bound to a different endpoint; clear and re-enter it."
                )
        except ValueError as err:
            if str(err).startswith("The provider API key"):
                raise
            raise ValueError("The saved provider endpoint binding is invalid.") from err
    return normalized


def model_catalog_signature(
    endpoint: str, api_key: str | None, provider_kind: str
) -> str:
    normalized = normalize_provider_endpoint(endpoint)
    host = endpoint_host(normalized)
    fingerprint = (
        hashlib.sha256(str(api_key).encode("utf-8")).hexdigest()
        if api_key
        else "anonymous"
    )
    return f"{provider_kind}:{host}:{normalized}:{fingerprint}"


__all__ = [
    "ALLOWED_ENDPOINT_SCHEMES",
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_OPENAI_BASE_URL",
    "endpoint_host",
    "endpoint_identity",
    "is_loopback_endpoint",
    "model_catalog_signature",
    "normalize_endpoint",
    "normalize_ollama_endpoint",
    "normalize_openai_endpoint",
    "normalize_provider_endpoint",
    "validate_credential_endpoint",
    "validate_endpoint_url",
    "validate_local_endpoint",
    "validate_provider_endpoint",
]
