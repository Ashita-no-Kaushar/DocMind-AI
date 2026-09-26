import contextlib
import ipaddress
import os
import re

DEFAULT_BIND_ADDRESS = "127.0.0.1"
BIND_ADDRESS_ENV = "DOCMIND_BIND_ADDRESS"
REMOTE_BIND_OPT_IN_ENV = "DOCMIND_ALLOW_REMOTE_BIND"
ALLOW_REMOTE_BIND_ENV = REMOTE_BIND_OPT_IN_ENV
REMOTE_BIND_OPT_IN_ENV_ALIASES = (
    "DOCMIND_ALLOW_REMOTE_BINDING",
    "DOCMIND_REMOTE_BIND_OPT_IN",
    "DOCMIND_ALLOW_NON_LOOPBACK",
    "DOCMIND_ENABLE_REMOTE_BIND",
)
REMOTE_DEPLOYMENT_CONTRACT = (
    "Non-loopback or wildcard binding requires DOCMIND_ALLOW_REMOTE_BIND=true and a "
    "user-configured authenticated reverse proxy; DocMind does not provide built-in "
    "authentication."
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})
_WILDCARD_VALUES = frozenset({"0.0.0.0", "::", "*", "[::]", "0:0:0:0:0:0:0:0"})
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f]")


class RuntimePolicyError(ValueError):
    pass


def _environment(environ=None):
    return os.environ if environ is None else environ


def _loopback_address(value: str) -> bool:
    host = str(value).strip().lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if "%" in host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_bind_address(address=None, environ=None) -> str:
    env = _environment(environ)
    raw = default_value = (
        env.get(BIND_ADDRESS_ENV)
        or env.get("DOCMIND_SERVER_ADDRESS")
        or env.get("STREAMLIT_SERVER_ADDRESS")
        or DEFAULT_BIND_ADDRESS
    )
    if address is not None:
        raw = address
    if raw is None:
        raw = default_value
    try:
        value = str(raw).strip()
    except (TypeError, ValueError) as err:
        raise RuntimePolicyError("The runtime bind address is invalid.") from err
    if (
        not value
        or _CONTROL_OR_SPACE.search(value)
        or any(character in value for character in "/?#@")
    ):
        raise RuntimePolicyError("The runtime bind address is invalid.")
    wildcard = value.lower() in _WILDCARD_VALUES
    with contextlib.suppress(ValueError):
        wildcard = wildcard or ipaddress.ip_address(value.strip("[]")).is_unspecified
    opt_in_value = next(
        (
            env.get(name)
            for name in (REMOTE_BIND_OPT_IN_ENV, *REMOTE_BIND_OPT_IN_ENV_ALIASES)
            if env.get(name) is not None
        ),
        None,
    )
    if opt_in_value is not None:
        opt_in_text = str(opt_in_value).strip().lower()
        if opt_in_text not in _TRUE_VALUES | _FALSE_VALUES:
            raise RuntimePolicyError(
                f"{REMOTE_BIND_OPT_IN_ENV} must be one of true, false, 1, 0, yes, no, on, or off."
            )
        remote_opt_in = opt_in_text in _TRUE_VALUES
    else:
        remote_opt_in = False
    if _loopback_address(value):
        return value
    if not remote_opt_in:
        if wildcard:
            raise RuntimePolicyError(
                "Wildcard runtime binding is disabled. Set "
                f"{REMOTE_BIND_OPT_IN_ENV}=true only behind an authenticated reverse proxy."
            )
        raise RuntimePolicyError(
            "Non-loopback binding is disabled. Set "
            f"{REMOTE_BIND_OPT_IN_ENV}=true only behind an authenticated reverse proxy."
        )
    return value


def enforce_runtime_bind_policy(address=None, environ=None) -> str:
    return resolve_bind_address(address=address, environ=environ)


def runtime_bind_contract() -> str:
    return REMOTE_DEPLOYMENT_CONTRACT


__all__ = [
    "ALLOW_REMOTE_BIND_ENV",
    "BIND_ADDRESS_ENV",
    "DEFAULT_BIND_ADDRESS",
    "REMOTE_BIND_OPT_IN_ENV",
    "REMOTE_BIND_OPT_IN_ENV_ALIASES",
    "REMOTE_DEPLOYMENT_CONTRACT",
    "RuntimePolicyError",
    "enforce_runtime_bind_policy",
    "resolve_bind_address",
    "runtime_bind_contract",
]
