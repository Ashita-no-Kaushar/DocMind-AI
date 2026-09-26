import contextlib
import errno
import functools
import os
import threading
import time
from pathlib import Path

DEFAULT_INGESTION_LOCK_TIMEOUT_SECONDS = 120.0
INGESTION_LOCK_TIMEOUT_SECONDS = DEFAULT_INGESTION_LOCK_TIMEOUT_SECONDS
DEFAULT_INGESTION_LOCK_ENV = "DOCMIND_INGESTION_LOCK_PATH"
DEFAULT_INGESTION_LOCK_NAME = ".docmind-ingestion.lock"
_LOCK_MAP_GUARD = threading.Lock()
_LOCK_MAP = {}


class IngestionLockTimeout(TimeoutError):
    pass


def default_ingestion_lock_path() -> Path:
    configured = os.environ.get(DEFAULT_INGESTION_LOCK_ENV) or os.environ.get(
        "DOCMIND_LOCK_PATH"
    )
    if configured:
        return Path(configured).expanduser().absolute()
    return Path.cwd() / DEFAULT_INGESTION_LOCK_NAME


def _timeout(value):
    if value is None:
        value = os.environ.get(
            "DOCMIND_INGESTION_LOCK_TIMEOUT_SECONDS",
            os.environ.get(
                "DOCMIND_LOCK_TIMEOUT_SECONDS",
                DEFAULT_INGESTION_LOCK_TIMEOUT_SECONDS,
            ),
        )
    try:
        timeout = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError("Ingestion lock timeout must be a positive number.") from err
    if timeout < 0:
        raise ValueError("Ingestion lock timeout must be a positive number.")
    return timeout


def _local_lock(key):
    process_key = (os.getpid(), key)
    with _LOCK_MAP_GUARD:
        entry = _LOCK_MAP.get(process_key)
        if entry is None:
            entry = threading.RLock()
            _LOCK_MAP[process_key] = entry
        return entry


def _try_os_lock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as err:
        if err.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise
    return True


def _unlock_os(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_metadata(handle, state):
    payload = f"{state}\n{os.getpid()}\n{time.time()}\n".encode("ascii")
    handle.seek(0)
    handle.truncate()
    handle.write(payload)
    handle.flush()
    with contextlib.suppress(OSError):
        os.fsync(handle.fileno())


@contextlib.contextmanager
def ingestion_lock(path=None, timeout=None):
    lock_path = (
        Path(path).expanduser().absolute() if path else default_ingestion_lock_path()
    )
    timeout_value = _timeout(timeout)
    if lock_path.is_symlink():
        raise OSError("The ingestion lifecycle lock path must not be a symlink.")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(str(lock_path))
    local_lock = _local_lock(key)
    started = time.monotonic()
    remaining = timeout_value
    if not local_lock.acquire(timeout=remaining):
        raise IngestionLockTimeout(
            "Timed out waiting for the ingestion lifecycle lock."
        )
    depth = getattr(_local_state, "depths", {}).get(key, 0)
    stack = contextlib.ExitStack()
    try:
        if depth:
            _local_state.depths[key] = depth + 1
            try:
                yield
            finally:
                depths = getattr(_local_state, "depths", {})
                depths[key] = max(0, depths.get(key, 1) - 1)
            return
        try:
            handle = stack.enter_context(lock_path.open("a+b"))
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
        except OSError as err:
            raise OSError("The ingestion lifecycle lock could not be opened.") from err
        while not _try_os_lock(handle):
            remaining = timeout_value - (time.monotonic() - started)
            if remaining <= 0:
                raise IngestionLockTimeout(
                    "Timed out waiting for the ingestion lifecycle lock."
                )
            time.sleep(min(0.05, max(0.001, remaining)))
        _write_metadata(handle, "held")
        if not hasattr(_local_state, "depths"):
            _local_state.depths = {}
        _local_state.depths[key] = 1
        try:
            yield
        finally:
            _write_metadata(handle, "released")
            _unlock_os(handle)
            depths = getattr(_local_state, "depths", {})
            depths.pop(key, None)
    finally:
        with contextlib.suppress(OSError):
            stack.close()
        local_lock.release()


_local_state = threading.local()


def lifecycle_lock(function=None, *, path=None, timeout=None):
    def decorate(target):
        @functools.wraps(target)
        def wrapped(*args, **kwargs):
            with ingestion_lock(path=path, timeout=timeout):
                return target(*args, **kwargs)

        return wrapped

    if function is None:
        return decorate
    return decorate(function)


def ingestion_lock_path(path=None) -> Path:
    return Path(path).expanduser().absolute() if path else default_ingestion_lock_path()


__all__ = [
    "DEFAULT_INGESTION_LOCK_TIMEOUT_SECONDS",
    "INGESTION_LOCK_TIMEOUT_SECONDS",
    "IngestionLockTimeout",
    "default_ingestion_lock_path",
    "ingestion_lock",
    "ingestion_lock_path",
    "lifecycle_lock",
]
