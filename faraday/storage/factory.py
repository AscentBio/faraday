from __future__ import annotations

from typing import Optional

from faraday.config import get_backend_value
from faraday.storage.base import StorageBackend
from faraday.storage.sqlite_backend import SQLiteStorageBackend


_BACKEND_SINGLETON: Optional[StorageBackend] = None
_BACKEND_KEY: Optional[str] = None


def _resolve_backend_key() -> str:
    return get_backend_value("db", default="auto")


def get_storage_backend(force_refresh: bool = False) -> StorageBackend:
    """Return a process-global storage backend instance."""
    global _BACKEND_SINGLETON, _BACKEND_KEY

    backend_key = _resolve_backend_key()
    if (
        not force_refresh
        and _BACKEND_SINGLETON is not None
        and _BACKEND_KEY == backend_key
    ):
        return _BACKEND_SINGLETON

    if backend_key in ("sqlite", "auto", ""):
        backend = SQLiteStorageBackend.from_env()
    else:
        raise ValueError(
            f"Unsupported config backends.db={backend_key!r}. "
            "Expected one of: 'auto', 'sqlite'."
        )

    backend.initialize()
    _BACKEND_SINGLETON = backend
    _BACKEND_KEY = backend_key
    return backend
