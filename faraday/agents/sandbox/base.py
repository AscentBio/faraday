"""Backend interface and registry for sandbox execution environments.

To add a new backend:

1. Create a class that satisfies :class:`SandboxBackend` (implement every method).
2. Call :func:`register_backend` with a short name and your class.
3. Set ``execution.backend: your_name`` in ``faraday.yaml`` (or pass
   ``execution_backend="your_name"`` programmatically).

Example::

    from faraday.agents.sandbox.base import SandboxBackend, register_backend
    from faraday.agents.sandbox.config import SandboxConfig, SandboxState

    class K8sSandboxManager:
        def __init__(self, config: SandboxConfig):
            self.config = config
            self.state = SandboxState.UNINITIALIZED
            self.sandbox_id = None
            self.error_message = None
            self.initialization_time = None
            ...

        # implement every method from SandboxBackend
        ...

    register_backend("k8s", K8sSandboxManager)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, runtime_checkable

from typing import Protocol

from faraday.agents.sandbox.config import SandboxConfig, SandboxState


@runtime_checkable
class SandboxBackend(Protocol):
    """Structural interface that every sandbox backend must satisfy.

    All three attributes (``state``, ``sandbox_id``, ``error_message``) and every
    method below are accessed by :class:`~faraday.agents.execution.runtime.SandboxRuntime`.
    A backend that omits any of them will fail at construction time with a clear
    ``TypeError``.
    """

    state: SandboxState
    sandbox_id: Optional[str]
    error_message: Optional[str]
    initialization_time: Optional[float]

    def start_initialization(self) -> None:
        """Begin (possibly asynchronous) environment setup."""
        ...

    def wait_for_ready(self, timeout: float = 120.0) -> bool:
        """Block until ``state`` is READY or timeout. Return True if ready."""
        ...

    def get_sandbox(self) -> Any:
        """Return an object suitable for ``exec()`` calls, or ``None``."""
        ...

    def exec(self, *args: Any) -> Any:
        """Run an arbitrary command in the sandbox. Return a process-like object."""
        ...

    def exec_python_code(self, code: str, timeout: float = 300.0) -> Tuple[str, str, bool]:
        """Execute Python code, returning ``(stdout, stderr, had_error)``."""
        ...

    def ensure_directories(self, sandbox: Any = None, directories: Any = None) -> bool:
        """Create agent output directories. Return True on success."""
        ...

    def get_outputs_root(self) -> str:
        """Return the path to the ``agent_outputs/`` root inside the sandbox."""
        ...

    def terminate(self) -> None:
        """Tear down the environment and release resources."""
        ...

    def reinitialize(self) -> None:
        """Tear down and re-create the environment from scratch."""
        ...

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for progress/status messages."""
        ...


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type] = {}

# The runtime will crash without these; registration fails if any are missing.
_REQUIRED_METHODS = {
    "start_initialization",
    "wait_for_ready",
    "get_sandbox",
    "ensure_directories",
    "terminate",
    "set_progress_callback",
}

# Newer methods that the runtime handles gracefully when absent (via hasattr).
# Missing optional methods produce a warning, not an error.
_OPTIONAL_METHODS = {
    "exec",
    "exec_python_code",
    "get_outputs_root",
    "reinitialize",
}


def register_backend(name: str, manager_class: type) -> None:
    """Register *manager_class* under *name* so it can be selected via config.

    Raises ``TypeError`` if any **required** protocol methods are missing.
    Logs a warning for missing **optional** methods.
    """
    missing_required = [m for m in _REQUIRED_METHODS if not hasattr(manager_class, m)]
    if missing_required:
        raise TypeError(
            f"Cannot register {manager_class.__name__!r} as backend {name!r}: "
            f"missing required methods: {', '.join(sorted(missing_required))}"
        )
    missing_optional = [m for m in _OPTIONAL_METHODS if not hasattr(manager_class, m)]
    if missing_optional:
        import warnings
        warnings.warn(
            f"Backend {name!r} ({manager_class.__name__}) is missing optional "
            f"methods: {', '.join(sorted(missing_optional))}. Some features "
            f"may not work.",
            stacklevel=2,
        )
    _BACKEND_REGISTRY[name] = manager_class


def get_backend_class(name: str) -> Optional[type]:
    """Look up a registered backend by name. Returns ``None`` if not found."""
    return _BACKEND_REGISTRY.get(name)


def registered_backends() -> dict[str, type]:
    """Return a snapshot of all currently registered backends."""
    return dict(_BACKEND_REGISTRY)
