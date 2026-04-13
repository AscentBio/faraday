from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from faraday.agents.sandbox.base import get_backend_class, register_backend
from faraday.agents.sandbox.docker_backend import (
    DockerSandboxManager,
    docker_sidecar_requirements_message,
    running_in_app_docker,
)
from faraday.agents.execution.models import ExecutionPolicy
from faraday.agents.sandbox.local_backend import LocalSandboxManager
from faraday.agents.sandbox.config import SandboxConfig, SandboxState

try:
    from faraday.agents.sandbox.modal_backend import SandboxManager as _ModalSandboxManager
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name == "modal":
        _ModalSandboxManager = None  # type: ignore[assignment,misc]
    else:
        raise

# ---------------------------------------------------------------------------
# Register built-in backends
# ---------------------------------------------------------------------------
register_backend("docker", DockerSandboxManager)
register_backend("host", LocalSandboxManager)
if _ModalSandboxManager is not None:
    register_backend("modal", _ModalSandboxManager)


# Modal-only directory list — Docker and local managers declare their own concrete dirs.
_MODAL_SANDBOX_DIRECTORIES = [
    "/cloud-storage/agent_outputs/",
    "/cloud-storage/agent_outputs/plots/",
    "/cloud-storage/agent_outputs/data/",
    "/cloud-storage/agent_outputs/reports/",
    "/cloud-storage/agent_outputs/tex/",
    "/cloud-storage/agent_outputs/webpages/",
    "/cloud-storage/.modal_code_vars/",
]


@dataclass
class RuntimeConfig:
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    verbose: bool = True
    app_runtime: str = "host"
    execution_backend_name: str = "docker"
    enable_sandbox: bool = True
    lazy_sandbox: bool = True
    workspace_source_root: Optional[str] = None
    workspace_mount_path: str = "/workspace"
    # cloud_storage_root / cloud_storage_mode are Modal-only; ignored for Docker/local backends.
    cloud_storage_root: Optional[str] = None
    cloud_storage_mode: str = "disabled"
    modal_bucket_name: Optional[str] = None
    docker_image: Optional[str] = None
    docker_dockerfile: Optional[str] = None
    allow_host_runtime: bool = False
    allow_backend_fallback: bool = True
    fallback_order: Optional[list[str]] = None
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)


class SandboxRuntime:
    def __init__(self, config: RuntimeConfig, log_fn):
        self.config = config
        self._log = log_fn
        self._lifecycle_lock = threading.RLock()
        self._async_lifecycle_lock = asyncio.Lock()
        self.sandbox_manager = self._init_sandbox_manager()
        self.sb_index: Optional[str] = None

    def _init_sandbox_manager(self):
        if (not self.config.enable_sandbox) or self.config.execution_backend_name == "disabled":
            disabled_config = SandboxConfig(
                user_id=self.config.user_id,
                chat_id=self.config.chat_id,
                query_id=None,
                execution_backend="host",
                app_runtime=self.config.app_runtime,
                workspace_source_root=self.config.workspace_source_root,
                workspace_mount_path=self.config.workspace_mount_path,
                docker_image=self.config.docker_image,
                docker_memory=self.config.execution_policy.docker_memory,
                docker_cpus=self.config.execution_policy.docker_cpus,
                docker_pids_limit=self.config.execution_policy.docker_pids_limit,
                docker_shm_size=self.config.execution_policy.docker_shm_size,
                enable_sandbox=False,
                lazy_initialization=True,
                verbose=self.config.verbose,
            )
            return LocalSandboxManager(disabled_config)
        errors: list[str] = []
        for candidate in self._backend_candidates():
            try:
                manager = self._build_manager_for_backend(candidate)
                if self.config.verbose:
                    manager.set_progress_callback(lambda msg: self._log(msg))
                if candidate != self.config.execution_backend_name:
                    self._log(
                        f"Execution backend fallback: {self.config.execution_backend_name} -> {candidate}"
                    )
                    self.config.execution_backend_name = candidate
                return manager
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
        raise RuntimeError("No execution backend available. " + " | ".join(errors))

    def _build_sandbox_config(self, backend_name: str) -> SandboxConfig:
        """Build a :class:`SandboxConfig` populated from :attr:`config`."""
        is_modal = backend_name == "modal"
        return SandboxConfig(
            user_id=self.config.user_id,
            chat_id=self.config.chat_id,
            query_id=None,
            execution_backend=backend_name,
            app_runtime=self.config.app_runtime,
            workspace_source_root=self.config.workspace_source_root,
            workspace_mount_path=self.config.workspace_mount_path,
            cloud_storage_root=self.config.cloud_storage_root if is_modal else None,
            cloud_storage_mode=self.config.cloud_storage_mode if is_modal else "disabled",
            bucket_name=self.config.modal_bucket_name if is_modal else None,
            docker_image=self.config.docker_image,
            docker_dockerfile=self.config.docker_dockerfile,
            docker_memory=self.config.execution_policy.docker_memory,
            docker_cpus=self.config.execution_policy.docker_cpus,
            docker_pids_limit=self.config.execution_policy.docker_pids_limit,
            docker_shm_size=self.config.execution_policy.docker_shm_size,
            docker_network=self.config.execution_policy.docker_network,
            max_reinitialize_attempts=self.config.execution_policy.max_reinitialize_attempts,
            heartbeat_interval_sec=self.config.execution_policy.heartbeat_interval_sec,
            enable_sandbox=self.config.enable_sandbox,
            lazy_initialization=self.config.lazy_sandbox,
            verbose=self.config.verbose,
        )

    def _build_manager_for_backend(self, backend_name: str):
        if backend_name == "disabled":
            raise RuntimeError("Execution backend is disabled")

        # Pre-construction availability check for Docker (needs Docker daemon).
        if backend_name == "docker" and not DockerSandboxManager.is_docker_available():
            if self.config.app_runtime == "docker" or running_in_app_docker():
                raise RuntimeError(
                    "Docker CLI/daemon is unavailable inside the Faraday app container. "
                    + docker_sidecar_requirements_message()
                )
            raise RuntimeError("Docker CLI/daemon is unavailable")

        manager_cls = get_backend_class(backend_name)
        if manager_cls is None:
            raise RuntimeError(
                f"Unsupported execution backend: {backend_name!r}. "
                f"Register it with faraday.agents.sandbox.base.register_backend()."
            )
        return manager_cls(self._build_sandbox_config(backend_name))

    def _backend_candidates(self) -> list[str]:
        from faraday.agents.sandbox.base import registered_backends

        primary = (self.config.execution_backend_name or "docker").strip().lower()
        if primary == "disabled":
            return []
        defaults = {
            "docker": ["docker", "host"],
            "modal": ["modal", "docker", "host"],
            "host": ["host"],
        }
        if self.config.fallback_order:
            candidates = [primary] + list(self.config.fallback_order)
        else:
            candidates = defaults.get(primary, [primary])
        known = registered_backends()
        deduped: list[str] = []
        for value in candidates:
            normalized = (value or "").strip().lower()
            if normalized in known and normalized not in deduped:
                deduped.append(normalized)
        if primary == "docker" and (
            self.config.app_runtime == "docker" or running_in_app_docker()
        ):
            deduped = [value for value in deduped if value != "host"]
        if not self.config.allow_backend_fallback:
            return deduped[:1]
        return deduped

    def start(self) -> None:
        with self._lifecycle_lock:
            if not self.config.enable_sandbox:
                self._log("Sandbox disabled; skipping initialization")
                return
            self.sandbox_manager.start_initialization()
            timeout = self.config.execution_policy.sandbox_ready_timeout_sec
            if self.sandbox_manager.wait_for_ready(timeout=timeout):
                self.sb_index = self.sandbox_manager.sandbox_id
                init_time = self.sandbox_manager.initialization_time or 0
                self._log(f"Sandbox ready in {init_time:.2f}s with ID {self.sb_index}")
                return
            self.sb_index = None
            error_msg = self.sandbox_manager.error_message or "Unknown error"
            raise RuntimeError(f"Sandbox initialization failed: {error_msg}")

    def stop(self) -> None:
        with self._lifecycle_lock:
            self.sandbox_manager.terminate()
            self.sb_index = None

    def ensure_ready(self, timeout: Optional[float] = None) -> bool:
        with self._lifecycle_lock:
            if not self.config.enable_sandbox:
                return False
            if self.sandbox_manager.state == SandboxState.UNINITIALIZED:
                self.sandbox_manager.start_initialization()
            effective_timeout = (
                timeout
                if timeout is not None
                else self.config.execution_policy.sandbox_ready_timeout_sec
            )
            return self.sandbox_manager.wait_for_ready(timeout=effective_timeout)

    def get_sandbox(self) -> Any:
        with self._lifecycle_lock:
            sandbox = self.sandbox_manager.get_sandbox()
            if sandbox is None:
                manager_state = getattr(self.sandbox_manager, "state", None)
                manager_state_value = (
                    manager_state.value if hasattr(manager_state, "value") else str(manager_state)
                )
                manager_error = getattr(self.sandbox_manager, "error_message", None)
                diagnostic = f"sandbox unavailable (state={manager_state_value})"
                if manager_error:
                    diagnostic += f", error={manager_error}"
                raise RuntimeError(diagnostic)
            if self.sandbox_manager.sandbox_id != self.sb_index:
                self.sb_index = self.sandbox_manager.sandbox_id
            return sandbox

    def reinitialize(self) -> None:
        with self._lifecycle_lock:
            if hasattr(self.sandbox_manager, "reinitialize"):
                self.sandbox_manager.reinitialize()
                self.sb_index = self.sandbox_manager.sandbox_id
            else:
                self.start()

    async def start_async(self) -> None:
        async with self._async_lifecycle_lock:
            await asyncio.to_thread(self.start)

    async def stop_async(self) -> None:
        async with self._async_lifecycle_lock:
            await asyncio.to_thread(self.stop)

    async def ensure_ready_async(self, timeout: Optional[float] = None) -> bool:
        async with self._async_lifecycle_lock:
            return await asyncio.to_thread(self.ensure_ready, timeout)

    async def get_sandbox_async(self) -> Any:
        async with self._async_lifecycle_lock:
            return await asyncio.to_thread(self.get_sandbox)

    async def reinitialize_async(self) -> None:
        async with self._async_lifecycle_lock:
            await asyncio.to_thread(self.reinitialize)

    def ensure_directories(self, sb: Any) -> bool:
        """Create backend-specific output directories; return True on success."""
        try:
            # Each manager knows its own concrete directory layout.
            return self.sandbox_manager.ensure_directories(sb)
        except Exception as exc:
            self._log(f"Directory setup failed: {exc}")
            return False

    def sandbox_outputs_root(self, cloud_storage_available: bool = True) -> str:
        """Return the agent outputs root path as seen inside the sandbox."""
        manager = self.sandbox_manager
        if hasattr(manager, "get_outputs_root"):
            return manager.get_outputs_root()
        # Modal legacy: /cloud-storage mount
        return "/cloud-storage/agent_outputs" if cloud_storage_available else "/tmp/agent_outputs"

    def normalize_agent_outputs_path(
        self,
        content: str,
        cloud_storage_available: bool = True,
    ) -> str:
        """Rewrite ./agent_outputs references for Modal; no-op for Docker/local."""
        if self.config.execution_backend_name != "modal":
            return content
        target_root = self.sandbox_outputs_root(cloud_storage_available)
        normalized = content.replace("./agent_outputs/", f"{target_root}/")
        return normalized.replace("./agent_outputs", target_root)

    @staticmethod
    def is_transient_exec_failure(error_text: str) -> bool:
        lowered = (error_text or "").lower()
        transient_markers = (
            "oci runtime exec failed",
            "unable to start container process",
            "connection reset by peer",
            "broken pipe",
            "no such container",
            "container is not running",
            "cannot exec in a stopped state",
            "sandbox not available",
        )
        return any(marker in lowered for marker in transient_markers)
