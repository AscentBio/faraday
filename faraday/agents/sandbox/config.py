from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from faraday.agents.sandbox.docker_memory import normalize_docker_memory


class SandboxState(Enum):
    """Clear states for sandbox lifecycle management."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class SandboxConfig:
    """Centralized configuration for sandbox initialization.

    **Core fields** (used by all backends):
        ``user_id``, ``chat_id``, ``query_id``, ``execution_backend``,
        ``app_runtime``, ``workspace_source_root``, ``workspace_mount_path``,
        ``enable_sandbox``, ``lazy_initialization``, ``max_reinitialize_attempts``,
        ``heartbeat_interval_sec``, ``verbose``.

    **Docker / Modal fields** are kept for backward compatibility with the
    built-in backends.  New backends should read their own settings from
    :attr:`backend_config` instead of adding top-level fields here.

    **backend_config** is an open ``dict`` for backend-specific settings.
    Each backend documents which keys it reads::

        SandboxConfig(
            execution_backend="k8s",
            backend_config={"namespace": "science", "service_account": "faraday"},
        )
    """

    # --- Core (all backends) ------------------------------------------------
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    query_id: Optional[str] = None
    execution_backend: str = "docker"
    app_runtime: str = "host"
    workspace_source_root: Optional[str] = None
    workspace_mount_path: str = "/workspace"
    enable_sandbox: bool = True
    lazy_initialization: bool = True
    max_reinitialize_attempts: int = 3
    heartbeat_interval_sec: float = 30.0
    verbose: bool = True

    # --- Backend-specific (open dict for third-party backends) ---------------
    backend_config: Dict[str, Any] = field(default_factory=dict)

    # --- Docker / Modal built-in fields (kept for backward compat) ----------
    cloud_storage_root: Optional[str] = None
    cloud_storage_mode: str = "disabled"
    bucket_name: Optional[str] = None
    docker_image: Optional[str] = None
    docker_dockerfile: Optional[str] = None
    docker_memory: Optional[str] = None
    docker_cpus: Optional[float] = None
    docker_pids_limit: Optional[int] = None
    docker_shm_size: Optional[str] = None
    docker_network: Optional[str] = None
    app_name: str = "faraday-codeexec-sandbox01"
    python_version: str = "3.10"

    def __post_init__(self):
        runtime = (self.app_runtime or "host").strip().lower().replace("_", "-")
        self.app_runtime = "docker" if runtime in {"docker", "container"} else "host"
        mount_path = (self.workspace_mount_path or "/workspace").strip() or "/workspace"
        if not mount_path.startswith("/"):
            mount_path = f"/{mount_path}"
        self.workspace_mount_path = mount_path.rstrip("/") or "/"
        normalized_mode = (self.cloud_storage_mode or "disabled").strip().lower()
        if normalized_mode not in {"disabled", "optional", "required"}:
            normalized_mode = "disabled"
        self.cloud_storage_mode = normalized_mode
        self.docker_memory = normalize_docker_memory(self.docker_memory)
        if self.docker_cpus is not None:
            try:
                cpus = float(self.docker_cpus)
                self.docker_cpus = cpus if cpus > 0 else None
            except Exception:
                self.docker_cpus = None
        if self.docker_pids_limit is not None:
            try:
                pids_limit = int(self.docker_pids_limit)
                self.docker_pids_limit = pids_limit if pids_limit > 0 else None
            except Exception:
                self.docker_pids_limit = None
        shm_size = (self.docker_shm_size or "").strip()
        self.docker_shm_size = shm_size or None
        network = (self.docker_network or "").strip()
        self.docker_network = network or None
        self.max_reinitialize_attempts = max(0, int(self.max_reinitialize_attempts))
        self.heartbeat_interval_sec = max(0.0, float(self.heartbeat_interval_sec))

    @property
    def bucket_path(self) -> str:
        if self.user_id and self.chat_id and self.query_id:
            return f"{self.user_id}/{self.chat_id}/{self.query_id}/"
        if self.user_id and self.chat_id:
            return f"{self.user_id}/{self.chat_id}/"
        if self.chat_id:
            return f"{self.chat_id}/"
        return ""

    @property
    def needs_bucket(self) -> bool:
        if self.cloud_storage_mode == "disabled":
            return False
        return bool(self.chat_id and self.bucket_name)


def tprint(message: str, verbose: bool = True) -> None:
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
