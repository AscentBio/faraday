from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from faraday.agents.sandbox.docker_memory import normalize_docker_memory


@dataclass
class ProcessResult:
    return_code: Optional[int]
    stdout: Optional[str]
    stderr: Optional[str]
    timed_out: bool = False


@dataclass
class ExecutionResult:
    output_message: str
    status_message: str
    stdout: Optional[str]
    stderr: Optional[str]
    return_code: Optional[int] = None
    timed_out: bool = False

    @property
    def has_error(self) -> bool:
        if self.timed_out:
            return True
        if self.return_code not in (None, 0):
            return True
        return bool((self.stderr or "").strip())


@dataclass(frozen=True)
class ExecutionPolicy:
    """Typed runtime policy for execution and lifecycle behavior.

    Docker memory: unset values become ``512m`` (~512 MB per code-exec container); see
    ``faraday.agents.docker_memory.normalize_docker_memory``.
    """

    sandbox_ready_timeout_sec: float = 120.0
    python_exec_timeout_sec: float = 300.0
    bash_exec_timeout_sec: float = 60.0
    transient_retry_attempts: int = 1
    reinitialize_on_transient_failure: bool = True
    max_reinitialize_attempts: int = 3
    heartbeat_interval_sec: float = 30.0
    stdout_max_bytes: int = 256 * 1024
    stderr_max_bytes: int = 32 * 1024
    docker_memory: Optional[str] = None  # normalized to 512m by default
    docker_cpus: Optional[float] = None
    docker_pids_limit: Optional[int] = None
    docker_shm_size: Optional[str] = None
    docker_network: Optional[str] = None  # e.g. "none" to disable egress

    def __post_init__(self) -> None:
        # object.__setattr__ is required for frozen dataclasses.
        object.__setattr__(
            self,
            "sandbox_ready_timeout_sec",
            max(1.0, float(self.sandbox_ready_timeout_sec)),
        )
        object.__setattr__(
            self,
            "python_exec_timeout_sec",
            max(1.0, float(self.python_exec_timeout_sec)),
        )
        object.__setattr__(
            self,
            "bash_exec_timeout_sec",
            max(1.0, float(self.bash_exec_timeout_sec)),
        )
        object.__setattr__(
            self,
            "transient_retry_attempts",
            max(0, int(self.transient_retry_attempts)),
        )
        object.__setattr__(
            self,
            "max_reinitialize_attempts",
            max(0, int(self.max_reinitialize_attempts)),
        )
        object.__setattr__(
            self,
            "heartbeat_interval_sec",
            max(0.0, float(self.heartbeat_interval_sec)),
        )
        object.__setattr__(
            self,
            "stdout_max_bytes",
            max(1024, int(self.stdout_max_bytes)),
        )
        object.__setattr__(
            self,
            "stderr_max_bytes",
            max(1024, int(self.stderr_max_bytes)),
        )
        object.__setattr__(self, "docker_memory", normalize_docker_memory(self.docker_memory))
        network = (self.docker_network or "").strip() or None
        object.__setattr__(self, "docker_network", network)
        cpus = self.docker_cpus
        if cpus is not None:
            try:
                cpus_val = float(cpus)
                cpus = cpus_val if cpus_val > 0 else None
            except Exception:
                cpus = None
        object.__setattr__(self, "docker_cpus", cpus)
        pids_limit = self.docker_pids_limit
        if pids_limit is not None:
            try:
                pids_val = int(pids_limit)
                pids_limit = pids_val if pids_val > 0 else None
            except Exception:
                pids_limit = None
        object.__setattr__(self, "docker_pids_limit", pids_limit)
        shm_size = (self.docker_shm_size or "").strip()
        object.__setattr__(self, "docker_shm_size", shm_size or None)
