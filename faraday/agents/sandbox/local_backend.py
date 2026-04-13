from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from faraday.agents.sandbox.legacy import migrate_legacy_faraday_cloud_storage
from faraday.agents.sandbox.repl import REPL_SERVER_SCRIPT, ReplProcess
from faraday.agents.sandbox.config import SandboxConfig, SandboxState, tprint


class _ReadableTempFile:
    def __init__(self) -> None:
        self._file = tempfile.TemporaryFile()

    def fileno(self) -> int:
        return self._file.fileno()

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        self._file.flush()
        self._file.seek(0)
        return self._file.read(*args, **kwargs)

    def close(self) -> None:
        self._file.close()


class LocalProcess:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        stdout: _ReadableTempFile,
        stderr: _ReadableTempFile,
    ) -> None:
        self._process = process
        self.stdout = stdout
        self.stderr = stderr

    @property
    def returncode(self) -> Optional[int]:
        return self._process.returncode

    def poll(self) -> Optional[int]:
        return self._process.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        return self._process.wait(timeout=timeout)

    def terminate(self) -> None:
        self._process.terminate()


class LocalSandboxManager:
    """Local workspace-backed sandbox that mimics the DockerSandboxManager surface."""

    SANDBOX_DIRECTORIES: list[str] = []  # host dirs created at init, not container paths
    # Resolved at __init__ time once workspace_root is known.
    AGENT_OUTPUTS_ROOT: str = ""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.state = SandboxState.UNINITIALIZED
        self.sandbox_id: Optional[str] = None
        self.error_message: Optional[str] = None
        self.initialization_time: Optional[float] = None
        self.progress_callback = None
        self._directories_initialized = False
        self._repl: Optional[ReplProcess] = None

        self.workspace_root = Path(config.workspace_source_root or Path.cwd()).expanduser().resolve()
        self.workspace_mount_path = config.workspace_mount_path
        migrate_legacy_faraday_cloud_storage(self.workspace_root)
        _cwd_ws = Path.cwd().expanduser().resolve()
        if _cwd_ws != self.workspace_root.resolve():
            migrate_legacy_faraday_cloud_storage(_cwd_ws)

        self.AGENT_OUTPUTS_ROOT = str(self.workspace_root / "agent_outputs")

        if self.config.enable_sandbox and not self.config.lazy_initialization:
            self.start_initialization()

    def set_progress_callback(self, callback) -> None:
        self.progress_callback = callback

    def _update_progress(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        else:
            tprint(message, self.config.verbose)

    def _translate_text(self, text: str) -> str:
        """Translate legacy /cloud-storage virtual paths to concrete workspace paths."""
        ws = str(self.workspace_root)
        t = text
        t = t.replace("/cloud-storage/agent_outputs", f"{ws}/agent_outputs")
        t = t.replace("/project-storage", f"{ws}/agent_outputs")
        t = t.replace("/project-files", f"{ws}/agent_outputs")
        t = t.replace("/cloud-storage", f"{ws}/.faraday_runtime/cloud-storage-legacy")
        return t

    def _prepare_exec_args(self, args: tuple[Any, ...]) -> list[str]:
        prepared = [str(arg) for arg in args]
        if len(prepared) >= 3 and prepared[0] in {"bash", "sh"} and prepared[1] in {"-c", "-lc"}:
            prepared[2] = self._translate_text(prepared[2])
        elif len(prepared) >= 4 and prepared[0] in {"python", "python3"} and prepared[2] == "-c":
            prepared[3] = self._translate_text(prepared[3])
        else:
            for i, arg in enumerate(prepared):
                if "/cloud-storage" in arg or "/project-storage" in arg or "/project-files" in arg:
                    prepared[i] = self._translate_text(arg)
        return prepared

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "FARADAY_EXECUTION_BACKEND": "local",
                "FARADAY_WORKSPACE_ROOT": str(self.workspace_root),
                "FARADAY_CLOUD_STORAGE_ROOT": str(self.workspace_root / "agent_outputs"),
            }
        )
        return env

    def start_initialization(self) -> None:
        if not self.config.enable_sandbox:
            self.state = SandboxState.UNINITIALIZED
            return
        if self.state == SandboxState.READY:
            return
        start_time = time.time()
        self.state = SandboxState.INITIALIZING
        try:
            for path in (
                self.workspace_root,
                self.workspace_root / "agent_outputs",
                self.workspace_root / "agent_outputs" / "plots",
                self.workspace_root / "agent_outputs" / "data",
                self.workspace_root / "agent_outputs" / "webpages",
            ):
                path.mkdir(parents=True, exist_ok=True)
            self.sandbox_id = f"local:{self.workspace_root}"
            self.initialization_time = time.time() - start_time
            self.state = SandboxState.READY
            self.error_message = None
            self._update_progress(f"Local sandbox ready at {self.workspace_root}")
        except Exception as exc:
            self.state = SandboxState.ERROR
            self.error_message = str(exc)
            self._update_progress(f"Failed to initialize local sandbox: {exc}")

    def wait_for_ready(self, timeout: float = 120.0) -> bool:
        del timeout
        if not self.config.enable_sandbox:
            return False
        if self.state in {SandboxState.UNINITIALIZED, SandboxState.TERMINATED, SandboxState.ERROR}:
            self.start_initialization()
        return self.state == SandboxState.READY

    def get_sandbox(self):
        if not self.wait_for_ready():
            return None
        return self

    def get_outputs_root(self) -> str:
        return self.AGENT_OUTPUTS_ROOT

    def ensure_directories(self, _sandbox: Any = None, _directories: Any = None) -> bool:
        """Create agent output subdirectories on the host filesystem."""
        for subdir in ("plots", "data", "webpages"):
            (self.workspace_root / "agent_outputs" / subdir).mkdir(parents=True, exist_ok=True)
        self._directories_initialized = True
        return True

    def get_repl(self) -> "ReplProcess":
        """Return the running REPL, starting it if necessary."""
        if self._repl is None or not self._repl.is_alive():
            self._repl = self._start_repl()
        return self._repl

    def exec_python_code(self, code: str, timeout: float = 300.0) -> tuple:
        """Execute Python code in the persistent REPL, returning (stdout, stderr, had_error)."""
        repl = self.get_repl()
        return repl.execute(code, timeout=timeout)

    def _start_repl(self) -> "ReplProcess":
        if self.state != SandboxState.READY:
            self.start_initialization()
            if self.state != SandboxState.READY:
                raise RuntimeError(self.error_message or "local sandbox not ready for REPL start")
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", REPL_SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(self.workspace_root),
            env=self._base_env(),
        )
        return ReplProcess(proc)

    def _detect_project_files(self, _sandbox: Any) -> bool:
        return False

    def reinitialize(self) -> None:
        if self._repl is not None:
            self._repl.terminate()
            self._repl = None
        self.terminate()
        self.state = SandboxState.UNINITIALIZED
        self.start_initialization()

    def terminate(self) -> None:
        if self._repl is not None:
            self._repl.terminate()
            self._repl = None
        self.state = SandboxState.TERMINATED

    def exec(self, *args: Any) -> LocalProcess:
        if self.state != SandboxState.READY:
            self.start_initialization()
        command = self._prepare_exec_args(args)
        stdout_file = _ReadableTempFile()
        stderr_file = _ReadableTempFile()
        process = subprocess.Popen(
            command,
            cwd=str(self.workspace_root),
            env=self._base_env(),
            stdout=stdout_file,
            stderr=stderr_file,
        )
        return LocalProcess(process, stdout_file, stderr_file)
