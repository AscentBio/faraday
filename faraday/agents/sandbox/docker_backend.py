from __future__ import annotations

from datetime import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from faraday.agents.sandbox.legacy import migrate_legacy_faraday_cloud_storage
from faraday.agents.sandbox.repl import REPL_SERVER_SCRIPT, ReplProcess
from faraday.agents.sandbox.config import SandboxConfig, SandboxState, tprint

FARADAY_HOST_WORKSPACE_ROOT = "FARADAY_HOST_WORKSPACE_ROOT"
FARADAY_RUNNING_IN_APP_DOCKER = "FARADAY_RUNNING_IN_APP_DOCKER"
FARADAY_APP_WORKSPACE_MOUNT_PATH = "FARADAY_APP_WORKSPACE_MOUNT_PATH"


def running_in_app_docker() -> bool:
    return os.getenv(FARADAY_RUNNING_IN_APP_DOCKER, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def docker_sidecar_requirements_message() -> str:
    return (
        "Docker sidecar code execution requires Docker access from the Faraday app container "
        "(`/var/run/docker.sock` mounted), FARADAY_HOST_WORKSPACE_ROOT pointing at the host "
        "workspace path, and a prebuilt 'faraday-code-sandbox' image on the Docker host."
    )


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


class DockerProcess:
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


def _println_unredirected(message: str) -> None:
    """Print to the process's original stdout (bypasses Faraday CLI agent-init capture)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    real_out = getattr(sys, "__stdout__", None) or sys.stdout
    print(f"[{ts}] {message}", file=real_out, flush=True)


_SKIP_WORKSPACE_TREE_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
}


def _format_workspace_tree_lines(
    root: Path,
    *,
    max_depth: int = 4,
    max_lines: int = 250,
) -> list[str]:
    """Bounded tree listing under ``root`` for startup diagnostics."""
    root = root.resolve()
    lines: list[str] = []

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if len(lines) >= max_lines:
            return
        if depth >= max_depth:
            return
        try:
            entries = sorted(
                dir_path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except (OSError, PermissionError):
            return
        entries = [e for e in entries if e.name not in _SKIP_WORKSPACE_TREE_NAMES]
        for i, entry in enumerate(entries):
            if len(lines) >= max_lines:
                return
            branch = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{branch}{entry.name}")
            if entry.is_dir():
                next_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
                walk(entry, next_prefix, depth + 1)

    lines.append(f"{root.name}/")
    walk(root, "", 0)
    if len(lines) >= max_lines:
        lines.append("... (truncated)")
    return lines


def _container_name_for_config(config: SandboxConfig) -> str:
    """Produce a deterministic, session-scoped name for the code-sandbox *sidecar* container.

    Prefix ``faraday-code-sandbox-sidecar-`` distinguishes this from the Faraday *app*
    container (``faraday-app-*``, set by ``faraday --use-docker``) in ``docker ps``.
    """
    raw = (config.chat_id or "").strip()
    safe = re.sub(r"[^a-zA-Z0-9]", "-", raw).strip("-")[:20]
    if not safe:
        safe = uuid.uuid4().hex[:12]
    return f"faraday-code-sandbox-sidecar-{safe}"


class DockerSandboxManager:
    """Docker-backed sandbox manager compatible with LocalSandboxManager surface."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.state = SandboxState.UNINITIALIZED
        self.sandbox_id: Optional[str] = None
        self.error_message: Optional[str] = None
        self.initialization_time: Optional[float] = None
        self.progress_callback = None
        self._directories_initialized = False
        self._repl: Optional[ReplProcess] = None

        self._state_lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        self._reinitialize_count: int = 0
        self._workspace_tree_logged: bool = False

        self.workspace_root = Path(config.workspace_source_root or Path.cwd()).expanduser().resolve()
        self.workspace_mount_path = config.workspace_mount_path
        self._app_workspace_mount_path_env = os.getenv(FARADAY_APP_WORKSPACE_MOUNT_PATH, "").strip()
        self.app_workspace_mount_path = self._app_workspace_mount_path_env or self.workspace_mount_path
        self.agent_outputs_root = f"{self.workspace_mount_path}/agent_outputs"
        self.sandbox_directories = [
            f"{self.agent_outputs_root}/",
            f"{self.agent_outputs_root}/plots/",
            f"{self.agent_outputs_root}/data/",
            f"{self.agent_outputs_root}/reports/",
            f"{self.agent_outputs_root}/tex/",
            f"{self.agent_outputs_root}/webpages/",
        ]
        host_workspace_root = os.getenv(FARADAY_HOST_WORKSPACE_ROOT, "").strip()
        self.host_workspace_root = host_workspace_root or None
        self._running_in_app_docker = (
            (config.app_runtime or "host").strip().lower() == "docker"
            or running_in_app_docker()
        )
        self._mount_workspace_root = self._resolve_mount_workspace_root()
        self.image = (
            (config.docker_image or "").strip()
            or os.getenv("FARADAY_DOCKER_IMAGE", "faraday-code-sandbox").strip()
            or "faraday-code-sandbox"
        )
        self.container_name = _container_name_for_config(config)
        migrate_legacy_faraday_cloud_storage(self.workspace_root)
        # App-in-Docker mounts the repo into the sandbox at workspace_mount_path; workspace_root
        # may be a subdir (e.g. ./test_folder)
        # while obsolete ``.faraday_runtime/cloud-storage`` from older releases lives at the mount root.
        _cwd_ws = Path.cwd().expanduser().resolve()
        if _cwd_ws != self.workspace_root.resolve():
            migrate_legacy_faraday_cloud_storage(_cwd_ws)
        self._startup_probe_timeout_sec = 12.0
        self._debug_enabled = os.getenv("FARADAY_DOCKER_DEBUG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if self.config.enable_sandbox and not self.config.lazy_initialization:
            self.start_initialization()

        # Log workspace as soon as the Docker backend is chosen — not only when the sidecar
        # starts (lazy sandbox) or when code tools run (no tools => no init => no log).
        if self.config.enable_sandbox:
            try:
                self.workspace_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            self._log_workspace_tree_once()

    def _resolve_mount_workspace_root(self) -> str:
        if not self.host_workspace_root:
            return str(self.workspace_root)
        if not self._running_in_app_docker:
            return self.host_workspace_root
        # For legacy/test environments that only provide FARADAY_HOST_WORKSPACE_ROOT,
        # preserve previous behavior and mount the host workspace root directly.
        if not self._app_workspace_mount_path_env:
            return self.host_workspace_root

        host_root = Path(self.host_workspace_root).expanduser().resolve()
        app_mount = Path(self.app_workspace_mount_path).expanduser().resolve()
        workspace = self.workspace_root.resolve()
        try:
            rel = workspace.relative_to(app_mount)
        except ValueError:
            raise RuntimeError(
                f"workspace.source_root ({workspace}) must be inside app mount path "
                f"({app_mount}) when runtime.app is docker."
            )
        return str((host_root / rel).resolve())

    @staticmethod
    def is_docker_available() -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            probe = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            return probe.returncode == 0
        except Exception:
            return False

    def set_progress_callback(self, callback) -> None:
        self.progress_callback = callback

    def _update_progress(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        else:
            tprint(message, self.config.verbose)

    def _debug(self, message: str) -> None:
        """Temporary verbose diagnostics for docker sandbox stability issues."""
        if not self._debug_enabled:
            return
        self._update_progress(f"[docker-debug] {message}")

    def _sandbox_label(self) -> str:
        if self._running_in_app_docker:
            return "Docker sidecar sandbox"
        return "Docker sandbox"

    def _sandbox_container_label(self) -> str:
        if self._running_in_app_docker:
            return "code-exec sidecar container"
        return "docker sandbox container"

    def _validate_sidecar_contract(self) -> None:
        if self._running_in_app_docker and not self.host_workspace_root:
            raise RuntimeError(
                "Missing FARADAY_HOST_WORKSPACE_ROOT inside the Faraday app container. "
                + docker_sidecar_requirements_message()
            )

    def _log_workspace_tree_once(self) -> None:
        """Print a concise workspace summary once, with full tree only in debug mode."""
        if self._workspace_tree_logged:
            return
        self._workspace_tree_logged = True

        def emit(message: str) -> None:
            if self.progress_callback:
                self.progress_callback(message)
            # Always mirror to the real TTY: CLI wraps agent init in redirect_stdout(LogSink)
            # and never replays captured lines, so tprint() alone would be invisible.
            _println_unredirected(message)

        if not self._debug_enabled:
            return
        emit(
            f"Sandbox workspace: {self._mount_workspace_root} \u2192 {self.workspace_mount_path}"
        )
        try:
            for line in _format_workspace_tree_lines(self.workspace_root):
                emit(f"  {line}")
        except Exception as exc:
            emit(f"  (workspace tree unavailable: {exc})")

    @staticmethod
    def _is_default_local_sandbox_image_name(image: str) -> bool:
        """True for unqualified ``faraday-code-sandbox`` tags (not a registry path).

        That name is local-only; there is no public image to pull from Docker Hub.
        """
        name = (image or "").strip().lower().split(":", 1)[0]
        if not name or "/" in name:
            return False
        return name == "faraday-code-sandbox"

    def _find_sandbox_dockerfile(self) -> Optional[tuple[Path, Path]]:
        """Locate a Dockerfile for the code sandbox.

        Resolution order:
        1. ``config.docker_dockerfile`` (explicit path from YAML / CLI)
        2. ``Dockerfile.sandbox`` in workspace root
        3. ``Dockerfile.sandbox`` in cwd (if different)

        Returns ``(dockerfile_path, build_context)`` or *None*.  The build
        context defaults to the Dockerfile's parent directory.
        """
        configured = (self.config.docker_dockerfile or "").strip()
        if configured:
            df = Path(configured).expanduser().resolve()
            if df.is_file():
                return df, df.parent
            for base in [self.workspace_root, Path.cwd().resolve()]:
                candidate = (base / configured).resolve()
                if candidate.is_file():
                    return candidate, candidate.parent
            return None

        candidates = [self.workspace_root]
        cwd = Path.cwd().resolve()
        if cwd != self.workspace_root.resolve():
            candidates.append(cwd)
        for base in candidates:
            df = base / "Dockerfile.sandbox"
            if df.is_file():
                return df, base
        return None

    def _build_sandbox_image_from_dockerfile(
        self,
        dockerfile_path: Path,
        build_context: Path,
        *,
        after_pull_failed: bool,
    ) -> bool:
        """Return True if build succeeded."""
        build_label = "Dockerfile.sandbox (code execution dependencies)"
        if after_pull_failed:
            self._update_progress(
                f"Pull failed; building '{self.image}' from {dockerfile_path} ({build_label})"
            )
        else:
            self._update_progress(
                f"Docker image '{self.image}' not found locally; building from {dockerfile_path} "
                f"({build_label}). There is no public image with this name on Docker Hub."
            )
        build = subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                self.image,
                str(build_context),
            ],
            check=False,
        )
        return build.returncode == 0

    def _ensure_image(self) -> None:
        inspect = subprocess.run(
            ["docker", "image", "inspect", self.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspect.returncode == 0:
            return

        if self._running_in_app_docker:
            raise RuntimeError(
                f"Docker sidecar image '{self.image}' not found on the Docker host. "
                "Build it first on the host, e.g. "
                "`docker build -f Dockerfile.sandbox -t faraday-code-sandbox .`."
            )

        found = self._find_sandbox_dockerfile()

        # Default tag is local-only; never waste a pull on Docker Hub (confusing "access denied").
        if self._is_default_local_sandbox_image_name(self.image):
            if found is not None:
                df_path, ctx = found
                if self._build_sandbox_image_from_dockerfile(df_path, ctx, after_pull_failed=False):
                    return
                raise RuntimeError(
                    f"docker build failed for '{self.image}' from {df_path}. "
                    "Fix the build output above, or set execution.docker_image to a pullable image."
                )
            raise RuntimeError(
                f"Docker image '{self.image}' not found locally, and Dockerfile.sandbox "
                f"was not found under workspace source root ({self.workspace_root}) "
                f"or cwd ({Path.cwd()}). "
                "From your Faraday repo root run: "
                "`docker build -f Dockerfile.sandbox -t faraday-code-sandbox .` "
                "or set execution.docker_image to a pullable image."
            )

        self._update_progress(f"Docker image '{self.image}' missing; attempting pull...")
        pull = subprocess.run(["docker", "pull", self.image], check=False)
        if pull.returncode == 0:
            return

        if found is not None:
            df_path, ctx = found
            if self._build_sandbox_image_from_dockerfile(df_path, ctx, after_pull_failed=True):
                return
        raise RuntimeError(
            f"Unable to prepare docker image '{self.image}'. "
            "Provide a pullable image, or Dockerfile.sandbox under the workspace source root "
            "or repo root (e.g. docker build -f Dockerfile.sandbox -t faraday-code-sandbox .)."
        )

    def _docker_resource_flags(self) -> list[str]:
        flags: list[str] = []
        if self.config.docker_memory:
            flags.extend(["--memory", str(self.config.docker_memory)])
        if self.config.docker_cpus is not None:
            flags.extend(["--cpus", str(self.config.docker_cpus)])
        if self.config.docker_pids_limit is not None:
            flags.extend(["--pids-limit", str(self.config.docker_pids_limit)])
        if self.config.docker_shm_size:
            flags.extend(["--shm-size", str(self.config.docker_shm_size)])
        if self.config.docker_network:
            flags.extend(["--network", self.config.docker_network])
        flags.extend(["--security-opt", "no-new-privileges"])
        return flags

    def _container_exists(self) -> bool:
        if not self.container_name:
            return False
        check = subprocess.run(
            ["docker", "container", "inspect", self.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        exists = check.returncode == 0
        self._debug(f"_container_exists={exists} container={self.container_name}")
        return exists

    def _container_is_running(self) -> bool:
        if not self.container_name:
            return False
        check = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                self.container_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode != 0:
            return False
        state = check.stdout.decode("utf-8", errors="replace").strip().lower()
        is_running = state == "true"
        self._debug(
            f"_container_is_running={is_running} container={self.container_name} raw_state={state!r}"
        )
        return is_running

    def _container_is_healthy(self) -> bool:
        """Container is usable only when it both exists and is running."""
        return self._container_exists() and self._container_is_running()

    def _get_container_logs(self) -> str:
        if not self.container_name:
            return ""
        proc = subprocess.run(
            ["docker", "logs", "--tail", "50", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        logs = stdout or stderr
        return logs[:2000]

    def _wait_for_container_ready(self) -> bool:
        """Probe container until the Python runtime responds to a trivial no-op.

        Uses exponential backoff to avoid hammering Docker on slow starts.
        Probes the Python interpreter directly rather than just the filesystem so
        that a container whose entrypoint failed to start the runtime is detected
        before it is marked READY.
        """
        deadline = time.time() + self._startup_probe_timeout_sec
        probe_attempt = 0
        delay = 0.1
        while time.time() < deadline:
            probe_attempt += 1
            if not self._container_exists() or not self._container_is_running():
                self._debug(
                    f"startup probe attempt={probe_attempt} container not running yet"
                )
                sleep_for = min(delay, deadline - time.time())
                if sleep_for > 0:
                    time.sleep(sleep_for)
                delay = min(delay * 2, 2.0)
                continue
            try:
                probe = subprocess.run(
                    [
                        "docker",
                        "exec",
                        self.container_name,
                        "python3",
                        "-c",
                        "print('ok')",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                self._debug(f"startup probe attempt={probe_attempt} python probe timed out")
                probe = None  # type: ignore[assignment]

            if probe is not None:
                self._debug(
                    f"startup probe attempt={probe_attempt} "
                    f"python probe returncode={probe.returncode} "
                    f"stdout={probe.stdout.strip()!r}"
                )
                if probe.returncode == 0 and probe.stdout.strip() == b"ok":
                    return True

            sleep_for = min(delay, deadline - time.time())
            if sleep_for > 0:
                time.sleep(sleep_for)
            delay = min(delay * 2, 2.0)
        self._debug("startup probe timed out")
        return False

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        interval = self.config.heartbeat_interval_sec
        if interval <= 0:
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"faraday-hb-{self.container_name}",
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        interval = max(5.0, self.config.heartbeat_interval_sec)
        while not self._heartbeat_stop.wait(interval):
            try:
                with self._state_lock:
                    if self.state != SandboxState.READY:
                        continue
                # Run the Docker subprocess check outside the lock.
                healthy = self._container_is_healthy()
                if not healthy:
                    self._update_progress(
                        f"Heartbeat: {self._sandbox_container_label()} died; "
                        "will reinitialize on next exec."
                    )
                    with self._state_lock:
                        if self.state == SandboxState.READY:
                            self.state = SandboxState.UNINITIALIZED
                            self.sandbox_id = None
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def start_initialization(self) -> None:
        if not self.config.enable_sandbox:
            self.state = SandboxState.UNINITIALIZED
            return
        if self.state == SandboxState.READY and self._container_is_healthy():
            return
        if self._container_exists() and not self._container_is_running():
            # Remove stale stopped container before recreating with same name.
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if not self.is_docker_available():
            self.state = SandboxState.ERROR
            if self._running_in_app_docker:
                self.error_message = (
                    "Docker is unavailable inside the Faraday app container "
                    "(CLI missing, socket not mounted, or daemon unreachable). "
                    + docker_sidecar_requirements_message()
                )
            else:
                self.error_message = "Docker is unavailable (CLI missing or daemon not running)."
            self._update_progress(self.error_message)
            return

        start_time = time.time()
        self.state = SandboxState.INITIALIZING
        self._debug(
            f"start_initialization image={self.image} workspace_root={self.workspace_root} "
            f"mount_workspace_root={self._mount_workspace_root} container={self.container_name}"
        )
        try:
            self._validate_sidecar_contract()
            self.workspace_root.mkdir(parents=True, exist_ok=True)
            (self.workspace_root / "agent_outputs").mkdir(parents=True, exist_ok=True)
            self._ensure_image()

            boot_cmd = (
                f"mkdir -p {self.agent_outputs_root}/plots {self.agent_outputs_root}/data "
                f"{self.agent_outputs_root}/reports {self.agent_outputs_root}/tex "
                f"{self.agent_outputs_root}/webpages && tail -f /dev/null"
            )
            run_cmd = [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.container_name,
                *self._docker_resource_flags(),
                "-v",
                f"{self._mount_workspace_root}:{self.workspace_mount_path}",
                "-w",
                self.workspace_mount_path,
                self.image,
                "sh",
                "-lc",
                boot_cmd,
            ]
            proc = subprocess.run(
                run_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self._debug(
                f"docker run returncode={proc.returncode} stdout={proc.stdout.decode('utf-8', errors='replace').strip()!r} "
                f"stderr={proc.stderr.decode('utf-8', errors='replace').strip()!r}"
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"docker run failed: {stderr}")

            container_id = proc.stdout.decode("utf-8", errors="replace").strip()
            self.sandbox_id = container_id or self.container_name
            if not self._wait_for_container_ready():
                logs = self._get_container_logs()
                self._debug(f"startup probe failed; recent logs={logs!r}")
                raise RuntimeError(
                    "docker container failed startup probe"
                    + (f": {logs}" if logs else "")
                )
            self.initialization_time = time.time() - start_time
            with self._state_lock:
                self.state = SandboxState.READY
                self.error_message = None
            self._reinitialize_count = 0
            self._start_heartbeat()
            if self._running_in_app_docker:
                self._update_progress(
                    f"Docker sidecar sandbox ready ({self.sandbox_id[:12]}; "
                    f"app container -> {self.container_name})"
                )
            else:
                self._update_progress(f"Docker sandbox ready ({self.sandbox_id[:12]})")
        except Exception as exc:
            self.state = SandboxState.ERROR
            self.error_message = str(exc)
            self._debug(f"initialization exception={exc!r}")
            self._update_progress(f"Failed to initialize {self._sandbox_label().lower()}: {exc}")

    def wait_for_ready(self, timeout: float = 120.0) -> bool:
        del timeout
        if not self.config.enable_sandbox:
            return False
        self._debug(f"wait_for_ready state={self.state.value} container={self.container_name}")
        if self.state == SandboxState.READY and not self._container_is_healthy():
            # Container may have exited/stopped/been removed after initialization.
            self._update_progress(
                f"{self._sandbox_label()} container unavailable while marked ready; reinitializing."
            )
            self.state = SandboxState.UNINITIALIZED
            self.sandbox_id = None
        if self.state in {SandboxState.UNINITIALIZED, SandboxState.TERMINATED, SandboxState.ERROR}:
            self.start_initialization()
        ready = self.state == SandboxState.READY and self._container_is_healthy()
        self._debug(f"wait_for_ready result={ready} state={self.state.value}")
        return ready

    def get_sandbox(self):
        if not self.wait_for_ready():
            return None
        return self

    def get_outputs_root(self) -> str:
        return self.agent_outputs_root

    def ensure_directories(self, _sandbox: Any = None, _directories: Any = None) -> bool:
        """Create the agent output subdirectories inside the container."""
        for directory in self.sandbox_directories:
            self.exec("mkdir", "-p", directory).wait()
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
        if self.state != SandboxState.READY or not self._container_is_healthy():
            self.start_initialization()
            if self.state != SandboxState.READY:
                raise RuntimeError(self.error_message or "sandbox not ready for REPL start")
        proc = subprocess.Popen(
            [
                "docker", "exec", "-i", "-w", self.workspace_mount_path,
                self.container_name, "python3", "-u", "-c", REPL_SERVER_SCRIPT,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self._base_env(),
        )
        return ReplProcess(proc)

    def _detect_project_files(self, _sandbox: Any) -> bool:
        return False

    def reinitialize(self) -> None:
        max_attempts = self.config.max_reinitialize_attempts
        self._reinitialize_count += 1
        if max_attempts > 0 and self._reinitialize_count > max_attempts:
            msg = (
                f"{self._sandbox_label()} reinitialize limit reached "
                f"({self._reinitialize_count}/{max_attempts}); giving up."
            )
            self._update_progress(msg)
            self.state = SandboxState.ERROR
            self.error_message = msg
            return
        self.terminate()
        self.state = SandboxState.UNINITIALIZED
        self.start_initialization()

    def terminate(self) -> None:
        self._stop_heartbeat()
        if self._repl is not None:
            self._repl.terminate()
            self._repl = None
        if self._container_exists():
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        self.state = SandboxState.TERMINATED

    def _translate_path(self, text: str) -> str:
        """Map virtual /cloud-storage/... paths to workspace layout (./agent_outputs + runtime)."""
        t = text
        t = t.replace("/cloud-storage/agent_outputs", self.agent_outputs_root)
        t = t.replace(
            "/cloud-storage/.modal_code_vars",
            f"{self.workspace_mount_path}/.faraday_runtime/.modal_code_vars",
        )
        t = t.replace("/cloud-storage", f"{self.workspace_mount_path}/.faraday_runtime/cloud-storage-legacy")
        return t

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "FARADAY_EXECUTION_BACKEND": "local",
                "FARADAY_WORKSPACE_ROOT": self.workspace_mount_path,
                "FARADAY_CLOUD_STORAGE_ROOT": self.agent_outputs_root,
            }
        )
        return env

    def _prepare_exec_args(self, args: tuple[Any, ...]) -> list[str]:
        prepared = [str(arg) for arg in args]
        if len(prepared) >= 3 and prepared[0] in {"bash", "sh"} and prepared[1] in {"-c", "-lc"}:
            prepared[2] = self._translate_path(prepared[2])
        elif len(prepared) >= 4 and prepared[0] in {"python", "python3"} and prepared[2] == "-c":
            prepared[3] = self._translate_path(prepared[3])
        else:
            for i, arg in enumerate(prepared):
                if "/cloud-storage" in arg or "/project-storage" in arg or "/project-files" in arg:
                    prepared[i] = self._translate_path(arg)
        return prepared

    def exec(self, *args: Any) -> DockerProcess:
        self._debug(f"exec requested args={args!r} state={self.state.value}")
        if self.state != SandboxState.READY or not self._container_is_healthy():
            self.start_initialization()
            if self.state != SandboxState.READY or not self._container_is_healthy():
                raise RuntimeError(
                    self.error_message
                    or f"{self._sandbox_label()} is not ready (container unavailable)."
                )
        command = self._prepare_exec_args(args)
        self._debug(f"exec prepared command={command!r}")
        full_command = ["docker", "exec", "-w", self.workspace_mount_path, self.container_name, *command]
        stdout_file = _ReadableTempFile()
        stderr_file = _ReadableTempFile()
        process = subprocess.Popen(
            full_command,
            env=self._base_env(),
            stdout=stdout_file,
            stderr=stderr_file,
        )
        return DockerProcess(process, stdout_file, stderr_file)
