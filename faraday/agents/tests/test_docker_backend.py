"""Unit and integration tests for the Docker sidecar code-execution model.

**Sidecar integration tests** (``test_sidecar_*``) simulate Faraday running inside the
app container: ``FARADAY_RUNNING_IN_APP_DOCKER=1`` and
``FARADAY_HOST_WORKSPACE_ROOT`` set to the host workspace path that the Docker
daemon can bind-mount into the sibling ``faraday-code-sandbox`` container.

They require a working Docker daemon, the Docker CLI, and a prebuilt sandbox image
(default tag ``faraday-code-sandbox``, overridable via ``FARADAY_TEST_DOCKER_IMAGE``).
Build e.g.::

    docker build -f Dockerfile.sandbox -t faraday-code-sandbox .

Run (from repo root)::

    uv run pytest faraday/agents/tests/test_docker_backend.py -k sidecar -s

Integration tests skip at **fixture time** (not import time) if Docker or the image is
unavailable, so starting Docker Desktop and re-running is enough without restarting Python.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from faraday.agents.sandbox.docker_backend import (
    DockerSandboxManager,
    FARADAY_APP_WORKSPACE_MOUNT_PATH,
    FARADAY_HOST_WORKSPACE_ROOT,
    FARADAY_RUNNING_IN_APP_DOCKER,
)
from faraday.agents.execution import (
    ExecutionPolicy,
    HotfixService,
    ProcessRunner,
    PythonExecutor,
    RuntimeConfig,
    SandboxRuntime,
)
from faraday.agents.sandbox.local_backend import LocalSandboxManager
from faraday.agents.sandbox.config import SandboxConfig


def _sidecar_image_name() -> str:
    return (os.environ.get("FARADAY_TEST_DOCKER_IMAGE") or "faraday-code-sandbox").strip()


def _docker_images_q(reference: str) -> str:
    """Return non-empty image id(s) if ``docker images -q <ref>`` finds a match."""
    proc = subprocess.run(
        ["docker", "images", "-q", reference],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return proc.stdout.strip()


def _docker_has_faraday_code_sandbox_by_list() -> bool:
    """Match ``docker image ls`` output: any repository whose basename is faraday-code-sandbox."""
    proc = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if proc.returncode != 0:
        return False
    for line in proc.stdout.splitlines():
        repo = line.strip()
        if not repo or repo == "<none>":
            continue
        if repo.rsplit("/", 1)[-1] == "faraday-code-sandbox":
            return True
    return False


def _sidecar_integration_skip_reason() -> str | None:
    """Skip sidecar integration tests when Docker or the sandbox image is unavailable.

    Evaluated at test time (not import time) so Docker Desktop can start after the
    interpreter loads. We avoid ``docker image inspect`` alone: on some setups name
    resolution differs from ``docker image ls``; we use ``docker images -q`` and a
    repository list fallback instead.
    """
    if not DockerSandboxManager.is_docker_available():
        return "Docker CLI/daemon not available"
    explicit = (os.environ.get("FARADAY_TEST_DOCKER_IMAGE") or "").strip()
    if explicit:
        if _docker_images_q(explicit):
            return None
        return (
            f"Docker image {explicit!r} not found (`docker images -q {explicit}` was empty). "
            "Build or tag the image, or fix FARADAY_TEST_DOCKER_IMAGE."
        )

    for ref in ("faraday-code-sandbox:latest", "faraday-code-sandbox"):
        if _docker_images_q(ref):
            return None
    if _docker_has_faraday_code_sandbox_by_list():
        return None

    return (
        "No local faraday-code-sandbox image found. "
        "Build e.g. `docker build -f Dockerfile.sandbox -t faraday-code-sandbox .` "
        "or set FARADAY_TEST_DOCKER_IMAGE to the exact name you use in `docker image ls`."
    )


@pytest.fixture
def require_sidecar_docker_image() -> None:
    """Skip integration tests unless Docker is up and a sandbox image is present."""
    reason = _sidecar_integration_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _sidecar_test_log(message: str) -> None:
    print(f"[sidecar-docker-test] {message}", file=sys.stderr, flush=True)


def _make_python_executor_app_docker_sidecar(
    workspace: Path,
    *,
    docker_image: str | None = None,
) -> PythonExecutor:
    """Docker backend with the same env the ``faraday --use-docker`` path sets."""
    policy = ExecutionPolicy(
        sandbox_ready_timeout_sec=120.0,
        python_exec_timeout_sec=120.0,
        bash_exec_timeout_sec=30.0,
        transient_retry_attempts=0,
    )
    cfg = RuntimeConfig(
        execution_backend_name="docker",
        enable_sandbox=True,
        lazy_sandbox=False,
        workspace_source_root=str(workspace),
        allow_backend_fallback=False,
        execution_policy=policy,
        docker_image=docker_image or _sidecar_image_name(),
    )
    _sidecar_test_log(
        f"app_docker_sidecar workspace={workspace} docker_image={cfg.docker_image!r}"
    )
    runtime = SandboxRuntime(cfg, _sidecar_test_log)
    process_runner = ProcessRunner(
        _sidecar_test_log,
        stdout_max_bytes=policy.stdout_max_bytes,
        stderr_max_bytes=policy.stderr_max_bytes,
    )
    hotfix = HotfixService(llm_client=None, log_fn=_sidecar_test_log)
    return PythonExecutor(
        runtime=runtime,
        process_runner=process_runner,
        hotfix_service=hotfix,
        log_fn=_sidecar_test_log,
        auto_install_missing_modules=False,
        auto_install_missing_modules_timeout_sec=300,
        runtime_import_check_enabled=False,
        runtime_import_check_warn_optional=False,
    )


def _runtime_config(workspace_root: str) -> RuntimeConfig:
    return RuntimeConfig(
        execution_backend_name="docker",
        enable_sandbox=True,
        lazy_sandbox=True,
        workspace_source_root=workspace_root,
        allow_backend_fallback=True,
        execution_policy=ExecutionPolicy(transient_retry_attempts=0),
    )


def test_docker_sandbox_manager_prefers_host_workspace_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, "/host/workspace")

    manager = DockerSandboxManager(SandboxConfig(workspace_source_root=str(tmp_path), enable_sandbox=False))

    assert manager.host_workspace_root == "/host/workspace"
    assert manager._mount_workspace_root == "/host/workspace"


def test_docker_sandbox_manager_uses_configured_mount_path(tmp_path) -> None:
    manager = DockerSandboxManager(
        SandboxConfig(
            workspace_source_root=str(tmp_path),
            workspace_mount_path="/work",
            enable_sandbox=False,
        )
    )

    assert manager.workspace_mount_path == "/work"
    assert manager.get_outputs_root() == "/work/agent_outputs"
    assert "/work/agent_outputs/reports/" in manager.sandbox_directories
    assert "/work/agent_outputs/tex/" in manager.sandbox_directories
    translated = manager._translate_path("/cloud-storage/agent_outputs/report.txt")
    assert translated == "/work/agent_outputs/report.txt"


def test_docker_sidecar_mounts_workspace_subpath_when_source_root_is_nested(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, "/host/workspace")
    monkeypatch.setenv(FARADAY_APP_WORKSPACE_MOUNT_PATH, "/workspace")

    nested = Path("/workspace/.faraday_runtime/workspace-copies/run_001")
    manager = DockerSandboxManager(
        SandboxConfig(
            workspace_source_root=str(nested),
            workspace_mount_path="/workspace",
            enable_sandbox=False,
        )
    )
    assert manager._mount_workspace_root == "/host/workspace/.faraday_runtime/workspace-copies/run_001"


def test_docker_sandbox_manager_requires_prebuilt_image_inside_app_container(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, "/host/workspace")

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("faraday.agents.sandbox.docker_backend.subprocess.run", fake_run)

    manager = DockerSandboxManager(SandboxConfig(workspace_source_root=str(tmp_path), enable_sandbox=False))

    with pytest.raises(RuntimeError, match="Build it first on the host"):
        manager._ensure_image()

    assert calls == [["docker", "image", "inspect", "faraday-code-sandbox"]]


def test_ensure_image_skips_pull_for_default_faraday_code_sandbox_when_dockerfile_present(
    monkeypatch, tmp_path: Path
) -> None:
    """Local-only image name: build from Dockerfile.sandbox without attempting docker pull."""
    monkeypatch.delenv(FARADAY_RUNNING_IN_APP_DOCKER, raising=False)
    monkeypatch.delenv(FARADAY_HOST_WORKSPACE_ROOT, raising=False)
    (tmp_path / "Dockerfile.sandbox").write_text("FROM alpine:3.19\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1)
        if argv[:2] == ["docker", "pull"]:
            raise AssertionError("docker pull should not run for default local-only image name")
        if argv[:2] == ["docker", "build"]:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("faraday.agents.sandbox.docker_backend.subprocess.run", fake_run)

    manager = DockerSandboxManager(SandboxConfig(workspace_source_root=str(tmp_path), enable_sandbox=False))
    manager._ensure_image()

    assert any(c[:2] == ["docker", "build"] for c in calls)
    assert not any(c[:2] == ["docker", "pull"] for c in calls)


def test_ensure_image_raises_without_pull_when_default_image_missing_and_no_dockerfile(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(FARADAY_RUNNING_IN_APP_DOCKER, raising=False)
    monkeypatch.delenv(FARADAY_HOST_WORKSPACE_ROOT, raising=False)
    # Ensure fallback cwd search also finds nothing.
    monkeypatch.chdir(tmp_path)

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("faraday.agents.sandbox.docker_backend.subprocess.run", fake_run)

    manager = DockerSandboxManager(SandboxConfig(workspace_source_root=str(tmp_path), enable_sandbox=False))
    with pytest.raises(RuntimeError, match="Dockerfile.sandbox"):
        manager._ensure_image()

    assert not any(c[:2] == ["docker", "pull"] for c in calls)


def test_runtime_falls_back_to_host_when_not_running_inside_app_container(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(FARADAY_RUNNING_IN_APP_DOCKER, raising=False)
    monkeypatch.delenv(FARADAY_HOST_WORKSPACE_ROOT, raising=False)
    monkeypatch.setattr(DockerSandboxManager, "is_docker_available", staticmethod(lambda: False))

    runtime = SandboxRuntime(_runtime_config(str(tmp_path)), lambda _msg: None)

    assert isinstance(runtime.sandbox_manager, LocalSandboxManager)
    assert runtime.config.execution_backend_name == "host"


def test_runtime_does_not_fall_back_to_host_inside_app_container(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, "/host/workspace")
    monkeypatch.setattr(DockerSandboxManager, "is_docker_available", staticmethod(lambda: False))

    with pytest.raises(RuntimeError, match="Docker sidecar code execution requires"):
        SandboxRuntime(_runtime_config(str(tmp_path)), lambda _msg: None)


def test_docker_sandbox_manager_labels_sidecar_mode_explicitly(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, "/host/workspace")

    manager = DockerSandboxManager(SandboxConfig(workspace_source_root=str(tmp_path), enable_sandbox=False))

    assert manager._sandbox_label() == "Docker sidecar sandbox"
    assert manager._sandbox_container_label() == "code-exec sidecar container"


def test_docker_sandbox_manager_labels_standalone_docker_mode(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(FARADAY_RUNNING_IN_APP_DOCKER, raising=False)
    monkeypatch.delenv(FARADAY_HOST_WORKSPACE_ROOT, raising=False)

    manager = DockerSandboxManager(SandboxConfig(workspace_source_root=str(tmp_path), enable_sandbox=False))

    assert manager._sandbox_label() == "Docker sandbox"
    assert manager._sandbox_container_label() == "docker sandbox container"


# ---------------------------------------------------------------------------
# Integration: app-container sidecar (real Docker + sibling sandbox container)
# ---------------------------------------------------------------------------


def test_sidecar_app_docker_mode_python_prints(
    require_sidecar_docker_image, monkeypatch, tmp_path: Path
) -> None:
    """End-to-end: simulate ``faraday --use-docker`` env and run code in sandbox image."""
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, str(tmp_path.resolve()))

    executor = _make_python_executor_app_docker_sidecar(tmp_path)
    result = executor.run("print(40 + 2)", timeout=120.0)

    assert not result.timed_out
    assert result.return_code in (None, 0)
    assert not result.has_error
    assert "42" in (result.stdout or "").strip()


def test_sidecar_app_docker_mode_reads_workspace_file(
    require_sidecar_docker_image, monkeypatch, tmp_path: Path
) -> None:
    """Workspace bind-mount from host path is visible inside the sidecar container."""
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, str(tmp_path.resolve()))

    (tmp_path / "sidecar_probe.txt").write_text("sidecar-workspace-ok", encoding="utf-8")

    executor = _make_python_executor_app_docker_sidecar(tmp_path)
    result = executor.run(
        "from pathlib import Path\n"
        "print(Path('sidecar_probe.txt').read_text(encoding='utf-8').strip())\n",
        timeout=120.0,
    )

    assert not result.timed_out
    assert result.return_code in (None, 0)
    assert not result.has_error
    assert "sidecar-workspace-ok" in (result.stdout or "").strip()


def test_sidecar_app_docker_mode_has_pdflatex(
    require_sidecar_docker_image, monkeypatch, tmp_path: Path
) -> None:
    """The sandbox image exposes a LaTeX compiler for report generation."""
    monkeypatch.setenv(FARADAY_RUNNING_IN_APP_DOCKER, "1")
    monkeypatch.setenv(FARADAY_HOST_WORKSPACE_ROOT, str(tmp_path.resolve()))

    runtime = SandboxRuntime(_runtime_config(str(tmp_path)), _sidecar_test_log)
    try:
        assert runtime.ensure_ready(timeout=120.0)
        proc = runtime.sandbox_manager.exec("pdflatex", "--version")
        proc.wait(timeout=30.0)
        assert proc.returncode == 0
        stdout = proc.stdout.read().decode("utf-8", errors="replace")
        assert "pdfTeX" in stdout
    finally:
        runtime.stop()
