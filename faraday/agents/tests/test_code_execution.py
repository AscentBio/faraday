"""Integration tests: run Python via PythonExecutor for each execution backend.

Logs from the executor stack are emitted as ``[code-exec-test] ...`` on stderr.
Use ``pytest -s`` (or ``--capture=no``) to see them in the shell; by default pytest
captures stderr along with stdout.

**Agent sandbox Docker image (RDKit):** tests that use ``RuntimeConfig.docker_image`` default
to ``agent-sandbox:latest``. Override with env ``FARADAY_TEST_DOCKER_IMAGE``. The image must
exist locally (``docker image inspect <name>``). Build e.g.::

    docker build -f Dockerfile.sandbox -t agent-sandbox:latest .
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from faraday.agents.sandbox.docker_backend import DockerSandboxManager
from faraday.agents.execution import (
    ExecutionPolicy,
    HotfixService,
    ProcessRunner,
    PythonExecutor,
    RuntimeConfig,
    SandboxRuntime,
)


def _test_log(message: str) -> None:
    print(f"[code-exec-test] {message}", file=sys.stderr, flush=True)


def _agent_sandbox_test_image() -> str:
    return (os.environ.get("FARADAY_TEST_DOCKER_IMAGE") or "agent-sandbox:latest").strip()


def _skip_reason_agent_sandbox_docker_tests() -> str | None:
    """Skip agent-sandbox tests when Docker or the configured image is unavailable."""
    if not DockerSandboxManager.is_docker_available():
        return "Docker CLI/daemon not available"
    image = _agent_sandbox_test_image()
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"docker image inspect failed: {exc}"
    if proc.returncode != 0:
        return (
            f"Docker image {image!r} not present locally. "
            "Build it (e.g. docker build -f Dockerfile.sandbox -t agent-sandbox:latest .) "
            "or set FARADAY_TEST_DOCKER_IMAGE to an existing tag."
        )
    return None


_AGENT_SANDBOX_SKIP_REASON = _skip_reason_agent_sandbox_docker_tests()


def _make_python_executor(
    workspace: Path,
    *,
    execution_backend_name: str,
    enable_sandbox: bool = True,
    allow_backend_fallback: bool = False,
    lazy_sandbox: bool = False,
    docker_image: str | None = None,
) -> PythonExecutor:
    policy = ExecutionPolicy(
        sandbox_ready_timeout_sec=120.0,
        python_exec_timeout_sec=60.0,
        bash_exec_timeout_sec=30.0,
        transient_retry_attempts=0,
    )
    cfg = RuntimeConfig(
        execution_backend_name=execution_backend_name,
        enable_sandbox=enable_sandbox,
        lazy_sandbox=lazy_sandbox,
        workspace_source_root=str(workspace),
        allow_backend_fallback=allow_backend_fallback,
        execution_policy=policy,
        docker_image=docker_image,
    )
    _test_log(
        f"configure backend={execution_backend_name!r} workspace={workspace} "
        f"enable_sandbox={enable_sandbox} docker_image={docker_image!r}"
    )
    runtime = SandboxRuntime(cfg, _test_log)
    process_runner = ProcessRunner(
        _test_log,
        stdout_max_bytes=policy.stdout_max_bytes,
        stderr_max_bytes=policy.stderr_max_bytes,
    )
    hotfix = HotfixService(llm_client=None, log_fn=_test_log)
    return PythonExecutor(
        runtime=runtime,
        process_runner=process_runner,
        hotfix_service=hotfix,
        log_fn=_test_log,
        auto_install_missing_modules=False,
        auto_install_missing_modules_timeout_sec=300,
        runtime_import_check_enabled=False,
        runtime_import_check_warn_optional=False,
    )


def _assert_print_ok(result) -> None:
    assert not result.timed_out
    assert result.return_code in (None, 0)
    assert not result.has_error
    out = (result.stdout or "").strip()
    assert "42" in out


def _assert_stdout_contains(result, *substrings: str) -> None:
    assert not result.timed_out
    assert result.return_code in (None, 0)
    assert not result.has_error
    out = (result.stdout or "").strip()
    for s in substrings:
        assert s in out, f"expected {s!r} in stdout, got {out!r}"


def _log_run_result(label: str, result) -> None:
    """Summarize execution for shell visibility (stderr; use ``pytest -s``)."""
    _test_log(
        f"{label} rc={result.return_code} timed_out={result.timed_out} "
        f"has_error={result.has_error} stdout_chars={len(result.stdout or '')}"
    )
    if result.stdout:
        preview = (result.stdout.strip())[:400]
        _test_log(f"{label} stdout: {preview!r}{'…' if len(result.stdout) > 400 else ''}")
    if result.stderr:
        preview = (result.stderr.strip())[:400]
        _test_log(f"{label} stderr: {preview!r}{'…' if len(result.stderr) > 400 else ''}")


def test_code_execution_host_prints(tmp_path: Path) -> None:
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="host",
        allow_backend_fallback=False,
    )
    result = executor.run("print(40 + 2)", timeout=30.0)
    _log_run_result("host_prints", result)
    _assert_print_ok(result)


@pytest.mark.parametrize(
    "case_id,code,expected",
    [
        ("stdlib_import_math", "import math; print(int(math.factorial(5)))", "120"),
        ("stdlib_import_itertools", "import itertools; print(list(itertools.islice(itertools.count(10), 3)))", "[10, 11, 12]"),
        ("pathlib", "from pathlib import Path; print(Path('a/b').name)", "b"),
        ("json_roundtrip", "print(json.dumps({'layer': 'test'}))", "layer"),
        (
            "multiline_function",
            "def _add(a, b):\n    return a + b\nprint(_add(22, 20))",
            "42",
        ),
    ],
)
def test_code_execution_host_script_complexity(
    tmp_path: Path, case_id: str, code: str, expected: str
) -> None:
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="host",
        allow_backend_fallback=False,
    )
    _test_log(f"host_script_complexity case={case_id}")
    result = executor.run(code, timeout=30.0)
    _log_run_result(f"host_script_complexity[{case_id}]", result)
    _assert_stdout_contains(result, expected)


def test_code_execution_host_reads_workspace_file(tmp_path: Path) -> None:
    data_path = tmp_path / "input_data.txt"
    data_path.write_text("faraday-file-test\nsecond line", encoding="utf-8")
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="host",
        allow_backend_fallback=False,
    )
    _test_log("host_reads_workspace_file: read input_data.txt from cwd")
    result = executor.run(
        "from pathlib import Path\n"
        "text = Path('input_data.txt').read_text(encoding='utf-8')\n"
        "print(len(text.splitlines()))",
        timeout=30.0,
    )
    _log_run_result("host_reads_workspace_file", result)
    _assert_stdout_contains(result, "2")


def test_code_execution_host_writes_then_reads_file(tmp_path: Path) -> None:
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="host",
        allow_backend_fallback=False,
    )
    _test_log("host_writes_then_reads_file")
    result = executor.run(
        "from pathlib import Path\n"
        "Path('out.txt').write_text('written-by-exec', encoding='utf-8')\n"
        "print(Path('out.txt').read_text(encoding='utf-8'))",
        timeout=30.0,
    )
    _log_run_result("host_writes_then_reads_file", result)
    _assert_stdout_contains(result, "written-by-exec")
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "written-by-exec"


@pytest.mark.skipif(
    not DockerSandboxManager.is_docker_available(),
    reason="Docker CLI/daemon not available",
)
def test_code_execution_docker_prints(tmp_path: Path) -> None:
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="docker",
        allow_backend_fallback=False,
    )
    result = executor.run("print(40 + 2)", timeout=120.0)
    _log_run_result("docker_prints", result)
    _assert_print_ok(result)


@pytest.mark.skipif(
    not DockerSandboxManager.is_docker_available(),
    reason="Docker CLI/daemon not available",
)
def test_code_execution_docker_import_and_file_io(tmp_path: Path) -> None:
    (tmp_path / "dock_data.txt").write_text("docker-workspace-ok", encoding="utf-8")
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="docker",
        allow_backend_fallback=False,
    )
    _test_log("docker_import_and_file_io: import + read workspace file")
    result = executor.run(
        "import math\n"
        "from pathlib import Path\n"
        "s = Path('dock_data.txt').read_text(encoding='utf-8').strip()\n"
        "print(math.sqrt(9), s)",
        timeout=120.0,
    )
    _log_run_result("docker_import_and_file_io", result)
    _assert_stdout_contains(result, "3.0", "docker-workspace-ok")


@pytest.mark.skipif(
    _AGENT_SANDBOX_SKIP_REASON is not None,
    reason=_AGENT_SANDBOX_SKIP_REASON or "skipped",
)
def test_code_execution_docker_agent_sandbox_image_uses_configured_tag(tmp_path: Path) -> None:
    """Smoke-test that code runs in the agent sandbox image (see module docstring)."""
    image = _agent_sandbox_test_image()
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="docker",
        allow_backend_fallback=False,
        docker_image=image,
    )
    _test_log(f"docker_agent_sandbox_image: using image={image!r}")
    result = executor.run("print('agent-sandbox-ok')", timeout=120.0)
    _log_run_result("docker_agent_sandbox_image", result)
    _assert_stdout_contains(result, "agent-sandbox-ok")


@pytest.mark.skipif(
    _AGENT_SANDBOX_SKIP_REASON is not None,
    reason=_AGENT_SANDBOX_SKIP_REASON or "skipped",
)
def test_code_execution_docker_agent_sandbox_rdkit(tmp_path: Path) -> None:
    """RDKit import and minimal MOL API when the agent sandbox image includes it."""
    image = _agent_sandbox_test_image()
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="docker",
        allow_backend_fallback=False,
        docker_image=image,
    )
    _test_log(f"docker_agent_sandbox_rdkit: image={image!r}")
    result = executor.run(
        "from rdkit import Chem\n"
        "m = Chem.MolFromSmiles('CCO')\n"
        "print('RDKIT_HEAVY', m.GetNumHeavyAtoms())\n",
        timeout=120.0,
    )
    _log_run_result("docker_agent_sandbox_rdkit", result)
    _assert_stdout_contains(result, "RDKIT_HEAVY", "3")


@pytest.mark.skipif(
    os.environ.get("FARADAY_TEST_MODAL") != "1",
    reason="Set FARADAY_TEST_MODAL=1 to run Modal sandbox integration (network + Modal account)",
)
def test_code_execution_modal_prints(tmp_path: Path) -> None:
    pytest.importorskip("modal", reason="modal package required")
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="modal",
        allow_backend_fallback=False,
    )
    result = executor.run("print(40 + 2)", timeout=300.0)
    _log_run_result("modal_prints", result)
    _assert_print_ok(result)


def test_code_execution_disabled_does_not_run(tmp_path: Path) -> None:
    executor = _make_python_executor(
        tmp_path,
        execution_backend_name="disabled",
        enable_sandbox=False,
    )
    result = executor.run("print(40 + 2)", timeout=5.0)
    _log_run_result("disabled", result)
    assert result.has_error
    assert "Code execution is disabled" in (result.stderr or "")
