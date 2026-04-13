from __future__ import annotations

import importlib.util
import time
from typing import Optional

from faraday.agents.execution.models import ExecutionResult
from faraday.agents.sandbox.packages import CORE_BASE, CORE_BIO, CORE_CHEM, CORE_DOCUMENT
from faraday.agents.sandbox.local_backend import LocalSandboxManager


LOCAL_RUNTIME_REQUIRED_MODULES = CORE_BASE
LOCAL_RUNTIME_OPTIONAL_MODULES = CORE_CHEM + CORE_BIO + CORE_DOCUMENT

# Preamble run once per REPL session (or prepended for Modal where each exec is independent).
_REPL_INIT_CODE = """import os
import sys
import json
import re
"""


class PythonExecutor:
    def __init__(
        self,
        runtime,
        process_runner,
        hotfix_service,
        log_fn,
        auto_install_missing_modules: bool,
        auto_install_missing_modules_timeout_sec: int,
        runtime_import_check_enabled: bool,
        runtime_import_check_warn_optional: bool,
    ):
        self.runtime = runtime
        self.process_runner = process_runner
        self.hotfix_service = hotfix_service
        self._log = log_fn
        self.auto_install_missing_modules = auto_install_missing_modules
        self.auto_install_missing_modules_timeout_sec = auto_install_missing_modules_timeout_sec
        self.runtime_import_check_enabled = runtime_import_check_enabled
        self.runtime_import_check_warn_optional = runtime_import_check_warn_optional
        self._runtime_import_check_reported = False
        self._auto_install_missing_modules_attempted = False
        self._repl_initialized = False

    def report_runtime_import_health_once(self) -> None:
        if self._runtime_import_check_reported:
            return
        self._runtime_import_check_reported = True
        if not self.runtime_import_check_enabled:
            return
        if not isinstance(self.runtime.sandbox_manager, LocalSandboxManager):
            return

        missing_required = [
            module for module in LOCAL_RUNTIME_REQUIRED_MODULES if importlib.util.find_spec(module) is None
        ]
        missing_optional = [
            module for module in LOCAL_RUNTIME_OPTIONAL_MODULES if importlib.util.find_spec(module) is None
        ]
        if missing_required:
            self._log(
                "Local runtime missing required Python modules for execute_python_code: "
                + ", ".join(missing_required)
            )
        if missing_optional and self.runtime_import_check_warn_optional:
            self._log("Local runtime missing optional Python modules: " + ", ".join(missing_optional))

    def run(
        self,
        code_string: str,
        timeout: float = 300.0,
        enable_hotfix: bool = False,
        allow_transient_retry: bool = True,
        run_type: str = "normal",
        transient_retry_count: int = 0,
    ) -> ExecutionResult:
        if not self.runtime.config.enable_sandbox:
            return self._error_result(
                code_string,
                "Code execution is disabled. Sandbox was not initialized.",
                return_code=None,
            )
        try:
            sb = self.runtime.get_sandbox()
            self.runtime.ensure_directories(sb)
            self._auto_install_local_runtime_modules_once(sb)
            executable_code = self.runtime.normalize_agent_outputs_path(code_string)

            # Use REPL (persistent interpreter) when the backend supports it.
            if hasattr(sb, "exec_python_code"):
                stdout_content, stderr_content, had_error = self._run_via_repl(
                    sb, executable_code, timeout
                )
                return_code = 1 if had_error else 0
            else:
                # Modal / fallback: independent exec per call with preamble.
                complete_code = f"{_REPL_INIT_CODE}\n{executable_code}"
                process_result = self._execute_python_code(sb, complete_code, timeout)
                stderr_content = process_result.stderr
                stdout_content = process_result.stdout
                return_code = process_result.return_code

            if return_code not in (None, 0):
                combined_error = (
                    ((stderr_content or "").strip() + "\n" + (stdout_content or "").strip()).strip()
                )
                policy = self.runtime.config.execution_policy
                if (
                    allow_transient_retry
                    and policy.reinitialize_on_transient_failure
                    and self.runtime.is_transient_exec_failure(combined_error)
                    and transient_retry_count < policy.transient_retry_attempts
                ):
                    next_retry = transient_retry_count + 1
                    self._log(
                        "Transient python execution failure; "
                        f"reinitializing sandbox and retrying ({next_retry}/{policy.transient_retry_attempts})."
                    )
                    self._repl_initialized = False
                    self.runtime.reinitialize()
                    return self.run(
                        code_string=code_string,
                        timeout=timeout,
                        enable_hotfix=enable_hotfix,
                        allow_transient_retry=True,
                        run_type=f"{run_type}_retry{next_retry}",
                        transient_retry_count=next_retry,
                    )
                stderr_content = (
                    (stderr_content.strip() if stderr_content else "")
                    or (stdout_content.strip() if stdout_content else "")
                    or f"Execution failed with return code {return_code}"
                )

            result = self._build_result(
                code_string=code_string,
                stdout_content=stdout_content,
                stderr_content=stderr_content,
                return_code=return_code,
                timed_out=False,
            )
            if enable_hotfix and result.has_error:
                fixed_code = self.hotfix_service.generate_fix(code_string, result.stderr or "")
                if fixed_code and fixed_code != code_string:
                    fixed_result = self.run(
                        code_string=fixed_code,
                        timeout=timeout,
                        enable_hotfix=False,
                        allow_transient_retry=False,
                        run_type=f"{run_type}_hotfix",
                    )
                    if not fixed_result.has_error:
                        return fixed_result
            return result
        except Exception as exc:
            return self._error_result(
                code_string=code_string,
                error_message=f"Execution error: {exc}",
                return_code=None,
            )

    def _run_via_repl(self, sb, code: str, timeout: float):
        """Send code to the persistent REPL, running the init preamble once per session."""
        if not self._repl_initialized:
            sb.exec_python_code(_REPL_INIT_CODE, timeout=30.0)
            self._repl_initialized = True
        return sb.exec_python_code(code, timeout=timeout)

    def _execute_python_code(self, sb, complete_code: str, timeout: float):
        last_error = None
        for interpreter in ("python", "python3"):
            try:
                return self.process_runner.run(sb, [interpreter, "-u", "-c", complete_code], timeout)
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                if interpreter == "python" and ("no such file or directory" in error_text or "not found" in error_text):
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("No python interpreter available in sandbox")

    def _error_result(self, code_string: str, error_message: str, return_code: Optional[int]) -> ExecutionResult:
        stderr_section = f"\n\nstderr:\n\n```bash\n{error_message}\n```"
        output_message = (
            f"ran code:\n\n```python\n{code_string}\n```\n\noutput:\n\n```bash\nNone\n```{stderr_section}"
        )
        status_message = f"```python\n{code_string}\n```\n\noutput:\n\n```bash\nNone\n```{stderr_section}"
        return ExecutionResult(
            output_message=output_message,
            status_message=status_message,
            stdout=None,
            stderr=error_message,
            return_code=return_code,
            timed_out=False,
        )

    def _build_result(
        self,
        code_string: str,
        stdout_content: Optional[str],
        stderr_content: Optional[str],
        return_code: Optional[int],
        timed_out: bool,
    ) -> ExecutionResult:
        has_error = bool((stderr_content or "").strip())
        stderr_section = f"\n\nstderr:\n\n```bash\n{stderr_content}\n```" if has_error else ""
        output_message = (
            f"ran code:\n\n```python\n{code_string}\n```\n\noutput:\n\n```bash\n{stdout_content}\n```{stderr_section}"
        )
        status_message = f"```python\n{code_string}\n```\n\noutput:\n\n```bash\n{stdout_content}\n```{stderr_section}"
        return ExecutionResult(
            output_message=output_message,
            status_message=status_message,
            stdout=stdout_content,
            stderr=stderr_content if has_error else None,
            return_code=return_code,
            timed_out=timed_out,
        )

    def _select_python_interpreter(self, sb) -> Optional[str]:
        for interpreter in ("python", "python3"):
            try:
                proc = sb.exec(interpreter, "-c", "import sys; print(sys.executable)")
                proc.wait()
                if getattr(proc, "returncode", 1) == 0:
                    return interpreter
            except Exception:
                continue
        return None

    def _auto_install_local_runtime_modules_once(self, sb) -> None:
        if self._auto_install_missing_modules_attempted:
            return
        self._auto_install_missing_modules_attempted = True
        if not self.auto_install_missing_modules:
            return
        if not isinstance(self.runtime.sandbox_manager, LocalSandboxManager):
            return

        interpreter = self._select_python_interpreter(sb)
        if not interpreter:
            self._log("auto_install_missing_modules is enabled but no python interpreter was found.")
            return

        install_script = f"""
import importlib.util
import subprocess
import sys
required = {list(LOCAL_RUNTIME_REQUIRED_MODULES)!r}
missing = [name for name in required if importlib.util.find_spec(name) is None]
if not missing:
    print("FARADAY_AUTO_INSTALL:already_satisfied")
else:
    print("FARADAY_AUTO_INSTALL:installing:" + ",".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing])
    print("FARADAY_AUTO_INSTALL:done")
"""
        proc = sb.exec(interpreter, "-u", "-c", install_script)
        start_wait = time.time()
        while proc.poll() is None:
            if time.time() - start_wait > self.auto_install_missing_modules_timeout_sec:
                try:
                    proc.terminate()
                except Exception:
                    pass
                self._log(
                    f"Auto-install missing module timeout after {self.auto_install_missing_modules_timeout_sec}s."
                )
                return
            time.sleep(0.2)
