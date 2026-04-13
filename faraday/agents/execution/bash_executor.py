from __future__ import annotations

from faraday.agents.execution.models import ExecutionResult


class BashExecutor:
    def __init__(self, runtime, process_runner, log_fn):
        self.runtime = runtime
        self.process_runner = process_runner
        self._log = log_fn

    def run(
        self,
        bash_command: str,
        timeout: float = 60.0,
        allow_transient_retry: bool = True,
        transient_retry_count: int = 0,
    ) -> ExecutionResult:
        if not self.runtime.config.enable_sandbox:
            return self._error_result("Bash command execution is disabled. Sandbox was not initialized.")
        if not bash_command.strip():
            return self._error_result("Empty bash command provided.")
        is_valid, validation_message = self._validate_command(bash_command)
        if not is_valid:
            return self._error_result(validation_message or "Invalid bash command.")
        if not self.runtime.ensure_ready():
            return self._error_result("Sandbox is not ready for bash command execution.")
        try:
            sb = self.runtime.get_sandbox()
            cloud_storage_available = self.runtime.ensure_directories(sb)
            translated_command = self.runtime.normalize_agent_outputs_path(
                bash_command,
                cloud_storage_available=cloud_storage_available,
            )
            process_result = self.process_runner.run(sb, ["bash", "-c", translated_command], timeout)
            status_message = self._status_from_process(translated_command, process_result.return_code, process_result.stdout, process_result.stderr)
            stderr_content = process_result.stderr
            if process_result.return_code not in (None, 0) and not stderr_content:
                stderr_content = f"Command failed with return code {process_result.return_code}"
            return ExecutionResult(
                output_message=status_message,
                status_message=status_message,
                stdout=process_result.stdout,
                stderr=stderr_content,
                return_code=process_result.return_code,
                timed_out=process_result.timed_out,
            )
        except Exception as exc:
            policy = self.runtime.config.execution_policy
            if (
                allow_transient_retry
                and policy.reinitialize_on_transient_failure
                and self.runtime.is_transient_exec_failure(str(exc))
                and transient_retry_count < policy.transient_retry_attempts
            ):
                next_retry = transient_retry_count + 1
                self._log(
                    "Transient bash failure; "
                    f"reinitializing sandbox and retrying ({next_retry}/{policy.transient_retry_attempts})."
                )
                self.runtime.reinitialize()
                return self.run(
                    bash_command=bash_command,
                    timeout=timeout,
                    allow_transient_retry=True,
                    transient_retry_count=next_retry,
                )
            return self._error_result(f"Error executing bash command: {exc}")

    def _status_from_process(self, command: str, return_code, stdout_content, stderr_content) -> str:
        if return_code == 0:
            status = f"Bash command executed successfully: {command}"
            if stdout_content and stdout_content.strip():
                status += f"\nOutput: {len(stdout_content.splitlines())} lines"
            else:
                status += "\nNo output produced"
            return status
        status = f"Bash command failed with return code {return_code}: {command}"
        if stderr_content:
            preview = stderr_content[:200] if len(stderr_content) > 200 else stderr_content
            status += f"\nError: {preview}"
        return status

    def _error_result(self, message: str) -> ExecutionResult:
        return ExecutionResult(
            output_message=message,
            status_message=message,
            stdout=None,
            stderr=message,
            return_code=None,
            timed_out=False,
        )

    def _validate_command(self, bash_command: str):
        dangerous_patterns = ["&&", ";", "`"]
        if self.runtime.config.execution_backend_name == "modal":
            dangerous_patterns.extend(["||", "$("])
        for pattern in dangerous_patterns:
            if pattern in bash_command:
                blocked = ", ".join(dangerous_patterns)
                return False, (
                    "Command contains potentially dangerous characters "
                    f"({blocked}). Use safer shell syntax."
                )
        return True, None
