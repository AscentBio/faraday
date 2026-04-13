from __future__ import annotations

from datetime import datetime
import asyncio
import os
from pathlib import Path
import shutil
import uuid
from typing import Dict, Optional

from faraday.config import (
    _first_set,
    get_bool_value,
    get_config_value,
    get_execution_backend,
    get_runtime_app,
    get_workspace_mount_path,
    get_workspace_source_root,
    get_path_value,
    get_string_value,
    normalize_execution_backend,
    normalize_runtime_app,
)
from faraday.agents.execution import (
    BashExecutor,
    CODE_TOOLS,
    CODE_TOOLS_DICT,
    ExecutionPolicy,
    FilesystemStateTracker,
    HotfixService,
    ProcessRunner,
    PythonExecutor,
    RuntimeConfig,
    RuntimeOutputSync,
    SandboxRuntime,
)


def tprint(text: str):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\t{text}')


class BaseCodeExecutionAgent:
    def __init__(
        self,
        user_id: str = None,
        chat_id: str = None,
        query_id: str = None,
        verbose: bool = True,
        enable_sandbox: bool = True,
        lazy_sandbox: bool = True,
        enable_auto_hotfix: bool = False,
        mode: str = "dev",
        project_label: str = "",
        save_conversation: bool = True,
        app_runtime: Optional[str] = None,
        execution_backend: Optional[str] = None,
        workspace_source_root: Optional[str] = None,
        workspace_mount_path: Optional[str] = None,
        cloud_storage_root: Optional[str] = None,
    ):
        self.user_id = user_id
        self.chat_id = chat_id
        self.query_id = query_id or str(uuid.uuid4())[:8]
        self.verbose = verbose
        self.enable_auto_hotfix = enable_auto_hotfix
        self.lazy_sandbox = lazy_sandbox
        self.save_conversation = save_conversation
        self.project_label = project_label or ""
        self.mode = self._normalize_mode(mode)

        os.environ["ENV_SETTING_TYPE"] = self.mode
        if app_runtime is not None and str(app_runtime).strip():
            self.app_runtime = normalize_runtime_app(str(app_runtime))
        else:
            self.app_runtime = get_runtime_app(default="host")
        if execution_backend is not None and str(execution_backend).strip():
            self.execution_backend_name = normalize_execution_backend(str(execution_backend))
        else:
            self.execution_backend_name = get_execution_backend(default="docker")
        if get_config_value("execution", "docker_for_local", default=None) is not None:
            self._log(
                "Deprecated config key execution.docker_for_local is ignored; "
                "use app.mode + sandbox.backend to select runtime behavior."
            )
        self.enable_sandbox = enable_sandbox and self.execution_backend_name != "disabled"
        self.allow_backend_fallback = bool(
            _first_set(
                ("sandbox", "allow_backend_fallback"),
                ("execution", "allow_backend_fallback"),
                default=True,
            )
        )
        self.fallback_order = self._parse_fallback_order(
            _first_set(
                ("sandbox", "fallback_order"),
                ("execution", "fallback_order"),
            ),
        )

        self.workspace_root = workspace_source_root or get_workspace_source_root(default=None)
        self.workspace_mount_path = workspace_mount_path or get_workspace_mount_path(default="/workspace")
        self.workspace_init_mode = (
            str(_first_set(
                ("app", "workspace", "init_mode"),
                ("runtime", "workspace", "init_mode"),
                default="bind",
            ) or "bind")
        ).strip().lower()
        self.workspace_copy_root = get_path_value("app", "workspace", "copy_root", default=None) or get_path_value("runtime", "workspace", "copy_root", default=None)
        self.workspace_keep_copy = bool(
            _first_set(
                ("app", "workspace", "keep_copy"),
                ("runtime", "workspace", "keep_copy"),
                default=False,
            )
        )
        self.workspace_copy_path: Optional[Path] = None
        self.cloud_storage_root = cloud_storage_root or get_path_value("execution", "cloud_storage_root", default=None)
        self.cloud_storage_mode = str(
            _first_set(
                ("sandbox", "modal", "cloud_storage_mode"),
                ("execution", "modal_cloud_storage_mode"),
                default="disabled",
            ) or "disabled"
        )
        enable_cloud_mount = get_bool_value("execution", "enable_cloud_bucket_mount", default=None)
        if enable_cloud_mount is True and self.cloud_storage_mode == "disabled":
            self.cloud_storage_mode = "optional"

        self.modal_bucket_name = str(
            _first_set(
                ("sandbox", "modal", "bucket_name"),
                ("execution", "bucket_name"),
            ) or ""
        ) or None
        self.docker_image = str(
            _first_set(
                ("sandbox", "docker_image"),
                ("execution", "docker_image"),
            ) or ""
        ).strip() or None
        self.docker_dockerfile = str(
            _first_set(
                ("sandbox", "dockerfile"),
                ("execution", "dockerfile"),
            ) or ""
        ).strip() or None
        self.allow_host_runtime = bool(
            _first_set(
                ("sandbox", "allow_host_runtime"),
                ("execution", "allow_host_runtime"),
                default=False,
            )
        )
        self._apply_workspace_initialization_mode()

        self.runtime_import_check_enabled = get_bool_value(
            "execution",
            "runtime_import_check",
            default=(
                self.execution_backend_name == "host"
                or self.execution_backend_name == "docker"
            ),
        )
        self.runtime_import_check_warn_optional = get_bool_value(
            "execution",
            "runtime_import_check_warn_optional",
            default=False,
        )
        self.auto_install_missing_modules = get_bool_value("execution", "auto_install_missing_modules", default=False)
        timeout_cfg = get_config_value("execution", "auto_install_missing_modules_timeout_sec", default=300)
        try:
            self.auto_install_missing_modules_timeout_sec = max(30, int(timeout_cfg))
        except Exception:
            self.auto_install_missing_modules_timeout_sec = 300
        self.execution_policy = self._build_execution_policy()

        runtime_config = RuntimeConfig(
            user_id=self.user_id,
            chat_id=self.chat_id,
            verbose=self.verbose,
            app_runtime=self.app_runtime,
            execution_backend_name=self.execution_backend_name,
            enable_sandbox=self.enable_sandbox,
            lazy_sandbox=self.lazy_sandbox,
            workspace_source_root=self.workspace_root,
            workspace_mount_path=self.workspace_mount_path,
            cloud_storage_root=self.cloud_storage_root,
            cloud_storage_mode=self.cloud_storage_mode,
            modal_bucket_name=self.modal_bucket_name,
            docker_image=self.docker_image,
            docker_dockerfile=self.docker_dockerfile,
            allow_host_runtime=self.allow_host_runtime,
            allow_backend_fallback=self.allow_backend_fallback,
            fallback_order=self.fallback_order,
            execution_policy=self.execution_policy,
        )
        self.runtime = SandboxRuntime(runtime_config, self._log)
        self.sandbox_manager = self.runtime.sandbox_manager
        self.sb_index = self.runtime.sb_index

        self.process_runner = ProcessRunner(
            self._log,
            stdout_max_bytes=self.execution_policy.stdout_max_bytes,
            stderr_max_bytes=self.execution_policy.stderr_max_bytes,
        )
        self.hotfix_service = HotfixService(llm_client=self._get_optional_llm_client(), log_fn=self._log)
        self.python_executor = PythonExecutor(
            runtime=self.runtime,
            process_runner=self.process_runner,
            hotfix_service=self.hotfix_service,
            log_fn=self._log,
            auto_install_missing_modules=self.auto_install_missing_modules,
            auto_install_missing_modules_timeout_sec=self.auto_install_missing_modules_timeout_sec,
            runtime_import_check_enabled=self.runtime_import_check_enabled,
            runtime_import_check_warn_optional=self.runtime_import_check_warn_optional,
        )
        self.bash_executor = BashExecutor(self.runtime, self.process_runner, self._log)
        self.filesystem_state = FilesystemStateTracker(self.runtime, self._log)
        self.output_sync = RuntimeOutputSync(self.runtime, self._log)

        self.first_storage_check = True
        self.pre_file_storage_state_root = []
        self.pre_file_storage_state_data = []
        self.pre_file_storage_state_plots = []
        self.pre_file_storage_state_webpages = []
        self.usage_tracker = None
        if self.user_id and self.chat_id:
            self.bucket_path = f"{self.user_id}/{self.chat_id}/"
            self.bucket_path_messages = f"{self.user_id}/{self.chat_id}/{self.query_id}/"
        elif self.chat_id:
            self.bucket_path = f"{self.chat_id}/"
            self.bucket_path_messages = f"{self.chat_id}/{self.query_id}/"
        else:
            self.bucket_path = ""
            self.bucket_path_messages = ""

        self.python_executor.report_runtime_import_health_once()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = (mode or "dev").strip().lower()
        if normalized in {"production", "prod"}:
            return "prod"
        if normalized in {"development", "dev"}:
            return "dev"
        return "dev"

    @staticmethod
    def _parse_fallback_order(raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            parsed = [item.strip().lower().replace("_", "-") for item in raw_value.split(",")]
            parsed = [item for item in parsed if item]
        elif isinstance(raw_value, list):
            parsed = [str(item).strip().lower().replace("_", "-") for item in raw_value]
            parsed = [item for item in parsed if item]
        else:
            return None
        normalized: list[str] = []
        for item in parsed:
            mapped = normalize_execution_backend(item)
            if mapped in {"docker", "modal", "host"} and mapped not in normalized:
                normalized.append(mapped)
        return normalized or None

    def get_runtime_info(self) -> dict[str, str]:
        workspace = self.workspace_root or os.getcwd()
        mount_path = self.workspace_mount_path or "/workspace"
        image = self.docker_image or "faraday-code-sandbox"
        return {
            "app": self.app_runtime,
            "backend": self.execution_backend_name,
            "image": image,
            "workspace": workspace,
            "mount": mount_path,
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            tprint(message)

    @staticmethod
    def _coerce_float(value, default: float, minimum: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return default
        return max(minimum, parsed)

    @staticmethod
    def _coerce_int(value, default: int, minimum: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return default
        return max(minimum, parsed)

    @staticmethod
    def _coerce_optional_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _coerce_optional_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    def _build_execution_policy(self) -> ExecutionPolicy:
        raw_policy = get_config_value("execution", "policy", default={})
        policy_cfg = raw_policy if isinstance(raw_policy, dict) else {}

        sandbox_ready_timeout_sec = self._coerce_float(
            policy_cfg.get(
                "sandbox_ready_timeout_sec",
                get_config_value("execution", "sandbox_ready_timeout_sec", default=120.0),
            ),
            default=120.0,
            minimum=1.0,
        )
        python_exec_timeout_sec = self._coerce_float(
            policy_cfg.get(
                "python_exec_timeout_sec",
                get_config_value("execution", "python_exec_timeout_sec", default=300.0),
            ),
            default=300.0,
            minimum=1.0,
        )
        bash_exec_timeout_sec = self._coerce_float(
            policy_cfg.get(
                "bash_exec_timeout_sec",
                get_config_value("execution", "bash_exec_timeout_sec", default=60.0),
            ),
            default=60.0,
            minimum=1.0,
        )
        transient_retry_attempts = self._coerce_int(
            policy_cfg.get(
                "transient_retry_attempts",
                get_config_value("execution", "transient_retry_attempts", default=1),
            ),
            default=1,
            minimum=0,
        )
        reinitialize_on_transient_failure = bool(
            policy_cfg.get(
                "reinitialize_on_transient_failure",
                get_bool_value("execution", "reinitialize_on_transient_failure", default=True),
            )
        )
        docker_memory = policy_cfg.get(
            "docker_memory",
            get_string_value("execution", "docker_memory", default=None),
        )
        docker_cpus = self._coerce_optional_float(
            policy_cfg.get(
                "docker_cpus",
                get_config_value("execution", "docker_cpus", default=None),
            )
        )
        docker_pids_limit = self._coerce_optional_int(
            policy_cfg.get(
                "docker_pids_limit",
                get_config_value("execution", "docker_pids_limit", default=None),
            )
        )
        docker_shm_size = policy_cfg.get(
            "docker_shm_size",
            get_string_value("execution", "docker_shm_size", default=None),
        )
        docker_network = policy_cfg.get(
            "docker_network",
            get_string_value("execution", "docker_network", default=None),
        )
        max_reinitialize_attempts = self._coerce_int(
            policy_cfg.get(
                "max_reinitialize_attempts",
                get_config_value("execution", "max_reinitialize_attempts", default=3),
            ),
            default=3,
            minimum=0,
        )
        heartbeat_interval_sec = self._coerce_float(
            policy_cfg.get(
                "heartbeat_interval_sec",
                get_config_value("execution", "heartbeat_interval_sec", default=30.0),
            ),
            default=30.0,
            minimum=0.0,
        )
        stdout_max_bytes = self._coerce_int(
            policy_cfg.get(
                "stdout_max_bytes",
                get_config_value("execution", "stdout_max_bytes", default=256 * 1024),
            ),
            default=256 * 1024,
            minimum=1024,
        )
        stderr_max_bytes = self._coerce_int(
            policy_cfg.get(
                "stderr_max_bytes",
                get_config_value("execution", "stderr_max_bytes", default=32 * 1024),
            ),
            default=32 * 1024,
            minimum=1024,
        )
        return ExecutionPolicy(
            sandbox_ready_timeout_sec=sandbox_ready_timeout_sec,
            python_exec_timeout_sec=python_exec_timeout_sec,
            bash_exec_timeout_sec=bash_exec_timeout_sec,
            transient_retry_attempts=transient_retry_attempts,
            reinitialize_on_transient_failure=reinitialize_on_transient_failure,
            max_reinitialize_attempts=max_reinitialize_attempts,
            heartbeat_interval_sec=heartbeat_interval_sec,
            stdout_max_bytes=stdout_max_bytes,
            stderr_max_bytes=stderr_max_bytes,
            docker_memory=docker_memory,
            docker_cpus=docker_cpus,
            docker_pids_limit=docker_pids_limit,
            docker_shm_size=docker_shm_size,
            docker_network=docker_network,
        )

    def _apply_workspace_initialization_mode(self) -> None:
        mode = self.workspace_init_mode
        if mode not in {"bind", "copy"}:
            self._log(
                f"Unknown runtime.workspace.init_mode={mode!r}; falling back to 'bind'."
            )
            mode = "bind"
            self.workspace_init_mode = mode
        if mode != "copy":
            return

        source_root = Path(self.workspace_root or Path.cwd()).expanduser().resolve()
        if not source_root.exists() or not source_root.is_dir():
            raise RuntimeError(
                f"Workspace source root does not exist or is not a directory: {source_root}"
            )

        copy_root = Path(self.workspace_copy_root or "./.faraday_runtime/workspace-copies").expanduser()
        if not copy_root.is_absolute():
            copy_root = (Path.cwd() / copy_root).resolve()
        else:
            copy_root = copy_root.resolve()
        copy_root.mkdir(parents=True, exist_ok=True)

        copy_name = f"ws_{self.chat_id or 'chat'}_{self.query_id}_{uuid.uuid4().hex[:8]}"
        target_root = copy_root / copy_name
        ignore_fn = self._workspace_copy_ignore(source_root=source_root, copy_root=copy_root)
        shutil.copytree(source_root, target_root, ignore=ignore_fn)
        self.workspace_copy_path = target_root.resolve()
        self.workspace_root = str(self.workspace_copy_path)
        self._log(
            f"Workspace init_mode=copy: using isolated workspace at {self.workspace_copy_path}"
        )

    @staticmethod
    def _workspace_copy_ignore(source_root: Path, copy_root: Path):
        """Avoid recursive copy when copy_root lives under source_root."""
        try:
            rel_copy_root = copy_root.resolve().relative_to(source_root.resolve())
        except Exception:
            return None
        rel_parts = rel_copy_root.parts
        if not rel_parts:
            return None

        def _ignore(path: str, names: list[str]) -> set[str]:
            try:
                current_rel = Path(path).resolve().relative_to(source_root.resolve())
            except Exception:
                return set()
            depth = len(current_rel.parts)
            if depth >= len(rel_parts):
                return set()
            candidate = rel_parts[depth]
            return {candidate} if candidate in names else set()

        return _ignore

    def _get_optional_llm_client(self):
        try:
            from faraday.openai_clients import llm_client

            return llm_client
        except Exception as exc:
            self._log(f"Hotfix client unavailable; auto-hotfix disabled ({exc}).")
            return None

    def init_sandbox(self):
        self.runtime.start()
        self.sb_index = self.runtime.sb_index

    async def init_sandbox_async(self):
        await self.runtime.start_async()
        self.sb_index = self.runtime.sb_index

    def stop_sandbox(self):
        self.runtime.stop()
        self.sb_index = None

    async def stop_sandbox_async(self):
        await self.runtime.stop_async()
        self.sb_index = None

    def wait_for_sandbox_ready(self, max_wait_seconds: Optional[float] = None, poll_interval=0.5):
        del poll_interval
        return self.runtime.ensure_ready(timeout=max_wait_seconds)

    async def wait_for_sandbox_ready_async(
        self,
        max_wait_seconds: Optional[float] = None,
        poll_interval=0.5,
    ):
        del poll_interval
        return await self.runtime.ensure_ready_async(timeout=max_wait_seconds)

    def ensure_sandbox_ready(self, max_wait_seconds: Optional[float] = None):
        return self.runtime.ensure_ready(timeout=max_wait_seconds)

    async def ensure_sandbox_ready_async(self, max_wait_seconds: Optional[float] = None):
        return await self.runtime.ensure_ready_async(timeout=max_wait_seconds)

    def clear_saved_variables(self):
        """Reset code execution state for this session.

        For REPL-backed backends (Docker/local), clears the interpreter namespace.
        For Modal, deletes the pickle file used for cross-execution variable persistence.
        """
        if not self.enable_sandbox:
            return False
        try:
            sb = self.runtime.get_sandbox()
            if hasattr(sb, "get_repl"):
                repl = sb.get_repl()
                repl.clear_namespace()
                # Reset the executor's init flag so preamble re-runs on next call.
                self.python_executor._repl_initialized = False
            else:
                # Modal: delete the pickle file.
                proc = sb.exec("rm", "-f", "/cloud-storage/.modal_code_vars/variables.pkl")
                proc.wait()
            return True
        except Exception as exc:
            self._log(f"Error clearing saved variables: {exc}")
            return False

    def run_code(
        self,
        code_string: str,
        type_of_run: str = "normal",
        timeout: Optional[float] = None,
        enable_hotfix: bool = False,
        _allow_transient_retry: bool = True,
    ):
        effective_timeout = (
            timeout
            if timeout is not None
            else self.execution_policy.python_exec_timeout_sec
        )
        result = self.python_executor.run(
            code_string=code_string,
            timeout=effective_timeout,
            enable_hotfix=enable_hotfix,
            allow_transient_retry=_allow_transient_retry,
            run_type=type_of_run,
        )
        return result.output_message, result.status_message, result.stdout, result.stderr

    def bash_command(
        self,
        bash_command: str,
        timeout: Optional[float] = None,
        _allow_transient_retry: bool = True,
    ):
        effective_timeout = (
            timeout
            if timeout is not None
            else self.execution_policy.bash_exec_timeout_sec
        )
        result = self.bash_executor.run(
            bash_command=bash_command,
            timeout=effective_timeout,
            allow_transient_retry=_allow_transient_retry,
        )
        return result.status_message, result.stdout, result.stderr

    async def _execute_single_code_block_python(self, code_block: Dict):
        code_content = code_block["content"]
        output_message, status_message, stdout_content, stderr_content = await asyncio.to_thread(
            self.run_code, code_content, "normal", None, self.enable_auto_hotfix
        )
        has_error = bool((stderr_content or "").strip())
        if has_error:
            markdown_output = f"**Error:**\n```\n{stderr_content.strip()}\n```"
        elif stdout_content and str(stdout_content).strip():
            markdown_output = f"**Output:**\n```\n{str(stdout_content).strip()}\n```"
        else:
            markdown_output = "✓ Code executed"
        return {
            "markdown_output": markdown_output,
            "output_message": output_message,
            "status_message": status_message,
            "stdout": stdout_content,
            "stderr": stderr_content,
            "has_error": has_error,
        }

    async def _execute_single_code_block_bash(self, code_block: Dict):
        code_content = code_block["content"]
        status_message, stdout_content, stderr_content = await asyncio.to_thread(self.bash_command, code_content)
        has_error = bool((stderr_content or "").strip())
        if has_error:
            markdown_output = f"**Error:**\n```\n{stderr_content.strip()}\n```"
        elif stdout_content and str(stdout_content).strip():
            markdown_output = f"**Output:**\n```\n{str(stdout_content).strip()}\n```"
        else:
            markdown_output = "✓ Command completed"
        return {
            "markdown_output": markdown_output,
            "status_message": status_message,
            "stdout": stdout_content,
            "stderr": stderr_content,
            "has_error": has_error,
        }

    def check_file_storage_root(self):
        return self.filesystem_state.check_root()

    def check_file_storage_data(self):
        return self.filesystem_state.check_data()

    def check_file_storage_plots(self):
        return self.filesystem_state.check_plots()

    def check_file_storage(self):
        return self.filesystem_state.check_storage_summary()

    def get_filesystem_state_for_llm(self, include_summary=True, include_details=True):
        self._sync_filesystem_attributes_from_tracker()
        return self.filesystem_state.render_for_llm(include_summary=include_summary, include_details=include_details)

    def get_filesystem_state_compact(self):
        self._sync_filesystem_attributes_from_tracker()
        return self.filesystem_state.render_compact()

    async def refresh_filesystem_state(self):
        refreshed = await self.filesystem_state.refresh()
        self._sync_filesystem_attributes_from_tracker()
        return refreshed

    async def cleanup(self):
        try:
            if self.runtime.sb_index:
                await self.stop_sandbox_async()
            if self.workspace_copy_path is not None and not self.workspace_keep_copy:
                shutil.rmtree(self.workspace_copy_path, ignore_errors=True)
                self._log(f"Removed workspace copy: {self.workspace_copy_path}")
            self._log("Code execution cleanup completed")
        except Exception as exc:
            self._log(f"Error during cleanup: {exc}")

    def sync_runtime_agent_outputs_to_workspace(self) -> int:
        return self.output_sync.sync_to_workspace()

    def _sync_filesystem_attributes_from_tracker(self) -> None:
        self.pre_file_storage_state_root = list(self.filesystem_state.root_files)
        self.pre_file_storage_state_data = list(self.filesystem_state.data_files)
        self.pre_file_storage_state_plots = list(self.filesystem_state.plot_files)
        self.pre_file_storage_state_webpages = list(self.filesystem_state.webpage_files)
