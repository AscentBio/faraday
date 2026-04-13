from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Optional

try:
    from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
except ImportError as exc:  # pragma: no cover - exercised only outside Harbor
    _HARBOR_IMPORT_ERROR = exc

    class BaseInstalledAgent:  # type: ignore[override]
        pass

    def with_prompt_template(func):  # type: ignore[misc]
        return func

else:
    _HARBOR_IMPORT_ERROR = None


EVENT_PREFIX = "FARADAY_EVENT\t"
RESULT_PREFIX = "FARADAY_RESULT\t"


def _ensure_harbor_available() -> None:
    if _HARBOR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Harbor is not installed. Install Harbor in the target environment before "
            "using faraday.integrations.harbor.agent:FaradayHarborAgent."
        ) from _HARBOR_IMPORT_ERROR


def _emit_context_event(context: Any, event: dict[str, Any]) -> None:
    for method_name in ("add_event", "append_event", "record_event", "log_event"):
        method = getattr(context, method_name, None)
        if callable(method):
            try:
                method(event)
                return
            except Exception:
                pass
    events = getattr(context, "events", None)
    if isinstance(events, list):
        events.append(event)


def _set_context_solution(context: Any, solution: str) -> None:
    for attr_name in ("final_output", "final_answer", "output_text", "result"):
        if hasattr(context, attr_name):
            try:
                setattr(context, attr_name, solution)
                return
            except Exception:
                pass
    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        metadata["faraday_solution"] = solution


async def _write_answer_file(environment: Any, solution: str) -> dict[str, Any]:
    escaped_solution = shlex.quote(solution)
    command = (
        f"printf '%s\\n' {escaped_solution} > answer.txt; "
        f"printf '%s\\n' {escaped_solution} > /workspace/answer.txt 2>/dev/null || true; "
        f"printf '%s\\n' {escaped_solution} > /logs/answer.txt 2>/dev/null || true"
    )
    result = await environment.exec(command=command)
    return {
        "return_code": getattr(result, "return_code", None),
        "stdout": getattr(result, "stdout", "") or "",
        "stderr": getattr(result, "stderr", "") or "",
        "command": command,
    }


class FaradayHarborAgent(BaseInstalledAgent):
    _VENV_DIR = "/tmp/faraday-venv"
    _SRC_DIR = "/tmp/faraday-src"

    def __init__(
        self,
        logs_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        _ensure_harbor_available()
        agent_extra_env = dict(kwargs.pop("extra_env", {}) or {})
        for env_var in ("OPENAI_API_KEY",):
            env_val = os.getenv(env_var)
            if env_val and env_var not in agent_extra_env:
                agent_extra_env[env_var] = env_val
        if agent_extra_env:
            kwargs["extra_env"] = agent_extra_env
        if model_name is not None:
            kwargs.setdefault("model_name", model_name)
        resolved_logs_dir = Path(logs_dir) if logs_dir else Path(".")
        super().__init__(resolved_logs_dir, **kwargs)
        # Harbor's factory passes these kwargs to import-path agents.
        # Keep them for compatibility even though this adapter currently does
        # not need them directly.
        self.logs_dir = resolved_logs_dir
        self.model_name = model_name
        self._init_kwargs = dict(kwargs)
        self._events: list[dict[str, Any]] = []
        self._result: dict[str, Any] = {}

    @staticmethod
    def name() -> str:
        return "faraday"

    def version(self) -> Optional[str]:
        return "0.1.0"

    async def install(self, environment) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        env = dict(getattr(self, "_extra_env", {}) or {})
        env.setdefault("FARADAY_CONFIG", f"{self._SRC_DIR}/faraday.yaml")
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && apt-get install -y "
                "bash build-essential git python3-pip python3-venv"
            ),
        )
        await environment.upload_dir(str(repo_root), self._SRC_DIR)
        await self.exec_as_agent(
            environment,
            command=(
                f"set -euo pipefail; "
                f"python3 -m venv {self._VENV_DIR} && "
                f"{self._VENV_DIR}/bin/python -m pip install --upgrade pip && "
                # Bootstrap config parsing deps before resolving dynamic package set.
                f"{self._VENV_DIR}/bin/python -m pip install pyyaml && "
                f"RUNTIME_DEPS=$({self._VENV_DIR}/bin/python - <<'PY'\n"
                "import shlex\n"
                "import sys\n"
                f"sys.path.insert(0, {self._SRC_DIR!r})\n"
                "from faraday.config import get_runtime_dependency_packages\n"
                "packages = get_runtime_dependency_packages(default_backend='docker')\n"
                "print(' '.join(shlex.quote(pkg) for pkg in packages))\n"
                "PY\n"
                ") && "
                "if [ -n \"$RUNTIME_DEPS\" ]; then "
                f"{self._VENV_DIR}/bin/python -m pip install $RUNTIME_DEPS; "
                "fi && "
                f"{self._VENV_DIR}/bin/python -m pip install --no-deps -e {self._SRC_DIR}"
            ),
            env=env or None,
        )

    @with_prompt_template
    async def run(self, instruction: str, environment, context) -> None:
        self._events = []
        self._result = {}
        command = (
            f"{self._VENV_DIR}/bin/python -m faraday.integrations.harbor.runtime "
            f"--instruction {shlex.quote(instruction)} "
            "--launch-mode host "
            "--execution-backend docker "
            "--workspace-source /workspace "
            "--execution-workspace-path /workspace "
            "--artifacts-dir /logs/artifacts "
            "--trajectory-path /logs/artifacts/trajectory.json"
        )
        env = dict(getattr(self, "_extra_env", {}) or {})
        result = await environment.exec(
            command=f"set -o pipefail; {command}",
            env=env or None,
        )
        stdout = getattr(result, "stdout", "") or ""
        for line in stdout.splitlines():
            if line.startswith(EVENT_PREFIX):
                payload = json.loads(line[len(EVENT_PREFIX):])
                self._events.append(payload)
                _emit_context_event(context, payload)
            elif line.startswith(RESULT_PREFIX):
                self._result = json.loads(line[len(RESULT_PREFIX):])
        if getattr(result, "return_code", 0) != 0 and not self._result:
            stderr = getattr(result, "stderr", "") or ""
            self._result = {
                "solution": "",
                "event_count": len(self._events),
                "error": (
                    "Harbor runtime command failed "
                    f"(exit {getattr(result, 'return_code', 'unknown')}): "
                    f"stdout={stdout[:2000]!r} stderr={stderr[:2000]!r}"
                ),
            }
        solution = str(self._result.get("solution", "") or "").strip()
        if solution:
            _set_context_solution(context, solution)
            write_result = await _write_answer_file(environment, solution)
            metadata = getattr(context, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                setattr(context, "metadata", metadata)
            metadata["faraday_answer_write"] = write_result
            if write_result.get("return_code") not in (0, None):
                metadata["faraday_error"] = (
                    "Failed to write answer.txt after Harbor runtime completion"
                )

    def populate_context_post_run(self, context) -> None:
        for event in self._events:
            _emit_context_event(context, event)
        solution = str(self._result.get("solution", "") or "").strip()
        if solution:
            _set_context_solution(context, solution)
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            metadata["faraday_event_count"] = len(self._events)
            if self._result.get("error"):
                metadata["faraday_error"] = self._result["error"]
            if self._result.get("trajectory_path"):
                metadata["faraday_trajectory_path"] = self._result["trajectory_path"]
            if self._result.get("events_path"):
                metadata["faraday_events_path"] = self._result["events_path"]
            if self._result.get("result_path"):
                metadata["faraday_result_path"] = self._result["result_path"]
            if self._result.get("metadata_path"):
                metadata["faraday_metadata_path"] = self._result["metadata_path"]
