from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any


class RuntimeOutputSync:
    def __init__(self, runtime, log_fn):
        self.runtime = runtime
        self._log = log_fn

    def sync_to_workspace(self) -> int:
        if not self.runtime.config.enable_sandbox:
            return 0

        try:
            sb = self.runtime.get_sandbox()
        except Exception:
            return 0

        # Docker and local managers expose agent_outputs directly on the workspace
        # filesystem, so there is nothing to copy back.
        if hasattr(sb, "get_outputs_root"):
            return 0

        # Modal: outputs live inside the cloud sandbox; copy them out.
        workspace_outputs_dir = self._resolve_workspace_root() / "agent_outputs"
        copied_files = 0
        for runtime_dir in ("/cloud-storage/agent_outputs", "/tmp/agent_outputs"):
            try:
                copied_files += self._copy_sandbox_directory(
                    sb=sb,
                    sandbox_dir=runtime_dir,
                    destination_root=workspace_outputs_dir,
                )
            except Exception as exc:
                self._log(f"Failed syncing runtime outputs from {runtime_dir}: {exc}")
        if copied_files > 0:
            self._log(f"Synchronized {copied_files} runtime output file(s) to {workspace_outputs_dir}")
        return copied_files

    def _resolve_workspace_root(self) -> Path:
        sandbox_manager = getattr(self.runtime, "sandbox_manager", None)
        manager_workspace = getattr(sandbox_manager, "workspace_root", None)
        if manager_workspace:
            return Path(manager_workspace).expanduser().resolve()
        configured_workspace = getattr(self.runtime.config, "workspace_source_root", None)
        if configured_workspace:
            return Path(configured_workspace).expanduser().resolve()
        return Path.cwd().resolve()

    def _copy_sandbox_directory(
        self,
        sb: Any,
        sandbox_dir: str,
        destination_root: Path,
    ) -> int:
        interpreter = self._select_python_interpreter(sb) or "python"
        tar_stream_script = (
            "import os, sys, tarfile\n"
            f"path = {sandbox_dir!r}\n"
            "if os.path.isdir(path):\n"
            "    with tarfile.open(fileobj=sys.stdout.buffer, mode='w') as archive:\n"
            "        archive.add(path, arcname='.')\n"
        )
        process = sb.exec(interpreter, "-u", "-c", tar_stream_script)
        process.wait()
        stderr_text = ""
        if getattr(process, "stderr", None):
            raw_err = process.stderr.read()
            if raw_err:
                stderr_text = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, bytes) else str(raw_err)
        if getattr(process, "returncode", 1) != 0:
            self._log(f"Skipping sandbox output sync from {sandbox_dir}: {stderr_text.strip()}")
            return 0
        stdout_raw = b""
        if getattr(process, "stdout", None):
            stdout_raw = process.stdout.read() or b""
        if not stdout_raw:
            return 0
        if not isinstance(stdout_raw, bytes):
            stdout_raw = str(stdout_raw).encode("utf-8", errors="replace")
        return self._extract_tar_bytes(stdout_raw, destination_root)

    def _extract_tar_bytes(self, tar_bytes: bytes, destination_root: Path) -> int:
        extracted_files = 0
        destination_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                target = destination_root / member_path
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as out_f:
                    out_f.write(extracted.read())
                extracted_files += 1
        return extracted_files

    def _select_python_interpreter(self, sb):
        for interpreter in ("python", "python3"):
            try:
                proc = sb.exec(interpreter, "-c", "import sys; print(sys.executable)")
                proc.wait()
                if getattr(proc, "returncode", 1) == 0:
                    return interpreter
            except Exception:
                continue
        return None
