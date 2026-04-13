from __future__ import annotations

import time
from typing import Any, Sequence

from faraday.agents.execution.models import ProcessResult

_DEFAULT_STDOUT_MAX_BYTES = 256 * 1024
_DEFAULT_STDERR_MAX_BYTES = 32 * 1024


class ProcessRunner:
    def __init__(
        self,
        log_fn,
        stdout_max_bytes: int = _DEFAULT_STDOUT_MAX_BYTES,
        stderr_max_bytes: int = _DEFAULT_STDERR_MAX_BYTES,
    ):
        self._log = log_fn
        self._stdout_max_bytes = max(1024, stdout_max_bytes)
        self._stderr_max_bytes = max(1024, stderr_max_bytes)

    def run(self, sb: Any, argv: Sequence[str], timeout: float) -> ProcessResult:
        proc = sb.exec(*argv)
        start_wait = time.time()
        while proc.poll() is None:
            if time.time() - start_wait > timeout:
                try:
                    proc.terminate()
                except Exception:
                    pass
                return ProcessResult(
                    return_code=None,
                    stdout=None,
                    stderr=f"Execution timed out after {timeout} seconds",
                    timed_out=True,
                )
            time.sleep(0.1)

        stdout_content = self._read_stream(proc, "stdout", self._stdout_max_bytes)
        stderr_content = self._read_stream(proc, "stderr", self._stderr_max_bytes)
        return ProcessResult(
            return_code=getattr(proc, "returncode", None),
            stdout=stdout_content,
            stderr=stderr_content,
            timed_out=False,
        )

    def _read_stream(self, proc: Any, stream_name: str, max_bytes: int):
        try:
            stream = getattr(proc, stream_name, None)
            if stream:
                raw = stream.read()
                if raw:
                    text = (
                        raw.decode("utf-8", errors="replace")
                        if isinstance(raw, bytes)
                        else str(raw)
                    )
                    if len(text.encode("utf-8")) > max_bytes:
                        truncated = raw[:max_bytes].decode("utf-8", errors="replace")
                        omitted = len(raw) - max_bytes
                        text = (
                            truncated
                            + f"\n[...{omitted:,} bytes truncated]"
                        )
                    return text
        except Exception as exc:
            self._log(f"Error reading {stream_name}: {exc}")
        return None
