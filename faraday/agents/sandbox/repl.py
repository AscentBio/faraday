"""Persistent Python REPL for stateful code execution inside a sandbox.

Each sandbox backend (Docker, local) starts one long-lived Python process per
session and communicates via newline-delimited JSON over stdin/stdout.  This
replaces the old pickle-file approach, eliminating order-dependence, pickle
fragility, and the need for a /cloud-storage/.modal_code_vars directory.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Server script (sent as ``python -u -c <REPL_SERVER_SCRIPT>`` to start the
# REPL process inside the sandbox).  The script must be self-contained and
# compatible with the Python available in the sandbox.
# ---------------------------------------------------------------------------

REPL_SERVER_SCRIPT = r"""import sys, json, traceback, io

_ns = {"__name__": "__main__", "__builtins__": __builtins__}
_in  = sys.stdin.buffer
_out = sys.stdout.buffer

while True:
    try:
        line = _in.readline()
    except Exception:
        break
    if not line:
        break
    try:
        req = json.loads(line)
    except Exception as exc:
        _out.write((json.dumps({"stdout": "", "stderr": f"REPL bad request: {exc}", "error": True}) + "\n").encode())
        _out.flush()
        continue
    code = req.get("code", "")
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    sys_stdout, sys_stderr = sys.stdout, sys.stderr
    sys.stdout = buf_out
    sys.stderr = buf_err
    had_error = False
    try:
        exec(compile(code, "<cell>", "exec"), _ns)
    except SystemExit:
        pass
    except BaseException:
        buf_err.write(traceback.format_exc())
        had_error = True
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr
    _out.write(
        (json.dumps({"stdout": buf_out.getvalue(), "stderr": buf_err.getvalue(), "error": had_error}) + "\n").encode()
    )
    _out.flush()
"""


# ---------------------------------------------------------------------------
# ReplProcess
# ---------------------------------------------------------------------------

class ReplProcess:
    """Wraps a long-lived Python REPL subprocess (inside or outside a container)."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, code: str, timeout: float = 300.0) -> Tuple[str, str, bool]:
        """Execute *code* in the REPL, returning ``(stdout, stderr, had_error)``."""
        with self._lock:
            return self._execute_locked(code, timeout)

    def clear_namespace(self) -> None:
        """Reset the REPL namespace, discarding all previously defined variables."""
        self.execute(
            "_ns_keys = [k for k in list(globals().keys()) if not k.startswith('_') and k not in ('__builtins__',)]\n"
            "for _k in _ns_keys:\n"
            "    globals().pop(_k, None)\n"
        )

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def terminate(self) -> None:
        if not self.is_alive():
            return
        try:
            self._process.stdin.close()
        except Exception:
            pass
        self._process.terminate()
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_locked(self, code: str, timeout: float) -> Tuple[str, str, bool]:
        if not self.is_alive():
            return "", "REPL process is not running", True

        request = (json.dumps({"code": code}) + "\n").encode()
        try:
            self._process.stdin.write(request)
            self._process.stdin.flush()
        except OSError as exc:
            return "", f"REPL stdin write failed: {exc}", True

        line_holder: list[Optional[bytes]] = [None]
        done = threading.Event()

        def _read() -> None:
            try:
                line_holder[0] = self._process.stdout.readline()
            except Exception as exc:
                line_holder[0] = (
                    json.dumps({"stdout": "", "stderr": f"REPL read error: {exc}", "error": True}) + "\n"
                ).encode()
            finally:
                done.set()

        threading.Thread(target=_read, daemon=True).start()
        if not done.wait(timeout):
            return "", f"REPL execution timed out after {timeout}s", True

        raw = line_holder[0]
        if not raw:
            return "", "REPL: no response", True
        try:
            resp = json.loads(raw)
            return resp.get("stdout", ""), resp.get("stderr", ""), bool(resp.get("error", False))
        except json.JSONDecodeError as exc:
            return "", f"REPL: bad response ({exc}): {raw!r}", True
