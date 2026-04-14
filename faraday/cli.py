from __future__ import annotations

"""CLI entrypoint for running the Faraday agent."""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from faraday.config import (
    _first_set,
    get_config_value,
    get_bool_value,
    get_execution_backend,
    get_runtime_app,
    get_runtime_launch,
    get_runtime_app_docker_image,
    get_path_value,
    get_run_output_dir,
    get_runtime_config_path,
    get_string_value,
    get_workspace_mount_path,
    get_workspace_source_root,
    normalize_execution_backend,
    normalize_runtime_app,
)
from faraday.openai_clients import get_client_settings, get_llm_model, get_llm_provider


def _default_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_trajectory_payload(trajectory_path: str) -> tuple[dict, Path]:
    path = Path(trajectory_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Trajectory payload must be a JSON object.")
    return payload, path


def _trajectory_to_conversation_history(payload: dict) -> list[dict]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Trajectory payload missing `steps` array.")

    conversation_history: list[dict] = []
    for step in steps:
        if not isinstance(step, dict):
            continue

        source = str(step.get("source", "")).strip().lower()
        message = step.get("message")
        if message is None:
            message = ""
        if not isinstance(message, str):
            message = str(message)

        if source == "user":
            if message:
                conversation_history.append({"role": "user", "content": message})
            continue

        if source == "agent":
            tool_calls = step.get("tool_calls")
            if isinstance(tool_calls, list):
                for i, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    call_id = str(tool_call.get("tool_call_id") or "").strip() or f"trajectory_call_{len(conversation_history)}_{i}"
                    function_name = str(tool_call.get("function_name") or "unknown_tool").strip() or "unknown_tool"
                    arguments = tool_call.get("arguments", {})
                    conversation_history.append(
                        {
                            "type": "function_call",
                            "name": function_name,
                            "arguments": arguments if isinstance(arguments, dict) else {"raw": str(arguments)},
                            "call_id": call_id,
                        }
                    )

            observation = step.get("observation")
            if isinstance(observation, dict):
                results = observation.get("results")
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        if not isinstance(result, dict):
                            continue
                        source_call_id = str(result.get("source_call_id") or "").strip()
                        if not source_call_id:
                            source_call_id = f"trajectory_call_output_{len(conversation_history)}_{i}"
                        output = result.get("content")
                        if output is None:
                            output = ""
                        if not isinstance(output, str):
                            output = str(output)
                        conversation_history.append(
                            {
                                "type": "function_call_output",
                                "call_id": source_call_id,
                                "output": output,
                            }
                        )

            if message:
                conversation_history.append({"role": "assistant", "content": message})
            continue

        if source == "system" and message:
            conversation_history.append({"role": "assistant", "content": message})

    if not conversation_history:
        raise ValueError("Trajectory payload did not yield any usable conversation history.")
    return conversation_history


def _resolve_trajectory_path(agent) -> Path | None:
    resolver = getattr(agent, "_resolve_trajectory_output_path", None)
    if not callable(resolver):
        return None
    try:
        return Path(resolver()).expanduser()
    except Exception:
        return None


def _resolve_artifacts_dir(args: argparse.Namespace, agent) -> Path:
    if args.artifacts_dir:
        return Path(args.artifacts_dir).expanduser().resolve()
    # Use the agent's canonical run_artifacts_dir from the new run output structure
    run_artifacts_dir = getattr(agent, "run_artifacts_dir", None)
    if run_artifacts_dir is not None:
        return run_artifacts_dir.resolve()
    # Fallback for older agent builds
    trajectory_path = _resolve_trajectory_path(agent)
    if trajectory_path is not None:
        return trajectory_path.parent.resolve()
    run_id = f"{(agent.chat_id or 'chat').replace('/', '_')}__{(agent.query_id or 'query').replace('/', '_')}"
    return (Path.cwd() / "run_outputs" / run_id / "run_artifacts").resolve()


def _resolve_collection_dir(args: argparse.Namespace) -> Path | None:
    if args.collect_artifacts_dir:
        return Path(args.collect_artifacts_dir).expanduser().resolve()
    cfg_dir = get_path_value("artifacts", "collect_dir", default=None)
    if cfg_dir:
        return Path(cfg_dir).expanduser().resolve()
    default_harbor_dir = Path(
        get_path_value("artifacts", "harbor_dir", default="/logs/artifacts")
        or "/logs/artifacts"
    )
    collect_to_harbor = bool(get_bool_value("artifacts", "collect_to_harbor", default=False))
    if default_harbor_dir.exists() or collect_to_harbor:
        return default_harbor_dir
    return None


def _collect_artifacts(artifacts: dict[str, Path], destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for artifact_name, source_path in artifacts.items():
        if not source_path.exists():
            manifest.append(
                {
                    "name": artifact_name,
                    "source": str(source_path),
                    "destination": "",
                    "type": "file",
                    "status": "missing",
                }
            )
            continue
        destination = destination_dir / source_path.name
        shutil.copy2(source_path, destination)
        manifest.append(
            {
                "name": artifact_name,
                "source": str(source_path),
                "destination": str(destination),
                "type": "file",
                "status": "ok",
            }
        )
    manifest_path = destination_dir / "manifest.json"
    _write_json(manifest_path, {"artifacts": manifest, "generated_at": _iso_now()})
    return manifest_path


def _coerce_batch_prompt_list(raw: Any, source_label: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{source_label} must be a list of prompt strings.")
    prompts: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"{source_label} contains a non-string prompt.")
        normalized = item.strip()
        if normalized:
            prompts.append(normalized)
    return prompts


def _load_batch_prompts_from_file(batch_file: str) -> list[str]:
    batch_path = Path(batch_file).expanduser().resolve()
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch prompt file not found: {batch_path}")

    suffix = batch_path.suffix.lower()
    raw_text = batch_path.read_text(encoding="utf-8")

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            prompts: list[str] = []
            for line in raw_text.splitlines():
                trimmed = line.strip()
                if not trimmed:
                    continue
                prompts.append(trimmed)
            return prompts

        payload = json.loads(raw_text)
        return _coerce_batch_prompt_list(
            payload,
            source_label=f"batch file {batch_path}",
        )

    prompts = []
    for line in raw_text.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        prompts.append(trimmed)
    return prompts


def _resolve_batch_queries(args: argparse.Namespace) -> list[str]:
    cli_queries = _coerce_batch_prompt_list(
        getattr(args, "batch_query", []) or [],
        source_label="--batch-query",
    )
    file_queries: list[str] = []
    if args.batch_file:
        file_queries = _load_batch_prompts_from_file(args.batch_file)

    if cli_queries or file_queries:
        return cli_queries + file_queries

    enabled_in_config = bool(get_bool_value("batch", "enabled", default=False))
    config_prompts = _coerce_batch_prompt_list(
        get_config_value("batch", "prompts", default=[]),
        source_label="batch.prompts",
    )
    config_batch_file = get_path_value("batch", "prompts_file", default="") or ""
    if config_batch_file:
        config_prompts.extend(_load_batch_prompts_from_file(config_batch_file))

    if enabled_in_config or config_prompts:
        return config_prompts
    return []


def _resolve_batch_output_root(args: argparse.Namespace) -> Path:
    if args.batch_output_root:
        return Path(args.batch_output_root).expanduser().resolve()
    config_root = get_path_value("batch", "output_root", default="") or ""
    if config_root:
        return Path(config_root).expanduser().resolve()
    return Path(get_run_output_dir()).expanduser().resolve()


def _resolve_batch_max_concurrency(args: argparse.Namespace) -> int:
    cli_value = getattr(args, "batch_max_concurrency", None)
    raw_value: Any = cli_value if cli_value is not None else get_config_value(
        "batch",
        "max_concurrency",
        default=1,
    )
    if isinstance(raw_value, bool):
        raise ValueError("batch.max_concurrency must be an integer >= 1.")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch.max_concurrency must be an integer >= 1.") from exc
    if value < 1:
        raise ValueError("batch.max_concurrency must be >= 1.")
    return value


def _resolve_batch_max_retries(args: argparse.Namespace) -> int:
    cli_value = getattr(args, "batch_max_retries", None)
    raw_value: Any = cli_value if cli_value is not None else get_config_value(
        "batch",
        "max_retries",
        default=2,
    )
    if isinstance(raw_value, bool):
        raise ValueError("batch.max_retries must be an integer >= 0.")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch.max_retries must be an integer >= 0.") from exc
    if value < 0:
        raise ValueError("batch.max_retries must be >= 0.")
    return value


def _resolve_batch_retry_delay_settings(args: argparse.Namespace) -> tuple[float, float, float]:
    cli_base = getattr(args, "batch_retry_base_delay_seconds", None)
    cli_max = getattr(args, "batch_retry_max_delay_seconds", None)
    cli_jitter = getattr(args, "batch_retry_jitter_seconds", None)
    raw_base: Any = (
        cli_base
        if cli_base is not None
        else get_config_value("batch", "retry_base_delay_seconds", default=2.0)
    )
    raw_max: Any = (
        cli_max
        if cli_max is not None
        else get_config_value("batch", "retry_max_delay_seconds", default=30.0)
    )
    raw_jitter: Any = (
        cli_jitter
        if cli_jitter is not None
        else get_config_value("batch", "retry_jitter_seconds", default=0.5)
    )
    try:
        base_delay = float(raw_base)
        max_delay = float(raw_max)
        jitter = float(raw_jitter)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "batch retry delay settings must be numeric values."
        ) from exc
    if base_delay <= 0:
        raise ValueError("batch.retry_base_delay_seconds must be > 0.")
    if max_delay < base_delay:
        raise ValueError(
            "batch.retry_max_delay_seconds must be >= batch.retry_base_delay_seconds."
        )
    if jitter < 0:
        raise ValueError("batch.retry_jitter_seconds must be >= 0.")
    return base_delay, max_delay, jitter


def _read_batch_run_error(run_artifacts_dir: Path) -> str:
    result_path = run_artifacts_dir / "result.json"
    if not result_path.exists():
        return ""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    return error if isinstance(error, str) else ""


def _is_retryable_batch_error(error_message: str) -> tuple[bool, bool]:
    if not error_message:
        return False, False
    normalized = error_message.strip().lower()
    if not normalized:
        return False, False

    rate_limit_signals = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "quota exceeded",
    )
    transient_signals = (
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection error",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "overloaded",
    )
    is_rate_limited = any(token in normalized for token in rate_limit_signals)
    is_transient = any(token in normalized for token in transient_signals)
    return is_rate_limited or is_transient, is_rate_limited


async def _run_batch(args: argparse.Namespace, queries: list[str]) -> int:
    batch_id = _default_id("batch")
    batch_root = _resolve_batch_output_root(args) / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)

    continue_on_error = bool(args.batch_continue_on_error) or bool(
        get_bool_value("batch", "continue_on_error", default=False)
    )
    max_concurrency = _resolve_batch_max_concurrency(args)
    max_retries = _resolve_batch_max_retries(args)
    retry_base_delay, retry_max_delay, retry_jitter = _resolve_batch_retry_delay_settings(args)
    semaphore = asyncio.Semaphore(max_concurrency)
    stop_after_failure = asyncio.Event()
    state_lock = asyncio.Lock()
    rate_limit_state: dict[str, float] = {"block_until": 0.0}
    started_at = _iso_now()
    batch_t0 = perf_counter()
    summary_path = batch_root / "batch_summary.json"
    run_summaries_by_index: dict[int, dict[str, Any]] = {}

    _println()
    _println(f"  {_bold('Batch run')} {_dim(batch_id)}")
    _println(f"  {_dim(f'Prompts: {len(queries)}')}")
    _println(f"  {_dim(f'Max concurrency: {max_concurrency}')}")
    _println(f"  {_dim(f'Max retries: {max_retries}')}")
    _println(f"  {_dim(f'Output root: {batch_root}')}")
    _println()

    async def _update_batch_summary(final: bool = False) -> None:
        runs = [run_summaries_by_index[idx] for idx in sorted(run_summaries_by_index)]
        completed_runs = sum(1 for run in runs if run["status"] in {"ok", "error"})
        skipped_runs = sum(1 for run in runs if run["status"] == "skipped")
        failures = sum(1 for run in runs if run["status"] == "error")
        _write_json(
            summary_path,
            {
                "batch_id": batch_id,
                "started_at": started_at,
                "finished_at": _iso_now() if final else None,
                "duration_seconds": round(perf_counter() - batch_t0, 3),
                "prompt_count": len(queries),
                "completed_runs": completed_runs,
                "failed_runs": failures,
                "skipped_runs": skipped_runs,
                "pending_runs": len(queries) - len(runs),
                "continue_on_error": continue_on_error,
                "max_concurrency": max_concurrency,
                "max_retries": max_retries,
                "retry_base_delay_seconds": retry_base_delay,
                "retry_max_delay_seconds": retry_max_delay,
                "retry_jitter_seconds": retry_jitter,
                "runs": runs,
            },
        )

    async def _wait_for_global_rate_limit_window() -> None:
        wait_seconds = 0.0
        async with state_lock:
            now = perf_counter()
            wait_seconds = max(0.0, rate_limit_state["block_until"] - now)
        if wait_seconds > 0:
            _println(
                _yellow(
                    f"  Rate-limit backoff active; waiting {wait_seconds:.1f}s before next attempt."
                )
            )
            await asyncio.sleep(wait_seconds)

    async def _record_run(index: int, summary: dict[str, Any]) -> None:
        async with state_lock:
            run_summaries_by_index[index] = summary
            await _update_batch_summary(final=False)

    async def _run_single(index: int, query: str) -> tuple[int, dict[str, Any]]:
        if not continue_on_error and stop_after_failure.is_set():
            summary = {
                "run_name": f"run_{index:03d}",
                "status": "skipped",
                "exit_code": None,
                "query": query,
                "attempts": 0,
                "max_retries": max_retries,
                "skip_reason": "stopped_after_failure",
            }
            await _record_run(index, summary)
            return index, summary

        async with semaphore:
            if not continue_on_error and stop_after_failure.is_set():
                summary = {
                    "run_name": f"run_{index:03d}",
                    "status": "skipped",
                    "exit_code": None,
                    "query": query,
                    "attempts": 0,
                    "max_retries": max_retries,
                    "skip_reason": "stopped_after_failure",
                }
                await _record_run(index, summary)
                return index, summary

            run_name = f"run_{index:03d}"
            run_dir = batch_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "agent_outputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "run_artifacts").mkdir(parents=True, exist_ok=True)

            run_args = argparse.Namespace(**vars(args))
            run_args.chat_id = f"{batch_id}_{run_name}_chat"
            run_args.query_id = f"{batch_id}_{run_name}_query"
            run_args.artifacts_dir = str(run_dir / "run_artifacts")
            run_args.trajectory_path = ""  # agent writes trajectory to run_artifacts/trajectory.json
            run_args.collect_artifacts_dir = ""

            _println(f"  {_bold(f'[{index}/{len(queries)}]')} {query}")
            attempt = 0
            final_exit_code = 1
            last_error = ""
            retryable_failure = False

            while True:
                await _wait_for_global_rate_limit_window()
                attempt += 1
                if attempt > 1:
                    _println(_dim(f"  Retrying {run_name} (attempt {attempt}/{max_retries + 1})"))

                final_exit_code = await _run_once(run_args, query)
                last_error = _read_batch_run_error(run_dir / "run_artifacts")

                if final_exit_code == 0:
                    retryable_failure = False
                    break

                retryable_failure, is_rate_limited = _is_retryable_batch_error(last_error)
                if not retryable_failure or attempt > max_retries:
                    break

                backoff_seconds = min(
                    retry_max_delay,
                    retry_base_delay * (2 ** (attempt - 1)),
                ) + random.uniform(0.0, retry_jitter)
                if is_rate_limited:
                    async with state_lock:
                        rate_limit_state["block_until"] = max(
                            rate_limit_state["block_until"],
                            perf_counter() + backoff_seconds,
                        )
                _println(
                    _yellow(
                        f"  {run_name} failed with retryable error; "
                        f"retrying in {backoff_seconds:.1f}s."
                    )
                )
                await asyncio.sleep(backoff_seconds)

            summary = {
                "run_name": run_name,
                "status": "ok" if final_exit_code == 0 else "error",
                "exit_code": final_exit_code,
                "query": query,
                "run_dir": str(run_dir),
                "run_artifacts_dir": str(run_dir / "run_artifacts"),
                "agent_outputs_dir": str(run_dir / "agent_outputs"),
                "attempts": attempt,
                "max_retries": max_retries,
                "retried": attempt > 1,
                "last_error": last_error or None,
                "retryable_failure": retryable_failure,
            }
            await _record_run(index, summary)

            if final_exit_code != 0 and not continue_on_error:
                async with state_lock:
                    if not stop_after_failure.is_set():
                        stop_after_failure.set()
                        _println(_yellow("  Stopping batch after first failing prompt."))
            return index, summary

    tasks = [
        asyncio.create_task(_run_single(index=index, query=query))
        for index, query in enumerate(queries, start=1)
    ]
    indexed_results = await asyncio.gather(*tasks)
    run_summaries = [summary for _, summary in sorted(indexed_results, key=lambda item: item[0])]

    async with state_lock:
        await _update_batch_summary(final=True)

    failures = [run for run in run_summaries if run["status"] == "error"]
    skipped_runs = sum(1 for run in run_summaries if run["status"] == "skipped")
    _println()
    _println(_dim(f"  Batch summary written to {summary_path}"))
    if skipped_runs:
        _println(
            _dim(
                f"  Batch complete: {len(run_summaries) - len(failures)} succeeded, "
                f"{len(failures)} failed, {skipped_runs} skipped."
            )
        )
    else:
        _println(
            _dim(
                f"  Batch complete: {len(run_summaries) - len(failures)} succeeded, "
                f"{len(failures)} failed."
            )
        )
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_HRULE_50 = "\u2500" * 50
_ARROW = "\u2192"
_ELLIPSIS = "\u2026"
_HRULE_60 = "\u2500" * 60
_CHECK = "\u2713"
_CROSS = "\u2717"
_WARN = "\u26a0"
_TERM_LOCK = threading.Lock()

_CC_ARROW = "\u25b8"   # ▸  tool call marker
_CC_CONT = "\u23bf"    # ⎿  output continuation
_CC_THINK = "\u25cf"   # ●  thinking indicator
_CC_VBAR = "\u2502"    # │  code line prefix
_ELLIPSIS = "\u2026"   # …  truncation marker


def _tty():
    """Return the real terminal stream (not the redirected one)."""
    return sys.__stdout__ if sys.__stdout__ is not None else sys.stdout


def _color_ok() -> bool:
    return _tty().isatty() and not os.getenv("NO_COLOR")


def _c(text: str, code: str) -> str:
    if not _color_ok():
        return text
    return f"\033[{code}m{text}\033[0m"


def _dim(text: str) -> str:
    return _c(text, "90")


def _bold(text: str) -> str:
    return _c(text, "1")


def _green(text: str) -> str:
    return _c(text, "32")


def _red(text: str) -> str:
    return _c(text, "31")


def _yellow(text: str) -> str:
    return _c(text, "33")


def _clear_line() -> None:
    out = _tty()
    if out.isatty():
        with _TERM_LOCK:
            out.write("\r\033[2K")
            out.flush()


def _write(text: str) -> None:
    out = _tty()
    with _TERM_LOCK:
        out.write(text)
        out.flush()


def _println(text: str = "") -> None:
    _write(text + "\n")


# ---------------------------------------------------------------------------
# Docker passthrough
# ---------------------------------------------------------------------------

_DOCKER_ENV_PASSTHROUGH = [
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "EXA_API_KEY",
    "TURBOPUFFER_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
]


def _resolve_docker_config(args: argparse.Namespace) -> Path | None:
    """Return a host config path to bind-mount into the container, or None."""
    if args.config and Path(args.config).expanduser().exists():
        return Path(args.config).expanduser().resolve()
    for name in ("faraday.yaml", "faraday.yml"):
        candidate = Path.cwd() / name
        if candidate.exists():
            return candidate
    return None


def _is_truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _docker_exec(args: argparse.Namespace) -> int:
    """Re-run the current faraday invocation inside a Docker container."""
    import subprocess

    image = args.docker_image or "faraday-oss"
    host_workspace = Path.cwd().resolve()
    docker_socket = Path("/var/run/docker.sock")

    # Ensure the image exists; build it automatically from Dockerfile.main if not.
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    if inspect.returncode != 0:
        _println(
            f"  {_c('docker', '1;96')} Image '{image}' not found — building from Dockerfile.main..."
        )
        build = subprocess.run(
            ["docker", "build", "-f", "Dockerfile.main", "-t", image, "."]
        )
        if build.returncode != 0:
            _println(_red(f"  docker build failed (exit {build.returncode})"))
            return build.returncode

    tty_flags = ["-it"] if sys.stdin.isatty() else ["-i"]
    # Name makes the app container easy to spot vs the code-exec sidecar
    # (faraday-code-sandbox-sidecar-* from DockerSandboxManager).
    app_container_name = f"faraday-app-{uuid.uuid4().hex[:10]}"
    cmd = ["docker", "run", "--rm", "--name", app_container_name] + tty_flags

    workspace_mount_path = (
        args.workspace_mount_path
        or get_workspace_mount_path(default="/workspace")
    )
    # Bind-mount the current directory as the agent workspace.
    cmd += ["-v", f"{host_workspace}:{workspace_mount_path}"]
    cmd += ["-w", workspace_mount_path]
    # The image bakes `faraday/` at build time; mount the host package so local edits
    # (and PYTHONPATH) apply without rebuilding `faraday-oss` after every code change.
    host_faraday_pkg = host_workspace / "faraday"
    if host_faraday_pkg.is_dir():
        cmd += ["-v", f"{host_faraday_pkg}:/app/faraday:ro"]
        cmd += ["-e", "PYTHONPATH=/app"]
    if docker_socket.exists():
        cmd += ["-v", f"{docker_socket}:{docker_socket}"]
    cmd += ["-e", f"FARADAY_HOST_WORKSPACE_ROOT={host_workspace}"]
    cmd += ["-e", f"FARADAY_APP_WORKSPACE_MOUNT_PATH={workspace_mount_path}"]
    cmd += ["-e", "FARADAY_RUNNING_IN_APP_DOCKER=1"]

    # Mount a local config file when one is available; image defaults are used otherwise.
    config_host = _resolve_docker_config(args)
    if config_host is not None:
        cmd += ["-v", f"{config_host}:/app/config/faraday.yaml:ro"]
        cmd += ["-e", f"FARADAY_CONFIG_DISPLAY_NAME={config_host.name}"]

    # Mount file-path tool modules into the container and build a host→container
    # path mapping so the forwarded argv uses the container paths.
    _tool_module_path_map: dict[str, str] = {}
    for idx, ref in enumerate(getattr(args, "tool_modules", [])):
        if _is_file_path_module(ref):
            host_path = Path(ref).resolve()
            container_path = f"/app/_tool_modules/{host_path.name}"
            # Disambiguate when two modules share a filename.
            if container_path in _tool_module_path_map.values():
                container_path = f"/app/_tool_modules/{idx}_{host_path.name}"
            cmd += ["-v", f"{host_path}:{container_path}:ro"]
            _tool_module_path_map[str(host_path)] = container_path

    # Forward API keys and credentials present in the current environment.
    for var in _DOCKER_ENV_PASSTHROUGH:
        val = os.environ.get(var)
        if val:
            cmd += ["-e", f"{var}={val}"]

    # Dockerfile.main sets ENTRYPOINT ["faraday"], but older/custom tags may omit it.
    # Without an entrypoint, Docker treats the query string as the executable and fails with
    # "executable file not found" for the prompt text. Force the CLI entrypoint explicitly.
    cmd += ["--entrypoint", "faraday"]
    cmd.append(image)

    # Reconstruct the faraday argv, stripping flags that only apply to the host invocation.
    passthrough: list[str] = []
    skip_next = False
    rewrite_next_tool_module = False
    has_app_runtime_flag = False
    config_was_mounted = config_host is not None
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if rewrite_next_tool_module:
            rewrite_next_tool_module = False
            if _is_file_path_module(arg):
                resolved = str(Path(arg).resolve())
                passthrough.append(_tool_module_path_map.get(resolved, arg))
            else:
                passthrough.append(arg)
            continue
        if arg == "--use-docker":
            continue
        if arg.startswith("--docker-image="):
            continue
        if arg == "--docker-image":
            skip_next = True
            continue
        # Strip --config when we already mounted the file; the container finds it at the default path.
        if config_was_mounted and arg.startswith("--config="):
            continue
        if config_was_mounted and arg == "--config":
            skip_next = True
            continue
        if arg == "--app-runtime" or arg.startswith("--app-runtime="):
            has_app_runtime_flag = True
        # Rewrite --tool-module file paths to their container mount points.
        if arg.startswith("--tool-module="):
            val = arg.split("=", 1)[1]
            if _is_file_path_module(val):
                resolved = str(Path(val).resolve())
                passthrough.append(f"--tool-module={_tool_module_path_map.get(resolved, val)}")
            else:
                passthrough.append(arg)
            continue
        if arg == "--tool-module":
            passthrough.append(arg)
            rewrite_next_tool_module = True
            continue
        passthrough.append(arg)

    if not has_app_runtime_flag:
        passthrough += ["--app-runtime", "docker"]

    cmd += passthrough

    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Step label / prefix helpers
# ---------------------------------------------------------------------------

def _show_detailed_ui(args: argparse.Namespace) -> bool:
    return bool(args.debug or args.show_agent_logs or args.debug_raw_events)


def _step_label(event: dict) -> str:
    step_index = event.get("step_index")
    current_step = str(event.get("current_step", "")).strip()
    if step_index in (None, "", 0, "0"):
        if current_step and current_step not in {"initial", "0"}:
            return current_step
        return ""

    label = f"Step {step_index}"
    if current_step and current_step not in {"initial", "0", str(step_index)}:
        label = f"{label} \u2014 {current_step}"
    return label


def _status_label(event: dict, detailed: bool) -> str:
    if detailed:
        return _step_label(event)

    event_type = str(event.get("type", "")).strip()
    role = str(event.get("role", "")).strip()
    tool_name = str(
        event.get("display_tool_name") or event.get("tool_call_name") or ""
    ).strip()

    if tool_name:
        if event_type == "tool_plan":
            return f"Using {tool_name}"
        if event_type in {"tool_status", "tool_output"}:
            return f"Reviewing {tool_name}"

    if event_type in {"warning", "error"}:
        return "Handling an issue"
    if event_type == "solution" or role == "user":
        return ""
    return "Thinking"


def _step_key(event: dict, detailed: bool) -> str:
    if detailed:
        return f"{event.get('step_index')}|{str(event.get('current_step', '')).strip()}"

    event_type = str(event.get("type", "")).strip()
    tool_name = str(
        event.get("display_tool_name") or event.get("tool_call_name") or ""
    ).strip()
    if tool_name:
        return f"{event_type}|{tool_name}|{event.get('progress_id', '')}"
    return (
        f"{event_type}|{event.get('step_index')}|"
        f"{str(event.get('current_step', '')).strip()}"
    )


def _make_prefix(event: dict, run_t0: float, step_t0: float) -> str:
    total = _format_elapsed(perf_counter() - run_t0)
    step = _format_elapsed(perf_counter() - step_t0)
    timer = _c(total, "36")
    step_dur = _dim(f"(step {step})")
    label = _step_label(event)
    if label:
        label_str = _c(f" {label} ", "95;1")
        return f"  {timer} {step_dur}{label_str}"
    return f"  {timer} {step_dur}"


def _strip_agent_tags(content: str) -> str:
    """Remove XML-style agent tags from content."""
    if not content:
        return ""
    cleaned = re.sub(
        r"</?(feedback|thought|reflection|solution)>",
        "",
        content,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _has_agent_tags(content: str) -> bool:
    """True when content was wrapped by the agent's response-tag processor."""
    return bool(re.search(r"<(feedback|thought|reflection)>", content, re.IGNORECASE))


def _single_line(text: str, max_len: int = 140) -> str:
    if not text:
        return ""
    line = " ".join(str(text).split())
    if len(line) <= max_len:
        return line
    return line[: max(1, max_len - 3)] + "\u2026"


def _extract_code_preview(content: str, max_lines: int = 4) -> list[str]:
    """Pull code lines out of a markdown fenced block inside tool_plan content."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
    if not match:
        return []
    code = match.group(1).strip()
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return lines
    return lines[:max_lines] + ["\u2026"]


def _format_tool_args_inline(event: dict, max_len: int = 100) -> str:
    """Format tool arguments as a compact inline string like key="val", key2="val2"."""
    raw = event.get("tool_call_arguments", "")
    parsed: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except Exception:
            return ""
    if not isinstance(parsed, dict):
        return ""
    parts: list[str] = []
    for key, value in parsed.items():
        if isinstance(value, str):
            val_str = f'"{value}"' if len(value) <= 36 else f'"{value[:33]}\u2026"'
        elif isinstance(value, (dict, list)):
            dumped = json.dumps(value, ensure_ascii=True)
            val_str = dumped if len(dumped) <= 40 else dumped[:37] + "\u2026"
        else:
            val_str = str(value)
        parts.append(f"{key}={val_str}")
    result = ", ".join(parts)
    if len(result) > max_len:
        result = result[: max_len - 1] + "\u2026"
    return result


def _tool_output_preview(content: str, max_len: int = 120) -> str:
    """First meaningful line of tool output, cleaned and truncated."""
    cleaned = _strip_agent_tags(content)
    if not cleaned:
        return ""
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 3 and not all(c in "-=_~" for c in stripped):
            return _single_line(stripped, max_len=max_len)
    return ""


# ---------------------------------------------------------------------------
# Event rendering — Claude Code style
# ---------------------------------------------------------------------------

def _print_event(
    event: dict,
    content: str,
    run_t0: float,
    step_t0: float,
    detailed: bool,
    render_state: dict | None = None,
    debug_raw_events: bool = False,
) -> None:
    """Render a single agent event to the terminal.

    In normal mode the output mimics Claude Code's compact shell transcript:
      ● Thinking…              — agent reasoning
      ▸ tool_name(args…)       — tool invocation
        ⎿  ✓ output preview    — tool result
    """
    _clear_line()
    raw_content = content if isinstance(content, str) else str(content)
    content = raw_content.strip()

    # ── Detailed mode (--debug / --show-agent-logs) — unchanged ──
    if detailed:
        if debug_raw_events:
            prefix = _make_prefix(event, run_t0, step_t0)
            event_payload = dict(event)
            if "content" not in event_payload and raw_content:
                event_payload["content"] = raw_content
            _println(f"{prefix}  {json.dumps(event_payload, ensure_ascii=True)}")
            return
        if not content:
            return
        prefix = _make_prefix(event, run_t0, step_t0)
        lines = content.splitlines()
        _println(f"{prefix}  {lines[0]}")
        if len(lines) > 1:
            pad = " " * 20
            for line in lines[1:]:
                _println(f"{pad}{line}")
        return

    if not content:
        return

    # ── Normal mode — Claude Code style ──
    role = str(event.get("role", "")).strip()
    etype = str(event.get("type", "")).strip()
    clean = _strip_agent_tags(content)

    if role == "user":
        return

    # Thinking / reflection
    if etype == "thought":
        total = _format_elapsed(perf_counter() - run_t0)
        _println()
        thinking_label = "Thinking" + _ELLIPSIS
        _println(f"  {_c(_CC_THINK, '36')} {_c(thinking_label, '36;1')}  {_dim(total)}")
        if clean:
            for line in clean.splitlines()[:2]:
                preview = _single_line(line, max_len=120)
                if preview:
                    _println(f"    {_dim(preview)}")
        return

    # Agent feedback (from <feedback> tags) — show as commentary
    if etype == "feedback":
        if _has_agent_tags(content) and clean:
            _println()
            for line in clean.splitlines()[:4]:
                stripped = line.strip()
                if stripped:
                    _println(f"  {stripped}")
        return

    # Tool call start
    if etype == "tool_plan":
        tool_name = str(
            event.get("display_tool_name") or event.get("tool_call_name") or "tool"
        ).strip()
        call_id = str(event.get("tool_call_id", "")).strip()
        if render_state is not None and call_id:
            render_state.setdefault("tool_started", {})[call_id] = perf_counter()

        is_code = "code" in tool_name.lower() or "execute" in tool_name.lower()
        _println()

        if is_code:
            _println(f"  {_c(_CC_ARROW, '36')} {_bold(tool_name)}")
            for cl in _extract_code_preview(content):
                _println(f"    {_c(_CC_VBAR, '90')} {_dim(cl)}")
        else:
            args_str = _format_tool_args_inline(event)
            if args_str:
                _println(
                    f"  {_c(_CC_ARROW, '36')} {_bold(tool_name)}"
                    f"({_dim(args_str)})"
                )
            else:
                _println(f"  {_c(_CC_ARROW, '36')} {_bold(tool_name)}")
        return

    # Tool output (success)
    if etype == "tool_output":
        call_id = str(event.get("tool_call_id", "")).strip()
        elapsed_str = ""
        if render_state is not None and call_id:
            started = render_state.get("tool_started", {}).pop(call_id, None)
            if started is not None:
                secs = perf_counter() - started
                elapsed_str = (
                    _dim(f" {secs:.1f}s")
                    if secs < 60
                    else _dim(f" {_format_elapsed(secs)}")
                )
        preview = _tool_output_preview(clean)
        mark = _green(_CHECK)
        cont = _c(_CC_CONT, "90")
        if preview:
            _println(f"    {cont}  {mark} {preview}{elapsed_str}")
        else:
            _println(f"    {cont}  {mark} {_dim('done')}{elapsed_str}")
        return

    # Tool error
    if etype == "error" and str(event.get("tool_call_id", "")).strip():
        call_id = str(event.get("tool_call_id", "")).strip()
        elapsed_str = ""
        if render_state is not None:
            started = render_state.get("tool_started", {}).pop(call_id, None)
            if started is not None:
                secs = perf_counter() - started
                elapsed_str = _dim(f" {secs:.1f}s")
        preview = _tool_output_preview(clean) or "failed"
        cont = _c(_CC_CONT, "90")
        _println(f"    {cont}  {_red(_CROSS)} {preview}{elapsed_str}")
        return

    # Status updates
    if etype == "update" and clean:
        _println(f"  {_dim(_single_line(clean, max_len=140))}")
        return

    # Warnings & standalone errors
    if etype in {"warning", "error"}:
        icon = _WARN if etype == "warning" else _CROSS
        color = _yellow if etype == "warning" else _red
        lines = (clean or content).splitlines()
        _println(f"  {color(icon)} {color(lines[0])}")
        for line in lines[1:]:
            _println(f"    {line}")
        return


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

def _run_spinner(stop: threading.Event, state: dict) -> None:
    out = _tty()
    if not out.isatty():
        return

    frames = ("\u28fe", "\u28f7", "\u28ef", "\u28df", "\u287f", "\u28bf", "\u28fb", "\u28fd")
    idx = 0
    while not stop.is_set():
        label = state.get("label", "")
        detailed = bool(state.get("detailed"))
        dot = _c(frames[idx % len(frames)], "36")

        if detailed:
            total = _format_elapsed(perf_counter() - state["run_t0"])
            step_t0 = state.get("step_t0", state["run_t0"])
            step = _format_elapsed(perf_counter() - step_t0)
            msg = _dim(f"{total} (step {step})")
            label_part = _dim(f"  {label}") if label else ""
            line = f"  {dot} {msg}{label_part}"
        else:
            label_part = _dim(f" {label}{_ELLIPSIS}") if label else ""
            line = f"  {dot}{label_part}"

        with _TERM_LOCK:
            out.write(f"\r\033[2K{line}")
            out.flush()
        idx += 1
        stop.wait(0.08)

    _clear_line()


# ---------------------------------------------------------------------------
# Log capture (stdout + stderr + Python logging)
# ---------------------------------------------------------------------------

class _LogSink:
    """Captures both print() output and Python logging records."""

    def __init__(self, show_live: bool = False):
        self.show_live = show_live
        self._lines: list[str] = []
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._ingest(line)
        return len(text)

    def flush(self) -> None:
        pass

    def make_handler(self) -> logging.Handler:
        sink = self

        class _H(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                    sink._ingest(msg)
                except Exception:
                    pass

        h = _H()
        h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        return h

    def _ingest(self, line: str) -> None:
        clean = line.rstrip()
        if not clean.strip():
            return
        self._lines.append(clean)
        if self.show_live:
            _clear_line()
            _println(_dim(f"    {clean}"))

    def finalize(self) -> None:
        if self._buf:
            self._ingest(self._buf)
            self._buf = ""

    def render_collapsed(self) -> None:
        if self.show_live or not self._lines:
            return
        n = len(self._lines)
        _println(_dim(
            f"  \u25b8 {n} agent log line{'s' if n != 1 else ''}"
            " hidden \u2014 use --show-agent-logs to expand"
        ))


class AgentInitError(RuntimeError):
    """Raised when agent initialization fails before the CLI can start."""

    def __init__(self, message: str, log_lines: list[str] | None = None):
        super().__init__(message)
        self.log_lines = log_lines or []


# ---------------------------------------------------------------------------
# Tool check: categories, descriptions, required env keys
# ---------------------------------------------------------------------------

TOOL_CATEGORIES = {
    "Literature & Knowledge": [
        "scientific_literature_search",
    ],
    "Web & Data": [
        "pharma_web_search",
        "general_web_search",
        "general_web_search_question_answering",
        "read_webpage",
        "name_to_smiles",
    ],
    "Code Execution": [
        "execute_python_code",
        "execute_bash_code",
    ],
}

TOOL_SHORT_DESCRIPTIONS = {
    "scientific_literature_search": "Search scientific papers and journals",
    "pharma_web_search": "Search pharma and biotech web sources",
    "general_web_search": "General purpose web search",
    "general_web_search_question_answering": "Web search with answer extraction",
    "read_webpage": "Read and extract content from a URL",
    "name_to_smiles": "Convert molecule names to SMILES notation",
    "execute_python_code": "Run Python in a sandboxed environment",
    "execute_bash_code": "Run bash commands in a sandboxed environment",
}

CATEGORY_PREFIXES = {
    "Literature & Knowledge": "[LIT]",
    "Web & Data": "[WEB]",
    "Code Execution": "[CODE]",
}

from faraday.faraday_agent import TOOL_REQUIRED_KEYS  # single source of truth

MODAL_REQUIRED_KEYS = ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]


def _check_key(name: str) -> bool:
    return bool(os.getenv(name))


def _resolved_agent_requirements() -> list[tuple[str, bool, str]]:
    """Return startup requirements for the configured LLM provider.

    Each tuple is: (label, is_satisfied, note).
    """
    provider = get_llm_provider()
    try:
        settings = get_client_settings()
    except Exception:
        settings = {}

    requirements: list[tuple[str, bool, str]] = []
    api_key_env = settings.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        api_key_env = {
            "azure": "AZURE_OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(provider, "OPENAI_API_KEY")

    requirements.append((api_key_env, _check_key(api_key_env), "required to start Faraday"))

    if provider == "azure":
        has_base_url = bool(settings.get("base_url"))
        requirements.append(
            (
                "AZURE_OPENAI_BASE_URL or llm.base_url",
                has_base_url,
                "required for Azure provider",
            )
        )

    return requirements


def _resolved_exec_backend(args: argparse.Namespace) -> str:
    if args.execution_backend:
        raw = args.execution_backend.strip().lower().replace("_", "-")
        return normalize_execution_backend(raw)
    return get_execution_backend(default="docker")


def _resolved_app_runtime(args: argparse.Namespace) -> str:
    if args.app_runtime:
        raw = args.app_runtime.strip().lower().replace("_", "-")
        return normalize_runtime_app(raw)
    return get_runtime_app(default="host")


def _resolve_sandbox_docker_image(args: argparse.Namespace) -> str:
    """Return the sandbox Docker image name that DockerSandboxManager will use."""
    cli_backend = _resolved_exec_backend(args)
    if cli_backend != "docker":
        return ""
    image = (
        str(_first_set(("sandbox", "docker_image"), ("execution", "docker_image")) or "").strip()
        or os.getenv("FARADAY_DOCKER_IMAGE", "faraday-code-sandbox").strip()
        or "faraday-code-sandbox"
    )
    return image


def _preflight_check_docker_image(args: argparse.Namespace) -> None:
    """Fast check that the sandbox Docker image exists before starting the agent.

    Exits with a clear message and build instructions instead of letting the
    agent silently block for 5+ minutes on ``docker build``.
    """
    import subprocess as _sp

    image = _resolve_sandbox_docker_image(args)
    if not image:
        return

    if shutil.which("docker") is None:
        _println()
        _println(f"  {_yellow(_WARN)} Docker CLI not found on PATH.")
        _println(
            f"    The execution backend is {_bold('docker')} but the "
            f"{_bold('docker')} command is not available."
        )
        _println(f"    Install Docker or set {_bold('execution.backend: host')} in your config.")
        _println()
        raise SystemExit(1)

    probe = _sp.run(
        ["docker", "info"],
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
        check=False,
        timeout=5,
    )
    if probe.returncode != 0:
        _println()
        _println(f"  {_yellow(_WARN)} Docker daemon is not running.")
        _println(
            f"    The execution backend is {_bold('docker')} but the Docker "
            f"daemon did not respond."
        )
        _println(f"    Start Docker Desktop / the daemon, or set {_bold('execution.backend: host')}.")
        _println()
        raise SystemExit(1)

    inspect = _sp.run(
        ["docker", "image", "inspect", image],
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
        check=False,
    )
    if inspect.returncode != 0:
        _println()
        _println(
            f"  {_yellow(_WARN)} Docker image {_bold(repr(image))} not found locally."
        )
        _println()
        _println(f"    Build it first (from the repo root):")
        _println()
        _println(f"      {_c('docker build -f Dockerfile.sandbox -t ' + image + ' .', '1;97')}")
        _println()
        _println(
            f"    Or pull a pre-built image, or set {_bold('execution.backend: host')} "
            f"to skip Docker."
        )
        _println()
        raise SystemExit(1)


def _run_check_tools(agent, args: argparse.Namespace, init_error: str = "") -> int:
    """Print tool + backend env-key health status. Returns exit code."""
    selected_backend = _resolved_exec_backend(args)
    active_names: set[str] = set()
    if agent is not None:
        active_names = {
            t.get("name") or t.get("function", {}).get("name", "")
            for t in agent.active_tools
        }

    ok = _green(_CHECK)
    fail = _red(_CROSS)
    warn = _yellow(_WARN)
    hr = _dim("\u2500" * 52)
    missing_by_key: dict[str, list[str]] = {}
    total_tools = 0
    ready_tools = 0
    agent_requirements = _resolved_agent_requirements()
    core_missing = [label for label, ok_state, _note in agent_requirements if not ok_state]
    core_ready = not core_missing

    # ── Header ────────────────────────────────────────
    _println()
    _println(f"  {_c('Faraday Tool Check', '1;96')}")
    _println(f"  {hr}")
    _println()

    # ── Configuration ─────────────────────────────────
    _println(f"  {_c('Configuration', '1;97')}")
    _println(f"    Backend   {_c(selected_backend.upper(), '1;96')}")
    _println(f"    Runtime   {_c(_resolved_app_runtime(args).upper(), '1;96')}")
    _println()

    # ── Core requirements ─────────────────────────────
    all_ok = True
    _println(f"  {_c('Core Requirements', '1;97')}")
    for label, ok_state, note in agent_requirements:
        if ok_state:
            _println(f"    {ok} {label}")
        else:
            _println(f"    {fail} {label}  {_dim(f'({note})')}")
            all_ok = False
    _println()

    # ── Execution backend ─────────────────────────────
    _println(f"  {_c('Execution Backend', '1;97')}")
    if selected_backend == "modal":
        missing_modal = [k for k in MODAL_REQUIRED_KEYS if not _check_key(k)]
        if missing_modal:
            _println(f"    {fail} modal credentials")
            for k in missing_modal:
                _println(f"        {fail} {k}")
            _println(f"        {_dim('Get credentials with: modal token new')}")
            all_ok = False
        else:
            _println(f"    {ok} modal credentials")
    elif selected_backend in {"docker", "host"}:
        _println(f"    {ok} {selected_backend} backend (no Modal token required)")
        if not (args.workspace_source_root or get_workspace_source_root(default="")):
            _println(f"    {warn} set {_bold('runtime.workspace.source_root')} for predictable local runs")
    elif selected_backend == "disabled":
        _println(f"    {warn} execution backend disabled — code tools unavailable")
    else:
        _println(f"    {warn} backend '{selected_backend}' has no dedicated check")
    _println()

    for category, tool_names in TOOL_CATEGORIES.items():
        if not tool_names:
            continue
        prefix = CATEGORY_PREFIXES.get(category, "[TOOL]")

        ready_in_cat: list[tuple[str, str]] = []
        missing_in_cat: list[tuple[str, str, list[str]]] = []

        for name in tool_names:
            desc = TOOL_SHORT_DESCRIPTIONS.get(name, "")
            required = TOOL_REQUIRED_KEYS.get(name, [])
            missing = [k for k in required if not _check_key(k)]
            total_tools += 1
            if missing:
                missing_in_cat.append((name, desc, missing))
                for key in missing:
                    missing_by_key.setdefault(key, []).append(name)
            else:
                ready_in_cat.append((name, desc))
                ready_tools += 1

    if core_ready:
        # ── Per-category tools ────────────────────────────
        _println(f"  {hr}")
        _println()

        for category, tool_names in TOOL_CATEGORIES.items():
            if not tool_names:
                continue
            prefix = CATEGORY_PREFIXES.get(category, "[TOOL]")

            ready_in_cat: list[tuple[str, str]] = []
            missing_in_cat: list[tuple[str, str, list[str]]] = []

            for name in tool_names:
                desc = TOOL_SHORT_DESCRIPTIONS.get(name, "")
                required = TOOL_REQUIRED_KEYS.get(name, [])
                missing = [k for k in required if not _check_key(k)]
                if missing:
                    missing_in_cat.append((name, desc, missing))
                else:
                    ready_in_cat.append((name, desc))

            cat_ready = len(ready_in_cat)
            cat_missing = len(missing_in_cat)
            cat_summary = _dim(f"({cat_ready} ready")
            if cat_missing:
                cat_summary += _dim(", ") + _red(f"{cat_missing} blocked") + _dim(")")
            else:
                cat_summary += _dim(")")
            _println(f"  {_dim(prefix)} {_c(category, '1;93')} {cat_summary}")

            max_name_len = 0
            for name, _desc in ready_in_cat:
                max_name_len = max(max_name_len, len(name))
            for name, _desc, _missing_keys in missing_in_cat:
                max_name_len = max(max_name_len, len(name))

            for name, desc in ready_in_cat:
                padded_name = name.ljust(max_name_len)
                line = f"    {ok} {_c(padded_name, '97')}"
                if desc:
                    line += f"  {_dim(desc)}"
                _println(line)

            for name, desc, _missing_keys in missing_in_cat:
                padded_name = name.ljust(max_name_len)
                line = f"    {fail} {_c(padded_name, '90')}"
                if desc:
                    line += f"  {_dim(desc)}"
                _println(line)
                all_ok = False

            _println()

        optional_missing = {
            key: tools for key, tools in missing_by_key.items() if key not in core_missing
        }
        if optional_missing:
            _println(f"  {_c('Optional Integrations Disabled', '1;97')}")
            max_key_len = max(len(key) for key in optional_missing)
            for key, tools in sorted(optional_missing.items()):
                tool_list = ", ".join(sorted(tools))
                _println(f"    {fail} {key.ljust(max_key_len)}  {_dim('enables:')} {tool_list}")
            _println()

        # Uncategorized tools (e.g. from plugins)
        all_known = {n for names in TOOL_CATEGORIES.values() for n in names}
        uncategorized = active_names - all_known
        if uncategorized:
            _println(f"  {_dim('[EXT]')} {_c('Other', '1;93')}")
            for name in sorted(uncategorized):
                _println(f"    {ok} {_c(name, '97')}")
                ready_tools += 1
                total_tools += 1
            _println()
    else:
        optional_missing = {
            key: tools for key, tools in missing_by_key.items() if key not in core_missing
        }
        if optional_missing:
            _println(f"  {_c('Optional Integrations', '1;97')}")
            _println(f"    {_dim('After Faraday can start, these extra keys enable additional tools:')}")
            max_key_len = max(len(key) for key in optional_missing)
            for key, tools in sorted(optional_missing.items()):
                tool_list = ", ".join(sorted(tools))
                _println(f"    {warn} {key.ljust(max_key_len)}  {_dim('enables:')} {tool_list}")
            _println()

        _println(f"  {hr}")
        _println(f"  {fail} {_red('Faraday cannot start until required core keys are set.')}")
        _println(f"  {_dim('Set the missing core requirements above, then re-run `faraday --check-tools`.')}")
        _println()
        return 1

    # ── Init error (if agent failed to load) ──────────
    if init_error:
        clean_error = init_error.strip()
        missing_agent_keys = [label for label, ok_state, _note in _resolved_agent_requirements() if not ok_state]
        if clean_error and clean_error != "1":
            _println(f"  {warn} Agent failed to initialize: {_dim(clean_error)}")
            _println()
        elif not missing_agent_keys:
            _println(f"  {warn} Agent failed to initialize (LLM configuration error)")
            _println()

    # ── Summary ───────────────────────────────────────
    _println(f"  {hr}")
    excluded = total_tools - ready_tools
    if all_ok:
        _println(f"  {ok} {_green(f'All {total_tools} tools ready')}")
    else:
        sep = _dim("\u00b7")
        _println(
            f"  {_green(f'{ready_tools}/{total_tools} tools ready')}"
            f" {sep} {_red(f'{excluded} excluded')}{_dim(' (missing keys)')}"
        )
        _println(f"  {_dim('The agent works with available tools. Export missing keys to enable more.')}")
    _println()

    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Startup banner (lightweight, no tool listing)
# ---------------------------------------------------------------------------

def _shorten_path(path: str, max_len: int = 45) -> str:
    """Collapse a long absolute path to ``…/last_two_components``."""
    if len(path) <= max_len:
        return path
    parts = Path(path).parts
    if len(parts) <= 2:
        return path
    return f"{_ELLIPSIS}/{'/'.join(parts[-2:])}"


def _print_startup_banner(
    agent,
    config_path: str,
    init_time: float,
    detailed: bool,
) -> None:
    active_count = len(agent.active_tools)
    ri_fn = getattr(agent, "get_runtime_info", None)
    ri = ri_fn() if callable(ri_fn) else {}
    arrow = _dim(_ARROW)

    _println()
    _println(f"  {_c(_CC_THINK, '96')} {_c('Faraday', '1;96')} {_dim(f'v{agent.app_version}')}")
    _println()

    _println(f"    {_dim('Model')}      {_c(agent.model, '1;97')}")
    if config_path:
        display_name = os.environ.get("FARADAY_CONFIG_DISPLAY_NAME") or os.path.basename(config_path)
        _println(f"    {_dim('Config')}     {display_name}")

    if ri:
        backend = ri.get("backend", "docker")
        image = ri.get("image", "")
        workspace = _shorten_path(ri.get("workspace", ""))
        mount = ri.get("mount", "/workspace")
        if agent.enable_sandbox:
            sandbox_str = backend
            if image and backend == "docker":
                sandbox_str += f" {_dim(f'({image})')}"
        else:
            sandbox_str = "disabled"
        _println(f"    {_dim('Sandbox')}    {sandbox_str}")
        if workspace:
            _println(f"    {_dim('Workspace')}  {workspace} {arrow} {mount}")

    if detailed:
        _println()
        _println(f"    {_dim('Max steps')}  {agent.max_total_steps}")
        rag_label = "in-memory" if agent.in_memory_rag_enabled else "none"
        _println(f"    {_dim('RAG')}        {rag_label}")
        _println(f"    {_dim('Tools')}      {active_count} active")
        _println(f"    {_dim('Init')}       {init_time:.2f}s")

    _println()


def _print_agent_init_failure(err: AgentInitError) -> None:
    core_missing = [label for label, ok_state, _note in _resolved_agent_requirements() if not ok_state]

    _println()
    _println(f"  {_red(_CROSS)} {_red('Faraday could not start.')}")
    if core_missing:
        missing_keys = ", ".join(core_missing)
        _println(f"  {_dim('Missing required startup requirement(s):')} {_red(missing_keys)}")
        _println(f"  {_dim('Run `faraday --check-tools` for a full readiness report.')}")
        _println()
        return

    message = str(err).strip()
    if message:
        _println(f"  {_dim('Startup error:')} {message}")
    if err.log_lines:
        _println(f"  {_dim('Use --show-agent-logs for full initialization logs.')}")
    _println()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _bootstrap_faraday_config_from_argv(argv: list[str] | None = None) -> None:
    """Apply ``--config`` to ``FARADAY_CONFIG`` before :func:`parse_args` runs.

    Config precedence for one process:

    1. Built-in / code defaults used only when building the YAML schema (inside ``get_*`` helpers).
    2. Values from the active YAML file (``FARADAY_CONFIG`` or first discovered ``faraday.yaml``).
    3. Explicit CLI flags (e.g. ``--model``), which argparse applies on top of YAML-derived defaults.

    Without this bootstrap, ``parse_args`` computed defaults (model, execution backend, …) from
    the wrong file whenever ``--config`` was passed, because ``FARADAY_CONFIG`` was set only after
    parsing.
    """
    if argv is None:
        argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("error: --config requires a path")
            raw = argv[i + 1]
            path = Path(raw).expanduser()
            path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
            if not path.is_file():
                raise SystemExit(f"error: config file not found: {path}")
            os.environ["FARADAY_CONFIG"] = str(path)
            return
        if arg.startswith("--config="):
            raw = arg.split("=", 1)[1]
            if not raw.strip():
                raise SystemExit("error: --config= requires a path")
            path = Path(raw).expanduser()
            path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
            if not path.is_file():
                raise SystemExit(f"error: config file not found: {path}")
            os.environ["FARADAY_CONFIG"] = str(path)
            return
        i += 1


def parse_args() -> argparse.Namespace:
    default_model = get_llm_model(default="gpt-5")
    default_app_runtime = get_runtime_app(default="host")
    default_app_docker_image = get_runtime_app_docker_image(default="faraday-oss")
    default_execution_backend = get_execution_backend(default="docker")
    default_workspace_source_root = get_workspace_source_root(
        default=str(Path.cwd().expanduser().resolve())
    ) or str(Path.cwd().expanduser().resolve())
    default_workspace_mount_path = get_workspace_mount_path(default="/workspace")
    default_trajectory_path = ""  # persistence.trajectory_path is deprecated; use outputs.root
    default_previous_context = get_path_value("persistence", "previous_context", default="") or ""
    default_artifacts_dir = get_path_value("artifacts", "output_dir", default="") or ""
    default_collect_artifacts_dir = get_path_value("artifacts", "collect_dir", default="") or ""

    parser = argparse.ArgumentParser(
        description="Run Faraday from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to run. If omitted, uses config `query` when set; otherwise interactive mode.",
    )

    # -- Core -------------------------------------------------------------------
    core = parser.add_argument_group("core options")
    core.add_argument(
        "--model",
        default=default_model,
        help="Model name (default: config `model` or gpt-5).",
    )
    core.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum agent steps before stopping (default: 20).",
    )
    core.add_argument(
        "--config",
        default="",
        help="Path to Faraday YAML config (also supported via FARADAY_CONFIG env var).",
    )
    core.add_argument(
        "--chat-id",
        default=None,
        help="Optional chat ID. Auto-generated when omitted.",
    )
    core.add_argument(
        "--query-id",
        default=None,
        help="Optional query ID. Auto-generated when omitted.",
    )

    # -- Runtime ----------------------------------------------------------------
    runtime = parser.add_argument_group("runtime & execution")
    runtime.add_argument(
        "--app-runtime",
        default=default_app_runtime,
        help="Where Faraday app runs: host or docker.",
    )
    runtime.add_argument(
        "--execution-backend",
        default=default_execution_backend,
        help="Sandbox backend: docker, modal, host, or disabled (local is an alias for host).",
    )
    runtime.add_argument(
        "--workspace-source-root",
        default=default_workspace_source_root,
        help="Workspace source directory for docker/host execution backends (default: current working directory).",
    )
    runtime.add_argument(
        "--workspace-mount-path",
        default=default_workspace_mount_path,
        help="Path where the workspace is mounted inside docker runtimes.",
    )
    runtime.add_argument(
        "--use-docker",
        action="store_true",
        help="Re-run this command inside a Docker container.",
    )
    runtime.add_argument(
        "--docker-image",
        default=default_app_docker_image,
        help="Docker image for --use-docker (default: config runtime.app_image or faraday-oss).",
    )

    # -- Output & artifacts -----------------------------------------------------
    output = parser.add_argument_group("output & artifacts")
    output.add_argument(
        "--artifacts-dir",
        default=default_artifacts_dir,
        help="Output directory for run artifacts (events, result, metadata). "
             "Defaults to run_outputs/run_<id>/run_artifacts.",
    )
    output.add_argument(
        "--collect-artifacts-dir",
        default=default_collect_artifacts_dir,
        help="Copy generated artifacts here and write manifest.json (Harbor-style collection).",
    )
    output.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Disable automatic artifact generation for this run.",
    )
    output.add_argument(
        "--trajectory-path",
        default=default_trajectory_path,
        help="Output path for ATIF trajectory JSON.",
    )
    output.add_argument(
        "--previous-context",
        dest="previous_context",
        default=default_previous_context,
        help="Path to a prior trajectory.json to seed context from.",
    )

    # -- Batch ------------------------------------------------------------------
    batch = parser.add_argument_group("batch execution")
    batch.add_argument(
        "--batch-file",
        default="",
        help="Run a batch from a prompt file (.txt/.json/.jsonl), one prompt per line/item.",
    )
    batch.add_argument(
        "--batch-query",
        action="append",
        default=[],
        help="Add a prompt to the batch run (repeatable).",
    )
    batch.add_argument(
        "--batch-output-root",
        default="",
        help="Root directory for batch outputs. A new batch_<id> dir is created per run.",
    )
    batch.add_argument(
        "--batch-continue-on-error",
        action="store_true",
        help="Continue remaining batch prompts when one fails.",
    )
    batch.add_argument(
        "--batch-max-concurrency",
        type=int,
        default=None,
        help="Max concurrent batch prompts (default: config batch.max_concurrency or 1).",
    )
    batch.add_argument(
        "--batch-max-retries",
        type=int,
        default=None,
        help="Max retries per prompt (default: config batch.max_retries or 2).",
    )
    batch.add_argument(
        "--batch-retry-base-delay-seconds",
        type=float,
        default=None,
        help="Base backoff delay in seconds (default: 2.0).",
    )
    batch.add_argument(
        "--batch-retry-max-delay-seconds",
        type=float,
        default=None,
        help="Max backoff delay in seconds (default: 30.0).",
    )
    batch.add_argument(
        "--batch-retry-jitter-seconds",
        type=float,
        default=None,
        help="Jitter added to retry backoff in seconds (default: 0.5).",
    )

    # -- Debugging --------------------------------------------------------------
    debug = parser.add_argument_group("debugging")
    debug.add_argument(
        "--debug",
        action="store_true",
        help="Enable full debug profile (raw events + live logs + transient events).",
    )
    debug.add_argument(
        "--debug-raw-events",
        action="store_true",
        help="Print raw JSON event payloads.",
    )
    debug.add_argument(
        "--debug-include-transient-events",
        action="store_true",
        help="Include transient events in events.jsonl artifact output.",
    )
    debug.add_argument(
        "--show-agent-logs",
        action="store_true",
        help="Expand internal agent logs instead of collapsed summary.",
    )
    debug.add_argument(
        "--check-tools",
        action="store_true",
        help="Verify available tools and API keys, then exit.",
    )

    # -- Extensions -------------------------------------------------------------
    ext = parser.add_argument_group("extensions")
    ext.add_argument(
        "--tool-module",
        action="append",
        default=[],
        dest="tool_modules",
        metavar="MODULE",
        help=(
            "Python module to load (dotted name or .py path). Must expose "
            "EXTRA_TOOLS and/or EXTRA_TOOL_HANDLERS. Repeatable."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tool-module loader
# ---------------------------------------------------------------------------

def _is_file_path_module(ref: str) -> bool:
    """True when *ref* looks like a filesystem path rather than a dotted module name."""
    return ref.endswith(".py") or Path(ref).is_file()


def _load_tool_modules(module_refs: list[str]) -> tuple[list, dict]:
    """Import tool modules and collect their EXTRA_TOOLS / EXTRA_TOOL_HANDLERS.

    Each entry in *module_refs* is either a dotted Python module name
    (``my_pkg.tools``) or a path to a ``.py`` file.  The module must expose
    one or both of:

    - ``EXTRA_TOOLS``         – list of OpenAI-style tool spec dicts
    - ``EXTRA_TOOL_HANDLERS`` – dict mapping tool name → callable

    Modules are processed in order; later entries override earlier ones for
    duplicate handler names.  After all modules are loaded the collected specs
    are deduplicated by ``name`` (last spec wins) and every spec name is
    verified to have a matching handler.
    """
    import importlib
    import importlib.util

    extra_tools: list = []
    extra_handlers: dict = {}

    for idx, ref in enumerate(module_refs):
        if _is_file_path_module(ref):
            p = Path(ref)
            mod_name = f"_faraday_tool_module_{idx}"
            spec = importlib.util.spec_from_file_location(mod_name, str(p.resolve()))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load tool module from path: {ref!r}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        else:
            mod = importlib.import_module(ref)

        extra_tools.extend(getattr(mod, "EXTRA_TOOLS", []))
        extra_handlers.update(getattr(mod, "EXTRA_TOOL_HANDLERS", {}))

    # Deduplicate specs by tool name (last definition wins).
    seen: dict[str, dict] = {}
    for tool_spec in extra_tools:
        name = tool_spec.get("name")
        if name:
            seen[name] = tool_spec
    extra_tools = list(seen.values())

    # Cross-validate: every spec must have a handler registered.
    spec_names = {t.get("name") for t in extra_tools}
    missing = spec_names - set(extra_handlers)
    if missing:
        raise ValueError(
            f"Tool spec(s) {sorted(missing)} have no matching handler in "
            f"EXTRA_TOOL_HANDLERS. Check that the spec 'name' field and the "
            f"handler dict key match exactly."
        )

    return extra_tools, extra_handlers


# ---------------------------------------------------------------------------
# Agent init
# ---------------------------------------------------------------------------

def _create_agent(args: argparse.Namespace, chat_id: str, query_id: str):
    from faraday.faraday_agent import FaradayAgent, FaradayAgentConfig

    config_tool_modules = get_config_value("tool_modules", default=None) or []
    if isinstance(config_tool_modules, str):
        config_tool_modules = [config_tool_modules]
    cli_tool_modules = getattr(args, "tool_modules", []) or []
    all_tool_modules = list(config_tool_modules) + cli_tool_modules
    extra_tools, extra_tool_handlers = _load_tool_modules(all_tool_modules)

    return FaradayAgent(
        config=FaradayAgentConfig(
            model=args.model,
            max_total_steps=args.max_steps,
            chat_id=chat_id,
            query_id=query_id,
            debug_print=args.debug,
            verbose=True,
            conversation_history=getattr(args, "seed_conversation_history", None),
            app_runtime=args.app_runtime or None,
            execution_backend=args.execution_backend or None,
            workspace_source_root=args.workspace_source_root or None,
            workspace_mount_path=args.workspace_mount_path or None,
            trajectory_path=args.trajectory_path or None,
            extra_tools=extra_tools or None,
            extra_tool_handlers=extra_tool_handlers or None,
        )
    )


def _init_agent_quiet(args: argparse.Namespace, chat_id: str, query_id: str):
    """Create the agent while capturing all its init printouts."""
    init_sink = _LogSink(show_live=bool(args.show_agent_logs))
    init_handler = init_sink.make_handler()
    root = logging.getLogger()
    prev_handlers = root.handlers[:]
    root.handlers = [init_handler]

    t0 = perf_counter()
    agent = None
    init_error: Exception | SystemExit | None = None
    try:
        with contextlib.redirect_stdout(init_sink), contextlib.redirect_stderr(init_sink):
            agent = _create_agent(args, chat_id, query_id)
    except (Exception, SystemExit) as exc:
        init_error = exc
    finally:
        root.handlers = prev_handlers
        init_sink.finalize()

    if init_error is not None:
        message = str(init_error).strip()
        if not message or message == "1":
            for line in reversed(init_sink._lines):
                clean = line.strip()
                if clean:
                    message = clean
                    break
        if not message:
            message = init_error.__class__.__name__
        raise AgentInitError(message, log_lines=list(init_sink._lines)) from init_error

    init_time = perf_counter() - t0
    return agent, init_time


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

async def _run_once(args: argparse.Namespace, query: str, agent=None, init_time: float = 0) -> int:
    detailed_ui = _show_detailed_ui(args)
    if agent is None:
        chat_id = args.chat_id or _default_id("chat")
        query_id = args.query_id or _default_id("query")
        try:
            agent, init_time = _init_agent_quiet(args, chat_id, query_id)
        except AgentInitError as exc:
            _print_agent_init_failure(exc)
            return 1
        _print_startup_banner(agent, args.config, init_time, detailed=detailed_ui)

    artifacts_enabled = not args.no_artifacts
    artifacts_dir: Path | None = None
    events_path: Path | None = None
    result_path: Path | None = None
    metadata_path: Path | None = None
    trajectory_path: Path | None = _resolve_trajectory_path(agent)
    events_handle = None
    if artifacts_enabled:
        artifacts_dir = _resolve_artifacts_dir(args, agent)
        # When artifacts_dir is explicitly overridden (e.g. batch mode), sync the
        # agent's run directory tree so that trajectory and artifacts land together.
        if args.artifacts_dir and hasattr(agent, "run_artifacts_dir"):
            explicit_dir = artifacts_dir.resolve()
            if explicit_dir != agent.run_artifacts_dir.resolve():
                agent.run_artifacts_dir = explicit_dir
                agent.run_output_root = explicit_dir.parent
                agent.agent_outputs_dir = agent.run_output_root / "agent_outputs"
                agent._trajectory_output_path = None
                agent._trajectory_live_output_path = None
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        events_path = artifacts_dir / "events.jsonl"
        result_path = artifacts_dir / "result.json"
        metadata_path = artifacts_dir / "metadata.json"
        events_handle = events_path.open("w", encoding="utf-8")

    final_solution = None
    error_message = ""
    event_count = 0
    started_at = _iso_now()
    run_t0 = perf_counter()
    step_t0 = run_t0
    cur_key = ""
    spinner_state = {
        "run_t0": run_t0,
        "step_t0": step_t0,
        "label": "Starting" if detailed_ui else "Thinking",
        "detailed": detailed_ui,
    }
    render_state: dict[str, Any] = {"tool_started": {}}

    spinner_stop = threading.Event()
    spinner_thread = threading.Thread(
        target=_run_spinner,
        args=(spinner_stop, spinner_state),
        daemon=True,
    )
    spinner_thread.start()

    sink = _LogSink(show_live=args.show_agent_logs)
    log_handler = sink.make_handler()
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    root_logger.handlers = [log_handler]

    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            async for event in agent.run(query):
                event_count += 1
                etype = event.get("type", "")
                content = event.get("content", "")
                is_transient = bool(event.get("transient", False))
                if events_handle is not None and (
                    not is_transient or args.debug_include_transient_events
                ):
                    events_handle.write(json.dumps(event, ensure_ascii=True) + "\n")
                ekey = _step_key(event, detailed=detailed_ui)
                if ekey != cur_key:
                    cur_key = ekey
                    step_t0 = perf_counter()
                    spinner_state["step_t0"] = step_t0
                    spinner_state["label"] = _status_label(event, detailed=detailed_ui)

                # Keep spinner label current during long tool executions.
                # Progress-style feedback ("Running tool… (45s)") updates the
                # spinner; agent-generated <feedback> tags are rendered inline.
                if etype == "feedback" and not detailed_ui and content:
                    if not _has_agent_tags(content):
                        spinner_state["label"] = _strip_agent_tags(content).splitlines()[0][:80]

                if etype == "solution":
                    final_solution = content
                should_render = (
                    detailed_ui and args.debug_raw_events
                ) or (
                    etype in {"thought", "feedback", "update", "warning", "error", "tool_plan", "tool_output"}
                    and bool(content)
                )
                if should_render:
                    _print_event(
                        event,
                        content,
                        run_t0,
                        step_t0,
                        detailed=detailed_ui,
                        render_state=render_state,
                        debug_raw_events=args.debug_raw_events,
                    )
    except Exception as exc:
        error_message = str(exc)
        _clear_line()
        _println(f"  {_red('Error:')} {error_message}")
    finally:
        if events_handle is not None:
            events_handle.close()
        spinner_stop.set()
        spinner_thread.join(timeout=0.5)
        root_logger.handlers = old_handlers
        sink.finalize()
        if detailed_ui:
            sink.render_collapsed()

    total = _format_elapsed(perf_counter() - run_t0)

    if final_solution:
        clean_solution = _strip_agent_tags(final_solution).strip() or final_solution.strip()
        _println()
        _println(f"  {_c(_HRULE_60, '90')}")
        _println()
        for line in clean_solution.splitlines():
            _println(f"  {line}")
        _println()
    _println(f"  {_dim(total)}")
    _println()

    if artifacts_enabled and artifacts_dir is not None and result_path is not None and metadata_path is not None:
        trajectory_path = _resolve_trajectory_path(agent) or trajectory_path
        duration_seconds = round(perf_counter() - run_t0, 3)
        result_payload = {
            "solution": (final_solution or "").strip(),
            "event_count": event_count,
            "status": "error" if error_message else "ok",
            "error": error_message or None,
            "chat_id": agent.chat_id,
            "query_id": agent.query_id,
            "trajectory_path": str(trajectory_path) if trajectory_path else None,
            "duration_seconds": duration_seconds,
        }
        _write_json(result_path, result_payload)
        metadata_payload = {
            "query": query,
            "model": args.model,
            "max_steps": args.max_steps,
            "chat_id": agent.chat_id,
            "query_id": agent.query_id,
            "started_at": started_at,
            "finished_at": _iso_now(),
            "cwd": str(Path.cwd()),
            "config_path": args.config or None,
            "app_runtime": args.app_runtime or None,
            "execution_backend": args.execution_backend or None,
            "workspace_source_root": args.workspace_source_root or None,
            "workspace_mount_path": args.workspace_mount_path or None,
        }
        _write_json(metadata_path, metadata_payload)
        artifacts_for_collection = {
            "events": events_path,
            "result": result_path,
            "metadata": metadata_path,
        }
        if trajectory_path is not None:
            artifacts_for_collection["trajectory"] = trajectory_path
        collection_dir = _resolve_collection_dir(args)
        if collection_dir is not None:
            manifest_path = _collect_artifacts(artifacts_for_collection, collection_dir)
            _println(_dim(f"  Artifacts collected to {collection_dir} (manifest: {manifest_path.name})"))
        try:
            display_artifacts = str(artifacts_dir.relative_to(Path.cwd()))
        except ValueError:
            display_artifacts = str(artifacts_dir)
        _println(_dim(f"  Saved to {display_artifacts}"))

    return 0 if not error_message else 1


async def _interactive_loop(args: argparse.Namespace) -> int:
    chat_id = args.chat_id or _default_id("chat")
    query_id = args.query_id or _default_id("query")
    try:
        agent, init_time = _init_agent_quiet(args, chat_id, query_id)
    except AgentInitError as exc:
        _print_agent_init_failure(exc)
        return 1
    _print_startup_banner(
        agent,
        args.config,
        init_time,
        detailed=_show_detailed_ui(args),
    )

    while True:
        try:
            query = input(f"  {_c('faraday', '1;96')}> ").strip()
        except (EOFError, KeyboardInterrupt):
            _println()
            return 0

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return 0

        await _run_once(args, query, agent=agent, init_time=init_time)


async def _amain() -> int:
    # Load a .env file if present so users can store API keys there without
    # manually exporting them. Existing shell variables always take precedence.
    # We intentionally follow the standard `.env` convention rather than any
    # project-specific dotenv filename.
    try:
        from dotenv import load_dotenv as _load_dotenv

        _load_dotenv(override=False)
    except ImportError:
        pass  # python-dotenv not installed; rely on shell environment

    _bootstrap_faraday_config_from_argv()

    args = parse_args()
    args.seed_conversation_history = None

    if args.debug:
        # Debug profile: emit the fullest runtime visibility by default.
        args.show_agent_logs = True
        args.debug_raw_events = True
        args.debug_include_transient_events = True

    # Launch policy is separate from runtime behavior: when config asks for
    # docker launch and we're currently on host, re-exec via docker automatically.
    if not args.use_docker:
        launch_mode = get_runtime_launch(default="host")
        running_in_app_docker = _is_truthy_env(os.getenv("FARADAY_RUNNING_IN_APP_DOCKER"))
        if launch_mode == "docker" and not running_in_app_docker:
            _println(
                _dim(
                    f"  runtime.launch=docker: relaunching inside Docker image "
                    f"'{args.docker_image or 'faraday-oss'}'"
                )
            )
            return _docker_exec(args)

    if args.use_docker:
        return _docker_exec(args)

    if not args.config:
        resolved_config = get_runtime_config_path()
        if resolved_config is not None:
            args.config = str(resolved_config)

    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.is_file():
            raise SystemExit(f"error: config file not found: {cfg_path}")
        os.environ["FARADAY_CONFIG"] = str(cfg_path)

    # Default query from YAML when the CLI positional is omitted (CLI wins when both are set).
    if not args.query:
        config_query = get_string_value("query", default=None)
        if config_query:
            args.query = config_query

    if args.check_tools:
        chat_id = args.chat_id or _default_id("chat")
        query_id = args.query_id or _default_id("query")
        agent = None
        init_error = ""
        try:
            agent, _ = _init_agent_quiet(args, chat_id, query_id)
        except (Exception, SystemExit) as exc:
            init_error = str(exc)
        return _run_check_tools(agent, args=args, init_error=init_error)

    _preflight_check_docker_image(args)

    if args.previous_context:
        try:
            payload, resolved_trajectory_path = _load_trajectory_payload(args.previous_context)
            args.seed_conversation_history = _trajectory_to_conversation_history(payload)
        except Exception as exc:
            raise SystemExit(f"Invalid --previous-context value: {exc}") from exc

        _println(
            _dim(
                "  Loaded prior context from trajectory file: "
                f"{resolved_trajectory_path}"
            )
        )

    try:
        batch_queries = _resolve_batch_queries(args)
    except Exception as exc:
        raise SystemExit(f"Invalid batch configuration: {exc}") from exc

    if batch_queries:
        return await _run_batch(args, batch_queries)

    if args.query:
        return await _run_once(args, args.query)
    return await _interactive_loop(args)


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))
