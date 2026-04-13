from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from faraday.config import (
    get_execution_backend,
    get_launch_mode,
    get_path_value,
    get_workspace_mount_path,
    get_workspace_source_root,
)
from faraday.openai_clients import get_llm_model


EVENT_PREFIX = "FARADAY_EVENT\t"
RESULT_PREFIX = "FARADAY_RESULT\t"
DEFAULT_HARBOR_ARTIFACT_DIR = "/logs/artifacts"


def parse_args() -> argparse.Namespace:
    default_model = get_llm_model(default="gpt-5")
    default_launch_mode = get_launch_mode(default="host")
    default_execution_backend = get_execution_backend(default="docker")
    default_workspace_source_root = get_workspace_source_root(default="/workspace") or "/workspace"
    default_workspace_mount_path = get_workspace_mount_path(default="/workspace")
    default_harbor_artifacts_dir = (
        get_path_value("artifacts", "harbor_dir", default=DEFAULT_HARBOR_ARTIFACT_DIR)
        or DEFAULT_HARBOR_ARTIFACT_DIR
    )

    parser = argparse.ArgumentParser(description="Run Faraday inside a Harbor task.")
    parser.add_argument("--instruction", required=True, help="Task instruction to execute.")
    parser.add_argument(
        "--artifacts-dir",
        default=default_harbor_artifacts_dir,
        help="Artifact directory. Defaults to config `artifacts.harbor_dir` or /logs/artifacts.",
    )
    parser.add_argument("--events-path", default="", help="Optional JSONL path for emitted events.")
    parser.add_argument("--result-path", default="", help="Optional JSON path for the final result.")
    parser.add_argument("--metadata-path", default="", help="Optional JSON path for run metadata.")
    parser.add_argument("--trajectory-path", default="", help="Optional ATIF trajectory JSON output path.")
    parser.add_argument("--config", default="", help="Optional Faraday config path.")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--launch-mode", default=default_launch_mode)
    parser.add_argument("--execution-backend", default=default_execution_backend)
    parser.add_argument("--workspace-source", default=default_workspace_source_root)
    parser.add_argument("--execution-workspace-path", default=default_workspace_mount_path)
    return parser.parse_args()


def _write_json(path_value: str, payload: dict) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_event(path_value: str, event: dict) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _resolve_artifact_paths(args: argparse.Namespace) -> dict[str, str]:
    base_dir = args.artifacts_dir.strip() or (
        get_path_value("artifacts", "harbor_dir", default=DEFAULT_HARBOR_ARTIFACT_DIR)
        or DEFAULT_HARBOR_ARTIFACT_DIR
    )
    base = Path(base_dir).expanduser()
    return {
        "events_path": args.events_path or str(base / "events.jsonl"),
        "result_path": args.result_path or str(base / "result.json"),
        "metadata_path": args.metadata_path or str(base / "metadata.json"),
        "trajectory_path": args.trajectory_path or str(base / "trajectory.json"),
    }


async def _amain() -> int:
    args = parse_args()
    artifact_paths = _resolve_artifact_paths(args)
    if args.config:
        os.environ["FARADAY_CONFIG"] = args.config
    # Harbor runs may not provide chat/query identifiers. Ensure these are always
    # valid strings because Faraday's event models validate them as str.
    chat_id = f"harbor-chat{uuid.uuid4().hex[:12]}".strip()
    query_id = f"harbor-{uuid.uuid4().hex[:12]}".strip()

    events: list[dict] = []
    final_solution = ""
    error_message = ""
    agent = None

    try:
        from faraday.faraday_agent import FaradayAgent, FaradayAgentConfig

        agent = FaradayAgent(
            config=FaradayAgentConfig(
                model=args.model,
                chat_id=chat_id,
                query_id=query_id,
                max_total_steps=args.max_steps,
                verbose=False,
                debug_print=False,
                app_runtime=args.launch_mode or None,
                execution_backend=args.execution_backend or None,
                workspace_source_root=args.workspace_source or None,
                workspace_mount_path=args.execution_workspace_path or None,
                trajectory_path=artifact_paths["trajectory_path"] or None,
            )
        )
        async for event in agent.run(args.instruction):
            events.append(event)
            _append_event(artifact_paths["events_path"], event)
            if event.get("type") == "solution":
                final_solution = str(event.get("content", "")).strip()
            print(EVENT_PREFIX + json.dumps(event, ensure_ascii=True), flush=True)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        if agent is not None:
            try:
                await agent.cleanup()
            except Exception as cleanup_exc:
                if not error_message:
                    error_message = f"{type(cleanup_exc).__name__}: {cleanup_exc}"

    result_payload = {
        "solution": final_solution,
        "event_count": len(events),
        "error": error_message or None,
        "events_path": artifact_paths["events_path"],
        "result_path": artifact_paths["result_path"],
        "metadata_path": artifact_paths["metadata_path"],
        "trajectory_path": artifact_paths["trajectory_path"],
    }
    _write_json(artifact_paths["result_path"], result_payload)
    _write_json(
        artifact_paths["metadata_path"],
        {
            "instruction": args.instruction,
            "model": args.model,
            "max_steps": args.max_steps,
            "launch_mode": args.launch_mode,
            "execution_backend": args.execution_backend,
            "workspace_source": args.workspace_source,
            "execution_workspace_path": args.execution_workspace_path,
            "chat_id": chat_id,
            "query_id": query_id,
            "event_count": len(events),
        },
    )
    print(RESULT_PREFIX + json.dumps(result_payload, ensure_ascii=True), flush=True)

    if final_solution:
        print(final_solution, flush=True)
    return 0 if not error_message else 1


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
