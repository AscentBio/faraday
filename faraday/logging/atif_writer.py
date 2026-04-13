import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AtifTrajectoryWriter:
    """Collects agent events and writes an ATIF trajectory JSON file."""

    def __init__(
        self,
        *,
        session_id: str,
        agent_name: str,
        agent_version: str,
        model_name: str,
        output_dir: str,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        query_id: Optional[str] = None,
    ):
        self.session_id = session_id
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.model_name = model_name
        self.output_dir = output_dir
        self.user_id = user_id or ""
        self.chat_id = chat_id or ""
        self.query_id = query_id or ""

        self._steps: List[Dict[str, Any]] = []
        self._step_id = 0
        self._started_at = self._iso_now()
        self._system_prompt_saved = False
        self.output_path: Optional[str] = None

    def _iso_now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _to_iso_timestamp(self, timestamp: Any) -> str:
        if isinstance(timestamp, str) and timestamp:
            # AgentMessage timestamps use "%Y-%m-%d %H:%M:%S.%f" (UTC).
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            except ValueError:
                # If already ISO or unknown, keep as-is.
                return timestamp
        return self._iso_now()

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    def add_system_prompt(self, prompt: str) -> None:
        if self._system_prompt_saved:
            return
        self._system_prompt_saved = True
        self._steps.append(
            {
                "step_id": self._next_step_id(),
                "timestamp": self._iso_now(),
                "source": "system",
                "message": prompt or "",
                "extra": {},
            }
        )

    def add_message(self, message: Dict[str, Any]) -> None:
        msg_type = str(message.get("type") or "")
        role = str(message.get("role") or "")
        timestamp = self._to_iso_timestamp(message.get("timestamp"))
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = str(content)

        if role == "user":
            self._steps.append(
                {
                    "step_id": self._next_step_id(),
                    "timestamp": timestamp,
                    "source": "user",
                    "message": content,
                    "extra": {},
                }
            )
            return

        # Tool plan is represented as an agent step with a tool_calls entry.
        if msg_type == "tool_plan":
            call_id = message.get("tool_call_id") or ""
            function_name = message.get("tool_call_name") or message.get("display_tool_name") or "unknown_tool"
            raw_args = message.get("tool_call_arguments") or "{}"
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args) if raw_args.strip() else {}
                except Exception:
                    arguments = {"raw_arguments": raw_args}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            else:
                arguments = {"raw_arguments": str(raw_args)}

            self._steps.append(
                {
                    "step_id": self._next_step_id(),
                    "timestamp": timestamp,
                    "source": "agent",
                    "model_name": self.model_name,
                    "message": content,
                    "tool_calls": [
                        {
                            "tool_call_id": call_id,
                            "function_name": function_name,
                            "arguments": arguments,
                        }
                    ],
                    "extra": {},
                }
            )
            return

        # Tool outputs are represented as observations.
        if msg_type == "tool_output":
            call_id = message.get("tool_call_id") or ""
            self._steps.append(
                {
                    "step_id": self._next_step_id(),
                    "timestamp": timestamp,
                    "source": "agent",
                    "model_name": self.model_name,
                    "message": "",
                    "observation": {
                        "results": [
                            {
                                "source_call_id": call_id,
                                "content": content,
                            }
                        ]
                    },
                    "extra": {},
                }
            )
            return

        self._steps.append(
            {
                "step_id": self._next_step_id(),
                "timestamp": timestamp,
                "source": "agent",
                "model_name": self.model_name,
                "message": content,
                "extra": {},
            }
        )

    def to_dict(self, *, total_usage: Optional[Dict[str, int]] = None, notes: str = "") -> Dict[str, Any]:
        usage = total_usage or {}
        total_prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        total_completion_tokens = int(usage.get("output_tokens", 0) or 0)
        total_cached_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)

        return {
            "schema_version": "ATIF-v1.6",
            "session_id": self.session_id,
            "agent": {
                "name": self.agent_name,
                "version": self.agent_version,
                "model_name": self.model_name,
                "extra": {
                    "user_id": self.user_id,
                    "chat_id": self.chat_id,
                    "query_id": self.query_id,
                },
            },
            "notes": notes or "Faraday ATIF trajectory export",
            "final_metrics": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_cached_tokens": total_cached_tokens,
                "total_steps": len(self._steps),
                "extra": {
                    "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
                },
            },
            "steps": self._steps,
            "extra": {
                "started_at": self._started_at,
            },
        }

    def write(self, *, total_usage: Optional[Dict[str, int]] = None, notes: str = "") -> Optional[str]:
        if not self.output_dir:
            return None

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"trajectory_{self.session_id}_{timestamp}.json"
        path = os.path.join(self.output_dir, filename)

        payload = self.to_dict(total_usage=total_usage, notes=notes)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        self.output_path = path
        return path
