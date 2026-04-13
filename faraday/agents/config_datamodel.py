
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional

@dataclass
class FaradayAgentConfig:
    """Configuration for constructing a `FaradayAgent`."""
    model: str = "gpt-5"
    user_id: Optional[str] = None
    mode: str = "dev"
    conversation_history: Optional[List[Dict[str, Any]]] = None
    chat_id: Optional[str] = None
    query_id: Optional[str] = None
    verbose: bool = True
    enable_sandbox: bool = True
    lazy_sandbox: bool = True
    enable_caching: bool = True
    enable_parallel_tools: bool = True
    debug_print: bool = False
    task_complexity: str = ""
    agent_needed_boolean: str = ""
    eval_model: bool = False
    enable_auto_hotfix: bool = True
    enable_context_precheck: bool = True
    max_turns_per_step: int = 2
    reasoning_level: str = "medium"
    project_label: str = ""
    prompt_configuration: str = ""
    disabled_tools: Optional[List[str]] = None
    extra_tools: Optional[List[Dict[str, Any]]] = None
    extra_tool_handlers: Optional[Mapping[str, Callable[..., Any]]] = None
    save_conversation: bool = True
    app_runtime: Optional[str] = None
    execution_backend: Optional[str] = None
    workspace_source_root: Optional[str] = None
    workspace_mount_path: Optional[str] = None
    cloud_storage_root: Optional[str] = None
    trajectory_path: Optional[str] = None
    max_tokens: int = 8000
    max_total_steps: int = 50

