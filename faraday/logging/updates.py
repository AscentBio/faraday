from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import time
from datetime import datetime, timezone
import uuid
import threading

def _get_current_timestamp() -> str:
    """Factory function to generate current timestamp with millisecond precision in UTC timezone"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Remove last 3 digits to get milliseconds instead of microseconds

# Thread-safe counter for message ID generation
_message_counter_lock = threading.Lock()
_message_counter = 0

def _generate_message_id() -> str:
    """Factory function to generate unique message IDs"""
    global _message_counter
    with _message_counter_lock:
        _message_counter += 1
        timestamp_ms = int(time.time() * 1000)
        # Format: counter_timestamp_uuid
        return f"{_message_counter}_{timestamp_ms}_{uuid.uuid4().hex[:8]}"

MESSAGE_TYPES = [
    "thought",
    "code",
    "tool",
    "tool_plan",
    "tool_output",
    "tool_status",
    "data",
    "plot",
    "chat",
    "solution",
    "error",
    "status",
    "planning",
    "update",
    "step_solution",
    "warning",  # Added for timeout warnings
    "feedback",  # Added to allow feedback messages to be propagated
    
]

class MessageType(str, Enum):
    FEEDBACK = "feedback" # user facing messages that let the user know what's going on
    THOUGHT = "thought"
    CODE = "code"
    TOOL = "tool"
    TOOL_PLAN = "tool_plan"
    TOOL_OUTPUT = "tool_output"
    TOOL_STATUS = "tool_status"
    DATA = "data"
    PLOT = "plot"
    CHAT = "chat"
    SOLUTION = "solution"
    ERROR = "error"
    STATUS = "status"
    PLANNING = "planning"
    UPDATE = "update"
    STEP_SOLUTION = "step_solution"
    WARNING = "warning"
    
class AgentMessage(BaseModel):
    chat_id: str = Field(default="")
    user_id: str = Field(default="")
    query_id: str = Field(default="")
    type: MessageType = Field(default=MessageType.THOUGHT)
    content: str = Field(default="")
    timestamp: str = Field(default_factory=_get_current_timestamp)
    metadata: str = Field(default="")
    short_summary: str = Field(default="")
    display_tool_name: str = Field(default="")
    added_files: List[str] = Field(default=[])
    removed_files: List[str] = Field(default=[])
    focused_files: List[str] = Field(default=[])
    role: str = Field(default="")
    loop_index: int = Field(default=0)
    # reasoning_trajectory: List[str] = Field(default=[]) # for planning messages
    progress_id: str = Field(default="")
    step_index: int = Field(default=0)
    current_step: str = Field(default="")
    step_status: str = Field(default="awaiting")
    # template_urls: List[str] = Field(default=[])
    # message_data: Dict[List[str], List[str]] = Field(default={})
    message_id: str = Field(default_factory=_generate_message_id)
    
    # Tool call structured data (for proper LLM format reconstruction)
    tool_call_id: str = Field(default="")  # For tool calls and results
    tool_call_name: str = Field(default="")  # For tool calls
    tool_call_arguments: str = Field(default="")  # JSON string of arguments for tool calls
    transient: bool = Field(default=False)  # If True, emit live-only without persistence
    
    
    def model_dump(self, **kwargs):
        """Override model_dump to validate step_index consistency and convert MessageType to string"""
        data = super().model_dump(**kwargs)
        
        # Convert MessageType enum to string value
        if isinstance(data.get("type"), MessageType):
            data["type"] = data["type"].value
        
        # Validate step_index against reasoning_trajectory if both are present
        if data.get("step_index") and data.get("reasoning_trajectory"):
            step_index = data["step_index"]
            trajectory_length = len(data["reasoning_trajectory"])
            
            # Ensure step_index is within valid range (1-based indexing)
            if step_index < 1 or step_index > trajectory_length:
                print(f"Warning: step_index {step_index} not found in reasoning_trajectory (length: {trajectory_length})")
                # Adjust to valid range
                data["step_index"] = min(max(1, step_index), trajectory_length)
                
        return data



class PlanningMessage(AgentMessage):
    knowledge_categories: List[str] = Field(default=[])
    capabilities_required: List[str] = Field(default=[])
    data_sources_required: List[str] = Field(default=[])
    initial_planning_content: str = Field(default="")
    thought_process: str = Field(default="")
    task_statement: str = Field(default="")
    summary_reasoning_steps: List[str] = Field(default=[])
    