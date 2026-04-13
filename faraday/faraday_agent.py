
from datetime import datetime, timezone
import asyncio
import copy
import inspect
import queue
import re
import json
import time
import os
import shutil
import traceback
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, TYPE_CHECKING
import hashlib
import uuid
from dataclasses import dataclass, replace

from tenacity import (
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
    AsyncRetrying
)

from faraday.agents.config_datamodel import FaradayAgentConfig

# Import the base code execution agent
from faraday.agents.code_execution_agent import (
    BaseCodeExecutionAgent,
    tprint,
    CODE_TOOLS,
    CODE_TOOLS_DICT
)
from faraday.agents.sandbox.config import SandboxState

from faraday.logging.updates import (
    AgentMessage,
    MessageType,
    _get_current_timestamp
)


######################################################
# PROMPTS - GPT-5 Family
######################################################

from faraday.prompts.agent_system_prompt import create_configurable_prompt_main
from faraday.prompts.agent_system_prompt_initial import create_configurable_prompt_initial
from faraday.utils.summarize_utils import get_headline_summary
from faraday.config import get_backend_value, get_bool_value, get_path_value, get_run_output_dir, get_string_value


######################################################
# IMPORT TOOLS
######################################################


######################################################
# PARALLEL TOOL EXECUTION CONFIGURATION
######################################################

SEQUENTIAL_TOOLS = {
    "execute_python_code",
    "execute_bash_code",
}

MAX_PARALLEL_TOOL_CONCURRENCY = 3

_CODE_TOOL_MAP: dict[str, tuple[str, str]] = {
    "execute_python_code": ("python_code", "python"),
    "execute_bash_code":   ("bash_code",   "bash"),
}


def is_parallel_safe(tool_name: str) -> bool:
    """Determine if a tool can be safely run in parallel."""
    return tool_name not in SEQUENTIAL_TOOLS



######################################################
# TOOL REGISTRY (lazy) + RAG / ENV HELPERS
######################################################

# Disabled when memory RAG tools are off (see FaradayAgent tool filtering).
MEMORY_RAG_TOOL_NAMES = frozenset({
    "search_memory",
    "search_chat",
    "get_project_summary",
    "search_filebase",
})

_tool_registry_cache: Optional[tuple[list, dict]] = None


@dataclass
class ToolResult:
    """Normalized internal result for all tool executions."""
    output: str
    has_error: bool = False


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_rag_backend() -> str:
    """Normalize `rag` backend config: auto | in-memory | none."""
    raw = get_backend_value("rag", default="auto")
    if raw in {"in_memory", "inmemory"}:
        return "in-memory"
    if raw in {"auto", "in-memory", "none"}:
        return raw
    return "auto"


def _load_tool_registry() -> tuple[list, dict]:
    """Built-in search tools (`faraday.search`) + code tools, merged once per process.

    Per-instance tools are merged in :meth:`FaradayAgent._get_filtered_tools` and
    :meth:`FaradayAgent._merged_tools_dict` via ``extra_tools`` / ``extra_tool_handlers``.
    """
    global _tool_registry_cache
    if _tool_registry_cache is None:
        from faraday.search import ALL_SEARCH_TOOLS, SEARCH_TOOL_DICT

        tools = ALL_SEARCH_TOOLS + CODE_TOOLS
        by_name = SEARCH_TOOL_DICT | CODE_TOOLS_DICT
        _tool_registry_cache = (tools, by_name)
    return _tool_registry_cache


def get_all_faraday_agent_tools() -> list:
    return _load_tool_registry()[0]


def get_all_faraday_agent_tools_dict() -> dict:
    return _load_tool_registry()[1]


######################################################
# MAIN MODEL CLASS
######################################################

class FaradayAgent(BaseCodeExecutionAgent):
    def __init__(self, config: Optional[FaradayAgentConfig] = None, **config_overrides: Any):
        if config is None:
            config = FaradayAgentConfig()
        elif not isinstance(config, FaradayAgentConfig):
            raise TypeError("config must be a FaradayAgentConfig instance or None")

        # override config with passed in args
        if config_overrides:
            config = replace(config, **config_overrides)

        self.config = config

        # Initialize parent class
        super().__init__(
            user_id=config.user_id,
            chat_id=config.chat_id,
            query_id=config.query_id,
            verbose=config.verbose,
            enable_sandbox=config.enable_sandbox,
            lazy_sandbox=config.lazy_sandbox,
            enable_auto_hotfix=config.enable_auto_hotfix,
            mode=config.mode,
            project_label=config.project_label,
            save_conversation=config.save_conversation,
            app_runtime=config.app_runtime,
            execution_backend=config.execution_backend,
            workspace_source_root=config.workspace_source_root,
            workspace_mount_path=config.workspace_mount_path,
            cloud_storage_root=config.cloud_storage_root,
        )
        
        self.model = config.model
        # Re-initialize usage tracker with model name (model is set after parent __init__)
        if self.usage_tracker:
            # Update the existing tracker's model
            self.usage_tracker.model = config.model
        self.app_version = 0.80  # V8 version
        self.disabled_tools = config.disabled_tools
        self.extra_tools = list(config.extra_tools) if config.extra_tools else []
        self.extra_tool_handlers = dict(config.extra_tool_handlers) if config.extra_tool_handlers else {}
        self.save_conversation = config.save_conversation
        self.tz = timezone.utc
        self.max_tokens = config.max_tokens
        self.reasoning_level = config.reasoning_level  # GPT-5 specific

        # Initialize LLM client
        self._init_llm_client()
        
        self.max_total_steps = config.max_total_steps
        self.enable_caching = config.enable_caching
        self.enable_parallel_tools = config.enable_parallel_tools
        self.debug_print = config.debug_print
        self.task_complexity = config.task_complexity
        self.agent_needed_boolean = config.agent_needed_boolean
        self.final_check_boolean = False 
        self.error_early_stop = False
        self.enable_context_precheck = config.enable_context_precheck
        self.task_string = ""
        self.eval_model = config.eval_model
        self.max_turns_per_step = config.max_turns_per_step
        self.project_label = config.project_label or ""

        # Memory/file RAG tools can be driven by a SkyPilot-like YAML config:
        # backends:
        #   rag: auto | in-memory | none
        rag_backend = _resolve_rag_backend()
        if rag_backend == "in-memory":
            in_memory_default = True
        elif rag_backend == "none":
            in_memory_default = False
        else:
            # Auto defaults to in-process RAG now that external vector storage is removed.
            in_memory_default = True

        self.in_memory_rag_enabled = _env_flag("ENABLE_IN_MEMORY_RAG", in_memory_default)
        memory_tools_default = self.in_memory_rag_enabled
        self.memory_rag_enabled = _env_flag("ENABLE_MEMORY_RAG_TOOLS", memory_tools_default)

        # Filter tools
        self.active_tools = self._get_filtered_tools()
        
        # Step tracking
        self.step_ready_for_evaluation = False
        self.reasoning_trajectory = None
        self.current_step = None
        self.current_step_index = 1
        self.current_step_position = 0
        self.problem_keywords = None
        self.all_steps_completed = False
        self.step_status = {}
        self._finalization_done = False
        self._usage_record_finalized = False
        self.tag_path = []
        
        # Progress tracking
        self._meaningful_work_done_this_iteration = False
        self._previous_context_hash = None
        self._previous_file_state_hash = None
        self._last_assessment_step_position = -1
        self._step_context_lengths = {}
        
        # Sandbox management (handled by base class)
        self.sandbox_pool = None
        self._sandbox_verified = False  
        self._last_sandbox_check = 0   
        self._sandbox_check_interval = 30  
        self._sandbox_consecutive_failures = 0  
        self._max_consecutive_failures = 2
        self.loop_index = 0
        
        # Token tracking
        self.total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        # Setup system prompt
        self.prompt_configuration = config.prompt_configuration
        self.full_system_prompt = self._setup_system_prompt()

        # Initialize conversation history
        self.conversation_history = []
        if config.conversation_history:
            self.conversation_history.extend(config.conversation_history)
        
        # Iteration tracking
        self.step_iteration_count = {}
        self._step_analysis_tasks = {}
        self._skipped_steps_for_updates = []
        
        # Initialize trajectory storage (sync config only - no network calls)
        _storage_start = time.time()
        self._init_trajectory_storage(trajectory_path=config.trajectory_path)
        tprint(f"[INIT TIMING] Storage config init: {time.time() - _storage_start:.3f}s")
        
        # Track async initialization state
        self._async_init_done = False
        self._history_restored = bool(self.conversation_history)

    def _init_llm_client(self):
        """Initialize the OpenAI LLM client with upfront config validation."""
        from faraday.openai_clients import (
            llm_client,
            validate_client_config,
            LLMConfigError,
        )

        try:
            warnings = validate_client_config()
        except LLMConfigError as exc:
            tprint(f"  ✗ LLM configuration error:\n    {exc}")
            raise SystemExit(1) from exc

        if warnings:
            for w in warnings:
                tprint(f"  ✗ LLM config problem: {w}")
            raise SystemExit(
                f"Aborting — {len(warnings)} LLM configuration "
                f"problem(s) detected. Fix the issues above and retry."
            )

        self.llm_client = llm_client
        print(f'✓ Initialized OpenAI client for model: {self.model}')

    def _setup_system_prompt(self) -> str:
        """Build the default full system prompt (see `self.prompt_configuration` for future variants)."""
        return create_configurable_prompt_main()

    def _get_prompt_tools_and_model_for_loop(self) -> tuple[str, list, str]:
        """Select prompt/tools/model per loop iteration for V8 GPT-5."""
        if self.loop_index == 0:
            return create_configurable_prompt_initial(), [], "gpt-5.2"
        return create_configurable_prompt_main(), self.active_tools, self.model

    def _merged_tools_dict(self) -> dict:
        """Built-in registry plus per-instance handlers (adds or overrides search tools)."""
        return get_all_faraday_agent_tools_dict() | self.extra_tool_handlers

    def _get_filtered_tools(self) -> list:
        """Filter out disabled tools."""
        disabled = set(self.disabled_tools or [])
        if not self.memory_rag_enabled:
            disabled.update(MEMORY_RAG_TOOL_NAMES)

        base_tools = get_all_faraday_agent_tools() + self.extra_tools

        if not disabled:
            return base_tools

        return [tool for tool in base_tools if tool.get("name") not in disabled]

    def dprint(self, *args, **kwargs):
        """Debug print helper."""
        if self.debug_print:
            print(*args, **kwargs)

    ######################################################
    # OPENAI-SPECIFIC METHODS
    ######################################################

    def _format_conversation_for_openai(self, conversation_history: List[Dict]) -> List[Dict]:
        """
        Convert internal conversation history to OpenAI Responses API input format.
        
        Internal format stores:
        - {"type": "function_call", "name": "...", "arguments": "...", "call_id": "..."}
        - {"type": "function_call_output", "call_id": "...", "output": "..."}
        - {"role": "user"/"assistant", "content": "..."}
        
        Responses API input format:
        - Regular messages: {"role": "user/assistant", "content": "text"}
        - Function calls: {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
        - Function outputs: {"type": "function_call_output", "call_id": "...", "output": "..."}
        
        Also validates that every function_call_output has a matching function_call,
        removing orphaned outputs to prevent OpenAI API errors.
        """
        # First pass: collect all function_call call_ids
        valid_call_ids = set()
        for entry in conversation_history:
            if entry.get("type") == "function_call":
                call_id = entry.get("call_id")
                if call_id:
                    valid_call_ids.add(call_id)
        
        openai_messages = []
        orphaned_outputs_removed = 0
        
        for entry in conversation_history:
            entry_type = entry.get("type")
            
            if entry_type == "function_call":
                # Responses API format for function calls
                arguments = entry.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                
                openai_messages.append({
                    "type": "function_call",
                    "call_id": entry["call_id"],
                    "name": entry["name"],
                    "arguments": arguments
                })
                
            elif entry_type == "function_call_output":
                # Validate that this output has a matching function_call
                call_id = entry.get("call_id")
                if call_id not in valid_call_ids:
                    orphaned_outputs_removed += 1
                    self.dprint(f"[FORMAT WARNING] Removing orphaned function_call_output with call_id: {call_id}")
                    continue
                
                # Responses API format for function outputs
                output = entry.get("output", "")
                if not isinstance(output, str):
                    output = str(output)
                
                openai_messages.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output
                })
                
            else:
                # Regular message (user/assistant)
                role = entry.get("role")
                content = entry.get("content")
                
                # Ensure content is never None
                if content is None:
                    content = ""
                
                # Handle list content (convert to string)
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = " ".join(text_parts)
                
                # Only add messages with a role and non-empty content
                if role and content:
                    openai_messages.append({
                        "role": role,
                        "content": content
                    })
        
        if orphaned_outputs_removed > 0:
            self.dprint(f"[FORMAT WARNING] Removed {orphaned_outputs_removed} orphaned function_call_output(s) without matching function_call")
        self.dprint(f"[RESPONSES API FORMAT] Converted {len(conversation_history)} entries to {len(openai_messages)} input items")
        return openai_messages

    async def _unpack_openai_response(self, response):
        """Unpack and process OpenAI/GPT-5 model response, yielding messages."""
        # print(f'response: {response}')
        if hasattr(response, 'output') and response.output:
            # GPT-5 responses API format
            total_items = len(response.output)
            item_types = {}
            
            for item in response.output:
                item_type = item.type
                item_types[item_type] = item_types.get(item_type, 0) + 1
            
            self.dprint(f'Processing OpenAI response with {total_items} output items')
            self.dprint(f'Item type breakdown: {dict(item_types)}')

            for i, output_item in enumerate(response.output):
                item_id = getattr(output_item, 'id', 'N/A')
                content_count = len(output_item.content) if hasattr(output_item, 'content') and output_item.content else 0
                self.dprint(f'Output item {i}: type={output_item.type}, id={item_id}, content_items={content_count}')
                
                if output_item.type == "reasoning":
                    # GPT-5 reasoning output
                    self.dprint(f'Reasoning item with {content_count} content pieces')
                    if output_item.content:
                        for j, content_item in enumerate(output_item.content):
                            text_length = len(content_item.text) if hasattr(content_item, 'text') else 0
                            self.dprint(f'  Reasoning content {j}: text_length={text_length}')
                            
                            # Add to conversation history
                            self.conversation_history.append({
                                "role": "assistant",
                                "content": content_item.text
                            })

                            # Yield reasoning message
                            reasoning_content = content_item.text
                            message = await self._yield_and_save_message(AgentMessage(
                                chat_id=self.chat_id,
                                query_id=self.query_id,
                                type=MessageType.UPDATE,
                                content=reasoning_content,
                                short_summary=get_headline_summary(reasoning_content),
                                step_index=self.current_step_index,
                                current_step=self.current_step or "1",
                                role="assistant",
                                loop_index=self.loop_index
                            ).model_dump())
                            yield message
                        
                elif output_item.type == "message":
                    self.dprint(f'Message item with {content_count} content pieces')
                    if output_item.content:
                        for j, content_item in enumerate(output_item.content):
                            if hasattr(content_item, 'text'):
                                response_content = content_item.text
                                self.dprint(f'  Message content {j}: text_length={len(response_content)}')
                                
                                # Add to conversation history
                                if response_content:
                                    self.conversation_history.append({
                                        "role": "assistant",
                                        "content": response_content
                                    })
                                    self.dprint(f'[CONVERSATION HISTORY] Added GPT-5 response')
                                
                                async for result in self.process_response_tags_parallel(response_content):
                                    yield result
                                    
                elif output_item.type == "function_call":
                    # Collect all function calls and process them together
                    function_calls = [item for item in response.output if item.type == "function_call"]
                    function_names = [getattr(fc, 'name', 'unknown') for fc in function_calls]
                    self.dprint(f'Function call item - collecting {len(function_calls)} total function calls: {function_names}')
                    
                    # Format for internal handling
                    formatted_calls = []
                    for fc in function_calls:
                        formatted_calls.append({
                            'name': fc.name,
                            'arguments': json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments,
                            'call_id': fc.call_id
                        })
                    
                    async for message in self._handle_tool_calls(formatted_calls):
                        yield message
                    break  # Exit loop after processing all function calls
                    
                else:
                    self.dprint(f'Unknown output type: {output_item.type}')
            
            # Fallback to output_text if no items were processed
            if not any(item.type in ["message", "function_call"] for item in response.output):
                self.dprint('No processable items found, falling back to output_text')
                if hasattr(response, 'output_text') and response.output_text:
                    async for result in self.process_response_tags_parallel(response.output_text):
                        yield result
        
        elif hasattr(response, 'choices') and response.choices:
            # Standard OpenAI chat completions format
            choice = response.choices[0]
            message = choice.message
            
            # Handle tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                formatted_calls = []
                for tc in message.tool_calls:
                    formatted_calls.append({
                        'name': tc.function.name,
                        'arguments': json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                        'call_id': tc.id
                    })
                
                async for msg in self._handle_tool_calls(formatted_calls):
                    yield msg
            
            # Handle text content
            elif hasattr(message, 'content') and message.content:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content
                })
                async for result in self.process_response_tags_parallel(message.content):
                    yield result

    ######################################################
    # LLM EXECUTION
    ######################################################

    async def _execute_llm_and_process_response(self, context_window: List[Dict]):
        """Execute LLM call and process the response, yielding messages."""
        self.dprint("\n\n" + "="*100 + f"\nMain LLM call ({self.model})\n" + "="*100 + "\n\n")
        
        # Format for OpenAI
        openai_messages = self._format_conversation_for_openai(context_window)
        
        if self.debug_print:
            print(f'\n\n\nopenai_messages:')
            for msg in openai_messages[:5]:
                print(f'{msg}')
            print(f'\n\n\n')
        
        system_prompt, tools, model = self._get_prompt_tools_and_model_for_loop()

        # Build API params
        api_params = {
            "model": model,
            "input": openai_messages,
            "instructions": system_prompt,
            "tools": tools,
        }
        
        # Add reasoning for GPT-5 models
        if model in ["gpt-5", "gpt-5-mini", "gpt-5.2", "gpt-5.2-codex"]:
            api_params['reasoning'] = {'effort': self.reasoning_level}
        elif model == "gpt-4.1":
            api_params['temperature'] = 0.5

        print(f'api_params: {api_params}')
        
        # Call OpenAI API
        try:
            response = self.llm_client.responses.create(**api_params)
        except Exception as api_err:
            import openai as _openai
            if isinstance(api_err, _openai.APIStatusError):
                from faraday.openai_clients import diagnose_api_error
                diagnosis = diagnose_api_error(api_err)
                raise type(api_err)(
                    f"{api_err}\n\n--- Troubleshooting ---\n{diagnosis}",
                    response=api_err.response,
                    body=getattr(api_err, "body", None),
                ) from api_err
            raise

        print(f'✓ LLM call completed for model: {self.model}')

        print(f'response: {response}')
        
        # Extract and track usage from response
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0) or 0
            output_tokens = getattr(usage, 'output_tokens', 0) or 0
            
            # Update cumulative totals
            self.total_usage["input_tokens"] += input_tokens
            self.total_usage["output_tokens"] += output_tokens
            
            self.dprint(f"[USAGE] input_tokens={input_tokens}, output_tokens={output_tokens}, total={input_tokens + output_tokens}")
            
            # Log to usage tracker
            if self.usage_tracker:
                self.usage_tracker.log_llm_call(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time=0
                )
        
        # Process response
        async for message in self._unpack_openai_response(response):
            yield message

    async def call_llm_with_circuit_breaker(self, **kwargs):
        """LLM call with retry mechanism."""
        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(min=2, max=120),
            stop=stop_after_attempt(8)
        ):
            with attempt:
                return await self._call_llm_impl(**kwargs)

    async def _call_llm_impl(self, **kwargs):
        """Implementation of LLM call."""
        response = self.llm_client.responses.create(**kwargs)
        
        # Extract usage - OpenAI uses prompt_tokens/completion_tokens OR input_tokens/output_tokens
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            # Try OpenAI standard names first, then fallback to alternative names
            input_tokens = getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', 0) or 0
            
            # Update cumulative totals
            self.total_usage["input_tokens"] += input_tokens
            self.total_usage["output_tokens"] += output_tokens
            
            # Log to usage tracker
            if self.usage_tracker:
                model = kwargs.get('model', 'unknown')
                self.usage_tracker.log_llm_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    execution_time=0
                )
        
        return response

    ######################################################
    # TOOL HANDLING
    ######################################################

    async def handle_multiple_tool_calls(self, tool_calls: List[Dict]) -> List[ToolResult]:
        """Handle multiple tool calls with smart parallel execution."""
        if not self.enable_parallel_tools or len(tool_calls) == 1:
            return [
                await self.handle_single_tool_call(tc['name'], tc['arguments'], tc['call_id'])
                for tc in tool_calls
            ]

        parallel_tools, sequential_tools = [], []
        for i, tc in enumerate(tool_calls):
            (parallel_tools if is_parallel_safe(tc['name']) else sequential_tools).append((i, tc))

        self.dprint(f"[PARALLEL TOOLS] {len(parallel_tools)} parallel-safe, {len(sequential_tools)} sequential")

        results_by_index: Dict[int, ToolResult] = {}

        if parallel_tools:
            semaphore = asyncio.Semaphore(MAX_PARALLEL_TOOL_CONCURRENCY)

            async def run_with_semaphore(idx: int, tool_call: Dict):
                async with semaphore:
                    return idx, await self.handle_single_tool_call(
                        tool_call['name'], tool_call['arguments'], tool_call['call_id']
                    )

            parallel_results = await asyncio.gather(
                *[run_with_semaphore(idx, tc) for idx, tc in parallel_tools],
                return_exceptions=True,
            )
            for task_idx, item in enumerate(parallel_results):
                original_idx, tool_name = parallel_tools[task_idx][0], parallel_tools[task_idx][1]['name']
                if isinstance(item, Exception):
                    results_by_index[original_idx] = ToolResult(
                        output=f"Error in tool {tool_name}: {item}", has_error=True
                    )
                else:
                    results_by_index[original_idx] = item[1]

        for idx, tool_call in sequential_tools:
            results_by_index[idx] = await self.handle_single_tool_call(
                tool_call['name'], tool_call['arguments'], tool_call['call_id']
            )

        return [results_by_index[i] for i in range(len(tool_calls))]

    async def handle_single_tool_call(self, toolname: str, tool_args: dict, tool_call_id: str) -> ToolResult:
        """Handle individual tool call."""
        start_time = time.time()
        result = await self._execute_tool(toolname, tool_args, tool_call_id)
        if self.usage_tracker:
            self.usage_tracker.log_tool_call(toolname, tool_args, time.time() - start_time)
        return result

    def _coerce_tool_result(self, raw_result: Any) -> ToolResult:
        """Normalize tool outputs to a single internal shape."""
        if isinstance(raw_result, ToolResult):
            return raw_result
        if isinstance(raw_result, dict):
            if "output" in raw_result:
                return ToolResult(output=str(raw_result["output"]), has_error=bool(raw_result.get("has_error", False)))
            if "markdown_output" in raw_result:
                return ToolResult(output=str(raw_result["markdown_output"]), has_error=bool(raw_result.get("has_error", False)))
            if "error" in raw_result:
                return ToolResult(output=str(raw_result["error"]), has_error=True)
        return ToolResult(output=str(raw_result))

    async def _execute_tool(self, tool_name: str, tool_args: dict, tool_call_id: str) -> ToolResult:
        """Execute a single tool."""
        self.dprint(f'tool_name: {tool_name}')
        self.dprint(f'tool_args: {tool_args}')

        if tool_name in _CODE_TOOL_MAP:
            arg_key, code_type = _CODE_TOOL_MAP[tool_name]
            code_content = tool_args.get(arg_key, "")
            if not code_content.strip():
                return ToolResult(
                    output=f"Error: No {code_type} code provided. Received args: {list(tool_args.keys())}",
                    has_error=True,
                )
            return await self._handle_code_execution_tool(code_content=code_content, code_type=code_type)

        tools_dict = self._merged_tools_dict()
        if tool_name not in tools_dict:
            raise ValueError(f'Tool {tool_name} not found')
        function_handler = tools_dict[tool_name]

        params = inspect.signature(function_handler).parameters
        if 'chat_id' in params:
            tool_args['chat_id'] = self.chat_id
        # Many tool schemas intentionally omit `mode`, so we inject it here.
        if 'mode' in params and 'mode' not in tool_args:
            tool_args['mode'] = self.mode

        try:
            if asyncio.iscoroutinefunction(function_handler):
                result = await function_handler(**tool_args)
            else:
                result = function_handler(**tool_args)
        except Exception as e:
            return ToolResult(output=f"Error executing tool {tool_name}: {str(e)}", has_error=True)

        return self._coerce_tool_result(result)

    async def _handle_code_execution_tool(self, code_content: str, code_type: str = "python") -> ToolResult:
        """Handle code execution tool calls."""
        self._mark_meaningful_work_done()

        if self.verbose:
            tprint(f"Executing {code_type} code...")
        
        # Execute code
        if code_type == "python":
            code_execution_task = asyncio.create_task(
                self._execute_single_code_block_python({'content': code_content})
            )
        else:
            code_execution_task = asyncio.create_task(
                self._execute_single_code_block_bash({'content': code_content})
            )
        
        code_result = await code_execution_task
        enhanced_output_message = code_result['markdown_output']
        
        if self.verbose:
            if code_result['has_error']:
                tprint(f'Code execution error: {code_result["stderr"]}')
            else:
                tprint(f'Code executed successfully. Output: {code_result["stdout"][:200] if code_result["stdout"] else "No output"}...')
        
        return ToolResult(
            output=enhanced_output_message,
            has_error=bool(code_result.get("has_error", False)),
        )

    async def _handle_tool_calls(self, tool_calls_output):
        """Handle tool calls and yield progress messages."""
        tool_calls = tool_calls_output if isinstance(tool_calls_output, list) else [tool_calls_output]
        
        # Format calls
        formatted_calls = []
        for call in tool_calls:
            if isinstance(call, dict):
                formatted_calls.append(call)
            else:
                args = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
                formatted_calls.append({
                    'name': call.name,
                    'arguments': args,
                    'call_id': call.call_id
                })
        
        tool_progress_ids = [hashlib.sha256(str(call.get('arguments', '')).encode()).hexdigest()[:16] for call in formatted_calls]
        id2id_map = {call['call_id']: pid for pid, call in zip(tool_progress_ids, formatted_calls)}
        
        # Add tool calls to conversation history and yield plan messages
        for call, tool_progress_id in zip(formatted_calls, tool_progress_ids):
            call_name = call['name']
            call_arguments = call['arguments']
            call_id = call['call_id']
            
            sensitive_keys = {"user_id", "chat_id", "userId", "chatId"}
            if isinstance(call_arguments, dict):
                sanitized_args = {k: v for k, v in call_arguments.items() if k not in sensitive_keys}
            else:
                sanitized_args = call_arguments
            
            arguments_str = json.dumps(sanitized_args) if isinstance(sanitized_args, dict) else str(sanitized_args)
            
            # Add to conversation history
            self.conversation_history.append({
                "type": "function_call",
                "name": call_name,
                "arguments": call_arguments,
                "call_id": call_id
            })
            self.dprint(f"[CONVERSATION HISTORY] Added function_call: {call_name} (id: {call_id})")

            # Create tool plan message
            if "code" in call_name.lower():
                code_type = "bash" if 'bash' in call_name.lower() else "python"
                code = sanitized_args.get(f'{code_type}_code', '') if isinstance(sanitized_args, dict) else ''
                tool_plan_content = f"```{code_type}\n{code or sanitized_args}```"
            else:
                tool_plan_content = f"## Running {call_name} tool\n## Tool arguments: \n"
                if isinstance(sanitized_args, dict):
                    for param, value in sanitized_args.items():
                        tool_plan_content += f"\t- {param}: {value}\n"
                else:
                    tool_plan_content += f"\t-Raw arguments: {sanitized_args}\n"

            message = await self._yield_and_save_message(AgentMessage(
                chat_id=self.chat_id,
                query_id=self.query_id,
                type=MessageType.TOOL_PLAN,
                content=tool_plan_content,
                display_tool_name=call_name,
                short_summary=get_headline_summary(f"Running {call_name}..."),
                progress_id=str(tool_progress_id),
                step_index=self.current_step_index,
                current_step="1",
                role="assistant",
                loop_index=self.loop_index,
                tool_call_id=call_id,
                tool_call_name=call_name,
                tool_call_arguments=arguments_str
            ).model_dump())
            yield message
        
        # Execute tools
        self._mark_meaningful_work_done()

        tool_results = None
        execution_error = None

        # Hook into sandbox progress so status messages surface during long executions
        _progress_queue: queue.Queue = queue.Queue()
        _original_sb_callback = None
        if hasattr(self, 'sandbox_manager') and self.sandbox_manager is not None:
            _original_sb_callback = getattr(self.sandbox_manager, 'progress_callback', None)
            self.sandbox_manager.set_progress_callback(
                lambda msg, _q=_progress_queue: _q.put_nowait(msg)
            )

        _tool_names_str = ", ".join(c['name'] for c in formatted_calls)
        _exec_start = time.time()
        exec_task = asyncio.create_task(self.handle_multiple_tool_calls(formatted_calls))

        try:
            while not exec_task.done():
                done, _ = await asyncio.wait({exec_task}, timeout=3.0)
                if exec_task.done():
                    break
                elapsed = time.time() - _exec_start
                # Drain any sandbox progress messages queued from background threads
                sb_msgs = []
                while not _progress_queue.empty():
                    try:
                        sb_msgs.append(_progress_queue.get_nowait())
                    except Exception:
                        break
                detail = f" \u2014 {sb_msgs[-1]}" if sb_msgs else ""
                progress_content = f"Running {_tool_names_str}... ({elapsed:.0f}s){detail}"
                progress_msg = await self._yield_and_save_message(AgentMessage(
                    chat_id=self.chat_id,
                    query_id=self.query_id,
                    type=MessageType.FEEDBACK,
                    content=progress_content,
                    short_summary=progress_content,
                    step_index=self.current_step_index,
                    current_step="1",
                    role="assistant",
                    loop_index=self.loop_index,
                    transient=True,
                ).model_dump())
                yield progress_msg
        finally:
            if _original_sb_callback is not None and hasattr(self, 'sandbox_manager') and self.sandbox_manager is not None:
                self.sandbox_manager.set_progress_callback(_original_sb_callback)

        try:
            tool_results = await exec_task
        except Exception as e:
            execution_error = e
            self.dprint(f"⚠ [TOOL EXECUTION ERROR] {str(e)}")
            tool_results = [f"Error: Tool execution failed - {str(e)}" for _ in formatted_calls]
        
        # Add results to conversation history
        for i, result in enumerate(tool_results):
            call_id = formatted_calls[i]['call_id']
            tool_name = formatted_calls[i]['name']
            tool_progress_id = id2id_map[call_id]
            
            result_output = result.output
            
            self.conversation_history.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result_output
            })
            self.dprint(f"[CONVERSATION HISTORY] Added function_call_output for: {tool_name} (id: {call_id})")
            
            msg_type = MessageType.ERROR if execution_error or result.has_error else MessageType.TOOL_OUTPUT
            
            message = await self._yield_and_save_message(AgentMessage(
                chat_id=self.chat_id,
                query_id=self.query_id,
                type=msg_type,
                content=result_output,
                metadata=f"tool_call_id: {call_id}",
                display_tool_name=tool_name,
                short_summary=get_headline_summary(result_output),
                progress_id=tool_progress_id,
                step_index=self.current_step_index,
                current_step="1",
                role="assistant",
                loop_index=self.loop_index,
                tool_call_id=call_id
            ).model_dump())
            yield message
        
   
    ######################################################
    # RESPONSE TAG PROCESSING
    ######################################################

    async def process_response_tags_parallel(self, response_content: str):
        """Process response tags with parallel execution where possible."""
        self.tag_path = []
        self.dprint(f'\n\nraw response_content: {response_content}\n\n')

        # (msg_type, log_type, mark_work_done, wrap_tag)
        TAG_CONFIG = {
            'thought':    (MessageType.THOUGHT,  MessageType.THOUGHT,  False, 'thought'),
            'feedback':   (MessageType.FEEDBACK, MessageType.THOUGHT,  True,  'feedback'),
            'reflection': (MessageType.THOUGHT,  MessageType.THOUGHT,  True,  'thought'),
            'solution':   (MessageType.SOLUTION, MessageType.SOLUTION, False, 'solution'),
        }

        all_matches = []
        for tag_type in TAG_CONFIG:
            for match in re.finditer(rf'<{tag_type}>(.*?)</{tag_type}>', response_content, re.DOTALL):
                all_matches.append({'type': tag_type, 'start': match.start(), 'content': match.group(1)})

        # Handle unclosed <solution> tag
        if '<solution>' in response_content and not any(m['type'] == 'solution' for m in all_matches):
            solution_start = response_content.find('<solution>')
            content_after_tag = response_content[solution_start + len('<solution>'):].replace('</solution>', '').strip()
            if content_after_tag:
                all_matches.append({'type': 'solution', 'start': solution_start, 'content': content_after_tag})

        all_matches.sort(key=lambda x: x['start'])

        num_tags_in_response = len(all_matches)
        self.dprint(f'\n\nnumber of tags in response: {num_tags_in_response}\n\n')

        for match_info in all_matches:
            tag_type = match_info['type']
            content = match_info['content']
            self.tag_path.append(tag_type)

            msg_type, log_type, mark_work, wrap_tag = TAG_CONFIG[tag_type]

            try:
                if mark_work:
                    self._mark_meaningful_work_done()

                if self.verbose and tag_type == 'thought':
                    tprint(f'Thought: {content}')

                if self.usage_tracker:
                    self.usage_tracker.log_agent_message(log_type, content)

                if tag_type == 'solution':
                    if self.usage_tracker:
                        self.usage_tracker.complete_usage_record("completed")
                        self._usage_record_finalized = True
                    self.all_steps_completed = True

                result = AgentMessage(
                    chat_id=self.chat_id,
                    query_id=self.query_id,
                    type=msg_type,
                    content=content,
                    short_summary=get_headline_summary(content),
                    step_index="1",
                    current_step="1",
                    role="assistant",
                    loop_index=self.loop_index
                )

                if isinstance(result.content, str) and not result.content.strip().startswith(f'<{wrap_tag}>'):
                    result.content = f'<{wrap_tag}>{result.content}</{wrap_tag}>'

                message = await self._yield_and_save_message(result.model_dump())
                yield message

            except Exception as e:
                self.dprint(f'Error processing {tag_type} tag: {str(e)}')
                self.dprint(f'Traceback: {traceback.format_exc()}')
                error_msg = f'Error processing {tag_type} tag: {str(e)}'
                if self.verbose:
                    tprint(error_msg)
                message = await self._yield_and_save_message(AgentMessage(
                    chat_id=self.chat_id,
                    query_id=self.query_id,
                    type=MessageType.ERROR,
                    content=error_msg,
                    step_index="1",
                    current_step="1",
                    role="assistant",
                    loop_index=self.loop_index
                ).model_dump())
                yield message

        if self.verbose and num_tags_in_response > 0:
            tprint(f'Finished processing {num_tags_in_response} tags.')

    ######################################################
    # MAIN RUN LOOP
    ######################################################

    def _reset_iteration_tracking(self):
        """Reset tracking variables for a new iteration."""
        self._meaningful_work_done_this_iteration = False
        self.step_ready_for_evaluation = False

    def _mark_meaningful_work_done(self):
        """Mark that meaningful work was done in this iteration."""
        self._meaningful_work_done_this_iteration = True

    def _strip_context_tags_from_user_query(self, task_string: str) -> str:
        """Remove internal context tags from user-visible query text."""
        if not isinstance(task_string, str) or not task_string:
            return task_string

        cleaned = task_string
        cleaned = re.sub(r'<files_highlighted_by_user>.*?</files_highlighted_by_user>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'</?file_highlighted_by_user>', '', cleaned)
        cleaned = re.sub(r'</?files_highlighted_by_user>', '', cleaned)
        cleaned = re.sub(r'<user_attached_files>.*?</user_attached_files>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<workspace_context>.*?</workspace_context>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'</?user_query>', '', cleaned)
        cleaned = re.sub(r'<additional_context>.*?</additional_context>', '', cleaned, flags=re.DOTALL)

        cleaned = cleaned.strip()
        return cleaned if cleaned else task_string

    def _extract_user_query_text(self, content: str) -> str:
        """Extract content inside <user_query>...</user_query> if present."""
        if not isinstance(content, str) or not content:
            return content
        match = re.search(r'<user_query>(.*?)</user_query>', content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

    def _setup_initial_context(self, task_string: str, user_added_files: Optional[List[str]] = None):
        """Setup the initial context for the agent."""
        user_added_files = user_added_files or []
        parts = []
        
        parts.append(f"<user_query>{task_string}</user_query>")
        parts.append(f"<workspace_context>{'project' if self.project_label else 'standalone'}</workspace_context>")

        # Add user-attached files to context
        if user_added_files:
            files_lines = ["<user_attached_files>"]
            files_lines.append("The user has attached the following files with this query:")
            for filepath in user_added_files:
                files_lines.append(f"  - {filepath}")
            files_lines.append("</user_attached_files>")
            parts.append("\n".join(files_lines))


        full_initial_context = "\n".join(parts)
        return full_initial_context

    def _reset_per_run_state(self):
        """Reset state that must be fresh for each call to run().

        Without this, interactive-mode (multi-turn) sessions break because
        flags like all_steps_completed carry over from the previous query.
        """
        self.all_steps_completed = False
        self._finalization_done = False
        self._usage_record_finalized = False
        self.error_early_stop = False
        self.final_check_boolean = False
        self.loop_index = 0
        self.current_step_index = 1
        self.current_step_position = 0
        self.tag_path = []
        self._meaningful_work_done_this_iteration = False
        self.step_ready_for_evaluation = False

    async def run(self, task_string: str, user_added_files: Optional[List[str]] = None):
        """Main execution method."""
        self._reset_per_run_state()

        st_time = time.time()
        self.dprint(f'Starting task: {task_string}')
        user_added_files = user_added_files or []

        self.task_string = task_string
        self.revised_task = task_string

        # Initialize async components before adding the new user message.
        await self._initialize_run_session(task_string)

        initial_context = self._setup_initial_context(task_string, user_added_files)
        self.conversation_history.append({"role": "user", "content": initial_context})
        print("\033[95m[CONVERSATION HISTORY] Added initial context\033[0m")

        display_task = self._strip_context_tags_from_user_query(task_string)
        user_message = await self._yield_and_save_message(AgentMessage(
            chat_id=self.chat_id,
            query_id=self.query_id,
            type=MessageType.UPDATE,
            content=display_task,
            short_summary=get_headline_summary(display_task),
            role="user",
            loop_index=self.loop_index,
            step_index="0",
            current_step="initial"
        ).model_dump())
        yield user_message

        skip_usage_update = False
        try:
            for loop_index in range(self.max_total_steps):
                self.loop_index = loop_index
                self._reset_iteration_tracking()

                self.dprint(f'\033[92m(.run) executing main loop iteration\033[0m')
                async for message in self._execute_main_loop_iteration():
                    yield message

                if self.all_steps_completed:
                    elapsed = time.time() - st_time
                    self.dprint(f'\033[92m\n\nALL STEPS COMPLETED in {elapsed:.2f}s\n\n\033[0m')
                    break
            else:
                async for message in self._handle_max_steps_reached(self.loop_index):
                    yield message
                self._finalize_usage("completed")
                skip_usage_update = True

        except Exception as e:
            error_context = f"Error in main loop at iteration {self.loop_index}"
            self.dprint(f"Exception context: {error_context}")
            self._finalize_usage("failed", str(e))
            async for message in self._handle_execution_error(e, error_context):
                yield message
            skip_usage_update = True

        finally:
            await self._finalize_run_session(skip_usage_update=skip_usage_update)

    def _finalize_usage(self, status: str, error_message: str = None):
        """Finalize the usage record if not already done."""
        if self.usage_tracker and not self._usage_record_finalized:
            try:
                self.usage_tracker.complete_usage_record(status, error_message)
            except Exception as tracker_error:
                self.dprint(f"Failed to update usage record: {tracker_error}")
            finally:
                self._usage_record_finalized = True

    async def _initialize_run_session(self, task_string: str):
        """Initialize the run session."""
        await self.initialize_async()
        
        if self.usage_tracker:
            try:
                usage_record_id = self.usage_tracker.create_usage_record(task_string)
                self.dprint(f"Created usage record: {usage_record_id}")
            except Exception as e:
                tb = traceback.format_exc()
                tprint(f"[USAGE] Failed to create usage record: {e}")
                tprint(f"[USAGE] create_usage_record traceback:\n{tb}")

    def _dprint_conversation_history(self):
        self.dprint(f'\033[93mconversation_history: {len(self.conversation_history)} entries\033[0m')
        for i, entry in enumerate(self.conversation_history):
            content = entry.get("content", "")
            if content:
                preview = content[:100] if isinstance(content, str) else str(content)[:100]
                self.dprint(f'\033[93m\t-entry {i}: {preview}\033[0m')
            else:
                self.dprint(f'\033[93m\t-entry {i}: Tool Call\033[0m')

    async def _execute_main_loop_iteration(self):
        """Execute a single iteration of the main loop."""
        self.dprint(f'\n\n\n----------------------start of loop-----------------------\n\n\n')
        self._dprint_conversation_history()
        async for message in self._execute_llm_and_process_response(self.conversation_history):
            yield message

        self.dprint(f'\n\n\n----------------------end of loop-----------------------\n\n\n')

    async def _handle_execution_error(self, e: Exception, context: str = None):
        """Handle execution errors during the main loop."""
        import openai as _openai
        from faraday.openai_clients import diagnose_api_error

        if isinstance(e, RetryError):
            error_msg = "LLM API rate limit reached and retries exhausted"
        elif isinstance(e, _openai.APIStatusError):
            diagnosis = diagnose_api_error(e)
            error_msg = (
                f"LLM API error (HTTP {e.status_code}): {e}\n\n"
                f"--- Troubleshooting ---\n{diagnosis}"
            )
        else:
            full_traceback = traceback.format_exc()
            error_msg = f"Unexpected error: {str(e)}\n\nFull traceback:\n{full_traceback}"
        
        if context:
            error_msg = f"{context}\n{error_msg}"
        
        self.dprint(f'error_msg: {error_msg}')
        
        if self.usage_tracker and not self._usage_record_finalized:
            self.usage_tracker.complete_usage_record("failed", str(e)[:200])
            self._usage_record_finalized = True
        
        message = await self._yield_and_save_message(AgentMessage(
            chat_id=self.chat_id,
            query_id=self.query_id,
            type=MessageType.ERROR,
            content=error_msg,
            role="assistant",
            loop_index=self.loop_index
        ).model_dump())
    
        yield message

    async def _handle_max_steps_reached(self, loop_index: int):
        """Handle when maximum steps are reached."""
        self.dprint(f'Max loop index reached: {loop_index}. Generating graceful summary...')
        
        summary = "I've completed my analysis of this task. Please review the work done above. Let me know if you'd like me to continue exploring additional aspects."
        
        tagged_content = f"<solution>{summary}</solution>"
        
        message = await self._yield_and_save_message(AgentMessage(
            chat_id=self.chat_id,
            query_id=self.query_id,
            type=MessageType.SOLUTION,
            content=tagged_content,
            role="assistant",
            loop_index=self.loop_index
        ).model_dump())
        
        yield message

    async def _finalize_run_session(self, skip_usage_update: bool = False):
        """Finalize the run session."""
        if self._finalization_done:
            self.dprint("⚠️ Finalization already done, skipping")
            return
        
        self._finalization_done = True
        self.dprint("🔄 Starting finalization...")
        
        if not skip_usage_update and self.usage_tracker and not self._usage_record_finalized:
            try:
                if self.all_steps_completed:
                    self.usage_tracker.complete_usage_record("completed")
                else:
                    self.usage_tracker.complete_usage_record("incomplete")
                self._usage_record_finalized = True
            except Exception as e:
                self.dprint(f"Failed to complete usage record: {e}")
        
        await self._save_atif_trajectory()
        
        await self.cleanup()
        self.dprint("✅ Finalization complete")

    async def cleanup(self):
        """Cleanup resources."""
        await self._ensure_usage_record_finalized()
        await self._save_atif_trajectory()
        self._sync_agent_outputs_to_run_dir()
        
        await super().cleanup()
        
        if self.sandbox_pool and self.chat_id:
            await self.sandbox_pool.return_sandbox(self.chat_id)
        
        if self.usage_tracker:
            self.usage_tracker.cleanup()

    def _sync_agent_outputs_to_run_dir(self) -> None:
        """
        Mirror workspace-level ./agent_outputs into this run's agent_outputs directory.
        This is required for docker/local sandbox runs where tools write files to
        the sandbox working directory but result artifacts are expected under run_outputs/.
        """
        try:
            self.sync_runtime_agent_outputs_to_workspace()
        except Exception as exc:
            self.dprint(f"⚠ Failed to sync runtime outputs into workspace: {exc}")

        target_dir = getattr(self, "agent_outputs_dir", None)
        if target_dir is None:
            return

        target_dir = Path(target_dir)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        workspace_root = None
        sandbox_manager = getattr(self, "sandbox_manager", None)
        if sandbox_manager is not None:
            workspace_root = getattr(sandbox_manager, "workspace_root", None)

        if workspace_root is None:
            configured_workspace = getattr(self, "workspace_root", None)
            if configured_workspace:
                workspace_root = Path(configured_workspace).expanduser().resolve()
            else:
                workspace_root = Path.cwd().resolve()
        else:
            workspace_root = Path(workspace_root).expanduser().resolve()

        source_dir = workspace_root / "agent_outputs"
        if not source_dir.exists() or not source_dir.is_dir():
            return

        try:
            if source_dir.resolve() == target_dir.resolve():
                return
        except Exception:
            pass

        try:
            for source_path in source_dir.rglob("*"):
                relative = source_path.relative_to(source_dir)
                destination = target_dir / relative
                if source_path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
        except Exception as exc:
            self.dprint(f"⚠ Failed to sync agent_outputs to run dir: {exc}")

    async def _ensure_usage_record_finalized(self):
        """Ensure usage record has a terminal status."""
        if self._usage_record_finalized:
            return
        
        if not self.usage_tracker:
            return
        
        try:
            self.usage_tracker.complete_usage_record("failed", "Agent cleanup triggered without proper finalization")
            self._usage_record_finalized = True
        except Exception as e:
            self.dprint(f"Failed to finalize usage record during cleanup: {e}")

    def mark_as_failed(self, error_message: str = "Agent run failed"):
        """Mark the usage record as failed."""
        if self._usage_record_finalized:
            return
        
        if not self.usage_tracker:
            return
        
        try:
            self.usage_tracker.complete_usage_record("failed", error_message)
            self._usage_record_finalized = True
        except Exception as e:
            self.dprint(f"Failed to mark usage record as failed: {e}")

    def mark_as_killed(self, reason: str = "Terminated by user"):
        """Mark the usage record as killed."""
        if self._usage_record_finalized:
            return
        
        if not self.usage_tracker:
            return
        
        try:
            self.usage_tracker.complete_usage_record("killed", reason)
            self._usage_record_finalized = True
        except Exception as e:
            self.dprint(f"Failed to mark usage record as killed: {e}")

    ######################################################
    # STORAGE INITIALIZATION
    ######################################################

    def _init_trajectory_storage(self, trajectory_path: Optional[str] = None):
        """Initialize run output directories and ATIF trajectory export settings."""
        self._trajectory_events: List[Dict[str, Any]] = []
        self._trajectory_written = False
        self._trajectory_write_error: Optional[str] = None
        self._trajectory_output_path: Optional[Path] = None
        self._trajectory_live_output_path: Optional[Path] = None
        self._trajectory_last_saved_event_count = 0

        # persistence.trajectory_dir: when set, trajectories are written to that directory
        # with a per-run filename instead of the auto-generated run_output_root structure.
        # This lets YAML-only configs control trajectory location without needing the
        # trajectory_path constructor argument.
        _traj_dir_cfg = get_path_value("persistence", "trajectory_dir", default=None)
        self._trajectory_output_dir_override: Optional[Path] = (
            Path(_traj_dir_cfg).expanduser() if _traj_dir_cfg else None
        )

        # Warn on the still-deprecated exact-path key (exact path breaks batch runs)
        _dep_traj_path = get_path_value("persistence", "trajectory_path", default=None)
        if _dep_traj_path and self.verbose:
            tprint(
                "⚠ Config key `persistence.trajectory_path` is deprecated and ignored. "
                "Use `persistence.trajectory_dir` to set the output directory, or "
                "`outputs.root` to control the full run output tree."
            )

        yaml_enabled = get_bool_value("persistence", "atif_trajectory", default=True)
        env_enabled = os.getenv("ENABLE_ATIF_TRAJECTORY")
        if env_enabled is None:
            enabled = bool(yaml_enabled)
        else:
            enabled = env_enabled.strip().lower() not in {"0", "false", "no", "off"}
        self.trajectory_save_enabled = bool(self.save_conversation and enabled)

        # Build canonical run output directory: run_outputs/run_{ts}_{chat_id}_{run_id}/
        run_output_dir = get_run_output_dir()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        chat_token = (self.chat_id or "chat").replace("/", "_").replace(" ", "_")
        run_token = self.query_id.replace("/", "_").replace(" ", "_")
        self.run_output_root: Path = Path(run_output_dir) / f"run_{ts}_{chat_token}_{run_token}"
        self.agent_outputs_dir: Path = self.run_output_root / "agent_outputs"
        self.run_artifacts_dir: Path = self.run_output_root / "run_artifacts"

        # Explicit trajectory path override (e.g. passed via CLI --trajectory-path or batch runner)
        self.trajectory_path_override: Optional[str] = trajectory_path or None

        # Eagerly create output directories when trajectory saving is active
        if self.trajectory_save_enabled:
            try:
                self.run_output_root.mkdir(parents=True, exist_ok=True)
                self.agent_outputs_dir.mkdir(parents=True, exist_ok=True)
                self.run_artifacts_dir.mkdir(parents=True, exist_ok=True)
            except Exception as _dir_err:
                if self.verbose:
                    tprint(f"⚠ Could not create run output directories: {_dir_err}")
            if self._trajectory_output_dir_override is not None:
                try:
                    self._trajectory_output_dir_override.mkdir(parents=True, exist_ok=True)
                except Exception as _dir_err:
                    if self.verbose:
                        tprint(f"⚠ Could not create trajectory directory: {_dir_err}")

        # Legacy alias kept for any internal code that still reads trajectory_dir
        self.trajectory_dir = str(self.run_artifacts_dir)

    def _record_trajectory_event(self, message: dict):
        """Capture event payloads for ATIF trajectory export."""
        if not getattr(self, "trajectory_save_enabled", False):
            return
        if not isinstance(message, dict):
            return
        self._trajectory_events.append(copy.deepcopy(message))

    @staticmethod
    def _coerce_iso8601(ts: str) -> str:
        raw = str(ts or "").strip()
        if not raw:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            if " " in raw and "T" not in raw:
                parsed = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                return parsed.isoformat().replace("+00:00", "Z")
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat().replace("+00:00", "Z")
        except Exception:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _clean_message_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _is_transient_tool_feedback(event_type: str, content: str) -> bool:
        """Identify heartbeat-style tool progress updates for trajectory filtering."""
        if event_type not in {"feedback", "update"}:
            return False
        text = str(content or "").strip()
        if not text:
            return False
        return bool(re.match(r"^Running [A-Za-z0-9_,\s\-]+\.{3} \(\d+s\)(?:\s+—\s+.*)?$", text))

    def _build_atif_trajectory(self) -> Dict[str, Any]:
        """Convert captured Faraday events to ATIF-v1.4 JSON payload."""
        steps: List[Dict[str, Any]] = []
        pending_tool_steps: Dict[str, Dict[str, Any]] = {}

        for event in self._trajectory_events:
            role = str(event.get("role", "")).strip().lower()
            event_type = str(event.get("type", "")).strip().lower()
            source = "agent"
            if role == "user":
                source = "user"
            elif event_type == "status":
                source = "system"

            timestamp = self._coerce_iso8601(str(event.get("timestamp", "")))
            content = self._clean_message_text(event.get("content", ""))
            tool_call_id = str(event.get("tool_call_id", "")).strip()
            tool_name = str(event.get("tool_call_name") or event.get("display_tool_name") or "").strip()

            if self._is_transient_tool_feedback(event_type, content):
                continue

            if event_type == "tool_plan" and tool_call_id:
                arguments_raw = event.get("tool_call_arguments", "")
                if isinstance(arguments_raw, str):
                    try:
                        arguments = json.loads(arguments_raw) if arguments_raw.strip() else {}
                    except Exception:
                        arguments = {"raw": arguments_raw}
                elif isinstance(arguments_raw, dict):
                    arguments = arguments_raw
                else:
                    arguments = {}

                tool_step: Dict[str, Any] = {
                    "step_id": len(steps) + 1,
                    "timestamp": timestamp,
                    "source": "agent",
                    "message": content or f"Running tool {tool_name}",
                    "tool_calls": [
                        {
                            "tool_call_id": tool_call_id,
                            "function_name": tool_name or "unknown_tool",
                            "arguments": arguments,
                        }
                    ],
                }
                steps.append(tool_step)
                pending_tool_steps[tool_call_id] = tool_step
                continue

            if event_type in {"tool_output", "error"} and tool_call_id and tool_call_id in pending_tool_steps:
                tool_step = pending_tool_steps[tool_call_id]
                observation = tool_step.setdefault("observation", {"results": []})
                results = observation.setdefault("results", [])
                results.append(
                    {
                        "source_call_id": tool_call_id,
                        "content": content,
                    }
                )
                continue

            if not content:
                continue

            step: Dict[str, Any] = {
                "step_id": len(steps) + 1,
                "timestamp": timestamp,
                "source": source,
                "message": content,
            }
            if source == "agent" and event_type in {"thought", "planning"}:
                step["reasoning_content"] = content
            steps.append(step)

        total_prompt_tokens = int(self.total_usage.get("input_tokens", 0) or 0)
        total_completion_tokens = int(self.total_usage.get("output_tokens", 0) or 0)
        total_cached_tokens = int(self.total_usage.get("cache_read_input_tokens", 0) or 0)

        return {
            "schema_version": "ATIF-v1.4",
            "session_id": self.query_id or self.chat_id or f"session-{uuid.uuid4().hex[:8]}",
            "agent": {
                "name": "faraday",
                "version": str(self.app_version),
                "model_name": self.model,
                "extra": {
                    "chat_id": self.chat_id or "",
                    "query_id": self.query_id or "",
                    "mode": self.mode,
                },
            },
            "steps": steps,
            "final_metrics": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_cached_tokens": total_cached_tokens,
                "total_steps": len(steps),
            },
        }

    def _resolve_trajectory_output_path(self) -> Path:
        if self._trajectory_output_path is not None:
            return self._trajectory_output_path
        # Highest priority: explicit constructor / CLI argument
        if self.trajectory_path_override:
            self._trajectory_output_path = Path(self.trajectory_path_override).expanduser()
            return self._trajectory_output_path
        # YAML persistence.trajectory_dir: flat directory, one file per run
        if self._trajectory_output_dir_override is not None:
            run_token = self.query_id.replace("/", "_").replace(" ", "_")
            chat_token = (self.chat_id or "chat").replace("/", "_").replace(" ", "_")
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"trajectory_{chat_token}_{run_token}_{ts}.json"
            self._trajectory_output_path = self._trajectory_output_dir_override / filename
            return self._trajectory_output_path
        # Default: nested under the run output tree
        self._trajectory_output_path = self.run_artifacts_dir / "trajectory.json"
        return self._trajectory_output_path

    def _resolve_live_trajectory_output_path(self) -> Path:
        if self._trajectory_live_output_path is not None:
            return self._trajectory_live_output_path
        final_path = self._resolve_trajectory_output_path()
        suffix = final_path.suffix if final_path.suffix else ".json"
        live_name = f"{final_path.stem}.in_progress{suffix}"
        self._trajectory_live_output_path = final_path.with_name(live_name)
        return self._trajectory_live_output_path

    async def _write_atif_trajectory_snapshot(self, final: bool = False):
        """Persist a trajectory snapshot to disk during execution/finalization."""
        if not getattr(self, "trajectory_save_enabled", False) or not self._trajectory_events:
            return
        current_event_count = len(self._trajectory_events)
        if not final and current_event_count == self._trajectory_last_saved_event_count:
            return
        try:
            payload = self._build_atif_trajectory()
            output_path = (
                self._resolve_trajectory_output_path()
                if final
                else self._resolve_live_trajectory_output_path()
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                output_path.write_text,
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            self._trajectory_last_saved_event_count = current_event_count
            if final:
                self._trajectory_written = True
                live_path = self._resolve_live_trajectory_output_path()
                if live_path.exists():
                    await asyncio.to_thread(live_path.unlink)
            self._trajectory_write_error = None
            if self.verbose and final:
                tprint(f"ATIF trajectory written to: {output_path}")
        except Exception as e:
            self._trajectory_write_error = str(e)
            if self.verbose:
                tprint(f"Failed to write ATIF trajectory: {e}")

    async def _save_atif_trajectory(self):
        """Persist final ATIF trajectory JSON to disk."""
        await self._write_atif_trajectory_snapshot(final=True)

    ######################################################
    # ASYNC INITIALIZATION
    ######################################################

    async def initialize_async(self):
        """Async initialization — currently a no-op hook for subclass extensions."""
        if self._async_init_done:
            return
        self._async_init_done = True


    
    ######################################################
    # MESSAGE SAVING HELPERS
    ######################################################

    async def _yield_and_save_message(self, message: dict):
        """Helper to yield message and save to trajectory."""
        if not message.get("transient", False):
            self._record_trajectory_event(message)
            await self._write_atif_trajectory_snapshot()
        return self._get_user_facing_message(message)

    def _get_user_facing_message(self, message: dict) -> dict:
        """Return a user-facing version of the message content."""
        if not isinstance(message, dict):
            return message

        raw_content = message.get('content', '')
        if not isinstance(raw_content, str) or not raw_content:
            return message
        if message.get('role') == 'user':
            raw_content = self._extract_user_query_text(raw_content)

        markdown_content = re.sub(r'<additional_context>.*?</additional_context>', '', raw_content, flags=re.DOTALL)
        markdown_content = re.sub(r'<files_highlighted_by_user>.*?</files_highlighted_by_user>', '', markdown_content, flags=re.DOTALL)
        markdown_content = re.sub(r'</?(?:thought|solution|reflection|feedback|planning|execute|execute_python|execute_bash)>', '', markdown_content)
        markdown_content = markdown_content.strip()
        if markdown_content:
            markdown_content = '\n' + markdown_content

        if markdown_content == raw_content:
            return message

        return {**message, 'content': markdown_content}

    ######################################################
    # UTILITY METHODS
    ######################################################

    def list_available_tools(self, print_output: bool = True) -> List[Dict[str, str]]:
        """List all available tools."""
        tools_info = []
        
        for tool in self.active_tools:
            name = tool.get("name", "Unknown")
            description = tool.get("description", "No description available")
            tools_info.append({"name": name, "description": description})
        
        if print_output:
            print("\n" + "=" * 80)
            print(f"AVAILABLE TOOLS (GPT-5)")
            print("=" * 80)
            
            for i, tool_info in enumerate(tools_info, 1):
                print(f"\n{i:2d}. {tool_info['name']}")
                desc = tool_info['description']
                if len(desc) > 500:
                    desc = desc[:500] + "..."
                print(f"    {desc}")
            
            print("\n" + "=" * 80)
            print(f"Total: {len(tools_info)} tools available")
            print("=" * 80 + "\n")
        
        return tools_info

    def print_config(self):
        """Print agent configuration."""
        print("\n" + "="*70)
        print(f"{'FARADAY AGENT V8 GPT-5 CONFIGURATION':^70}")
        print("="*70)
        
        print(f"\n{'MODEL CONFIGURATION':^70}")
        print("-"*70)
        print(f"  Model:              {self.model}")
        print(f"  App Version:        {self.app_version}")
        print(f"  Reasoning Level:    {self.reasoning_level}")
        
        print(f"\n{'SESSION IDENTIFIERS':^70}")
        print("-"*70)
        print(f"  Chat ID:            {self.chat_id or 'None'}")
        print(f"  Query ID:           {self.query_id or 'None'}")
        
        print(f"\n{'FEATURE FLAGS':^70}")
        print("-"*70)
        print(f"  Caching:            {'✓ Enabled' if self.enable_caching else '✗ Disabled'}")
        print(f"  Parallel Tools:     {'✓ Enabled' if self.enable_parallel_tools else '✗ Disabled'}")
        print(f"  Sandbox:            {'✓ Enabled' if self.enable_sandbox else '✗ Disabled'}")
        print(f"  Execution Backend:  {self.execution_backend_name}")
        
        print("="*70 + "\n")
