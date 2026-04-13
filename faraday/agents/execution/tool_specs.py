EXECUTE_PYTHON_TOOL_DESCRIPTION = """
Execute Python code in a secure sandbox environment.

The runtime persists state across calls in a session, so defined variables and imports
can be reused in later executions.
"""

EXECUTE_BASH_CODE_TOOL_DESCRIPTION = """
Execute bash commands in a secure sandbox environment.

Use for filesystem operations, text processing, and short command execution tasks.
"""

execute_python_code_tool = {
    "type": "function",
    "name": "execute_python_code",
    "description": EXECUTE_PYTHON_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "strict": True,
        "properties": {
            "python_code": {
                "type": "string",
                "description": "Valid python code to execute in the sandbox.",
            },
        },
        "required": ["python_code"],
    },
}

execute_bash_code_tool = {
    "type": "function",
    "name": "execute_bash_code",
    "description": EXECUTE_BASH_CODE_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "strict": True,
        "properties": {
            "bash_code": {
                "type": "string",
                "description": "Valid bash code to execute in the sandbox.",
            },
        },
        "required": ["bash_code"],
    },
}

CODE_TOOLS = [
    execute_python_code_tool,
    execute_bash_code_tool,
]

CODE_TOOLS_DICT = {
    "execute_python_code": execute_python_code_tool,
    "execute_bash_code": execute_bash_code_tool,
}
