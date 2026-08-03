from typing import TypedDict, Union, Dict, Any


"""Tool definitions and schemas for FixCommand."""


class ReadFileArgs(TypedDict):
    filename: str


class ApplyFixArgs(TypedDict):
    filename: str
    line_number: int
    patch: str


READ_FILE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {"type": "string", "description": "Path to the file to read"},
    },
    "required": ["filename"],
}

APPLY_FIX_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {"type": "string", "description": "Path to the file to fix"},
        "line_number": {"type": "integer", "description": "1-indexed line number where the fix starts"},
        "patch": {"type": "string", "description": "Replacement code for the specified line"},
    },
    "required": ["filename", "line_number", "patch"],
}

# Immutable tuple of tool schemas (OpenAI SDK accepts any sequence)
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file by filename.",
            "parameters": READ_FILE_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_fix",
            "description": "Apply a code patch to a specific line in a file.",
            "parameters": APPLY_FIX_SCHEMA,
        },
    },
)


class ToolCallResult(TypedDict):
    tool: str
    arguments: Union[ReadFileArgs, ApplyFixArgs]
    # result field is placeholder - actual results returned by execute_tool_call
