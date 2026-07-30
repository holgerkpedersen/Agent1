import json
from typing import List, Dict, Any

from ..tools.definitions import ApplyFixArgs, ReadFileArgs, ToolCallResult


def parse_tool_calls(response_message: Dict[str, Any]) -> List[ToolCallResult]:
    """Parse structured tool calls from LLM response."""
    results: List[ToolCallResult] = []

    raw_calls = response_message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return results

    for call in raw_calls:
        if not isinstance(call, dict):
            continue

        tool_name = str(call.get("name", ""))
        function_data = call.get("function", {})
        arguments_str = ""

        if isinstance(function_data, dict):
            arguments_str = function_data.get("arguments", "") or ""

        try:
            parsed_args = json.loads(arguments_str) if arguments_str else {}
        except (json.JSONDecodeError, TypeError):
            parsed_args = {}

        result = _build_tool_call_result(tool_name, parsed_args)
        results.append(result)

    return results


def execute_tool_call(call: ToolCallResult) -> str:
    """Dispatch and execute a single tool call."""
    if call["tool"] == "read_file":
        args = call["arguments"]
        filename = args["filename"]
        return _execute_read_file(filename)

    elif call["tool"] == "apply_fix":
        args = call["arguments"]
        filename = args["filename"]
        line_number = args["line_number"]
        patch = args["patch"]
        return _execute_apply_fix(filename, line_number, patch)

    raise ValueError(f"Unknown tool: {call['tool']}")


def _build_tool_call_result(tool_name: str, parsed_args: Dict[str, Any]) -> ToolCallResult:
    """Construct a typed ToolCallResult from parsed arguments."""
    if tool_name == "read_file":
        filename = str(parsed_args.get("filename", ""))
        return {
            "tool": "read_file",
            "arguments": {"filename": filename},
            "result": "",
        }

    elif tool_name == "apply_fix":
        filename = str(parsed_args.get("filename", ""))
        line_number = int(parsed_args.get("line_number", 0))
        patch = str(parsed_args.get("patch", ""))
        return {
            "tool": "apply_fix",
            "arguments": {"filename": filename, "line_number": line_number, "patch": patch},
            "result": "",
        }

    raise ValueError(f"Unknown tool: {tool_name}")


def _execute_read_file(filename: str) -> str:
    """Read file contents."""
    try:
        with open(filename, "r") as f:
            return f.read()
    except IOError:
        return ""


def _execute_apply_fix(filename: str, line_number: int, patch: str) -> str:
    """Apply a fix to the specified file (1-indexed line numbers)."""
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        idx = line_number - 1  # Convert 1-indexed to 0-indexed
        if 0 <= idx < len(lines):
            lines[idx] = patch + "\n"
        elif idx >= len(lines):
            lines.append(patch + "\n")

        with open(filename, "w") as f:
            f.writelines(lines)

        return ""
    except IOError:
        return ""
