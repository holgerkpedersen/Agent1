import json
from typing import List, Dict, Any, cast

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

        # Handle unknown tool names gracefully
        try:
            result = _build_tool_call_result(tool_name, parsed_args)
            results.append(result)
        except ValueError:
            continue

    return results


def execute_tool_call(call: ToolCallResult) -> str:
    """Dispatch and execute a single tool call."""
    tool = call["tool"]
    args = call["arguments"]

    if tool == "read_file":
        # Type check to ensure args is dict-like for safe access
        if isinstance(args, dict):
            return _execute_read_file(args["filename"])
        raise ValueError("Invalid arguments for read_file")

    elif tool == "apply_fix":
        # Ensure args is dict-like for safe access of keys that exist in ApplyFixArgs
        if isinstance(args, dict):
            try:
                line_number = int(args.get("line_number", 0))
            except (ValueError, TypeError):
                line_number = 0
            return _execute_apply_fix(
                args["filename"], line_number, args["patch"]
            )
        raise ValueError("Invalid arguments for apply_fix")

    raise ValueError(f"Unknown tool: {tool}")


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
        # Handle potentially malformed line numbers by validating first
        raw_line_num = parsed_args.get("line_number")
        if isinstance(raw_line_num, int) or (isinstance(raw_line_num, str) and raw_line_num.isdigit()):
            line_number = int(raw_line_num)
        else:
            line_number = 0

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
    except IOError as exc:
        return f"Error reading file: {exc}"


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
    except IOError as exc:
        return f"Error applying fix: {exc}"
