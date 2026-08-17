"""Structured tool routing, schema validation, and LLM function-call parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Core Types & Exceptions
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class ToolExecutionError(Exception):
    """Raised when tool argument validation fails or routing encounters an unrecoverable state."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class RoutingError(Exception):
    """Raised when the router cannot match a prompt to a registered tool."""

    pass


# ---------------------------------------------------------------------------
# Tool Argument Models (Pydantic Validation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and JSON Schema for a single agent tool."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Relative or absolute path to the file within workspace")


class WriteFileArgs(BaseModel):
    path: str = Field(
        ..., description="Target file path for writing/overwriting"
    )
    content: str = Field(..., description="String content to write into the file")


class SearchFilesArgs(BaseModel):
    query: str = Field(..., description="Text or regex pattern to search for")
    file_pattern: str | None = Field(
        None, description="Glob filter like '*.py' or '*.md'"
    )


class ShellCommandArgs(BaseModel):
    command: str = Field(
        ..., description="Shell command string (sanitized externally)"
    )


# Registry mapping tool names -> Pydantic models for runtime validation
_VALIDATION_REGISTRY: dict[str, type[BaseModel]] = {
    "read_file": ReadFileArgs,
    "write_file": WriteFileArgs,
    "search_files": SearchFilesArgs,
    "run_command": ShellCommandArgs,
}

# Default tool schemas aligned with OpenAI function calling spec
DEFAULT_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description="Reads the content of a file within the workspace sandbox.",
        parameters_schema=ReadFileArgs.model_json_schema(),
    ),
    ToolDefinition(
        name="write_file",
        description="Writes or overwrites content in a workspace file.",
        parameters_schema=WriteFileArgs.model_json_schema(),
    ),
    ToolDefinition(
        name="search_files",
        description=(
            "Searches for text patterns across workspace files using "
            "grep/findstr."
        ),
        parameters_schema=SearchFilesArgs.model_json_schema(),
    ),
    ToolDefinition(
        name="run_command",
        description=(
            "Executes a sanitized shell command within the workspace environment."
        ),
        parameters_schema=ShellCommandArgs.model_json_schema(),
    ),
]


# ---------------------------------------------------------------------------
# Shell Command Handler
# ---------------------------------------------------------------------------

class ShellCommandHandler:
    """Handles execution of validated shell commands.

    Security decisions are DELEGATED to the shared modules — ``agent_core.
    security.allowlist`` (binary allow-list + structural shell-pattern
    rejection) — instead of the duplicated local allow-list / block-list of
    the old implementation (plan ARCH item 6).  Windows shell builtins
    (echo, dir, type, ...) have no ``.exe`` so they still run through the
    shell, but only AFTER the allow-list and metacharacter gates above —
    chaining, pipes, and redirection cannot survive those gates.
    """

    def execute(self, args: ShellCommandArgs) -> str | dict[str, Any]:
        """Execute a command after the shared security gates."""
        import shlex
        import subprocess
        from agent_core.security.allowlist import (
            find_unsafe_shell_pattern,
            is_command_allowed,
        )

        cmd = args.command.strip()
        if not cmd:
            raise ToolExecutionError(
                "Empty command", details={"command": cmd},
            )

        # Gate 1 — structural safety: no pipes / redirection / chaining.
        unsafe = find_unsafe_shell_pattern(cmd)
        if unsafe is not None:
            raise ToolExecutionError(
                f"Command '{cmd}' contains {unsafe}",
                details={"command": cmd},
            )

        # Gate 2 — parse and allow-list the binary (shared module).
        try:
            tokens = shlex.split(cmd)
        except ValueError as exc:
            raise ToolExecutionError(
                f"Invalid command: {cmd}",
                details={"command": cmd, "error": str(exc)},
            )
        if not tokens:
            raise ToolExecutionError(
                f"Invalid command: {cmd}", details={"command": cmd},
            )
        binary = Path(tokens[0]).name  # strip any path component for allowlist check
        if not is_command_allowed(binary):
            raise ToolExecutionError(
                f"Command '{cmd}' is not in the allowed command list",
                details={"command": cmd},
            )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return {"stdout": result.stdout.strip(), "returncode": 0}
            return {
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            raise ToolExecutionError(
                f"Command timed out after 30s: {cmd}",
                details={"command": cmd},
            )


# ---------------------------------------------------------------------------
# Router Implementation
# ---------------------------------------------------------------------------

class ToolRouter:
    """Routes LLM responses or natural language prompts to validated tool executions."""

    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {
            t.name: t for t in (tools or DEFAULT_TOOL_DEFINITIONS)
        }
        self._handlers: dict[str, Callable[[BaseModel], Any]] = {}

        # Register built-in handlers
        shell_handler = ShellCommandHandler()
        self.register_handler("run_command", lambda args: shell_handler.execute(args))  # type: ignore[arg-type]

    def register_tool(self, definition: ToolDefinition) -> None:
        """Register a new tool definition with the router."""
        if definition.name not in _VALIDATION_REGISTRY:
            raise ToolExecutionError(
                f"Validation model missing for tool '{definition.name}'"
            )
        self._tools[definition.name] = definition

    def register_handler(
        self, tool_name: str, handler: Callable[[BaseModel], Any]
    ) -> None:
        """Bind a runtime handler function to a tool name."""
        if tool_name not in self._tools:
            raise ToolExecutionError(
                f"Cannot bind handler to unregistered tool '{tool_name}'"
            )
        self._handlers[tool_name] = handler

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas for LLM injection."""
        return [t.to_openai_format() for t in self._tools.values()]

    # ------------------------------------------------------------------
    # Parsing Strategies
    # ------------------------------------------------------------------

    def parse_openai_function_call(
        self, raw_response: dict[str, Any]
    ) -> tuple[str, BaseModel]:
        """Parse OpenAI-style function call object and validate arguments."""
        func = raw_response.get("function")
        if not func:
            raise RoutingError("Missing 'function' key in LLM response")

        tool_name = func["name"]
        args_json = func.get("arguments", "{}")

        matched_name = self._resolve_tool_name(tool_name)
        return matched_name, self._validate_args(matched_name, args_json)

    def parse_explicit_command(self, prompt: str) -> tuple[str, BaseModel]:
        """Parse deterministic ``/tool:name key=value`` style commands."""
        match = re.match(
            r"^/tool:(\w+)\s+(.+)$", prompt.strip(), re.IGNORECASE
        )
        if not match:
            raise RoutingError(
                "Invalid explicit command format. Expected: /tool:<name> <args>"
            )

        tool_name, args_str = match.group(1), match.group(2)
        matched_name = self._resolve_tool_name(tool_name.lower())

        args_dict = self._parse_kv_string(args_str)
        return matched_name, self._validate_args(matched_name, args_dict)

    def parse_natural_language(self, prompt: str) -> tuple[str, BaseModel]:
        """Regex-backed fallback for varied natural language phrasing."""
        patterns: dict[str, str] = {
            r"\b(read|open|view)\s+(?:the\s+)?file[:\s]+(?P<path>\S+)": "read_file",
            (
                r"\b(write|save|create|overwrite)\s+(?:to\s+)?(?:the\s+)?file"
                r"[:\s]+(?P<path>\S+)(?:\s*with\s*(?P<content>.+))?$"
            ): "write_file",
            (
                r"\b(search|find|grep)\s+(?:for\s+)?['\"]?(?P<query>[^'\"]+)['\"]?"
                r"\s*(?:in\s+file[:\s]+(?P<file_pattern>\S+))?"
            ): "search_files",
        }

        for pattern, tool_name in patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                args_dict = {
                    k: v for k, v in match.groupdict().items() if v is not None
                }
                matched_name = self._resolve_tool_name(tool_name)
                return matched_name, self._validate_args(matched_name, args_dict)

        raise RoutingError(
            "Could not route natural language prompt to any registered tool"
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _resolve_tool_name(self, raw_name: str) -> str:
        """Fuzzy match input name against registered tools."""
        normalized = raw_name.lower().strip()
        if normalized in self._tools:
            return normalized

        # Prefix match: "read" matches "read_file"
        for tool_name in self._tools:
            if tool_name.startswith(normalized) or normalized.startswith(tool_name):
                return tool_name

        # Levenshtein distance for typo tolerance
        def levenshtein(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            prev = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr = [i + 1]
                for j, c2 in enumerate(s2):
                    curr.append(
                        prev[j + 1]
                        if c1 == c2
                        else min(prev[j], prev[j + 1], curr[-1]) + 1
                    )
                prev = curr
            return prev[-1]

        closest = min(self._tools.keys(), key=lambda k: levenshtein(k, normalized))
        if levenshtein(closest, normalized) <= 2:
            return closest

        raise RoutingError(
            f"Unknown tool '{raw_name}'. Available: {list(self._tools.keys())}"
        )

    def _validate_args(
        self, tool_name: str, raw_args: str | dict[str, Any]
    ) -> BaseModel:
        """Parse JSON/dict and validate against Pydantic model."""
        ModelClass = _VALIDATION_REGISTRY[tool_name]

        if isinstance(raw_args, str):
            try:
                import json as _json

                raw_args = _json.loads(raw_args)
            except ValueError as e:
                raise ToolExecutionError(
                    f"Invalid JSON arguments for {tool_name}",
                    details={"raw": raw_args},
                ) from e

        try:
            return ModelClass(**raw_args)  # type: ignore[return-value]
        except ValidationError as e:
            raise ToolExecutionError(
                f"Validation failed for {tool_name}", details=e.errors()
            ) from e

    @staticmethod
    def _parse_kv_string(kv_str: str) -> dict[str, Any]:
        """Parse simple key=value or key='value with spaces' strings into a dict."""
        result: dict[str, Any] = {}
        for match in re.finditer(
            r"""(\w+)=("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s]+)""", kv_str
        ):
            k, v = match.groups()
            if (v.startswith('"') and v.endswith('"')) or (
                v.startswith("'") and v.endswith("'")
            ):
                v = v[1:-1]  # strip quotes
            result[k] = v
        return result

    # ------------------------------------------------------------------
    # Execution Dispatch
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, args: BaseModel) -> Any:
        """Dispatch validated arguments to the bound handler."""
        if tool_name not in self._handlers:
            raise ToolExecutionError(
                f"No handler registered for tool '{tool_name}'"
            )
        return self._handlers[tool_name](args)

    def route_and_execute(
        self, prompt_or_response: str | dict[str, Any]
    ) -> tuple[str, BaseModel, Any]:
        """Unified entry point: parse, validate, and execute."""
        if isinstance(prompt_or_response, dict):
            tool_name, args = self.parse_openai_function_call(prompt_or_response)
        elif prompt_or_response.startswith("/tool:"):
            tool_name, args = self.parse_explicit_command(prompt_or_response)
        else:
            tool_name, args = self.parse_natural_language(prompt_or_response)

        result = self.execute(tool_name, args)
        return tool_name, args, result