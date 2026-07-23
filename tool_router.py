"""
tool_router.py
Structured tool routing, schema validation, and LLM function-call parsing for the Agent Framework.
Supports OpenAI-style function calling, regex-based natural language fallback, and explicit command prefixes.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# 📦 Core Types & Exceptions
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class ToolExecutionError(Exception):
    """Raised when tool argument validation fails or routing encounters an unrecoverable state."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class RoutingError(Exception):
    """Raised when the router cannot match a prompt to a registered tool."""
    pass


# ---------------------------------------------------------------------------
# 🛠️ Tool Argument Models (Pydantic Validation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and JSON Schema for a single agent tool."""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    
    def to_openai_format(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Relative or absolute path to the file within workspace")


class WriteFileArgs(BaseModel):
    path: str = Field(..., description="Target file path for writing/overwriting")
    content: str = Field(..., description="String content to write into the file")


class SearchFilesArgs(BaseModel):
    query: str = Field(..., description="Text or regex pattern to search for")
    file_pattern: Optional[str] = Field(None, description="Glob filter like '*.py' or '*.md'")


class ShellCommandArgs(BaseModel):
    command: str = Field(..., description="Shell command string (sanitized externally)")


# Registry mapping tool names -> Pydantic models for runtime validation
_VALIDATION_REGISTRY: Dict[str, type[BaseModel]] = {
    "read_file": ReadFileArgs,
    "write_file": WriteFileArgs,
    "search_files": SearchFilesArgs,
    "run_command": ShellCommandArgs,
}

# Default tool schemas aligned with OpenAI function calling spec
DEFAULT_TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description="Reads the content of a file within the workspace sandbox.",
        parameters_schema=ReadFileArgs.model_json_schema()
    ),
    ToolDefinition(
        name="write_file",
        description="Writes or overwrites content in a workspace file.",
        parameters_schema=WriteFileArgs.model_json_schema()
    ),
    ToolDefinition(
        name="search_files",
        description="Searches for text patterns across workspace files using grep/findstr.",
        parameters_schema=SearchFilesArgs.model_json_schema()
    ),
    ToolDefinition(
        name="run_command",
        description="Executes a sanitized shell command within the workspace environment.",
        parameters_schema=ShellCommandArgs.model_json_schema()
    ),
]

# ---------------------------------------------------------------------------
# 🧭 Router Implementation
# ---------------------------------------------------------------------------

class ToolRouter:
    """
    Routes LLM responses or natural language prompts to validated tool executions.
    Supports:
      1. OpenAI function-call JSON parsing
      2. Explicit `/tool:name args` command prefix
      3. Regex-backed natural language fallback
    """
    
    def __init__(self, tools: Optional[List[ToolDefinition]] = None):
        self._tools: Dict[str, ToolDefinition] = {t.name: t for t in (tools or DEFAULT_TOOL_DEFINITIONS)}
        self._handlers: Dict[str, Callable[[BaseModel], Any]] = {}
        
    def register_tool(self, definition: ToolDefinition) -> None:
        """Register a new tool definition with the router."""
        if definition.name not in _VALIDATION_REGISTRY:
            raise ToolExecutionError(f"Validation model missing for tool '{definition.name}'")
        self._tools[definition.name] = definition
        
    def register_handler(self, tool_name: str, handler: Callable[[BaseModel], Any]) -> None:
        """Bind a runtime handler function to a tool name."""
        if tool_name not in self._tools:
            raise ToolExecutionError(f"Cannot bind handler to unregistered tool '{tool_name}'")
        self._handlers[tool_name] = handler
        
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function schemas for LLM injection."""
        return [t.to_openai_format() for t in self._tools.values()]

    # ------------------------------------------------------------------
    # 🔍 Parsing Strategies
    # ------------------------------------------------------------------
    
    def parse_openai_function_call(self, raw_response: Dict[str, Any]) -> Tuple[str, BaseModel]:
        """Parse OpenAI-style function call object and validate arguments."""
        func = raw_response.get("function")
        if not func:
            raise RoutingError("Missing 'function' key in LLM response")
            
        tool_name = func["name"]
        args_json = func.get("arguments", "{}")
        
        # Fuzzy match tool name (handle minor hallucinations)
        matched_name = self._resolve_tool_name(tool_name)
        
        return matched_name, self._validate_args(matched_name, args_json)

    def parse_explicit_command(self, prompt: str) -> Tuple[str, BaseModel]:
        """Parse deterministic `/tool:name key=value` style commands."""
        match = re.match(r"^/tool:(\w+)\s+(.+)$", prompt.strip(), re.IGNORECASE)
        if not match:
            raise RoutingError("Invalid explicit command format. Expected: /tool:<name> <args>")
            
        tool_name, args_str = match.group(1), match.group(2)
        matched_name = self._resolve_tool_name(tool_name.lower())
        
        # Convert key=value pairs to dict (supports simple quoted values)
        args_dict = self._parse_kv_string(args_str)
        return matched_name, self._validate_args(matched_name, args_dict)

    def parse_natural_language(self, prompt: str) -> Tuple[str, BaseModel]:
        """Regex-backed fallback for varied natural language phrasing."""
        # Word-boundary matching prevents substring mangling (e.g., "research" != "search")
        patterns = {
            r"\b(read|open|view)\s+(?:the\s+)?file[:\s]+(?P<path>\S+)": "read_file",
            r"\b(write|save|create|overwrite)\s+(?:to\s+)?(?:the\s+)?file[:\s]+(?P<path>\S+)(?:\s*with\s*(?P<content>.+))?$": "write_file",
            r"\b(search|find|grep)\s+(?:for\s+)?['\"]?(?P<query>[^'\"]+)['\"]?\s*(?:in\s+file[:\s]+(?P<file_pattern>\S+))?": "search_files",
        }
        
        for pattern, tool_name in patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                args_dict = {k: v for k, v in match.groupdict().items() if v is not None}
                matched_name = self._resolve_tool_name(tool_name)
                return matched_name, self._validate_args(matched_name, args_dict)
                
        raise RoutingError("Could not route natural language prompt to any registered tool")

    # ------------------------------------------------------------------
    # ⚙️ Internal Helpers
    # ------------------------------------------------------------------
    
    def _resolve_tool_name(self, raw_name: str) -> str:
        """Fuzzy match input name against registered tools."""
        normalized = raw_name.lower().strip()
        if normalized in self._tools:
            return normalized
        
        # Levenshtein-like fallback for minor typos/hallucinations
        closest = min(self._tools.keys(), key=lambda k: sum(c1 != c2 for c1, c2 in zip(k, normalized)))
        if closest.lower() == normalized[:len(closest)]:
            return closest
            
        raise RoutingError(f"Unknown tool '{raw_name}'. Available: {list(self._tools.keys())}")

    def _validate_args(self, tool_name: str, raw_args: Union[str, Dict[str, Any]]) -> BaseModel:
        """Parse JSON/dict and validate against Pydantic model."""
        ModelClass = _VALIDATION_REGISTRY[tool_name]
        
        if isinstance(raw_args, str):
            try:
                import json
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise ToolExecutionError(f"Invalid JSON arguments for {tool_name}", details={"raw": raw_args}) from e
                
        try:
            return ModelClass(**raw_args)
        except ValidationError as e:
            raise ToolExecutionError(f"Validation failed for {tool_name}", details=e.errors()) from e

    @staticmethod
    def _parse_kv_string(kv_str: str) -> Dict[str, Any]:
        """Parse simple key=value or key='value with spaces' strings into a dict."""
        result = {}
        # Matches key=value or key="quoted value"
        for match in re.finditer(r"""(\w+)=("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s]+)""", kv_str):
            k, v = match.groups()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]  # strip quotes
            result[k] = v
        return result

    # ------------------------------------------------------------------
    # 🚀 Execution Dispatch
    # ------------------------------------------------------------------
    
    def execute(self, tool_name: str, args: BaseModel) -> Any:
        """Dispatch validated arguments to the bound handler."""
        if tool_name not in self._handlers:
            raise ToolExecutionError(f"No handler registered for tool '{tool_name}'")
        return self._handlers[tool_name](args)

    def route_and_execute(self, prompt_or_response: Union[str, Dict[str, Any]]) -> Tuple[str, BaseModel, Any]:
        """Unified entry point: parse, validate, and execute."""
        if isinstance(prompt_or_response, dict):
            tool_name, args = self.parse_openai_function_call(prompt_or_response)
        elif prompt_or_response.startswith("/tool:"):
            tool_name, args = self.parse_explicit_command(prompt_or_response)
        else:
            tool_name, args = self.parse_natural_language(prompt_or_response)
            
        result = self.execute(tool_name, args)
        return tool_name, args, result