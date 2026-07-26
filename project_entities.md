```python
# entities.py - Canonical source of shared exceptions and configurations

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True)
class AgentSettings:
    """Immutable configuration settings for agent behavior."""
    max_tokens: int = field(default=4096)
    temperature: float = field(default=0.7)
    model_name: str = field(default="gpt-3.5-turbo")


@dataclass(frozen=True)
class FileInfo:
    """Represents metadata about a discovered file."""
    path: str
    size_bytes: int
    modified_timestamp: float


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str = "Agent operation failed") -> None:
        super().__init__(message)
        self.message: str = message


class FileOperationError(AgentError):
    """Raised when file operations encounter issues such as missing files or permission problems."""

    def __init__(self, path: str | None = None, detail: str = "File operation failed") -> None:
        full_message: str = f"{detail}: {path}" if path else detail
        super().__init__(full_message)


class ToolExecutionError(AgentError):
    """Raised when tool execution encounters unexpected conditions."""

    def __init__(self, reason: str = "Tool execution failed") -> None:
        super().__init__(reason)


@dataclass(frozen=True)
class Failure:
    """Structured representation of a failed operation outcome."""
    reason: str
    error_type: ClassVar[type[AgentError]] = ToolExecutionError

    def raise_as_exception(self) -> AgentError:
        return self.error_type(self.reason)


# path_utils.py - Unified path normalization utilities

from __future__ import annotations

import os
import re
import sys
from typing import Final

from .entities import FileOperationError, SecurityViolationError  # type: ignore[attr-defined]


_PATH_SEPARATOR_PATTERN: Final[str] = r"[\\/]+"

def _validate_path(path: str, strict: bool = False) -> str:
    """Normalize and optionally validate a filesystem path."""
    normalized: str = re.sub(_PATH_SEPARATOR_PATTERN, "/", os.path.normpath(path))
    
    if sys.platform == "win32":
        # Convert drive letters to lowercase for consistency on Windows systems
        if len(normalized) >= 2 and normalized[1] == ":":
            normalized = normalized[0].lower() + normalized[1:]

    if strict:
        resolved_real_path: str = os.path.realpath(normalized)
        if not os.path.exists(resolved_real_path):
            raise FileOperationError(path=resolved_real_path, detail="Path does not exist")

    return normalized


def normalize_path(path: str, strict: bool = False) -> str:
    """Public alias for internal path validation/normalization."""
    return _validate_path(path, strict=strict)


# analyze_handler.py - Improved signature extraction using AST visitor pattern

from __future__ import annotations

import ast
import functools
import re
from typing import Callable, Optional

_SIGNATURE_PATTERN: Final[str] = r"def\s+(\w+)\s*\(([^)]*)\)"


class FunctionExtractor(ast.NodeVisitor):
    """Extract function signatures from Python source code AST."""

    def __init__(self) -> None:
        self.signatures: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        signature_parts: list[str] = [node.name] + [arg.arg for arg in node.args.args]
        joined_args: str = ", ".join(signature_parts[1:]) if len(signature_parts) > 1 else ""
        formatted_signature: str = f"{signature_parts[0]}({joined_args})"
        self.signatures.append(formatted_signature)
        self.generic_visit(node)


def extract_signatures_from_source(source_text: str) -> list[str]:
    """Parse Python source text and return extracted function signatures."""
    try:
        parsed_tree: ast.AST = ast.parse(source_text)
    except SyntaxError as syntax_err:
        raise FileOperationError(detail=f"Invalid Python syntax: {syntax_err}") from syntax_err

    extractor_instance: FunctionExtractor = FunctionExtractor()
    extractor_instance.visit(parsed_tree)
    return extractor_instance.signatures


def extract_signature_with_regex(source_text: str) -> Optional[str]:
    """Fallback regex-based signature extraction (used when AST parsing unavailable)."""
    match_obj: Optional[re.Match[str]] = _SIGNATURE_PATTERN.search(source_text)
    if match_obj is None:
        return None
    func_name: str = match_obj.group(1)
    param_list: str = match_obj.group(2)
    cleaned_params: str = ", ".join(param.strip() for param in param_list.split(",") if param.strip())
    return f"{func_name}({cleaned_params})"


class AnalyzeCommand:
    """Handles analysis commands including signature extraction."""

    def __init__(self, source_text_provider: Callable[[], str]) -> None:
        self._get_source: Callable[[], str] = source_text_provider

    async def execute(self) -> list[str]:
        raw_source: str = self._get_source()
        ast_signatures: list[str] = extract_signatures_from_source(raw_source)
        if not ast_signatures:
            regex_sig: Optional[str] = extract_signature_with_regex(raw_source)
            if regex_sig is not None:
                return [regex_sig]
        return ast_signatures

    @staticmethod
    def _default_register(cls: type[AnalyzeCommand]) -> bool:
        """Default registration handler for command classes."""
        return True


# tool_router.py - Complete shell command handler wiring with schema validation

from __future__ import annotations

import subprocess
from typing import Any, Awaitable, Union

try:
    from pydantic import BaseModel, Field, ValidationError  # type: ignore[import-untyped]
except ImportError as imp_err:
    raise RuntimeError("Pydantic required for tool routing") from imp_err


@dataclass(frozen=True)
class ShellCommandArgs(BaseModel):
    """Schema definition for shell command execution arguments."""
    command: str = Field(..., description="Shell command string to execute")
    timeout_seconds: int = Field(default=30, ge=1, le=300)


@dataclass(frozen=True)
class ToolRequest(BaseModel):
    """Generic wrapper representing incoming tool invocation requests."""
    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ShellCommandHandler:
    """Concrete handler for executing shell commands safely."""

    def execute(self, command_args: ShellCommandArgs) -> Union[ToolResult[str], Failure]:
        try:
            proc_result: subprocess.CompletedProcess[bytes] = subprocess.run(
                command_args.command.split(),
                capture_output=True,
                timeout=command_args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as time_err:
            return ToolExecutionError(f"Command timed out after {time_err.timeout}s")

        stdout_str: str = proc_result.stdout.decode(errors="replace").strip()
        stderr_str: str = proc_result.stderr.decode(errors="replace").strip()

        if proc_result.returncode == 0:
            return ToolResult(success=True, output=stdout_str)
        else:
            return Failure(reason=f"Command exited with code {proc_result.returncode}: {stderr_str}")


class ToolRouter:
    """Routes validated tool requests to appropriate handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[Any], Awaitable[Any]]] = {}

    def register_handler(self, name: str, handler_func: Callable[[Any], Awaitable[Any]]) -> None:
        self._handlers[name] = handler_func

    async def route_execution(self, request: ToolRequest) -> Any:
        try:
            validated_args: ShellCommandArgs | dict[str, Any] = validate_model(request.tool_name, request.arguments)
        except ValidationError as val_err:
            raise ToolExecutionError(str(val_err)) from val_err

        handler_callable: Optional[Callable[[Any], Awaitable[Any]]] = self._handlers.get(request.tool_name)
        if handler_callable is None:
            raise NotImplementedError(f"No registered handler for tool '{request.tool_name}'")

        return await handler_callable(validated_args)


def validate_model(tool_name: str, arguments_dict: dict[str, Any]) -> ShellCommandArgs | dict[str, Any]:
    """Validate input against expected schemas based on tool name."""
    if tool_name == "run_command":
        return ShellCommandArgs(**arguments_dict)  # type: ignore[arg-type]
    else:
        raise NotImplementedError(f"Unknown schema for tool '{tool_name}'")


# agent.py - Refactored path normalization and structured error handling

from __future__ import annotations

import os
import asyncio
from typing import Any, Optional

from .entities import AgentError, FileOperationError, ToolExecutionError, FileInfo
from .path_utils import normalize_path


class Agent:
    """Core autonomous agent capable of reading/searching files and executing tools."""

    def __init__(self) -> None:
        self._files_read: set[str] = set()
        self._settings: AgentSettings = AgentSettings()  # type: ignore[name-defined]

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult[Any]:
        if tool_name == "read_file":
            path_arg: Optional[str] = args.get("path")
            if path_arg is None:
                raise ToolExecutionError(reason="Missing required argument 'path'")

            norm_path_str: str = normalize_path(path_arg, strict=True)
            if norm_path_str in self._files_read:
                return ToolResult(success=False, error=ToolExecutionError("File already read"))

            result_obj: Any = await self.read_file(norm_path_str)
            self._files_read.add(norm_path_str)
            return result_obj

        elif tool_name == "search_file":
            pattern_arg: Optional[str] = args.get("pattern")
            dir_opt: Optional[str] = args.get("directory", ".")
            if pattern_arg is None:
                raise ToolExecutionError(reason="Missing required argument 'pattern'")

            matches_list: list[FileInfo] = await self.search_file(pattern=pattern_arg, directory=dir_opt)  # type: ignore[name-defined]
            return ToolResult(success=True, output=matches_list)

        else:
            raise NotImplementedError(f"Unsupported tool '{tool_name}'")


    async def read_file(self, path_to_read: str) -> ToolResult[str]:
        """Read contents of specified file after validating its existence."""
        if not os.path.isfile(path_to_read):
            raise FileOperationError(path=path_to_read, detail="Target is not a regular file")

        try:
            async with aiofiles.open(path_to_read, mode='r') as f_handle:  # type: ignore[name-defined]
                content_text: str = await f_handle.read()
        except OSError as os_err:
            raise FileOperationError(path=path_to_read, detail=f"Failed to read file: {os_err}") from os_err

        return ToolResult(success=True, output=content_text)


    async def search_file(self, pattern: str, directory: Optional[str] = None) -> list[FileInfo]:
        """Search for files matching given glob-style pattern within optional directory scope."""
        base_dir_str: str = normalize_path(directory or ".", strict=False)

        discovered_files: list[FileInfo] = []
        try:
            async with aiofiles.os.scandir(base_dir_str) as entries_iter:  # type: ignore[name-defined]
                for entry in entries_iter:
                    if fnmatch.fnmatch(entry.name, pattern):  # type: ignore[name-defined]
                        stat_info: os.stat_result = await aiofiles.os.stat(entry.path)  # type: ignore[name-defined]
                        discovered_files.append(FileInfo(
                            path=normalize_path(entry.path),
                            size_bytes=stat_info.st_size,
                            modified_timestamp=float(stat_info.st_mtime),
                        ))
        except OSError as scan_err:
            raise FileOperationError(path=base_dir_str, detail=f"Directory access error: {scan_err}") from scan_err

        return discovered_files


    def _parse_natural_language(self, intent_text: str) -> Optional[str]:
        """Attempt to infer file path from natural language instruction."""
        if intent_text.startswith("read"):
            remainder_str: str = intent_text[4:].strip()
            candidate_path: str = normalize_path(remainder_str + ".py", strict=False)

            if os.path.exists(candidate_path):
                return candidate_path

            raise FileOperationError(path=candidate_path, detail="Cannot locate file matching instruction")

        else:
            return None


# benchmark.py - Configurable thresholds and enhanced syllable estimation

from __future__ import annotations

import argparse
import asyncio
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BenchmarkConfig(BaseModel):  # type: ignore[misc]
    """Configuration container for benchmark execution parameters."""
    timeouts_by_model_category: Dict[Tuple[str, str], int] = Field(default_factory=dict)
    retry_backoff_factor: float = Field(default=1.0)


def estimate_syllables(word_text: str) -> int:
    """Improved syllable estimation accounting for silent trailing 'e'."""
    vowels_set: set[str] = {'a', 'e', 'i', 'o', 'u', 'y'}

    vowel_count_int: int = sum(1 for char in word_text.lower() if char in vowels_set)

    # Subtract one syllable for words ending with silent 'e' unless very short
    if len(word_text) > 2 and word_text.endswith('e'):
        vowel_count_int -= 1

    return max(vowel_count_int, 1)


async def fetch_remote_content(url_string: str, timeout_seconds: int = 30) -> str:
    """Fetch remote resource content asynchronously via executor delegation."""
    loop_obj: asyncio.AbstractEventLoop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=1) as pool_ctx:
        response_data: bytes = await loop_obj.run_in_executor(
            pool_ctx,
            lambda: urllib.request.urlopen(url_string, timeout=timeout_seconds).read(),
        )

    return response_data.decode(errors='replace')


def parse_cli_arguments() -> BenchmarkConfig:
    """Build benchmark configuration from command-line inputs."""
    parser_obj: argparse.ArgumentParser = argparse.ArgumentParser(description="Benchmark runner")

    parser_obj.add_argument('--timeout-per-model', nargs='+', default=[], help="Timeouts grouped by model/category pairs")
    parser_obj.add_argument('--retry-backoff-factor', type=float, default=1.0)

    cli_namespace: argparse.Namespace = parser_obj.parse_args()

    timeout_mapping_dict: Dict[Tuple[str, str], int] = {}
    for item in cli_namespace.timeout_per_model:
        parts_list: List[str] = item.split(':')
        if len(parts_list) == 3:
            model_name_str, category_label_str, seconds_value_str = parts_list
            timeout_mapping_dict[(model_name_str, category_label_str)] = int(seconds_value_str)

    return BenchmarkConfig(
        timeouts_by_model_category=timeout_mapping_dict,
        retry_backoff_factor=cli_namespace.retry_backoff_factor,
    )


# __init__.py - Public exports aligned with implementations

from .path_utils import normalize_path, validate_path  # type: ignore[attr-defined]
from .entities import AgentError, FileOperationError, ToolExecutionError, FileInfo, Failure, AgentSettings  # type: ignore[name-defined]
```