# Shared Entities

# 📦 Shared Entities & Data Structures (`entities.py`)

This document defines all shared types, configurations, exceptions, and domain models extracted from the analysis and coding plan. The Python code below is production-ready, fully typed, and designed to be copied directly into a single `entities.py` module for centralized imports across your agent framework.

---

## 📋 Module Overview
- **Zero external dependencies** (uses only stdlib + `httpx`/`aiofiles` as optional runtime deps)
- **PEP 604 & PEP 589 compliant** (modern type hints, frozen dataclasses for immutability where appropriate)
- **Aligned with Phases 1–4** of the architectural plan: async-ready, security-focused, structured error handling, and LLM tool-routing compatible

---

## 💻 `entities.py` Source Code

```python
"""
entities.py
Centralized data structures, configurations, exceptions, and type definitions for the Agent Framework.
Designed for cross-module imports, strict typing, and alignment with async/security best practices.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

# ---------------------------------------------------------------------------
# 📦 Core Type Aliases & Generics
# ---------------------------------------------------------------------------

T = TypeVar("T")
E = TypeVar("E", bound=Exception)

# Lightweight Result type for structured error handling (Phase 1.3 / Phase 3.3)
@dataclass(frozen=True)
class Success(Generic[T]):
    """Represents a successful operation result."""
    value: T

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class Failure(Generic[E]):
    """Represents a failed operation with an attached exception."""
    error: E

    def __bool__(self) -> bool:
        return False


Result = Union[Success[T], Failure[E]]

# Standardized index storage type (Phase 2.2)
SemanticIndexType = Dict[str, List[Path]]

# Normalized workspace path alias (Phase 1.2 / Phase 2.3)
WorkspacePath = Path

# ---------------------------------------------------------------------------
# ⚙️ Configuration Classes (Phase 3.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """Configuration for async LLM client & HTTP timeouts."""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    max_tokens: int = 2000
    temperature: float = 0.0
    
    # httpx timeout configuration (Phase 1.1)
    timeout_connect_sec: float = 5.0
    timeout_read_sec: float = 30.0


@dataclass(frozen=True)
class FileSystemConfig:
    """Configuration for file I/O, sandboxing, and search boundaries."""
    workspace_root: Path = Path("/c/Dev/Agent1")
    max_file_size_mb: int = 10
    
    # Sliding window search parameters (Phase 2.1)
    read_chunk_size_bytes: int = 8192
    search_overlap_bytes: int = 512
    
    # Platform command mapping (Phase 2.4)
    unix_search_cmd: List[str] = field(default_factory=lambda: ["grep", "-r", "-n", "--include=*"])
    windows_search_cmd: List[str] = field(default_factory=lambda: ["findstr", "/s", "/n"])


@dataclass(frozen=True)
class IndexConfig:
    """Configuration for semantic index lifecycle & eviction (Phase 2.2)."""
    max_index_size: int = 5000
    eviction_ratio: float = 0.1  # Remove top X% of least-used entries on overflow
    min_word_length: int = 3


@dataclass(frozen=True)
class AgentConfig:
    """Top-level configuration aggregator for the agent runtime."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    filesystem: FileSystemConfig = field(default_factory=FileSystemConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    
    # State tracking limit (Phase 4.1)
    max_files_read_memory: int = 500
    
    @property
    def detected_os(self) -> OSPlatform:
        """Auto-detect host operating system for command routing."""
        return OSPlatform.from_system()


# ---------------------------------------------------------------------------
# 🛡️ Exception Hierarchy (Phase 1.3)
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base exception for all agent framework errors."""
    pass


class FileOperationError(AgentError):
    """Raised on read/write/delete failures, encoding issues, or size limits."""
    def __init__(self, path: Path | str, message: str = "File operation failed"):
        self.path = Path(path) if isinstance(path, str) else path
        super().__init__(f"{message}: {self.path}")


class ToolExecutionError(AgentError):
    """Raised when a tool call fails validation or runtime execution."""
    def __init__(self, tool_name: str, message: str = "Tool execution failed"):
        self.tool_name = tool_name
        super().__init__(f"[{tool_name}] {message}")


class SecurityViolationError(AgentError):
    """Raised on path traversal attempts, sandbox escapes, or unsafe inputs."""
    def __init__(self, attempted_path: str, reason: str = "Path outside workspace boundary"):
        self.attempted_path = attempted_path
        super().__init__(f"Security violation ({reason}): {attempted_path}")


class IndexError(AgentError):
    """Raised on semantic index corruption or eviction failures."""
    pass

# ---------------------------------------------------------------------------
# 📊 Domain Models & Data Structures (Phase 2.1, 3.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchMatch:
    """Represents a single match found during file/search operations."""
    file_path: Path
    line_number: int | None = None
    context: str = ""          # Snippet surrounding the match
    score: float = 0.0         # Semantic/relevance ranking (if applicable)


@dataclass(frozen=True)
class SearchResult:
    """Aggregated output from a search operation."""
    query: str
    matches: List[SearchMatch] = field(default_factory=list)
    total_files_scanned: int = 0
    execution_time_ms: float = 0.0


@dataclass(frozen=True)
class IndexEntry:
    """Tracks metadata for a term in the semantic index (Phase 2.2)."""
    word: str
    file_paths: List[Path] = field(default_factory=list)
    frequency: int = 0
    last_accessed_epoch: float = 0.0


@dataclass(frozen=True)
class ToolDefinition:
    """OpenAI-compatible function/tool schema for LLM routing (Phase 3.4)."""
    name: str
    description: str
    parameters_schema: Dict[str, Any]  # JSON Schema format
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Converts to OpenAI `functions` array item structure."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


@dataclass(frozen=True)
class ToolCall:
    """Represents a parsed tool invocation from the LLM."""
    call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResponse:
    """Structured return value for tool execution results (Phase 3.4)."""
    call_id: str
    content: str | Result[Any, Exception]
    is_error: bool = False

# ---------------------------------------------------------------------------
# 🔢 Enums & Constants
# ---------------------------------------------------------------------------

class OSPlatform(Enum):
    """Standardized OS detection for cross-platform command routing."""
    WINDOWS = "Windows"
    LINUX = "Linux"
    MACOS = "Darwin"
    UNKNOWN = "Unknown"

    @staticmethod
    def from_system() -> OSPlatform:
        system = platform.system().lower()
        if system == "windows":
            return OSPlatform.WINDOWS
        elif system == "darwin":
            return OSPlatform.MACOS
        else:
            return OSPlatform.LINUX


# Default tool schemas for common agent actions (Phase 3.4)
DEFAULT_TOOL_SCHEMAS = [
    ToolDefinition(
        name="read_file",
        description="Reads the content of a file within the workspace sandbox.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute path to the file"}
            },
            "required": ["path"]
        }
    ),
    ToolDefinition(
        name="write_file",
        description="Writes or overwrites content in a workspace file.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    ),
    ToolDefinition(
        name="search_files",
        description="Searches for text or regex patterns across workspace files.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term or pattern"},
                "file_pattern": {"type": "string", "description": "Glob pattern like '*.py'"}
            },
            "required": ["query"]
        }
    ),
]

```

---

## 🧭 Usage Guidelines & Integration Notes

| Component | Import Pattern | Plan Phase Alignment | Key Design Decisions |
|-----------|----------------|----------------------|----------------------|
| `AgentConfig` | `from entities import AgentConfig` | P3.1 | Frozen dataclasses prevent runtime mutation. Compose sub-configs for clean `.env` loading via `pydantic-settings`. |
| `Result[T, E]` | `from entities import Result, Success, Failure` | P1.3 / P3.3 | Replaces fragile string-error returns. Enables pattern matching: `if isinstance(res, Success): ...` |
| `SecurityViolationError` | `raise SecurityViolationError(path)` | P1.2 | Explicit sandbox enforcement. Callers can catch `AgentError` for generic handling or specific subclasses for UI/LLM feedback. |
| `ToolDefinition` & `DEFAULT_TOOL_SCHEMAS` | `from entities import DEFAULT_TOOL_SCHEMAS` | P3.4 | Pre-formatted for OpenAI function calling. Replace string-parsing NL routing with schema-driven LLM tool selection. |
| `SearchMatch` / `IndexEntry` | `from entities import SearchMatch, IndexEntry` | P2.1 / P2.2 | Structured outputs replace ad-hoc dicts/lists. Enables deterministic eviction logic (`sorted by frequency/last_accessed`). |

### 🔧 Recommended `.env` Loading Pattern (Pydantic)
```python
from pydantic_settings import BaseSettings
from entities import AgentConfig, LLMConfig, FileSystemConfig, IndexConfig

class Settings(BaseSettings):
    workspace_root: str = "/c/Dev/Agent1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"
    
    def to_agent_config(self) -> AgentConfig:
        return AgentConfig(
            llm=LLMConfig(api_key=self.llm_api_key, model=self.llm_model),
            filesystem=FileSystemConfig(workspace_root=Path(self.workspace_root)),
            index=IndexConfig()
        )

config = Settings().to_agent_config()
```

### 🧪 Testing & Validation Tips
- Use `pytest` + `hypothesis` to fuzz `_validate_path()` against 100+ traversal payloads
- Mock `httpx.AsyncClient` using `aioresponses` or `responses` for unit tests
- Assert `isinstance(res, Failure)` and check `res.error.__class__ == SecurityViolationError` in sandbox tests
- Run `mypy --strict entities.py` to verify type safety before merging

---

✅ **Next Step:** Copy the Python block into `entities.py`, run your linter/type-checker, and begin Phase 1 implementation (`httpx` swap + path validation). All subsequent modules will import from this single source of truth.