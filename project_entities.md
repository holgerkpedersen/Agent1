# Shared Entities

# 📦 Shared Entities & Data Structures (`entities.py`)

This document defines the centralized shared entities, type aliases, exception hierarchy, context managers, and serialization structures for the Agent Core Framework. All definitions are Python 3.10+ compliant, strictly typed, and designed to be copied directly into a single `entities.py` module (or split per architectural phase if preferred).

---

## 🔑 1. Core Type Aliases
Standardized type hints used across path validation, logging, and async contexts.

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Modern union syntax (Python 3.10+)
FilePathLike = str | Path
CorrelationId = str
Timestamp = datetime[timezone.utc]
LogPayload = Dict[str, Any]
OptionalPath = Path | None
OptionalCorrelationId = CorrelationId | None
```

**Docstring**:  
> **Purpose**: Centralized type aliases to enforce consistency across modules. Replaces repetitive `Union`/`Optional` patterns and improves IDE autocomplete & static analysis accuracy.  
> **Usage**: Import via `from entities import FilePathLike, LogPayload`. Use in function signatures instead of inline unions.

---

## 🚨 2. Exception Hierarchy
Strict inheritance tree preventing built-in shadowing (e.g., `TimeoutError` → `ToolExecutionTimeoutError`). Includes serializable attributes for structured logging.

```python
class AgentError(Exception):
    """Base exception for all agent framework errors."""
    
    def __init__(self, message: str = "An unexpected agent error occurred.", code: Optional[str] = None) -> None:
        self.message = message
        self.code = code or type(self).__name__
        super().__init__(self.message)


class FileOperationError(AgentError):
    """Raised when filesystem operations fail (IO, permissions, missing targets)."""
    
    def __init__(self, path: FilePathLike, reason: str) -> None:
        self.path = Path(path) if isinstance(path, str) else path
        super().__init__(reason, code="FILE_OP_ERR")


class SecurityViolationError(AgentError):
    """Raised when path traversal, symlink escapes, or sandbox boundary violations are detected."""
    
    def __init__(self, raw_input: str, violation_reason: str) -> None:
        self.raw_input = raw_input
        self.violation_reason = violation_reason
        super().__init__(violation_reason, code="SEC_VIOLATION")


class ToolExecutionTimeoutError(AgentError):
    """Raised when an agent tool or subprocess exceeds its allowed execution time. 
    Renamed from built-in `TimeoutError` to prevent shadowing."""
    
    def __init__(self, operation: str, timeout_seconds: float) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Operation '{operation}' timed out after {timeout_seconds}s", code="TOOL_TIMEOUT")


class ContextCorruptionError(AgentError):
    """Raised when async context variables leak or fail to reset properly."""
    
    def __init__(self, ctx_name: str, details: str = "") -> None:
        self.ctx_name = ctx_name
        super().__init__(f"Context corruption in '{ctx_name}': {details}", code="CTX_CORRUPT")
```

**Docstring**:  
> **Purpose**: Provides a consistent, serializable exception hierarchy for structured logging and error handling. Each subclass attaches domain-specific attributes without mutating `__traceback__`.  
> **Usage**: Catch specific subclasses in agent dispatchers; log via `exc.__dict__` or custom encoders. Never shadow built-in exceptions.

---

## 🔄 3. Context & Correlation Management
Async-safe context variables and a production-ready context manager for distributed tracing compatibility.

```python
import contextvars
from typing import Generator, TypeVar

T = TypeVar("T")

# Global context variable for async/thread-safe correlation tracking
CORRELATION_ID_CTX: contextvars.ContextVar[CorrelationId] = \
    contextvars.ContextVar("agent_correlation_id", default=str(uuid.uuid4()))


class CorrelationIdContext:
    """Thread and task-safe context manager that scopes a correlation ID using `contextvars`.
    
    Attributes:
        _token: Internal token for safe context restoration on exit.
        _corr_id: The active correlation identifier string.
        
    Usage:
        with CorrelationIdContext("request-abc123") as cid:
            print(CORRELATION_ID_CTX.get())  # "request-abc123"
        # Automatically resets to previous value on exit, even if exceptions occur.
    """
    
    def __init__(self, corr_id: Optional[CorrelationId] = None) -> None:
        self._token: Optional[contextvars.Token[CorrelationId]] = None
        self._corr_id: CorrelationId = corr_id or str(uuid.uuid4())

    def __enter__(self) -> CorrelationId:
        """Activate the correlation ID in the current execution context."""
        self._token = CORRELATION_ID_CTX.set(self._corr_id)
        return self._corr_id

    def __exit__(self, exc_type: Optional[type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[Any]) -> None:
        """Restore previous context state. Safe under exceptions and async task switches."""
        if self._token is not None:
            CORRELATION_ID_CTX.reset(self._token)
```

**Docstring**:  
> **Purpose**: Ensures correlation IDs survive across `asyncio.create_task()`, thread pools, and nested calls without leakage. Uses `contextvars.Token` for deterministic cleanup.  
> **Usage**: Wrap agent request handlers or dispatch loops. Access current ID via `CORRELATION_ID_CTX.get()` in loggers/middleware.

---

## 📁 4. Workspace & Path Security Models
Dataclasses and configuration structures that drive sandbox validation policies. Designed to be consumed by `path_utils.py`.

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkspacePolicy:
    """Configuration contract for workspace sandboxing behavior.
    
    Attributes:
        allow_symlinks: Whether symbolic links inside the workspace are permitted.
                       If False, any symlink triggers SecurityViolationError.
        strict_boundary_enforcement: Enforces Path.relative_to() checks against resolved roots.
        max_path_depth: Optional limit on nested directory levels to prevent deep traversal abuse.
    """
    allow_symlinks: bool = True
    strict_boundary_enforcement: bool = True
    max_path_depth: int | None = None


@dataclass
class ValidatedPathResult:
    """Immutable result object returned by path validation routines.
    
    Attributes:
        raw_input: Original user/tool-supplied path string.
        resolved_path: Fully resolved, normalized pathlib.Path.
        is_within_workspace: True if resolved path satisfies sandbox boundary checks.
        contains_symlink: True if any component in the resolution chain was a symlink.
    """
    raw_input: str
    resolved_path: Path
    is_within_workspace: bool = False
    contains_symlink: bool = False


@dataclass
class WorkspaceSandboxConfig:
    """Runtime configuration for an active sandbox session.
    
    Attributes:
        root: Absolute path to the workspace boundary.
        policy: Validation rules applied to all I/O within this session.
        mounted_volumes: Optional dict mapping virtual prefixes to real paths (for chroot-like behavior).
    """
    root: Path
    policy: WorkspacePolicy = field(default_factory=WorkspacePolicy)
    mounted_volumes: Dict[str, Path] = field(default_factory=dict)
```

**Docstring**:  
> **Purpose**: Centralizes sandbox configuration and validation contracts. Enables declarative security policies without scattering boolean flags across utility functions.  
> **Usage**: Instantiate `WorkspacePolicy` once per agent session; pass to `_validate_path()` or sandbox context managers. Never mutate after initialization (`frozen=True`).

---

## 📜 5. Logging & Serialization Structures
Crash-proof JSON encoding and correlation-aware log filtering for structured observability pipelines.

```python
import json
import logging
from uuid import UUID


class SafeJsonEncoder(json.JSONEncoder):
    """JSON encoder that safely serializes common Python objects without raising TypeError.
    
    Handles: datetime, timezone-aware timestamps, pathlib.Path, Exception, UUID, and arbitrary objects with __dict__.
    Falls back to str() representation for unknown types instead of crashing the logging pipeline.
    """
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Exception):
            return {
                "type": type(obj).__name__,
                "message": str(obj),
                "args": obj.args
            }
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        # Fallback: prevent logging crashes on unserializable payloads
        try:
            return super().default(obj)
        except TypeError:
            return f"<unserializable:{type(obj).__name__}>"


class CorrelationIdFilter(logging.Filter):
    """Injects the current async-safe correlation ID into every log record."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Always allow the record through; mutation is safe in stdlib logging
        record.correlation_id = CORRELATION_ID_CTX.get()
        return True


class LogContextManager:
    """Temporary context that binds extra fields to all log records during execution."""
    
    def __init__(self, logger: logging.Logger, **extra_fields: Any) -> None:
        self.logger = logger
        self.extra_fields = extra_fields

    def __enter__(self) -> logging.Logger:
        # Attach via adapter for zero-overhead context binding
        self._adapter = logging.LoggerAdapter(self.logger, self.extra_fields)
        return self._logger_adapter  # type: ignore[name-defined]

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
```

**Docstring**:  
> **Purpose**: Eliminates `TypeError` crashes in production loggers when agents emit complex objects. Ensures every log line carries traceable correlation metadata without manual injection.  
> **Usage**: Configure via `logging.config.dictConfig()`. Attach `CorrelationIdFilter` to handlers and set `SafeJsonEncoder` on JSON formatters. Wrap critical sections with `LogContextManager` for request-scoped extra fields.

---

## 📌 Usage & Integration Guidelines

| Module | Import Pattern | Responsibility |
|--------|----------------|----------------|
| **Exceptions** | `from entities import AgentError, SecurityViolationError` | Error handling, dispatch routing, retry logic |
| **Context** | `from entities import CORRELATION_ID_CTX, CorrelationIdContext` | Async tracing, request scoping, middleware |
| **Workspace Models** | `from entities import WorkspacePolicy, ValidatedPathResult` | Path validation contracts, sandbox configuration |
| **Logging/Serialization** | `from entities import SafeJsonEncoder, CorrelationIdFilter` | Structured log pipelines, observability exporters |

### 🔧 Implementation Notes
1. **Strict Typing**: All signatures use Python 3.10+ union syntax (`|`). Run with `mypy --strict` for compliance.
2. **No Circular Imports**: These entities contain zero internal dependencies beyond stdlib. Safe to import from anywhere in the framework.
3. **Split Strategy (Optional)**: If following Phase 1 of the plan exactly, split this file into:
   - `agent_core/exceptions.py`
   - `agent_core/context.py`
   - `agent_core/models.py`
   - `agent_core/logging_types.py`
4. **Testing**: Mock `CORRELATION_ID_CTX` using `contextvars.ContextVar.set()` in pytest fixtures. Use `pyfakefs` for path model validation tests.

--- 
✅ *Ready for production integration. Copy contents into `entities.py`, run `ruff check` & `mypy`, and import across agent modules.*