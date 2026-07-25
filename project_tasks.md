# Task Plan: Agent Codebase Refactoring & Fixes  
**Dependency Order with Type-Checking Validation**

## Phase 1: Foundation Layer (No Dependencies)

### Task 1.1 – Create Centralized Exception Hierarchy (`agent_core/exceptions.py`) ⭐ **FOUNDATION**
```python
from __future__ import annotations
from typing import Final, Optional

class AgentBaseError(Exception):
    """Base exception for all agent-related errors."""
    
class FileOperationError(AgentBaseError):
    def __init__(self, message: str, path: Optional[str] = None) -> None:
        self.path = path
        super().__init__(message)

class SecurityViolationError(AgentBaseError): pass
class ConfigurationError(AgentBaseError): pass
class ToolExecutionError(AgentBaseError): 
    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")

# Type alias for consistent error handling
ErrorHandlerResult = tuple[bool, Optional[str]]  # (success, optional_error_message)
```

**Type Checking Validation:** ✅ Run `mypy --strict agent_core/exceptions.py`  
✅ **Status**: Ready for implementation

---

### Task 1.2 – Create Configuration Management (`agent_core/config.py`) ⭐ **FOUNDATION**
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

@dataclass(frozen=True)
class AgentSettings:
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    llm_api_url: str = "http://localhost:1234/v1"
    max_concurrent_tools: int = 5
    search_command_timeout_sec: float = 30.0
    compilation_check_timeout_sec: float = 30.0

# Global instance with validation at import time
def _validate_settings(settings: AgentSettings) -> None:
    if not settings.workspace_root.exists():
        raise ConfigurationError(f"Workspace root does not exist: {settings.workspace_root}")
    if settings.max_concurrent_tools <= 0:
        raise ConfigurationError("max_concurrent_tools must be positive")

DEFAULT_SETTINGS: Final[AgentSettings] = AgentSettings()
_validate_settings(DEFAULT_SETTINGS)
```

**Type Checking Validation:** ✅ Run `mypy --strict agent_core/config.py`  
✅ **Status**: Ready for implementation

---

## Phase 2: Core Utilities (Depends on Foundation)

### Task 2.1 – Fix Path Normalization (`agent_core/path_utils.py`) ⭐ **DEPENDS ON TASKS 1.1, 1.2**
```python
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Final, Optional, Union

from .exceptions import SecurityViolationError
from .config import AgentSettings

PLATFORM_WINDOWS: Final[bool] = sys.platform.startswith("win") or os.name == "nt"

def normalize_path(path_input: Union[str, Path], settings: AgentSettings) -> Optional[Path]:
    """Normalize and validate a path within the workspace."""
    if not isinstance(path_input, (str, Path)):
        raise TypeError(f"Expected str or Path, got {type(path_input).__name__}")
    
    raw_path = Path(str(path_input))
    
    # Handle Unix-style absolute paths on Windows/WSL
    if PLATFORM_WINDOWS and path_input.startswith("/"):
        normalized_path = _convert_unix_to_windows(raw_path)
    else:
        normalized_path = raw_path
    
    try:
        resolved = normalized_path.absolute().resolve(strict=False)
        if not _is_within_workspace(resolved, settings.workspace_root):
            raise SecurityViolationError(f"Path outside workspace: {resolved}")
        return resolved
    except FileNotFoundError as e:
        raise FileOperationError(f"File not found during resolution", str(raw_path)) from e

def _convert_unix_to_windows(path: Path) -> Path:
    """Convert Unix-style path to Windows equivalent."""
    # Implementation here...
    pass

def _is_within_workspace(resolved_path: Path, workspace_root: Path) -> bool:
    try:
        resolved_path.relative_to(workspace_root.resolve())
        return True
    except ValueError:
        return False
```

**Type Checking Validation:** ✅ Run `mypy --strict agent_core/path_utils.py`  
✅ **Status**: Ready for implementation

---

### Task 2.2 – Secure Async Subprocess Execution (`agent_core/subprocess_utils.py`) ⭐ **DEPENDS ON TASKS 1.1, 1.2**
```python
from __future__ import annotations
import asyncio
import logging
from typing import Final, Optional, Tuple

logger = logging.getLogger(__name__)

async def run_subprocess_with_timeout(
    cmd: list[str], 
    timeout_sec: float,
    cwd: Optional[str] = None
) -> Tuple[int, bytes, bytes]:
    """Run subprocess with timeout and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        return (proc.returncode or 0), stdout, stderr
    except asyncio.TimeoutError as e:
        logger.warning(f"Subprocess timed out after {timeout_sec}s: {' '.join(cmd)}")
        raise ToolExecutionError("subprocess", f"Timed out after {timeout_sec} seconds") from e
```

**Type Checking Validation:** ✅ Run `mypy --strict agent_core/subprocess_utils.py`  
✅ **Status**: Ready for implementation

---

## Phase 3: LLM Integration (Depends on Foundation)

### Task 3.1 – Consistent LLM Response Handling (`agent_core/llm_client.py`) ⭐ **DEPENDS ON TASKS 1.1, 2.2**
```python
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Final, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LlmResponse:
    content: str
    is_error: bool = False
    error_code: Optional[str] = None

class LLMClient:
    def __init__(self, api_url: str, timeout_sec: float = 30.0) -> None:
        self.api_url = api_url
        self.timeout_sec = timeout_sec
    
    async def chat(self, prompt: str) -> LlmResponse:
        """Send message to LLM and return structured response."""
        try:
            # Implementation using httpx/aiohttp for true async support
            pass
        except Exception as e:
            logger.exception("LLM request failed")
            return LlmResponse(
                content=f"Service error occurred", 
                is_error=True, 
                error_code="SERVICE_ERROR"
            )
```

**Type Checking Validation:** ✅ Run `mypy --strict agent_core/llm_client.py`  
✅ **Status**: Ready for implementation

---

## Phase 4: Command Handlers (Depends on Core Layers)

### Task 4.1 – Base Handler Interface (`agent_core/handlers/base_handler.py`) ⭐ **DEPENDS ON ALL PREVIOUS**
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Final, List, Protocol

class CommandHandler(Protocol):
    @abstractmethod
    async def handle(self, args: List[str]) -> int: ...  # Returns exit code

class BaseCommandHandler(ABC):
    """Abstract base class for command handlers."""
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    async def handle(self, args: List[str]) -> int: ...
```

**Type Checking Validation:** ✅ Run `mypy --strict agent_core/handlers/base_handler.py`  
✅ **Status**: Ready for implementation

---

### Task 4.2 – Analyze Command Handler (`agent_core/handlers/analyze_handler.py`) ⭐ **DEPENDS ON TASKS 1.1, 2.1, 3.1**
```python
from __future__ import annotations
import ast
import re
from pathlib import Path
from typing import Final, List

from ..base_handler import BaseCommandHandler
from ...path_utils import normalize_path
from ...config import AgentSettings
from ...exceptions import FileOperationError

class AnalyzeCommand(BaseCommandHandler):
    @property
    def name(self) -> str: return "analyze"
    
    async def handle(self, args: List[str]) -> int:
        # Implementation with proper error handling and logging
        pass
```

**Type Checking Validation:** ✅ Run `mypy --strict agent_core/handlers/analyze_handler.py`  
✅ **Status**: Ready for implementation

---

## Phase 5: Integration & Testing (Final Layer)

### Task 5.1 – Main Agent Entry Point Refactor (`agent.py`) ⭐ **DEPENDS ON ALL HANDLERS**
```python
from __future__ import annotations
import logging
from typing import Final, Dict, List

from agent_core.config import DEFAULT_SETTINGS
from agent_core.handlers.analyze_handler import AnalyzeCommand
# Import other handlers...

COMMAND_REGISTRY: Final[Dict[str, BaseCommandHandler]] = {
    "analyze": AnalyzeCommand(),
    # Add others...
}

async def run_interactive(args: List[str]) -> int:
    """Main entry point with clean delegation."""
    if not args:
        logging.error("No command provided")
        return 1
    
    command = args[0]
    handler = COMMAND_REGISTRY.get(command)
    if not handler:
        logging.error(f"Unknown command: {command}")
        return 1
    
    try:
        return await handler.handle(args[1:])
    except Exception as e:
        logging.exception("Command execution failed")
        return 1
```

**Type Checking Validation:** ✅ Run `mypy --strict agent.py`  
✅ **Status**: Ready for implementation

---

### Task 5.2 – Comprehensive Unit Test Suite ⭐ **DEPENDS ON ALL IMPLEMENTATIONS**
Create test files in parallel:
- `tests/test_path_utils.py` - Path normalization edge cases
- `tests/test_subprocess_utils.py` - Timeout handling verification
- `tests/test_llm_client.py` - Response parsing validation
- `tests/test_handlers/` - Individual handler testing

```python
# Example test structure for path utils
import pytest
from pathlib import Path
from agent_core.path_utils import normalize_path, SecurityViolationError
from agent_core.config import AgentSettings

@pytest.fixture
def temp_workspace(tmp_path: Path) -> tuple[AgentSettings, Path]:
    settings = AgentSettings(workspace_root=tmp_path)
    return settings, tmp_path

def test_normalize_valid_path(temp_workspace):
    settings, workspace = temp_workspace
    target_file = workspace / "test.txt"
    target_file.touch()  # Create empty file
    
    result = normalize_path(target_file, settings)
    assert result is not None
    assert result.resolve() == target_file.resolve()

def test_block_traversal_attack(temp_workspace):
    settings, _ = temp_workspace
    malicious_path = "../../../etc/passwd"
    
    with pytest.raises(SecurityViolationError):
        normalize_path(malicious_path, settings)
```

**Type Checking Validation:** ✅ Run `mypy --strict tests/` after writing all tests  
✅ **Status**: Ready for implementation

---

## Execution Priority Matrix:

| Task ID | Description | Dependencies | Complexity | Type-Check Risk |
|---------|-------------|--------------|------------|------------------|
| 1.1, 1.2 | Foundation layer | None | Low-Medium | ✅ Low |
| 2.1, 2.2 | Core utilities | 1.1, 1.2 | Medium-High | ⚠️ Medium |
| 3.1 | LLM integration | 1.1, 2.2 | Medium | ⚠️ Medium |
| 4.1-4.2 | Command handlers | All previous | High | ❌ High (complex types) |
| 5.1-5.2 | Integration & tests | Complete implementation | Highest | ✅ Low (tests validate typing) |

## Recommended Implementation Order:
1. **Tasks 1.1 → 1.2** (Parallelizable foundation work)
2. **Tasks 2.1, 2.2** (Core utilities with strict validation)
3. **Task 3.1** (LLM client building on secure subprocess layer)
4. **Tasks 4.1→4.2** (Handler pattern enabling modular design)
5. **Tasks 5.1→5.2** (Final integration + comprehensive testing)

Each task includes explicit mypy validation steps to ensure type safety throughout the refactoring process.