# Detailed Coding Plan: Agent Codebase Refactoring & Fixes

## Overview
This plan addresses critical bugs, architectural issues, and code quality concerns identified in the review. The approach prioritizes **mypy strict compliance**, modularization, consistency improvements, and safe concurrency practices.

---

## Phase 1: Immediate Bug Fixes (Critical Issues)

### Task 1.1 – Fix Path Normalization Logic (`agent.py`)
**Problem**: Incorrectly maps Unix paths like `/tmp/file` to Windows-style `C:\tmp\file`.

#### Changes Required:
- Replace hardcoded logic with platform-aware normalization using `os.name == 'nt'` or detect WSL via environment variables.
- Use standard libraries such as [`pathlib`](https://docs.python.org/3/library/pathlib.html), [`shutil.which()`](https://docs.python.org/3/library/shutil.html#shutil.which) where applicable.

```python
import os
from pathlib import Path

def _normalize_path_strict(self, path: str) -> Optional[Path]:
    if not isinstance(path, str):
        raise TypeError("Expected string for path")
    
    p = Path(path).resolve(strict=False)  # Safe resolution without requiring existence
    
    if os.name == "nt" and path.startswith("/"):
        normalized_path = self._convert_unix_to_windows(p)
        return normalized_path.resolve()

    elif sys.platform.startswith('linux'):
        # Assume native Linux behavior; no conversion needed unless explicitly configured otherwise.
        pass

    try:
        resolved = p.absolute().resolve(strict=True)  # Resolve symlinks safely
        if not self._is_within_workspace(resolved):
            logger.warning("Path outside workspace detected.")
            return None
        return resolved
    except FileNotFoundError as e:
        logger.error(f"File not found during strict resolve: {e}")
        return None
```

> ✅ Ensure all functions have proper type annotations (`Optional[Path]`, etc.)  
> 🔒 Add unit tests covering edge cases (symlinks, traversal attempts)

---

### Task 1.2 – Secure Subprocess Execution With Timeouts
**Problem**: `subprocess.run(...)` calls can block indefinitely.

#### Changes Required:
- Wrap all subprocess invocations with explicit timeouts and error handling.

```python
import asyncio.subprocess as async_subprocess

async def _run_compilation_check(self, filepath_str: str) -> bool:
    proc = await async_subprocess.create_exec(
        ["python", "-m", "py_compile", filepath_str],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("Compilation timed out after 30 seconds.")
        return False

    if proc.returncode != 0:
        logger.debug(stderr.decode())
        return False
    return True
```

> ⚠️ Migrate blocking calls to async equivalents or use executors with timeouts.  
> 🧪 Write mock-based tests simulating slow/hanging processes.

---

### Task 1.3 – Improve LLMClient.chat() Return Consistency
**Problem**: Inconsistent error reporting via strings instead of structured exceptions/types.

#### Changes Required:
- Define a typed result object or union type indicating success/failure states.

```python
from typing import Union, NamedTuple

class LlmResponse(NamedTuple):
    content: str
    is_error: bool = False

def chat(self, prompt: str) -> Union[LlmResponse, Exception]:
    try:
        response = requests.post(...)  # Or httpx/aiohttp for async support
        if not response.ok:
            return LlmResponse(content=str(response.text), is_error=True)
        
        data = response.json()
        choices = data.get("choices", [])
        if len(choices) == 0:
            return LlmResponse(content="No response from LLM.", is_error=True)

        first_choice = choices[0]
        text_response = first_choice["message"]["content"]
        return LlmResponse(content=text_response.strip(), is_error=False)

    except RequestException as e:
        logger.exception("HTTP request failed")
        return LlmResponse(content=f"[LM Studio error: {str(e)}]", is_error=True)
```

> 💡 Consider integrating with a unified exception hierarchy defined below.  
> 📊 Update callers to check `.is_error` flag rather than matching substrings.

---

### Task 1.4 – Fix Method Signature Regex Extraction (`extract_signatures`)
**Problem**: Regex only matches single-space indentation before `def`.

#### Changes Required:
- Broaden regex pattern to accept any whitespace including tabs/multiple spaces.

Before:
```regex
r'^\s+def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(.+?))?\s*:'
```

After:
```regex
r'^[\t ]+(async )?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(.*))?:\s*$'
```

Also, ensure that `re.MULTILINE` mode is enabled when scanning large files.

> ✅ Test against various Python formatting styles (PEP8-compliant vs tab-indented).  
> 🛠 Consider leveraging AST parsing as alternative strategy for robustness.

---

## Phase 2: Structural Improvements & Modularization

### Task 2.1 – Consolidate Duplicate Exception Definitions
**Problem**: Multiple modules define overlapping exception classes causing confusion and potential shadowing issues.

#### Strategy:
Create a centralized `exceptions.py` module inside `agent_core/` that defines all domain-specific exceptions once. Migrate other files to import from this location exclusively.

Example structure:

```python
# agent_core/exceptions.py
class AgentBaseError(Exception): ...
class FileOperationError(AgentBaseError): ...
class SecurityViolationError(AgentBaseError): ...

# In agent.py or anywhere else needing an error:
from .agent_core.exceptions import FileOperationError
raise FileOperationError("Invalid file access attempt")
```

> 🗑 Deprecate duplicate definitions in legacy modules (`entities.py`, `path_utils.py`).  
> ⚖️ Introduce backward compatibility shims temporarily if needed.

---

### Task 2.2 – Refactor Monolithic Functions Into Handlers
**Problem**: Massive functions like `run_interactive()` violate SRP and are hard to maintain/test.

#### Strategy:
Break down monolithic logic into discrete handler classes implementing a common interface.

```python
from abc import ABC, abstractmethod
from typing import List

class CommandHandler(ABC):
    @abstractmethod
    async def handle(self, args: List[str]) -> int: ...  # Returns exit code

class AnalyzeCommand(CommandHandler):
    async def handle(self, args: List[str]) -> int:
        ...

class ImplementCommand(CommandHandler):
    async def handle(self, args: List[str]) -> int:
        ...

COMMAND_REGISTRY = {
    "analyze": AnalyzeCommand(),
    "implement": ImplementCommand()
}
```

Update main entry point to route based on registry lookup.

> 🧵 Maintain backward compatibility with existing CLI usage patterns.  
> 📦 Encapsulate each subcommand in its own file/module for clarity.

---

### Task 2.3 – Centralize Configuration Management Using Dataclasses/Pydantic
**Problem**: Scattered constants and magic values make configuration brittle.

#### Strategy:
Define immutable settings class using `@dataclass(frozen=True)` or Pydantic BaseModel depending on validation needs.

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class AgentSettings:
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    llm_api_url: str = "http://localhost:1234/v1"
    max_concurrent_tools: int = 5
    search_command_timeout_sec: float = 30.0

settings = AgentSettings()
```

Pass instances around constructors or context managers instead of global access.

> 🛡 Validate inputs at startup time (e.g., check URL format, valid directories).  
> 📋 Document defaults clearly in docstrings/comments.

---

### Task 2.4 – Structured Logging Integration Across Modules
**Problem**: Print statements scattered throughout codebase hinder debugging and monitoring capabilities.

#### Strategy:
Replace `print()` with structured logging using Python’s built-in `logging` module integrated with existing config (`logging_config.py`).

```python
import logging
logger = logging.getLogger(__name__)

# Old way:
print("Processing file...")

# New way:
logger.info("Started processing", extra={"filename": filename})
```

Ensure consistent log levels (INFO/WARNING/ERROR) and structured metadata fields.

> 📈 Enable JSON-formatted logs for easier ingestion into observability stacks.  
> 🕵️‍♂️ Avoid logging sensitive information like full file contents or credentials.

---

### Task 2.5 – Async File I/O Migration Using `aiofiles`
**Problem**: Blocking disk operations inside async contexts degrade performance and responsiveness.

#### Strategy:
Replace synchronous reads/writes with `aiofiles.open()` wrapper functions.

```python
import aiofiles

async def read_file_async(filepath: str) -> Optional[str]:
    try:
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            return await f.read()
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
```

Apply this consistently across all I/O-heavy sections.

> 🔄 Audit entire codebase for remaining blocking calls and migrate them gradually.  
> 📊 Benchmark before/after to measure impact on throughput/latency.

---

## Phase 3: Testing Infrastructure Setup

### Task 3.1 – Establish Unit Tests for Core Components
**Problem**: Zero test coverage exposes risk in future changes.

#### Targets for Initial Coverage:
- Path normalization/validation logic (traversal protection, symlink resolution)
- LLM response parsing/error handling branches
- Tool execution dispatch flow (mock external dependencies)
- Semantic index cleanup behaviors under memory pressure scenarios

Use `pytest`, `unittest.mock` for mocks/stubs.

Sample test scaffold:

```python
import pytest
from unittest import mock
from agent_core.path_utils import _normalize_path_strict

@pytest.fixture
def temp_workspace(tmpdir):
    return tmpdir.mkdir("workspace")

def test_normalize_valid_absolute(temp_workspace):
    path = str(temp_workspace.joinpath("file.txt"))
    result = _normalize_path_strict(path)
    assert result is not None
```

> 🧪 Automate test runs via CI pipelines (GitHub Actions, GitLab CI).  
> 📈 Track coverage metrics using `coverage.py`.

---

## Phase 4: Cleanup & Finalization Tasks

### Task 4.1 – Resolve Circular Imports & Cross-Module Reference Issues
**Problem**: Implicit dependencies between modules create fragile import order requirements.

#### Strategy:
Audit all inter-module imports and restructure to avoid cycles:

- Move shared types/constants into lower-level utility packages (`agent_core/types.py`)
- Use lazy imports where necessary (only when actually used)
- Enforce unidirectional dependency graph through linting tools like `importchecker` or manual audits

> 🔄 Review `__init__.py` files carefully—avoid importing too much eagerly.  
> 📜 Apply PEP 484 typing discipline rigorously to catch mismatches early.

---

### Task 4.2 – Eliminate Bare Except Clauses & Improve Error Handling Granularity
**Problem**: Broad exception catches hide real bugs and complicate debugging.

#### Strategy:
Replace generic `except Exception:` blocks with specific ones targeting known failure modes.

```python
# Bad:
try:
    content = fp.read_text(...)
except Exception:
    pass

# Good:
try:
    content = fp.read_text(encoding="utf-8")
except FileNotFoundError as e:
    logger.warning(f"File disappeared unexpectedly: {e}")
except PermissionError as e:
    logger.error(f"Permission denied reading file: {e}")
else:
    process(content)
```

> 🚨 Never silently ignore unexpected exceptions.  
> 📋 Log contextual info whenever catching broad categories like `OSError`.

---

## Summary Checklist Before Merging

| Item | Status |
|------|--------|
| All modified functions pass mypy strict checks | ❏ |
| No bare except clauses remain unhandled | ❏ |
| Async file I/O replaces all blocking operations | ❏ |
| Unified exception hierarchy adopted across modules | ❏ |
| Structured logging replaces print statements | ❏ |
| Unit tests cover critical components (>70% coverage goal) | ❏ |
| Documentation updated for new APIs/configurations | ❏ |

--- 

Let me know which phase/task you'd like to start working on next!