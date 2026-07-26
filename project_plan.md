# Detailed Coding Plan: Fix Critical Bugs & Refactor Agent Core Module

## Overview
This plan addresses all identified issues from the code review analysis, prioritizing critical bugs that cause immediate runtime failures. The implementation must pass **mypy strict type checking**.

---

## Phase 1: Immediate Fixes (Critical Runtime Failures)

### Task 1.1 — Fix `normalize_path` Undefined Symbol  
**File**: `agent_core/path_utils.py`, `handlers/analyze_handler.py`

#### Action Steps:
1. Rename `_validate_path()` function to `normalize_path()` in `path_utils.py`.
2. Update export alias in `__init__.py`: 
   ```python
   from .path_utils import normalize_path as validate_path
   ```
3. Change handler imports to use correct name and relative depth:
   ```python
   # Before (broken):
   from ...path_utils import normalize_path

   # After (fixed):
   from ..path_utils import normalize_path  # Correct depth + symbol exists now
   ```

#### Mypy Compliance Notes:
- Ensure return type annotation is `str`.
- Add explicit parameter types (`Union[str, os.PathLike]`).

---

### Task 1.2 — Fix Async Blocking I/O in `_fallback_search`  
**File**: `handlers/analyze_handler.py`

#### Action Steps:
Replace synchronous file reading with async-safe pattern using executor:
```python
async def _read_chunk_async(filepath: str, chunk_size: int) -> Optional[str]:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, lambda: Path(filepath).open('r', encoding='utf-8').read(chunk_size)
        )
    except Exception as exc:
        logger.error("Failed to read chunk asynchronously", exc_info=exc)
        return None
```

Update `_fallback_search()` accordingly.

#### Mypy Compliance Notes:
- Use `Optional[str]` instead of raw strings where nullability possible.
- Avoid untyped lambda expressions; wrap logic in typed helper function.

---

### Task 1.3 — Fix `shlex.split` Misuse  
**File**: Anywhere user input parsing occurs (likely command dispatchers)

#### Action Steps:
Use POSIX mode explicitly for proper quote handling:
```python
def parse_user_input(user_input: str) -> List[str]:
    parts = shlex.split(user_input, posix=True)  # Ensures quoted args handled correctly
    if len(parts) == 0:
        raise ValueError("Empty input after parsing")
    return parts[:20]  # Truncate safely without losing semantics
```

#### Mypy Compliance Notes:
- Return `List[str]`, never bare list.
- Catch specific exception types rather than generic catch-all.

---

### Task 1.4 — Remove Redundant Compile Check  
**File**: Implement phase logic (likely in builder or writer modules)

#### Action Steps:
Remove second compilation step post-write unless conditional difference exists. If retained, ensure only one call per file modification cycle.

#### Mypy Compliance Notes:
- Guard against duplicate calls via state tracking flags (`bool` field).
- Log redundant attempts to debug level for audit trail visibility.

---

## Phase 2: Structural Refactoring (High Priority)

### Task 2.1 — Centralize Exceptions  
**Files**: `agent.py`, all submodules defining exceptions locally

#### Action Steps:
Create single source-of-truth module: `agent_core/exceptions.py`. Move definitions there and replace local redefinitions with centralized imports across project.

Example migration snippet:
```python
# In each affected file BEFORE refactor:
class AgentError(Exception): ...

# AFTER refactor:
from agent_core.exceptions import AgentError, FileOperationError
```

#### Mypy Compliance Notes:
- All exception classes inherit from `Exception`.
- Provide full docstrings and inheritance chain clarity.
- Export list maintained in `__init__.py` for convenience access.

---

### Task 2.2 — Correct Import Depth Errors  
**Files**: Handlers referencing core modules incorrectly (`handlers/analyze_handler.py`, tests)

#### Action Steps:
Audit all triple-dot imports (`...`) and adjust to double-dot (`..`) when targeting parent package level correctly. Validate actual module locations match expected paths.

Corrected example block:
```python
# Before (wrong depth):
from ...config import AgentSettings  
from ...exceptions import FileOperationError  

# After (corrected):
from ..config import AgentSettings  
from ..exceptions import FileOperationError  
```

#### Mypy Compliance Notes:
- No relative imports beyond valid package boundaries.
- Static verification through mypy `--strict` mode confirms resolution success.

---

### Task 2.3 — Resolve Circular Import Risks  
**Files**: `agent_core/__init__.py`, dependencies involving logging/context/config interplay

#### Action Steps:
Refactor implicit fallback logic in `logging_config.py`: replace conditional imports with explicit dependency injection pattern ensuring deterministic load order.

Proposed approach:
```python
# In logging_config.py BEFORE refactor:
try:
    from context_management import CORRELATION_ID_CTX  
except ImportError: ...  # Fragile divergence path introduced here

# AFTER refactor:
import inject
@inject.params(correlation_ctx='CORRELATION_ID_CTX')
def configure_logging(correlation_ctx: ContextType) -> None:
    ...  # Unified behavior regardless of standalone execution context
```

#### Mypy Compliance Notes:
- Explicit interface contracts defined via abstract base classes or protocols.
- Type annotations enforced even under dynamic injection scenarios.

---

## Phase 3: Typing & Safety Enhancements (Medium Priority)

### Task 3.1 — Enforce Structured Result Types  
**Files**: Public API functions returning raw strings/lists/objects inconsistently

#### Action Steps:
Adopt `Result[T]` pattern already partially implemented elsewhere—standardize usage everywhere applicable including handlers, parsers, validators.

Implementation scaffold:
```python
@dataclass(frozen=True)
class Success(Result):
    value: T

@dataclass(frozen=True)
class Failure(Result):
    error: str
    details: Optional[dict[str, Any]] = None
```

Apply consistently across interfaces previously returning ambiguous outputs.

#### Mypy Compliance Notes:
- Generics parameterized properly (`Generic[T]`).
- Discriminated unions supported with exhaustive pattern matching guarantees.

---

### Task 3.2 — Standardize Frozen Dataclass Defaults  
**File**: `entities.py` and other entity definitions violating immutability contract

#### Action Steps:
Audit all mutable default factories (e.g., lists) used inside frozen dataclasses—replace inconsistent patterns uniformly with `field(default_factory=list)` syntax compliant everywhere.

Before vs after comparison:
```python
# Before violation:
@dataclass(frozen=True)
class SomeEntity:
    tags = []  # Mutable default violates immutability!

# After compliance:
@dataclass(frozen=True)
class SomeEntity:
    tags: List[str] = field(default_factory=list)  # Safe immutable instantiation guaranteed
```

#### Mypy Compliance Notes:
- All fields annotated explicitly with appropriate generic container types.
- Zero runtime mutations allowed due to `frozen=True` enforcement combined with safe defaults strategy.

---

## Phase 4: Cleanup Pass (Low Priority Recommendations)

### Task 4.1 — Eliminate Duplicate Exception Class Definitions  
**Scope**: Full codebase sweep for redundant exception declarations post-centralized refactor completion.

#### Action Steps:
Run grep-based search identifying remaining instances of duplicated class definitions—remove them once confirmed superseded by central module imports established earlier in Phase 2.1.

#### Mypy Compliance Notes:
- Zero duplicate symbol occurrences detected after cleanup run completes successfully.

---

### Task 4.2 — Normalize Path Conversion Logic Duplication  
**Scope**: Identify repeated `/c/` → `C:\` conversion logic blocks scattered across multiple files (~7+). Consolidate into reusable utility function within `path_utils.py`.

#### Action Steps:
Extract shared normalization routine into dedicated helper method—invoke centrally from all call sites replacing inline copy-paste implementations previously present.

New consolidated signature proposal:
```python
def convert_to_windows_path(path_str: str) -> str:
    if path_str.startswith('/c/') or path_str.startswith('/C/'):
        return 'C:\\' + path_str.lstrip('/').replace('/', '\\')
    else:
        # Handle other drive-letter mappings similarly...
        pass
```

#### Mypy Compliance Notes:
- Single canonical implementation accepted by mypy strict checker without ambiguity.
- All callers updated to delegate transparently to unified service layer abstraction above.

---

## Verification Checklist Pre-Merge Submission

| Item | Status Required |
|------|----------------|
| ✅ `normalize_path` exists and importable from correct path in handlers/tests | Must pass |
| ✅ No blocking I/O inside async coroutines anywhere | Must pass |
| ✅ Proper POSIX mode enabled for all shlex operations | Must pass |
| ✅ Only one compile-check invocation per file write cycle | Must pass |
| ✅ All exceptions imported centrally—not locally redefined | Must pass |
| ✅ Relative import depths aligned with actual package structure | Must pass |
| ✅ Logging configuration deterministic via dependency injection—not fallback guessing | Must pass |
| ✅ Result[T] pattern adopted consistently across public APIs | Must pass |
| ✅ Frozen dataclasses avoid mutable defaults entirely | Must pass |
| ✅ mypy strict passes cleanly on entire codebase after changes applied | Must pass |

--- 

Let me know if you'd like this converted into actionable tickets or automated scripts targeting these fixes directly!