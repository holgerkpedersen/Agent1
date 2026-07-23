# Implementation Tasks

# 📋 Implementation Task Plan: `agent.py` Refactor & Stabilization

> **Note on Entities:** No `entities.py` exists in the current codebase. All tasks below are scoped to core modules, utilities, and configuration. Future entity models can be introduced without blocking these tasks.

## 🗂️ File Mapping Overview
| Target File/Module | Responsibility | Priority Focus |
|---|---|---|
| `config.py` (New) | Typed settings, `.env` loading, validation | P2 |
| `exceptions.py` (New) | Structured error hierarchy & custom exceptions | P0 |
| `path_utils.py` (New) | Workspace sandboxing, cross-platform normalization | P0 |
| `agent.py` (Existing) | Core loop, async I/O, search/index logic, tool execution | P0–P1 |
| `tool_router.py` (New) | LLM function schemas, argument validation, routing logic | P2 |
| `logging_config.py` (New) | Structured logging setup, formatters, correlation IDs | P2 |
| `.github/workflows/ci.yml` (New) | CI pipeline, linting, async & security testing | P3 |

---

## 📝 Task Breakdown by File

### 🔹 `config.py` (New)
#### `[CFG-01] Implement Pydantic Settings Configuration`
- **Priority:** P2 (Medium)
- **Dependencies:** None
- **Description:** Extract hardcoded magic strings, URLs, timeouts, and index caps into a typed configuration class for environment-aware deployment.
- **Steps:**
  1. Create `config.py` with `pydantic_settings.BaseSettings`.
  2. Define fields: `model_name`, `workspace_root`, `http_timeout_connect`, `http_timeout_read`, `max_index_size`, `chunk_overlap_bytes`, `log_level`.
  3. Add `.env.example` template and validation rules (e.g., workspace must be absolute).
  4. Export singleton instance for DI across modules.
- **Acceptance Criteria:** No magic strings remain in codebase. Config validates on startup. Dev/staging/prod environments swap via `.env`.

---

### 🔹 `exceptions.py` (New)
#### `[EXC-01] Define Structured Exception Hierarchy`
- **Priority:** P0 (Critical)
- **Dependencies:** None
- **Description:** Replace fragile string-based error returns with a typed exception hierarchy for predictable control flow.
- **Steps:**
  1. Create base `AgentError(Exception)` class.
  2. Define subclasses: `FileOperationError`, `ToolExecutionError`, `PathEscapeError`, `TimeoutError` (alias/wrap).
  3. Add context fields to each exception (`path`, `tool_name`, `raw_error`).
  4. Update imports in downstream modules.
- **Acceptance Criteria:** All tool methods raise exceptions instead of returning error strings. Callers catch via `except AgentError`. IDE autocomplete suggests correct types.

---

### 🔹 `path_utils.py` (New)
#### `[PATH-01] Implement Strict Workspace Sandboxing & Normalization`
- **Priority:** P0 (Critical)
- **Dependencies:** None
- **Description:** Consolidate path logic, enforce sandbox boundaries, and ensure cross-platform consistency.
- **Steps:**
  1. Create `_validate_path(raw: str, workspace_root: Path) -> Path`.
  2. Resolve both target and workspace using `Path.resolve()`. Assert prefix match; raise `PathEscapeError` on mismatch.
  3. Implement `_resolve_workspace_path()` handling POSIX/Windows slashes via `pathlib.PurePosixPath`/`PureWindowsPath`.
  4. Remove duplicate `_normalize_path_strict` and `_normalize_path` from `agent.py`.
- **Acceptance Criteria:** Any `..`, symlink, or absolute escape attempt fails fast. Single source of truth for paths. Cross-platform path tests pass.

---

### 🔹 `agent.py` (Existing)
#### `[AGT-01] Swap Synchronous HTTP Client for Async (`httpx`)`
- **Priority:** P0 (Critical)
- **Dependencies:** `[CFG-01]`, `[EXC-01]`
- **Description:** Replace blocking `urllib.request.urlopen()` with fully async HTTP client to prevent event loop starvation.
- **Steps:**
  1. Import `httpx.AsyncClient`. Instantiate via DI or context manager.
  2. Configure timeouts using values from `config.py`.
  3. Replace synchronous calls with `await client.post()`.
  4. Catch `httpx.TimeoutException` and `httpx.HTTPError`, map to `AgentError` subclasses.
- **Acceptance Criteria:** All LLM calls use `await`. Timeouts raise explicit exceptions. Event loop remains responsive under concurrent requests.

#### `[AGT-02] Unblock Async File I/O`
- **Priority:** P0 (Critical)
- **Dependencies:** `[PATH-01]`, `[EXC-01]`
- **Description:** Prevent blocking during large file reads/writes by delegating to async primitives.
- **Steps:**
  1. Replace all `open()` calls with `aiofiles.open()`.
  2. Wrap synchronous OS calls (`os.path.getsize`, `shutil`) in `asyncio.to_thread()`.
  3. Apply to `read_file`, `write_file`, `append_file`, and search prep routines.
- **Acceptance Criteria:** File operations >5MB no longer freeze concurrent tool execution. Verified via `pytest-asyncio` concurrency test.

#### `[AGT-03] Convert String Errors to Exceptions`
- **Priority:** P0 (Critical)
- **Dependencies:** `[EXC-01]`, `[PATH-01]`
- **Description:** Eliminate string-parsing anti-pattern for error handling across all tool methods.
- **Steps:**
  1. Audit all methods returning `"File not found..."` or similar strings.
  2. Replace returns with `raise FileOperationError(...)` or `ToolExecutionError(...)`.
  3. Update calling logic to use `try/except AgentError` and format LLM responses safely.
- **Acceptance Criteria:** Zero error strings returned from tool methods. Callers handle failures via structured exceptions.

#### `[AGT-04] Fix Chunked Search Boundary Flaw`
- **Priority:** P1 (High)
- **Dependencies:** `[AGT-02]`, `[PATH-01]`
- **Description:** Resolve false-negative search results caused by fixed 8KB chunk boundaries.
- **Steps:**
  1. Replace `f.read(CHUNK_SIZE)` with sliding window: read `CHUNK_SIZE + OVERLAP`.
  2. Strip overlap bytes before yielding/processing next iteration.
  3. Alternatively, implement `mmap.mmap()` for memory-mapped scanning on supported OS.
- **Acceptance Criteria:** Search terms spanning chunk boundaries return correct matches. False-negative rate drops to ~0%.

#### `[AGT-05] Rewrite Semantic Index Cleanup Logic`
- **Priority:** P1 (High)
- **Dependencies:** `[CFG-01]`
- **Description:** Fix unpredictable index eviction that drops high-frequency terms and keeps noise.
- **Steps:**
  1. Cap index at `config.max_index_size`.
  2. On insertion, check size limit. If exceeded, sort by frequency/access count ascending.
  3. Evict bottom 10% (`int(cap * 0.1)`). Delete from dict safely.
- **Acceptance Criteria:** Index never exceeds cap. High-frequency terms survive eviction. Memory stabilizes under continuous indexing.

#### `[AGT-06] Standardize Platform Commands & Shell Safety`
- **Priority:** P1 (High) → P3 (Low for `shell=False` migration)
- **Dependencies:** `[PATH-01]`
- **Description:** Ensure cross-platform search works reliably and safely without shell injection risks.
- **Steps:**
  1. Detect OS via `platform.system()`. Map to `grep -r` (Unix) / `findstr /s /c:` (Win).
  2. Replace `shell=True` with argument lists where possible.
  3. Apply `shlex.quote()` to all dynamic query arguments.
- **Acceptance Criteria:** Search works identically across platforms. Special characters in queries don't break subprocess calls.

#### `[AGT-07] Fix `_files_read` State Tracking & Async Input`
- **Priority:** P3 (Low)
- **Dependencies:** `[PATH-01]`, `[AGT-02]`
- **Description:** Resolve duplicate-read tracking inconsistencies and unblock interactive CLI.
- **Steps:**
  1. Normalize path *before* set lookup: `normalized = self._resolve_workspace_path(path); if normalized not in self._files_read:`.
  2. Replace blocking `input()` with `await asyncio.to_thread(input, prompt)`.
- **Acceptance Criteria:** Duplicate reads correctly tracked across OS variations. CLI remains responsive during agent execution.

---

### 🔹 `tool_router.py` (New)
#### `[ROUTER-01] Implement Structured Tool Routing & Schema Validation`
- **Priority:** P2 (Medium)
- **Dependencies:** `[EXC-01]`, `[CFG-01]`
- **Description:** Replace brittle string-based NL parsing with LLM function calling or regex-backed routing.
- **Steps:**
  1. Define OpenAI-style JSON schemas for each tool (`read_file`, `search`, `write_file`, etc.).
  2. Create router that matches prompts to schemas using regex word boundaries (`\bsearch\b`) or delegates to LLM function calling.
  3. Validate extracted arguments against Pydantic models before execution.
  4. Fallback to explicit `/tool:action args` prefix for deterministic control.
- **Acceptance Criteria:** Agent correctly routes 95%+ of varied phrasings. No substring mangling (`research` → `earch`). Tool arguments validated pre-execution.

---

### 🔹 `logging_config.py` (New) & Global Type Hints
#### `[LOG-01] Setup Structured Logging Across Codebase`
- **Priority:** P2 (Medium)
- **Dependencies:** None
- **Description:** Replace `print()` statements with configurable, production-ready logging.
- **Steps:**
  1. Create logger factory using `logging.getLogger(__name__)`.
  2. Configure JSON formatter for prod, human-readable for dev. Add correlation IDs per tool call.
  3. Replace all `print()` in `agent.py`, `path_utils.py`, `tool_router.py` with appropriate log levels.
- **Acceptance Criteria:** Logs contain timestamps, levels, module names. No stdout pollution. Errors traceable to specific agent iterations.

#### `[TYPE-01] Add Strict Type Hints & Return Contracts`
- **Priority:** P2 (Medium)
- **Dependencies:** All P0/P1 tasks complete
- **Description:** Enforce type safety across the entire module surface area.
- **Steps:**
  1. Annotate all method signatures with `typing` hints (`async def read_file(self, path: str) -> Result[str, FileOperationError]:`).
  2. Use custom `Result[T, E]` wrapper or explicit exception contracts.
  3. Run `mypy --strict` and resolve all violations.
- **Acceptance Criteria:** `mypy --strict` passes with zero errors. IDE autocomplete works flawlessly. Return contracts are explicit.

---

### 🔹 `.github/workflows/ci.yml` (New) & Test Suite
#### `[CI-01] Establish CI/CD Pipeline & Async Testing Framework`
- **Priority:** P3 (Low)
- **Dependencies:** All previous tasks merged to `main` or feature branch
- **Description:** Automate quality gates, async validation, and security checks.
- **Steps:**
  1. Create `.github/workflows/ci.yml` with Python matrix (3.10+).
  2. Add steps for: `ruff check`, `mypy --strict`, `pytest --asyncio-mode=auto -v`, coverage report (`>=80%`).
  3. Include concurrency stress tests & sandbox escape fuzzing in test suite.
  4. Configure branch protection requiring green CI.
- **Acceptance Criteria:** PRs fail if async blocking detected, paths unsanitized, or type hints missing. Green builds required for merge.

---

## 🔗 Dependency Graph (Execution Order)
```
[CFG-01] ──┐
           ├──▶ [EXC-01] ──▶ [PATH-01] ──▶ [AGT-03], [AGT-02], [AGT-07]
           │                        └──▶ [AGT-04], [AGT-05] (via CFG)
[EXC-01] ──┘                         └──▶ [ROUTER-01]
[PATH-01] ──▶ [AGT-01] ──▶ [LOG-01] ──▶ [TYPE-01] ──▶ [CI-01]
```

## ✅ Delivery Checklist
- [ ] Phase 1 (P0): `httpx` swap, sandboxing, exception hierarchy, async I/O
- [ ] Phase 2 (P1): Sliding-window search, LRU index cleanup, unified path resolver, cross-platform commands
- [ ] Phase 3 (P2): Config + `.env`, structured logging, strict type hints, tool routing schema
- [ ] Phase 4 (P3): State normalization fix, `shlex.quote()`, async input, CI/CD pipeline
- [ ] All phases covered by tests → `pytest --asyncio-mode=auto -v` passes green

> **Next Steps for Engineering:**  
> 1. Create feature branch `refactor/agent-stability-v2`  
> 2. Implement tasks in dependency order (P0 first)  
> 3. Run sandbox & concurrency tests after each `[AGT-*]` step before proceeding  
> 4. Open PR with incremental commits per task ID for review

---

## 🔌 Integration: Wire `agent.py` to Generated Modules

All 6 modules exist and compile. `agent.py` must import and use them.

---

### 🔹 `[INT-01]` Replace error strings with `exceptions.py`
- **Priority:** P0
- **Steps:**
  1. `from exceptions import FileOperationError, ToolExecutionError, SecurityViolationError`
  2. Replace all `return f"File not found: {path}"` with `raise FileOperationError(path, ...)`
  3. Replace `return f"Error writing file: {e}"` with `raise FileOperationError(path, raw_error=e)`
  4. In `execute_tool()` catch `AgentError` and format for LLM
- **Acceptance:** Zero error strings from tool methods. Only `raise`.

### 🔹 `[INT-02]` Replace path logic with `path_utils.py`
- **Priority:** P0
- **Steps:**
  1. `from path_utils import resolve_workspace_path, is_within_workspace`
  2. Replace `_safe_path()` body with `resolve_workspace_path(path, Path(self.workspace))`
  3. Replace `_normalize_path_strict()` / `_normalize_path()` with calls to `path_utils`
  4. Remove the old path methods entirely
- **Acceptance:** No path logic duplication. `../` escapes blocked.

### 🔹 `[INT-03]` Use `config.py` for settings
- **Priority:** P0
- **Steps:**
  1. `from config import get_settings`
  2. Replace `DEFAULT_WORKSPACE`, `DEFAULT_MODEL`, hardcoded timeouts with `settings = get_settings()`
  3. Remove hardcoded strings and magic values from agent.py
- **Acceptance:** No magic values in agent.py. Configurable via `.env`.

### 🔹 `[INT-04]` Use `logging_config.py` instead of `print()`
- **Priority:** P1
- **Steps:**
  1. `from logging_config import setup_logging, get_logger`
  2. Create `self.logger = get_logger(__name__)` in `Agent.__init__()`
  3. Replace all `print()` with `self.logger.info()` / `self.logger.error()`
  4. Call `setup_logging()` at startup
- **Acceptance:** No `print()` in agent.py. Structured log output.

### 🔹 `[INT-05]` Use `tool_router.py` for NL parsing
- **Priority:** P1
- **Steps:**
  1. `from tool_router import ToolRouter`
  2. Replace `_parse_natural_language()` body with `ToolRouter` dispatch
  3. Register tools via `router.register_tool(name, func)`
  4. Delete old fragile NL regex/substring logic
- **Acceptance:** `research` no longer becomes `earch`. LLM function calling supported.

### 🔹 `[INT-06]` Use `entities.py` for shared types
- **Priority:** P1
- **Steps:**
  1. `from entities import SearchMatch, SearchResult, ToolCall, ToolResponse`
  2. Replace `dict | str` returns with typed dataclasses
  3. Use `ToolDefinition.to_openai_format()` for LLM tool schemas
- **Acceptance:** All tool results are typed. IDE autocomplete on return values.

---

## 🔗 New Dependency Graph (Integration Phase)
```
[INT-01] (exceptions) ──┐
[INT-02] (path_utils)  ──┼──▶ [INT-03] (config) ──▶ [INT-05] (tool_router)
[INT-06] (entities)     ──┘                       └──▶ [INT-04] (logging)
```