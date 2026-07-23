Here is a **strictly dependency-ordered task plan**. Tasks are arranged so that each step’s prerequisites are fully completed before it begins. This guarantees safe incremental refactoring without breaking imports or runtime behavior.

---
### 🧱 Layer 0: Foundation & Shared Contracts
*Must be built first. All downstream modules import from these.*

| Step | Target File(s) | Dependencies | Key Actions |
|------|----------------|--------------|-------------|
| `0.1` | `legacy/` (new dir) | None | Move/archive top-level duplicates: `entities.py`, `exceptions.py`, `path_utils.py`, `logging_config.py`. Add `__init__.py` to mark as deprecated. |
| `0.2` | `agent_core/config.py` | `0.1` | Scaffold with `pydantic-settings`. Define `.env`/CLI overrides for: `workspace_root`, `llm_api_url`, `timeout_sec`, `similarity_threshold`, `max_index_size`. Add validation & defaults. |
| `0.3` | `agent_core/exceptions.py` | `0.1` | Unify error hierarchy: `AgentError(Base) → FileOperationError, ToolExecutionError, LLMClientError, ConfigValidationError`. Export cleanly in `__init__.py`. |
| `0.4` | `agent_core/path_utils.py` | `0.3` | Finalize sandboxing logic using `pathlib.Path.resolve()`. Add traversal guards (`..`, absolute escapes). Remove manual string slicing. Return validated `Path` objects. |
| `0.5` | `agent_core/__init__.py` | `0.2–0.4` | Export canonical APIs: `from .config import AgentConfig`, `from .exceptions import *`, `from .path_utils import safe_resolve_path`. Add `__future__` annotations to all files. |

---
### ⚡ Layer 1: Async Services & I/O Managers
*Consumes Layer 0. Replaces blocking/sync code in the old monolith.*

| Step | Target File(s) | Dependencies | Key Actions |
|------|----------------|--------------|-------------|
| `1.1` | `agent_core/llm_client.py` | `0.2, 0.3` | Replace `urllib.request` with `httpx.AsyncClient`. Implement connection pooling, `tenacity` retries, structured timeouts (`asyncio.wait_for`). Raise typed `LLMClientError`. |
| `1.2` | `agent_core/workspace_manager.py` | `0.4, 0.3, 0.2` | Wrap all disk I/O in `aiofiles` or `asyncio.to_thread()`. Use `safe_resolve_path()` for every read/write. Add sandbox boundary checks. Raise `FileOperationError`. |
| `1.3` | `agent_core/cache_manager.py` | `1.2, 0.2` | Extract cache logic from old `agent.py`. Implement async-safe in-memory/disk cache with TTL & size limits (`max_index_size`). Use workspace manager for persistence. |

---
### 🔄 Layer 2: Orchestration & Routing Engine
*Consumes Layer 1. Replaces the workflow/scoring logic inside the monolith.*

| Step | Target File(s) | Dependencies | Key Actions |
|------|----------------|--------------|-------------|
| `2.1` | `agent_core/tool_router.py` | `0.3, 1.1` | Unify tool routing & scoring. Remove duplicated error classes. Use `LLMClient` for semantic matching if needed. Return structured `ToolExecutionResult`. |
| `2.2` | `agent_core/orchestrator.py` | `1.1, 1.2, 1.3, 2.1, 0.2` | Build async workflow engine: prompt → route → execute tool/cache → format response → update context. Replace `subprocess.run()` compilation checks with `asyncio.to_thread()`. Preserve correlation IDs in logs. |
| `2.3` | `agent.py` (legacy wrapper) | `2.2, 0.5` | Slim down to a thin entry point that imports from `agent_core`. Add `DeprecationWarning` for direct usage. Keep backward-compatible CLI signatures temporarily. |

---
### 🖥️ Layer 3: Interface & Entry Points
*Consumes Layer 2. Replaces fragile `input().split()` parsing.*

| Step | Target File(s) | Dependencies | Key Actions |
|------|----------------|--------------|-------------|
| `3.1` | `cli/repl.py` (new) | `2.2, 0.2` | Implement interactive REPL using `prompt_toolkit`. Add history, tab-completion, safe flag parsing (`--workspace`, `--dry-run`). Bind commands to orchestrator methods. |
| `3.2` | `cli/script.py` or `main.py` (new) | `2.2, 0.2` | Implement headless execution with `argparse`. Support workflow queues, JSON output, and CI-friendly modes. Load config via CLI overrides. |
| `3.3` | `pyproject.toml` / `setup.cfg` | `3.1, 3.2` | Define entry points: `console_scripts = {"agent-cli": "cli.repl:main", "agent-run": "cli.script:main"}`. Pin dependencies (`httpx`, `prompt_toolkit`, `pydantic-settings`, etc.). |

---
### 🧪 Layer 4: Testing, Linting & CI/CD
*Consumes Layers 0–3. Validates the entire refactored stack.*

| Step | Target File(s) | Dependencies | Key Actions |
|------|----------------|--------------|-------------|
| `4.1` | `tests/conftest.py`, `tests/fixtures/` | All above | Configure `pytest-asyncio`, mock `httpx.AsyncClient`, create temp workspace sandbox, inject test config. |
| `4.2` | `tests/test_config.py`, `test_path_utils.py`, `test_exceptions.py` | `0.2–0.4` | Unit tests for validation, traversal guards, error inheritance, and env override precedence. |
| `4.3` | `tests/test_llm_client.py`, `test_workspace_manager.py` | `1.1, 1.2` | Async mocks (`respx`/`aioresponses`). Verify non-blocking behavior, retry logic, timeout handling, sandbox enforcement. |
| `4.4` | `tests/test_orchestrator.py`, `test_tool_router.py` | `2.1, 2.2` | Integration tests: mock LLM + workspace I/O → validate full pipeline execution & cache hits. Assert typed exception propagation. |
| `4.5` | `.github/workflows/ci.yml`, pre-commit hooks | All above | Add `ruff`, `mypy`, `black`, `pytest`. Enforce coverage ≥80% on `agent_core/`. Block PRs on type/lint failures or event-loop blocking warnings. |

---
### 🔗 Dependency Graph (Topological Sort)
```
legacy/ → config.py → exceptions.py → path_utils.py → __init__.py
                                    ↓              ↓
                              llm_client.py   workspace_manager.py
                                    \                /
                                     cache_manager.py
                                            ↓
                                      tool_router.py
                                            ↓
                                    orchestrator.py
                                            ↓
                               repl.py ←→ script.py (main entry)
                                            ↓
                                   tests/ + CI/CD pipeline
```

---
### ✅ Execution Guidelines
1. **Branch Strategy**: Work on `refactor/core-modernization`. Merge layer-by-layer via feature branches (`feat/config`, `feat/async-client`, etc.).
2. **Safety Net**: Until Layer 3 is complete, keep `agent.py` as a deprecated shim that delegates to `orchestrator.py`. Run both in parallel for 1 sprint.
3. **Async Verification**: Use `pytest-asyncio` + `aiostream` or custom tracers to assert the event loop never blocks >10ms during LLM/file ops.
4. **Rollback Point**: Each layer is independently testable. If Layer 2 breaks, revert to legacy shim without touching Layers 0–1.

Would you like this exported as a Markdown checklist with checkboxes, or converted into a GitHub Projects/Jira-compatible CSV/JSON format?