# Coding Plan

# 🏗️ Software Architecture Implementation Plan
**Project**: Agent Core Framework Refactoring & Hardening  
**Target Python Version**: 3.10+ (leveraging modern union syntax, `pathlib`, `contextvars`)  
**Priority Model**: P0 (Critical/Blocking) → P1 (High Impact) → P2 (Maintenance/Quality)

---

## 📊 Prioritization & Dependency Matrix
| Phase | Focus Area | Priority | Dependencies | Estimated Effort |
|-------|------------|----------|--------------|------------------|
| 1     | Module Separation & Core Structure | **P0** | None | 2-3 days |
| 2     | Path Security & Workspace Sandboxing | **P0** | Phase 1 | 3-4 days |
| 3     | Async Context Management & Logging | **P1** | Phase 1, 2 | 2-3 days |
| 4     | Type Safety, Linting & Cleanup | **P1** | Phase 1-3 | 2 days |
| 5     | Testing Strategy & CI Integration | **P0** | Phase 1-4 | 3-4 days |

---

## 📅 Phased Implementation Plan

### 🔹 Phase 1: Architectural Foundation & Module Separation (P0)
**Goal**: Eliminate duplication, enforce single source of truth, establish clean package boundaries.

| Step | Action | Deliverable |
|------|--------|-------------|
| 1.1 | Create `agent_core/` package structure with `__init__.py` exposing public API only | Clean namespace: `from agent_core import AgentError, validate_path, setup_logging` |
| 1.2 | Extract all exception classes into `exceptions.py`. Remove built-in name shadows (`TimeoutError` → `ToolExecutionTimeoutError`) | Single inheritance tree: `Exception → AgentError → {FileOperation, SecurityViolation, ToolExecutionTimeout, ...}` |
| 1.3 | Move path utilities to `path_utils.py`, models to `models.py`, logging config to `logging_config.py` | Modular structure with zero circular imports |
| 1.4 | Remove inline spec tags `(Phase 4.1)`, `[PATH-01]`. Migrate to module-level docstrings & `pyproject.toml` metadata | Linter-clean codebase, improved readability |

✅ **Verification**: `python -m compileall agent_core/`, `importlib.util.find_spec("agent_core")` succeeds, zero import conflicts.

---

### 🔹 Phase 2: Secure Path Validation & Workspace Sandboxing (P0)
**Goal**: Prevent directory traversal, handle symlinks deterministically, enforce strict workspace boundaries.

| Step | Action | Deliverable |
|------|--------|-------------|
| 2.1 | Implement `_validate_path(raw: str, workspace_root: Path, follow_symlinks: bool = True) -> Path` using `Path.resolve()` + `relative_to()` | Secure path resolver with configurable symlink policy |
| 2.2 | Add explicit boundary enforcement: raise `SecurityViolationError` on escape attempts or invalid inputs | Hardened sandboxing layer |
| 2.3 | Create `WorkspaceSandbox(workspace_root, policies)` context manager that auto-validates all I/O operations | Declarative security wrapper for agent file ops |
| 2.4 | Replace legacy string-based path checks across the codebase with `_validate_path()` | Zero raw `open()`, `Path()` usage outside validation layer |

✅ **Verification**: Unit tests pass for: absolute/relative inputs, `../` traversal, symlink loops, empty/malformed paths, workspace boundary enforcement.

---

### 🔹 Phase 3: Async Context Management & Structured Logging (P1)
**Goal**: Ensure async-safe correlation tracking, robust JSON serialization, and observability readiness.

| Step | Action | Deliverable |
|------|--------|-------------|
| 3.1 | Implement `CorrelationIdContext` with proper `contextvars.Token` lifecycle (`__enter__`/`__exit__`) | Thread/task-safe correlation scoping |
| 3.2 | Build `SafeJsonEncoder(json.JSONEncoder)` handling `datetime`, `Path`, `Exception`, `UUID` | Crash-free JSON logging under all payloads |
| 3.3 | Configure structured logger via `logging.config.dictConfig()` with `CorrelationIdFilter` and safe encoder | Production-ready log pipeline |
| 3.4 | Inject correlation ID into async task names & error traces for distributed tracing compatibility | OpenTelemetry-ready context propagation |

✅ **Verification**: Async isolation tests confirm no cross-task leakage; JSON logs parse without errors on edge payloads; `__exit__` cleanup verified under exceptions.

---

### 🔹 Phase 4: Type Safety, Modern Idioms & Code Quality (P1)
**Goal**: Enforce consistency, eliminate technical debt, prepare for static analysis pipelines.

| Step | Action | Deliverable |
|------|--------|-------------|
| 4.1 | Standardize type hints: `str \| Path`, `datetime \| None`, explicit return types on all helpers | Fully typed public & internal APIs |
| 4.2 | Replace legacy `typing.Union`/`Optional` with PEP 604 syntax (Python 3.10+) | Modern, readable type annotations |
| 4.3 | Run `ruff check --fix`, `mypy --strict`, resolve all violations | Zero lint/type errors in CI |
| 4.4 | Add module-level docstrings with responsibility boundaries & usage examples | Self-documenting codebase |

✅ **Verification**: `mypy` passes at strict level, `ruff` clean, type coverage >90% via `coverage run --source=agent_core`.

---

### 🔹 Phase 5: Testing Strategy & CI Integration (P0)
**Goal**: Guarantee reliability, security validation, and automated quality gates.

| Step | Action | Deliverable |
|------|--------|-------------|
| 5.1 | Write unit tests using `pytest`, `pyfakefs` for path resolution & sandbox enforcement | Deterministic FS mocking without real I/O |
| 5.2 | Add async context propagation tests with `pytest-asyncio` and task isolation verification | Context leak prevention guarantees |
| 5.3 | Integrate CI pipeline: lint → type-check → unit tests → coverage (>80%) → security scan (`bandit`) | Automated quality gate on PR/merge |
| 5.4 | Add property-based tests (`hypothesis`) for path validation edge cases | Fuzz-resistant boundary checks |

✅ **Verification**: CI green, coverage report generated, `bandit` clean, no flaky tests.

---

## 🔗 Dependency Graph & Parallelization Strategy
```
Phase 1 (Structure) 
    ├──→ Phase 2 (Path Security) ──┐
    ├──→ Phase 3 (Context/Logging) ├─→ Phase 4 (Type Safety/Linting) → Phase 5 (Tests/CI)
    └──→ Phase 4 can start early on non-security modules once imports are stable
```
**Parallelization Tip**: Once Phase 1 is merged, developers can work on Phases 2 & 3 simultaneously. Phase 4 should run continuously as a pre-commit hook.

---

## ⚠️ Risk Mitigation & Architectural Decisions

| Risk | Mitigation Strategy |
|------|---------------------|
| **Symlink traversal bypass** | `Path.resolve()` follows symlinks by default. Enforce boundary check on the *resolved* path. Add `follow_symlinks=False` policy mode that raises on symlink detection inside workspace. |
| **Context variable leakage in async pools** | Use `contextvars.copy_context()` when submitting tasks to thread/async executors. Wrap agent dispatchers in `CorrelationIdContext`. |
| **JSON logging performance overhead** | Batch log serialization, use `orjson` or `ujson` if throughput >10k req/s. Keep encoder lightweight; defer heavy object dumps to async sinks. |
| **Breaking changes during refactor** | Maintain backward-compatible aliases temporarily (`TimeoutError = ToolExecutionTimeoutError`). Bump version to `v1.0.0` post-refactor with clear migration guide. |

---

## ✅ Immediate Next Steps (Sprint 1: Days 1-5)
1. **Branch**: `refactor/core-architecture-v2`
2. **Execute Phase 1**: Create package structure, extract modules, clean imports
3. **Run baseline checks**: `ruff`, `mypy`, `python -c "import agent_core"`
4. **Commit & PR**: Request architecture review before proceeding to security hardening

Would you like me to generate the complete production-ready source for any specific module (e.g., `agent_core/path_utils.py` with symlink policy enforcement, or `agent_core/logging_config.py` with async-safe JSON pipeline)? I can also provide a ready-to-use `pyproject.toml` + CI workflow template aligned with this plan.