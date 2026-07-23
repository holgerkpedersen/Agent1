# 📋 Strategic Refactoring & Modernization Plan

Based on your expert code review, this plan transforms the fragmented prototype into a production-ready, maintainable AI agent framework. It is structured as a **4-phase roadmap** with clear deliverables, risk controls, and acceptance criteria.

---

## 🎯 Primary Objectives
1. Eliminate blocking I/O & event loop starvation in async contexts
2. Consolidate architecture around `agent_core/` as the single source of truth
3. Replace fragile parsing, hardcoded paths, and string-based error handling with robust, typed patterns
4. Establish automated testing, linting, and CI/CD guardrails
5. Maintain backward compatibility during transition via deprecation layers

---

## 🗺️ Phased Implementation Roadmap

### 🔹 Phase 1: Architecture Consolidation & Foundation (Days 1–3)
| Task | Description | Deliverable |
|------|-------------|-------------|
| **Archive duplicates** | Move top-level `entities.py`, `exceptions.py`, `path_utils.py`, `logging_config.py` to `legacy/` or delete after verification | Clean root directory; `agent_core/` declared canonical |
| **Unify exception hierarchy** | Define base `AgentError` + typed subclasses (`FileOperationError`, `ToolExecutionError`, `LLMClientError`) in `agent_core/exceptions.py` | Single, importable error module used across all modules |
| **Add future annotations** | Prepend `from __future__ import annotations` to every `.py` file | Forward-compatible type hints; no eager evaluation crashes |
| **Scaffold config layer** | Create `agent_core/config.py` using `pydantic-settings` with `.env` fallbacks for workspace, API URLs, timeouts, thresholds | Centralized, validated configuration object |

### 🔹 Phase 2: Async Migration & God Module Decomposition (Days 4–8)
| Task | Description | Deliverable |
|------|-------------|-------------|
| **Replace `urllib` with `httpx.AsyncClient`** | Migrate `LLMClient.chat()` to async HTTP with connection pooling, retries (`tenacity`), and structured timeouts | Non-blocking LLM client; proper `asyncio.TimeoutError` handling |
| **Extract God Module** | Split `agent.py` into: `llm_client.py`, `workspace_manager.py`, `tool_router.py`, `orchestrator.py`, `cache_manager.py` | Modular, testable components (~150 lines each) |
| **Fix cross-OS path translation** | Replace string slicing with `pathlib.Path(workspace).resolve() / user_path`. Add sandbox validation via `agent_core/path_utils.py` | Secure, OS-agnostic path resolution with traversal guards |
| **Offload blocking calls** | Wrap `subprocess.run()` and sync file reads in `asyncio.to_thread()` or migrate to `aiofiles` | Event loop remains responsive during compilation/file ops |

### 🔹 Phase 3: CLI/Config Unification & Error Handling Standardization (Days 9–12)
| Task | Description | Deliverable |
|------|-------------|-------------|
| **Modernize REPL** | Replace `input().split()` with `prompt_toolkit` or `cmd2`. Implement tab-completion, history, and flag parsing | Robust interactive CLI with predictable argument routing |
| **Add script-mode parser** | Integrate `argparse` for headless/workflow execution (`--workspace`, `--task`, `--dry-run`) | Dual-mode interface (REPL + CLI) |
| **Standardize error propagation** | Remove `except Exception: return f"[Error: {e}]"`. Raise typed exceptions or use `returns.Result[T, E]` for tool/LLM outputs | Consistent failure surfaces; stack traces preserved |
| **Centralize constants** | Move `DEFAULT_WORKSPACE`, `TIMEOUT`, `SIMILARITY_THRESHOLD`, `MAX_INDEX_SIZE` to config/dataclass | No magic numbers; runtime overrides via `.env` or CLI |

### 🔹 Phase 4: Testing, CI/CD & Validation (Days 13–15)
| Task | Description | Deliverable |
|------|-------------|-------------|
| **Setup test framework** | Configure `pytest`, `pytest-asyncio`, `respx`/`aioresponses` for mocking async HTTP | Automated test runner with async support |
| **Write unit tests** | Cover path validation, config loading, exception routing, tool scoring logic | ≥80% coverage on `agent_core/` |
| **Add integration tests** | Mock LLM responses + workspace I/O to validate full pipeline execution | End-to-end workflow verification |
| **Configure CI/CD & pre-commit** | Add `ruff`, `mypy`, `black`, `pytest` via GitHub Actions / GitLab CI | Automated quality gates on PRs |

---

## 🛠️ Recommended Tech Stack & Dependencies
| Category | Library | Purpose |
|----------|---------|---------|
| Async HTTP | `httpx>=0.27`, `tenacity` | Native async, retries, timeouts |
| Config | `pydantic-settings>=2.3` | `.env` parsing, validation, defaults |
| CLI/REPL | `prompt_toolkit` or `cmd2` | History, completion, safe flag parsing |
| Async I/O | `aiofiles`, `asyncio.to_thread()` | Non-blocking disk/process ops |
| Error Pattern | `returns` (optional) or custom `Result[T, E]` | Explicit success/failure typing |
| Testing | `pytest`, `pytest-asyncio`, `respx` | Async mocking & test automation |
| Linting/Typing | `ruff`, `mypy`, `black` | Fast formatting, strict type checking |

---

## ⚠️ Risk Mitigation & Rollback Strategy
| Risk | Mitigation | Rollback Plan |
|------|------------|---------------|
| Breaking existing workflows during refactor | Keep legacy `agent.py` as `agent_legacy.py` with deprecation warnings; run parallel for 1 sprint | Revert to `agent_legacy.py`; restore archived duplicates from `legacy/` |
| Async migration introduces subtle race conditions | Use `asyncio.run()` in isolated test harnesses; add structured logging with correlation IDs | Fall back to `run_in_executor()` wrapper until stabilized |
| CLI parsing changes break user muscle memory | Preserve shorthand aliases (`imp`, `wrk`) via `prompt_toolkit` completer | Keep old `input().split()` parser behind `--legacy-cli` flag temporarily |

---

## ✅ Acceptance Criteria & Success Metrics
- [ ] Event loop never blocks >10ms during LLM or file operations
- [ ] Zero duplicate class/function definitions across codebase
- [ ] All paths resolve safely with traversal guards (`..`, absolute escapes blocked)
- [ ] Typed exceptions replace 100% of string-based error returns
- [ ] `pytest` suite passes locally & in CI; core module coverage ≥80%
- [ ] Configuration fully externalized; no hardcoded magic values remain
- [ ] Documentation updated with new architecture diagram & usage examples

---

## 🚀 Immediate Next Steps (Days 1–3)
1. **Branch**: Create `refactor/core-modernization` from `main`
2. **Archive duplicates**: Move top-level parallel files to `legacy/`
3. **Scaffold config**: Implement `agent_core/config.py` with `pydantic-settings`
4. **Add annotations**: Run sed/pre-commit to inject `from __future__ import annotations`
5. **Verify baseline**: Ensure existing tests/scripts still run against legacy entry point before touching async code

Would you like this exported as a GitHub Project board template, Jira epic breakdown, or accompanied by starter boilerplate for any specific phase?