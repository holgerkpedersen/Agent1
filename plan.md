# Coding Plan

# 📐 Architectural Coding Plan for `agent.py`

This plan transforms your analysis into an executable, phased roadmap. It prioritizes **blocking stability & security** first, followed by **core logic correctness**, then **architectural hygiene**, and finally **production polish**. Each phase includes dependencies, concrete implementation steps, and acceptance criteria.

---

## 🎯 Prioritization Strategy
| Priority | Focus Area | Rationale |
|----------|------------|-----------|
| 🔴 P0 (Critical) | Async I/O swap, Path traversal fix, Exception-based error handling | Prevents event loop starvation, sandbox breaches, and fragile string-parsing failures. Blocks all downstream work. |
| 🟠 P1 (High) | Timeout correction, Chunked search fix, Index cleanup logic, Path normalization consolidation | Fixes data integrity issues and unpredictable behavior under load. |
| 🟡 P2 (Medium) | Config extraction, Logging, Type hints, NL parsing → Tool routing | Improves maintainability, observability, and LLM interaction reliability. |
| 🟢 P3 (Low) | `findstr` quoting, Async input, State tracking consistency, CI/CD setup | Production hardening & developer experience improvements. |

---

## 📅 Phase 1: Critical Stability & Security (Foundation)
**Goal:** Eliminate async blocking, enforce workspace sandboxing, and replace string-based error handling with structured exceptions.  
**Dependencies:** None  
**Estimated Effort:** 2-3 Days

| Step | Action | Technical Approach | Acceptance Criteria |
|------|--------|-------------------|---------------------|
| 1.1 | Replace `urllib` with async HTTP client | Use `httpx.AsyncClient` with `timeout=httpx.Timeout(connect=5.0, read=30.0)`. Wrap in context manager or inject via DI. | All LLM calls use `await`, timeouts raise `httpx.TimeoutException`, event loop remains responsive under concurrent requests. |
| 1.2 | Implement strict workspace sandboxing | Create `_validate_path(path: str) -> Path` using `Path.resolve()` and prefix check against resolved workspace root. Raise `ValueError` on escape attempts. | Any path containing `..` or symlinks outside workspace fails fast with explicit error. Fuzz-tested against 50+ traversal payloads. |
| 1.3 | Convert string errors to exceptions | Define custom exception hierarchy: `AgentError`, `FileOperationError`, `ToolExecutionError`. Replace all `"File not found..."` returns with `raise FileOperationError(...)`. | Callers use `try/except AgentError`. No error strings are returned from tool methods. LLM receives structured failure messages. |
| 1.4 | Unblock async file I/O | Wrap `open()` calls in `aiofiles.open()` or `await asyncio.to_thread(os.path.getsize, path)`. Apply to all read/write/append operations. | File operations >5MB no longer freeze concurrent tool execution. Verified via `pytest-asyncio` concurrency test. |

---

## 📅 Phase 2: Core Logic & Data Integrity
**Goal:** Fix search/index bugs, consolidate path logic, and resolve platform inconsistencies.  
**Dependencies:** Phase 1 (requires stable async I/O & sandboxing)  
**Estimated Effort:** 3-4 Days

| Step | Action | Technical Approach | Acceptance Criteria |
|------|--------|-------------------|---------------------|
| 2.1 | Fix chunked search boundary flaw | Replace fixed 8KB chunks with `mmap.mmap()` or sliding window (`chunk = f.read(CHUNK_SIZE + OVERLAP)`). Strip overlap on next iteration. | Search terms spanning chunk boundaries return correct matches. False-negative rate drops to ~0% in regression suite. |
| 2.2 | Rewrite semantic index cleanup | Cap at `MAX_INDEX_SIZE`. Evict least-used entries using sorted frequency/last-access: `to_remove = sorted(index.items(), key=lambda x: len(x[1]))[:int(cap*0.1)]`. | Index never exceeds cap. High-frequency terms survive eviction. Memory usage stabilizes under continuous indexing. |
| 2.3 | Consolidate path normalization | Merge `_normalize_path_strict` & `_normalize_path` into `_resolve_workspace_path(raw: str) -> Path`. Use `pathlib.PurePosixPath`/`PureWindowsPath` for cross-OS slash handling. | Single source of truth for paths. No duplicate normalization logic. Cross-platform path tests pass on Linux/macOS/Windows. |
| 2.4 | Standardize platform commands | Detect OS via `platform.system()`. Map search to `grep -r` (Unix) / `findstr /s /c:` (Win). Use `shlex.quote()` for all shell arguments. | Search works identically across platforms. Special characters in queries don't break subprocess calls. |

---

## 📅 Phase 3: Architecture & Maintainability
**Goal:** Introduce configuration management, structured logging, type safety, and robust tool routing.  
**Dependencies:** Phase 1 & 2 (requires stable base)  
**Estimated Effort:** 3-4 Days

| Step | Action | Technical Approach | Acceptance Criteria |
|------|--------|-------------------|---------------------|
| 3.1 | Extract configuration | Create `@dataclass` or `pydantic-settings.BaseSettings` for model, workspace, timeouts, index caps. Load from `.env`. | No magic strings/hardcoded paths. Config validated on startup. Easy to swap environments (dev/staging/prod). |
| 3.2 | Implement structured logging | Replace all `print()` with `logging.getLogger(__name__)`. Configure JSON formatter for production, human-readable for dev. Add correlation IDs per tool call. | Logs contain timestamps, levels, module names. No stdout pollution. Errors traceable to specific agent iterations. |
| 3.3 | Add comprehensive type hints | Annotate all methods: `async def read_file(self, path: str) -> Result[str, FileOperationError]:`. Use `typing.Result` or custom wrapper. | `mypy --strict` passes with zero errors. IDE autocomplete works flawlessly. Return contracts are explicit. |
| 3.4 | Replace NL parsing with tool routing | Define OpenAI-style function schema for tools (`read_file`, `search`, `write`). Use LLM function calling or regex router: `\b(read|open)\s+(\S+)`. Fallback to explicit command prefix `/tool:action args`. | Agent correctly routes 95%+ of varied phrasings. No substring mangling (`research` → `earch`). Tool arguments validated via schema before execution. |

---

## 📅 Phase 4: Polish & Production Readiness
**Goal:** Harden edge cases, fix async input, align state tracking, and establish CI/CD.  
**Dependencies:** Phases 1-3  
**Estimated Effort:** 2 Days

| Step | Action | Technical Approach | Acceptance Criteria |
|------|--------|-------------------|---------------------|
| 4.1 | Fix `_files_read` state tracking | Normalize paths *before* checking existence in set: `normalized = self._resolve_workspace_path(path); if normalized not in self._files_read:` | Duplicate reads correctly tracked across OS path variations (`./file.txt` vs `file.txt`). |
| 4.2 | Shell-quote `findstr` arguments | Use `shlex.quote(query)` for all subprocess shell=True calls. Prefer `shell=False` with argument lists where possible. | Queries with spaces, quotes, or `$` pass safely to OS without injection or syntax errors. |
| 4.3 | Unblock async console input | Replace `input()` with `await asyncio.to_thread(input, prompt)` or integrate `prompt_toolkit`. | CLI remains responsive during agent execution. No event loop starvation on interactive mode. |
| 4.4 | Establish CI/CD & Async Testing | Add `pytest-asyncio`, `ruff`, `mypy`, `coverage >= 80%`. Include concurrency stress tests & sandbox escape attempts in pipeline. | PRs fail if async blocking detected, paths unsanitized, or type hints missing. Green builds required for merge. |

---

## 🧪 Testing & Validation Strategy
| Test Type | Scope | Tools/Approach |
|-----------|-------|----------------|
| **Unit** | Path validation, index eviction, config parsing, timeout handling | `pytest`, `hypothesis` (fuzz paths), mock `httpx` & `aiofiles` |
| **Async Concurrency** | Simultaneous tool calls, event loop responsiveness | `pytest-asyncio`, `asyncio.gather()`, custom loop stall detector |
| **Integration** | Full agent loop with mocked LLM responses | Record/replay HTTP fixtures (`vcrpy`), sandboxed temp workspace |
| **Security** | Path traversal, shell injection, prompt leakage | OWASP path fuzzing suite, `shellcheck`, static analysis (`bandit`) |

---

## ⚠️ Risk Mitigation & Rollback Plan
| Risk | Mitigation | Rollback Trigger |
|------|------------|------------------|
| Breaking existing tool signatures | Use adapter layer initially: keep old methods as deprecated wrappers until callers migrated | >5% regression in agent success rate |
| `httpx`/`aiofiles` dependency bloat | Vendor only required modules; provide sync fallback feature flag if constrained environment | Memory/CPU overhead exceeds baseline by >15% |
| LLM tool routing misalignment | Start with explicit `/tool:` prefix + regex fallback. Gradually enable native function calling after schema validation passes | Tool misrouting rate >10% over 100 iterations |

---

## ✅ Implementation Checklist (Dev Handoff)
- [ ] Phase 1: `httpx` swap, `_validate_path`, exception hierarchy, `aiofiles` integration
- [ ] Phase 2: Sliding-window search, LRU index cleanup, unified path resolver, cross-platform commands
- [ ] Phase 3: `AgentConfig` + `.env`, `logging` setup, strict type hints, tool routing schema
- [ ] Phase 4: State normalization fix, `shlex.quote()`, async input, CI/CD pipeline
- [ ] All phases covered by tests → `pytest --asyncio-mode=auto -v` passes green

---

**Next Steps:**  
1. Create feature branch `refactor/agent-stability-v2`  
2. Implement Phase 1 steps in order (strict dependency chain)  
3. Run sandbox & concurrency tests after each step before proceeding  
4. Open PR with incremental commits per table row for review  

Would you like me to generate the exact code templates for any specific step (e.g., `_validate_path`, `httpx` client wrapper, or the tool routing schema)?