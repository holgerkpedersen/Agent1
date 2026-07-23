Based on your architecture review and refactoring plan, here is a clean extraction of **shared entities** structured to guarantee zero circular imports, followed by concrete patterns to enforce unidirectional dependencies.

---
### 📦 1. Extracted Shared Entities (Leaf Modules)
These modules must remain **pure**, contain no business logic, and import only from the standard library or third-party packages. They form the immutable foundation of `agent_core/`.

| Module | Shared Types / Classes | Purpose | Import Safety |
|--------|----------------------|---------|---------------|
| `entities.py` | `Message`, `ChatCompletion`, `ToolDefinition`, `ToolResult`, `AgentContext`, `IndexEntry`, `ScoredCandidate`, `Result[T, E]` | Canonical data contracts for LLM I/O, tool routing, and evaluation pipelines. | ✅ Zero imports. Pure dataclasses/TypedDicts. |
| `exceptions.py` | `AgentError(Exception)`, `FileOperationError`, `ToolExecutionError`, `LLMClientError`, `RoutingError`, `PathTraversalError`, `ConfigValidationError` | Typed error hierarchy replacing string returns and broad catches. | ✅ Zero imports. Inherit from builtins only. |
| `config.py` | `AppSettings(BaseSettings)`, `DEFAULTS(frozen dataclass)` | Centralized, validated configuration via `pydantic-settings`. Loads `.env`, CLI overrides, and timeouts/thresholds. | ⚠️ Imports only `pydantic_settings`, `pathlib`, `typing`. Never imports core logic. |
| `path_utils.py` | `resolve_safe_path(user_input: str) -> Path`, `is_within_workspace(path: Path, root: Path) -> bool`, `normalize_cross_os(path: str) -> str` | OS-agnostic path resolution with traversal guards (`..`/absolute escape blocking). | ✅ Imports only `pathlib`, `os`. Pure functions. |
| `context_management.py` | `CorrelationContext(ContextVar)`, `get_trace_id() -> str`, `set_span_context()` | Async-safe request/tracing context propagation across event loop boundaries. | ✅ Imports only `contextvars`, `uuid`, `typing`. |
| `logging_config.py` | `setup_logging(level: str, trace_ctx: bool) -> Logger`, `StructuredFormatter` | Centralized async-compatible logging with correlation ID injection. | ⚠️ Imports only `logging`, `json`, `sys`. Deferred initialization pattern. |

---
### 🌲 2. Strict Dependency Hierarchy (Anti-Circular Flow)
Enforce a **layered architecture** where imports only flow upward toward leaf modules:

```
[APP / ORCHESTRATOR]          ← imports from ↓
orchestrator.py, cli.py, agent_main.py
       ↑
[CORE SERVICES]               ← imports from ↓
llm_client.py, workspace_manager.py, tool_router.py, cache_manager.py
       ↑
[SHARED ENTITIES (LEAVES)]    ← NO OUTGOING IMPORTS TO APP/SERVICES
entities.py, exceptions.py, config.py, path_utils.py, context_management.py, logging_config.py
```

**Rules to Enforce:**
1. `agent_core/__init__.py` only re-exports leaves for convenience. Never contains logic.
2. Services (`llm_client`, etc.) may import **multiple** leaf modules but never each other directly (use orchestrator for coordination).
3. App/CLI layer imports services + leaves. Never the reverse.

---
### 🛡️ 3. Circular Import Prevention Tactics

| Risk Scenario | Mitigation Pattern | Example |
|---------------|-------------------|---------|
| Type hints referencing a class from another module at import time | Use `TYPE_CHECKING` guard | ```python\nfrom __future__ import annotations\nimport sys\nif sys.version_info < (3, 10): ... \n# Better:\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from agent_core.entities import ToolResult\n``` |
| Config loading triggers service initialization at module load time | Defer instantiation to runtime / use factory functions | ```python\ndef get_settings() -> AppSettings:  # Called in main(), not imported globally\n    return AppSettings()\n``` |
| Logging setup requires context management, which imports logging | Lazy import inside function scope or use `importlib` | Keep `logging_config.py` and `context_management.py` independent; inject formatter via dependency injection at startup. |
| Tool router needs entity types, entities need tool registry reference | Break symmetry: Entities define interfaces (`Protocol`), router implements them | ```python\n# entities.py\nclass ToolExecutor(Protocol):\n    async def run(self, ctx: AgentContext) -> Result[ToolResult, ToolExecutionError] ...\n``` |

---
### 📁 4. Ready-to-Use Skeleton & Import Patterns

**Directory Structure:**
```
agent_core/
├── __init__.py          # Re-exports only
├── entities.py          # Pure data contracts
├── exceptions.py        # Typed error hierarchy
├── config.py            # Pydantic settings
├── path_utils.py        # Safe path resolution
├── context_management.py# Async context vars
├── logging_config.py    # Structured logger setup
└── services/            # (Optional: keep flat for now)
    ├── llm_client.py
    ├── workspace_manager.py
    └── tool_router.py
```

**Safe Import Example (`services/tool_router.py`):**
```python
from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Sequence

# ✅ Safe: Leaf imports only
from agent_core.entities import ToolDefinition, AgentContext, Result, ToolResult
from agent_core.exceptions import RoutingError, ToolExecutionError
from agent_core.context_management import get_trace_id

if TYPE_CHECKING:  # ⚠️ Breaks potential cycles if entities ever reference router
    from agent_core.services.workspace_manager import WorkspaceManager

class ToolRouter:
    def __init__(self, tools: Sequence[ToolDefinition], workspace: WorkspaceManager) -> None:
        self.tools = {t.name: t for t in tools}
        self.workspace = workspace

    async def dispatch(self, ctx: AgentContext, tool_name: str, payload: dict) -> Result[ToolResult, ToolExecutionError]:
        trace_id = get_trace_id()
        # ... routing logic using only leaf types & injected services
```

**Startup Wiring (`agent_main.py`):**
```python
from agent_core.config import AppSettings
from agent_core.logging_config import setup_logging
from agent_core.context_management import set_span_context
# Services imported after leaves are fully loaded
from agent_core.services.llm_client import LLMClient
from agent_core.services.workspace_manager import WorkspaceManager

def bootstrap() -> None:
    cfg = AppSettings()  # Lazy instantiation
    logger = setup_logging(cfg.log_level, trace_ctx=True)
    
    llm = LLMClient(base_url=cfg.api_base_url, timeout=cfg.timeout_seconds)
    ws = WorkspaceManager(root=cfg.workspace_path)
    
    set_span_context(trace_id="init-seed")
    # ... wire orchestrator
```

---
### ✅ Verification Checklist Before Merging
- [ ] `grep -r "from.*services\|import.*orchestrator" agent_core/{entities,exceptions,path_utils,context_management}.py` → returns nothing.
- [ ] All type hints using cross-module references are wrapped in `TYPE_CHECKING`.
- [ ] No module-level function calls that instantiate services or load configs at import time.
- [ ] `python -c "import agent_core"` runs without side effects or network/file I/O.

This extraction guarantees a **strict DAG (Directed Acyclic Graph)** for imports, eliminates the fragmentation noted in your review, and aligns directly with Phase 1–2 of your roadmap. Let me know if you want automated linting rules (`ruff`/`mypy` configs) to enforce this hierarchy at CI time.