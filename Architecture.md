# Architecture of Agent1

## Overview

Agent1 is a Python AI agent framework with LLM integration via LM Studio (local), workspace management, tool execution, and a multi-agent orchestration layer. The system follows **SOLID** principles with clear separation of concerns.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent1 System                              │
│                                                                   │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │   LLM Layer  │   │  Agent Core  │   │   Command Layer      │  │
│  │  (lmstudio)  │──>│  (agent.py)  │<──│  (commands/)         │  │
│  └─────────────┘   └──────┬───────┘   └──────────────────────┘  │
│                            │                                       │
│          ┌─────────────────┼─────────────────┐                    │
│          │                 │                 │                    │
│   ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────────┐        │
│   │ FileSystem  │  │FileSearcher │  │ ToolDispatcher  │        │
│   │ read/write/ │  │ findstr/    │  │ registry-based  │        │
│   │ patch/edit  │  │ grep/fallback│ │ tool dispatch   │        │
│   └─────────────┘  └─────────────┘  └─────────────────┘        │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              src/agent1  (Multi-Agent Framework)           │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │   Core    │  │  Memory   │  │Orchestrat │  │Monitor   │  │   │
│  │  │ Agent     │  │ Store     │  │Scheduler  │  │Metrics   │  │   │
│  │  │ Messages  │  │ Vector DB │  │Workflow   │  │Dashboard │  │   │
│  │  │ Context   │  │ Semantic  │  │DepGraph   │  │Alerts    │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## 1. LLM Layer (`agent_core/llm/`)

Communicates with LM Studio's OpenAI-compatible API endpoint.

### Components

| Component | File | Role |
|---|---|---|
| `LLMProvider` | `provider.py` | Protocol defining the LLM interface |
| `LMStudioProvider` | `lmstudio.py` | Concrete implementation with chat, stream, analyze |
| `RetryPolicy` | `retry.py` | Exponential backoff retry on transient errors |
| `ToolLoopRunner` | `tool_loop.py` | Multi-step LLM interaction with tool result injection |

### Data Flow

```
LMStudioProvider.chat(messages, tools?) ──> _build_payload() ──> POST /chat/completions
                                          │
                                          ├── payload["tool_choice"] = "none" (if model disables tools)
                                          ├── payload["thinking"] = {"type": "disabled"} (if model flag set)
                                          └── RetryPolicy.execute_with_retry() on transient errors
```

### Model Configuration (`agent_core/constants.py`)

```python
KNOWN_MODELS = {
    "laguna-s-2.1": {"desc": "Laguna S 2.1 MoE A8B - fast agentic coding", "max_tokens": 100000, "tool_calling": False},
    "qwen3.6-27b-mtp": {"desc": "Qwen 3.6 27B - chat, codegen, large context", "max_tokens": 100000},
    "google/gemma-4-31b": {"desc": "Gemma 4 31B - chat, reasoning, fast generation", "max_tokens": 100000},
}
```

Models carry flags (`thinking`, `tool_calling`) that influence payload construction. `tool_calling=False` suppresses Laguna's native tool call output via `tool_choice: "none"`.

## 2. Agent Core (`agent.py`)

The `Agent` class is the central orchestrator. It composes extracted components and delegates operations.

### Constructor

```python
class Agent:
    def __init__(self, workspace, model_name):
        self.fs = FileSystem(workspace)       # file I/O
        self.searcher = FileSearcher()        # file search
        self.dispatcher = ToolDispatcher()    # tool dispatch
        self.llm = LLMClient(model_name)      # LLM wrapper

        # Memory
        self._files_read: set[str]            # tracked file paths
        self._file_mtimes: dict[str, float]   # mtime per tracked file
        self._semantic_index: dict[str, set]  # word → positions
        self._knowledge_graph: dict           # entity relationships
        self._working_memory: list            # task items
        self._history: list                   # interaction log
        self.chat_history: list               # LLM conversation
```

### Tool Dispatch

Tools are registered via `ToolDispatcher` (registry pattern, OCP-compliant):

```python
dispatcher.register("read_file", lambda args: self._tool_read_file(**args))
dispatcher.register("write_file", lambda args: self._tool_write_file(**args))
dispatcher.register("search", lambda args: self._tool_search(**args))
# All 8 tools registered: read_file, write_file, apply_patch, edit_file,
# search, list_files, delete_file, analyze_file
```

Execution is a one-liner:
```python
async def execute_tool(self, tool_name, arguments):
    return await self.dispatcher.execute(tool_name, arguments)
```

### Memory System

| Store | Type | Purpose | Detection |
|---|---|---|---|
| `_files_read` | `set[str]` | Tracks which files were read | — |
| `_file_mtimes` | `dict[str, float]` | mtime per tracked file | `check_stale_files()` |
| `_semantic_index` | `dict[str, set[int]]` | Word → position index | Invalidated on stale |
| `_knowledge_graph` | `dict` | Entity relationship graph | — |
| `_working_memory` | `list` | Active task items | — |
| `_history` | `list` | Interaction log | — |
| `chat_history` | `list` | LLM conversation context | — |

**Stale file detection**: Every file read stores its `os.path.getmtime()`. `check_stale_files()` compares current mtime vs stored — returns paths changed externally. `invalidate_stale()` purges stale entries and clears the semantic index.

**Memory stats** via `memory_stats()` returns a dict with all sizes + stale count for the `clear` command.

### Path Handling

```
to_windows_path() ──> _normalize_path() ──> _safe_path() ──> FileSystem.read()/write()
```

- `to_windows_path()`: Converts POSIX-style paths (`/c/Dev/Agent1`) to Windows paths (`C:\Dev\Agent1`)
- `_normalize_path()`: Resolves to absolute, validates
- `_safe_path()`: Strips `./` prefix, delegates to normalize

## 3. Command Layer (`agent_core/commands/`)

Commands follow the **Command pattern** with a registry for OCP-compliant dispatch.

### Architecture

```
                 ┌──────────────────┐
User input ──>   │ CommandRegistry  │
                 │  _commands: dict │
                 │  register()      │
                 │  execute()       │
                 └───────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐    ┌─────┴──────┐   ┌───┴──────────┐
    │ReadCmd  │    │ImplementCmd│   │WorkflowCmd   │
    │execute()│    │execute()   │   │execute()     │
    └─────────┘    └────────────┘   └──────────────┘
```

### Command ABC

```python
class Command(ABC):
    @property
    def name(self) -> str: ...
    @property
    def help_text(self) -> str: ...
    async def execute(self, args: list[str], agent: Agent) -> bool: ...
```

### Commands (14 total)

| Command | Class | Purpose |
|---|---|---|
| `read` | `ReadCommand` | Read file contents |
| `write` | `WriteCommand` | Write content to file |
| `search` | `SearchCommand` | Search files for text |
| `clear` | `ClearCommand` | Show memory stats + clear |
| `model` | `ModelCommand` | List/reload/switch LLM models |
| `analyze` | `AnalyzeCommand` | AI code analysis |
| `plan` | `PlanCommand` | Generate coding plan |
| `entities` | `EntitiesCommand` | Extract shared entities |
| `taskplan` | `TaskplanCommand` | Generate task list |
| `implement` | `ImplementCommand` | Generate files from taskplan |
| `fix` | `FixCommand` | Fix from traceback or description |
| `cleanup` | `CleanupCommand` | Show unreferenced files |
| `workflow` | `WorkflowCommand` | Full pipeline: analyze→plan→entities→tasks→implement |
| `optimize` | `OptimizeCommand` | Find and apply performance/quality improvements |
| `perf` | `PerfCommand` | Command performance dashboard |
| `paste` | `PasteCommand` | Paste multi-line text for analysis |
| `decide` | `DecideCommand` | Record, search, and enforce design decisions

### REPL Loop (`run_interactive()`)

```
while True:
    input ──> shlex.split() ──> command
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              registry cmd    quit/exit    natural language
              (13 commands)               ──> agent.process_query()
                                                  │
                                          _parse_natural_language()
                                          execute_tool()
```

## 4. File Operations

### FileSystem (`agent_core/file_system.py`)

```python
class FileSystem:
    normalize_path(path)    # resolve + validate
    safe_path(path)         # strip ./ then normalize
    read(path)              # read file contents
    write(path, content)    # create/write (mkdirs parents)
    apply_patch(path, f, r) # find-and-replace (unique match enforced)
    edit(path, content)     # overwrite
```

### FileSearcher (`agent_core/file_searcher.py`)

Platform-aware search with fallback:

```
Windows: findstr /S /N /C:query path ──> Python os.walk fallback
Linux:   grep -rn query path
```

## 5. LLMClient Wrapper

Thin wrapper around `LMStudioProvider` for backward compatibility:

```python
class LLMClient:
    def __init__(self, model_name, api_key):
        self._provider = LMStudioProvider(model_name, api_key)

    async def chat(self, messages, tools=None):     # -> self._provider.chat()
    async def chat_stream(self, messages):           # -> self._provider.chat_stream()
    async def chat_with_continuation(self, msgs, n): # auto-resume truncated
    async def analyze_code(self, code):              # -> self._provider.analyze_code()
```

## 6. Multi-Agent Framework (`src/agent1/`)

Generated via the `workflow → implement` pipeline. A composable framework for building multi-agent systems.

```
src/agent1/
├── core/
│   ├── __init__.py           # Shared types: AgentMessage, TaskNode, StorageBackend,
│   │                         #   VectorDatabase, EmbeddingService, PluginInterface, MetricType
│   ├── agent.py              # Agent class with message bus, memory, plugins
│   ├── message_bus.py        # Publish/subscribe inter-agent messaging
│   └── context_manager.py    # Shared context with locking
├── memory/
│   ├── __init__.py           # Re-exports core + MemoryStore, SemanticSearchEngine
│   ├── memory_store.py       # Persistent storage with cache + optional semantic search
│   ├── vector_db.py          # VectorDatabase + EmbeddingService facade
│   └── semantic_search.py    # Combine embedding + vector DB for semantic queries
├── orchestration/
│   ├── __init__.py           # Re-exports TaskScheduler, WorkflowEngine, DependencyGraph
│   ├── task_scheduler.py     # Schedule tasks with executors, timing, handlers
│   ├── workflow_engine.py    # Execute concurrent ready tasks via executor map
│   └── dependency_graph.py   # Directed graph with topological sort, cycle detection
├── plugins/
│   ├── __init__.py           # BasePlugin, PluginRegistry, PluginManager
│   ├── base_plugin.py        # Lifecycle: initialize → execute → cleanup
│   ├── plugin_manager.py     # Load/unload/execute lifecycle management
│   └── registry.py           # Plugin class registration + metadata
└── monitoring/
    ├── __init__.py           # Re-exports MetricsCollector, DashboardAPIServer, AlertSystem
    ├── metrics_collector.py  # Counters, gauges, histograms, timers (thread-safe)
    ├── dashboard_api.py      # HTTP API on port 8080 for querying metrics
    └── alert_system.py       # Alert rules with threshold checking + cooldown
```

### Key Design Decisions

- **Centralized types**: `core/__init__.py` defines all shared types (dataclasses, protocols, enums) to avoid circular imports.
- **Protocol-based interfaces**: `StorageBackend`, `PluginInterface`, `VectorEmbeddingModel` are Protocols for mypy-strict compliance.
- **Composition over inheritance**: Components like `MemoryStore` compose `SQLiteStorage`, `EmbeddingService`, `VectorDatabase` rather than inheriting.
- **Lifecycle management**: Plugins follow `initialize → execute → cleanup`, managed by `PluginManager`.

### Safety Architecture

The framework has multiple layers of protection against common AI agent failure modes:

**Stdlib Shadowing Prevention (3 layers):**
1. Workflow prompts warn the LLM against directory names matching stdlib modules (logging/, json/, types/)
2. Analysis verifier flags shadowed paths as `[UNVERIFIED]` before implementation
3. Implement command auto-redirects shadowed paths (`logging/` → `logging_utils/`)

**LLM Reasoning Stripping** (`agent_core/commands/reasoning_strip.py`):
- Automatic removal of LLM chain-of-thought, thinking tags, self-correction markers, and output generation badges from all workflow-generated files
- Two modes: `"analysis"` (aggressive section extraction + reasoning removal) and `"light"` (regex-only tag stripping for plan/entities/taskplan)
- Prevents corrupted output when models leak reasoning into responses

**Decision Tracking** (`agent_core/decisions.py`, `decide` command):
- `.decisions.json` stores design decisions with context, rationale, affected files, and tags
- Auto-extracts decision candidates from `workflow`, `implement`, and `fix` runs
- Past decisions injected as hard constraints into LLM prompts — prevents accidental contradictions
- LLM-powered contradiction detection (`decide check`) and resolution (`decide resolve`)
- Full lifecycle: make decisions → record them → enforce on future runs → detect drift

## 7. Test Architecture

```
tests/
├── test_agent_paths.py        # Path normalization + security
├── test_llm_client.py         # LLM client response types
├── test_path_utils.py         # to_windows_path edge cases
├── test_subprocess_utils.py   # Subprocess execution + timeout
├── test_tool_router.py        # Tool parsing, validation, dispatch
├── test_handlers/             # Analyze handler tests
├── unit/
│   ├── test_memory.py         # MemoryStore save/load/forget
│   ├── test_orchestration.py  # DependencyGraph, TaskScheduler
│   └── test_plugins.py        # PluginRegistry, PluginManager
├── integration/
│   └── test_multi_agent.py    # AgentMessage routing, multi-store isolation
└── performance/
    └── test_scaling.py        # Metrics throughput, vector DB search latency
```

86 tests, all passing.

## Summary

| Layer | Responsibility | Pattern |
|---|---|---|
| **LLM** (`llm/`) | Model communication, retry, streaming | Provider protocol + DIP |
| **Agent Core** (`agent.py`) | State, tool dispatch, memory | Composition + SRP |
| **Commands** (`commands/`) | User-facing REPL commands | Command pattern + Registry |
| **File Ops** (`file_system.py`, `file_searcher.py`) | Read/write/search with path safety | Extracted utility classes |
| **Src Agent1** (`src/agent1/`) | Multi-agent framework | Centralized types, protocols, composition |
| **Tests** (`tests/`) | Unit, integration, performance | pytest with anyio |
