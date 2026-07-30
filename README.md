# Agent1

Python AI agent framework with LLM integration, tool execution, workspace management, and multi-agent orchestration.

## Features

### Core Agent (`agent.py`)
- Interactive REPL with LM Studio (local) LLM integration
- File operations: read, write, patch, edit with path normalization
- Semantic search across project files
- Memory management with stale file detection
- Workspace-aware path handling (Windows + POSIX)

### LLM Integration (`agent_core/llm/`)
- LM Studio provider with OpenAI-compatible API
- Configurable retry policy with exponential backoff
- Tool loop runner for multi-step LLM interactions
- Streaming chat with real-time token output
- Multi-model support: Laguna S 2.1, Qwen, Gemma, etc.

### Commands (`agent_core/commands/`)
```
read <path>                      Read a file
write <path> <content>           Write content to file
search <query>                   Search files for string
analyze <file> [--desc "q"]      AI analysis — follows imports, answers specific questions
         --stdin [--desc "q"]   Paste multi-line text for analysis (--- to finish)
         --deep                  Follow imports with deep analysis
plan <analysis> <plan>           Generate coding plan
entities <analysis> <plan>       Generate shared entities
taskplan <analysis> <plan>       Generate implementation tasks
implement <taskplan> [opts]      Implement files from task plan
                                 --force            Overwrite existing files
                                 --keep             Skip files that compile OK
                                 --fix              Retry compilation errors
                                 --retry            Re-generate only missing files
                                 --review           Review new files for bugs + DRY violations
                                 --workspace <path> Target workspace
fix <traceback>                  Paste traceback to auto-fix root cause
    <file> --desc "text"        On-demand analysis — only sends top-5 files by relevance
    <file> --desc "text" --full  Send entire project context (old behavior)
cleanup                          Show unreferenced files
workflow <target> [opts]         Full pipeline: analyze → plan → entities → tasks → implement
                                 --from spec.md    Greenfield from specification
                                 --stdin           Paste multi-line spec (--- to finish)
                                 --desc "text"     Greenfield from description
                                 --features spec   Brownfield extension
                                 --brainstorm      Add 6th dimension: creative features
                                 --force           Skip existing file checks
                                 --workspace <p>   Target workspace
model [list|load|unload|reload|name]  Manage models via LM Studio API
clear [stats|--force]            Show memory stats, confirm then clear
```

### Workflow Pipeline

The `workflow` command runs a full analysis-to-implementation pipeline. Brownfield mode (`workflow .`) now uses multi-dimensional analysis across 5 dimensions:

| Dimension | Scope |
|---|---|
| **CODE QUALITY** | Bugs, edge cases, type safety, error handling gaps |
| **COMPLETENESS** | Missing tests, docs, error handling, underdeveloped features |
| **ARCHITECTURE** | DRY violations, circular dependencies, coupling, SRP breaks |
| **INNOVATION** | New capabilities that would make the system more useful/powerful |
| **PRODUCTION** | Logging, monitoring, configuration, security, deployment readiness |

Add `--brainstorm` for a 6th dimension: bold, creative, unconventional features.

The `implement` command uses smart context truncation: per-file excerpts (±400 chars around filename mentions) instead of repeating the full analysis/plan for every batch. Reduces context from ~50K to ~4K tokens per batch.

### Safety Guards

The `fix` and `implement` commands include automatic protections against common failure modes:

**fix** — Traceback cascade analysis:
- Finds the **first user file** in the import chain (skips stdlib and `<frozen>` entries) rather than the last frame where the error manifested
- Detects **stdlib shadowing**: when a local file (e.g. `types.py`) shadows a stdlib module, suggests renaming instead of trying to modify Python's own files
- **Refuses writes** to files under the Python installation directory to prevent `PermissionError`
- **On-demand mode** (`fix <file> --desc`): scores candidate files by keyword relevance, sends only the top 5 files as full source. The LLM can request additional files with `[READ: path]` and iterates up to 3 rounds. Use `--full` for the legacy "send everything" behavior.

**implement** — File safety (3 layers):
- **Prevention**: LLM prompts instruct the LLM to use sub-package paths (`agent_core/thing.py`) and avoid bare root-level filenames
- **Auto-repair**: All bare root-level filenames are auto-prefixed with `agent1/` — `config.py` → `agent1/config.py`. No file is written at workspace root.
- **Target protection**: Auto-repair skips overwriting files that already exist (>100 bytes), preventing accidental overwrite of project packages.
- **Categorized outcomes**: Final report groups files by status: auto-repaired, rejected (with reasons), skipped (target exists), and truly missing.

### Memory Management

The `clear` command shows what's stored before clearing:
- **chat history**: LLM conversation context
- **files read**: Tracked with mtime for staleness detection
- **stale files**: Files changed externally since last read
- **working memory**: Active task items
- **semantic index**: Word → position index for fast search
- **knowledge graph**: Entity relationships

Use `clear stats` to view without clearing, `clear --force` to skip confirmation.

### Src Agent1 — Multi-Agent Framework (`src/agent1/`)
| Module | Purpose |
|---|---|
| `core` | AgentMessage, MessageBus, SharedContext, SQLiteStorage, VectorDatabase, PluginInterface |
| `memory` | MemoryStore with caching, SemanticSearchEngine, VectorDatabase integration |
| `orchestration` | DependencyGraph, TaskScheduler, WorkflowEngine |
| `plugins` | BasePlugin, PluginRegistry, PluginManager with lifecycle management |
| `monitoring` | MetricsCollector, DashboardAPIServer, AlertSystem |

## Quick Start

### Prerequisites
- Python 3.12+
- LM Studio running on `http://localhost:1234`
- Model loaded in LM Studio (default: `laguna-s-2.1`)

### Run
```bash
python agent.py
```

### Switch model
```
> model qwen3.6-27b-mtp
```

## Project Structure
```
Agent1/
├── agent.py                      # Main agent + REPL
├── agent_core/
│   ├── constants.py              # Model definitions, defaults
│   ├── path_utils.py             # Path normalization utilities
│   ├── file_system.py            # File I/O operations
│   ├── file_searcher.py          # Platform-aware file search
│   ├── tool_dispatcher.py        # Registry-based tool dispatch
│   ├── llm/
│   │   ├── lmstudio.py           # LM Studio provider
│   │   ├── provider.py           # LLMProvider protocol
│   │   ├── retry.py              # RetryPolicy
│   │   └── tool_loop.py          # ToolLoopRunner
│   └── commands/
│       ├── base.py               # Command ABC
│       ├── registry.py           # CommandRegistry
│       ├── read_cmd.py           # Read file
│       ├── write_cmd.py          # Write file
│       ├── search_cmd.py         # Search files
│       ├── clear_cmd.py          # Memory management
│       ├── model_cmd.py          # Model switching
│       ├── analyze_cmd.py        # Code analysis
│       ├── plan_cmd.py           # Plan generation
│       ├── entities_cmd.py       # Entity extraction
│       ├── taskplan_cmd.py       # Task planning
│       ├── cleanup_cmd.py        # File cleanup
│       ├── implement_cmd.py      # File implementation
│       ├── fix_cmd.py            # Auto-fix errors
│       └── workflow_cmd.py       # Full pipeline
├── src/agent1/                   # Generated multi-agent framework
│   ├── core/                     # Agent, message bus, context manager
│   ├── memory/                   # Storage, vector DB, semantic search
│   ├── orchestration/            # Task scheduler, workflow engine, dep graph
│   ├── plugins/                  # Plugin system with lifecycle
│   └── monitoring/               # Metrics, dashboard, alerts
├── tests/
│   ├── test_agent_paths.py
│   ├── test_llm_client.py
│   ├── test_path_utils.py
│   ├── test_subprocess_utils.py
│   ├── test_tool_router.py
│   ├── unit/                     # Generated module unit tests
│   ├── integration/              # Multi-agent integration tests
│   └── performance/              # Scaling tests
└── README.md
```

## Configuration

Set environment variables or use `.env`:
```env
AGENT_MODEL=laguna-s-2.1
LMSTUDIO_URL=http://localhost:1234/v1
OPENAI_API_KEY=your-key
```

## Models

| Model | Description |
|---|---|
| `laguna-s-2.1` | Laguna S 2.1 MoE A8B — fast agentic coding |
| `qwen3.6-27b-mtp` | Qwen 3.6 27B — chat, codegen, large context |
| `google/gemma-4-31b` | Gemma 4 31B — chat, reasoning, fast generation |

## Testing

```bash
pytest tests/ -v
```

86 tests, all passing.
