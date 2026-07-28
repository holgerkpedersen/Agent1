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
read <path>                  Read a file
write <path> <content>       Write content to file
search <query>               Search files for string
analyze <file> [output]      AI analysis via LLM
plan <analysis> <plan>       Generate coding plan
entities <analysis> <plan>   Generate shared entities
taskplan <analysis> <plan>   Generate implementation tasks
implement <taskplan> [opts]  Implement files from task plan
fix <traceback>              Auto-fix from traceback
fix <file> --desc "text"     Describe issue, LLM fixes it
cleanup                      Show unreferenced files
workflow <target> [opts]     Full pipeline (analyze→plan→entities→tasks→implement)
model [list|reload|name]     Manage LLM models
clear [stats|--force]        Show/clear agent memory
```

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
