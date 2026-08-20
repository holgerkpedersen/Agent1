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
- **Multi-provider**: local LM Studio (OpenAI-compatible API) and the hosted opencode-go API (native tool calling, key from `OPENCODE_API_KEY` or opencode's auth.json); the persisted provider in `model.json` wins, model switches rebuild the provider
- **Enhanced executor** with parallel tool execution and error handling (fixcommand/core/executor/llm_executor.py)
- **Safe thinking-disable**: sends `reasoning: "off"` plus only the per-model `chat_template_kwargs` declared in `KNOWN_MODELS` (e.g. `enable_thinking: false` for Qwen/Laguna jinja templates). The aggressive knob set (`thinking.disabled` / `enableThinking` / `preserve_thinking`) was removed after probes on qwen/qwen3.8-27b showed it makes the model burn its whole output budget on `reasoning_content` and emit zero content

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
                                  --force            Overwrite existing files (wholesale)
                                  --keep             Skip files that compile OK
                                  --modify           Merge generated content into existing compile-OK modules as a reviewed diff
                                  --fix              Retry compilation errors
                                   --retry            Re-generate only missing files (cache auto-invalidates on taskplan change)
                                   --review           Review new files + offer to delete dangerous/unwired files
                                  --allow-rewrite    Opt in to wholesale rewrite of existing file under --modify
                                  --refresh          With --keep: rebuild the cached file list from the taskplan
                                  --no-history       Skip injecting PAST EXECUTION NOTES from past runs
                                  --workspace <path> Target workspace
fix <traceback>                  Paste traceback to auto-fix root cause
    <file> --desc "text"        On-demand — top-5 files by keyword match, LLM requests more with [READ:]
    <file> --desc "text" --full  Send entire project context (old behavior)
    [PATCH:] output              Prefer minimal diffs — shows changed lines, asks y/N before applying
cleanup                          Show unreferenced files
workflow <target> [opts]         Full pipeline: analyze → plan → entities → tasks → implement
                                 --from spec.md    Greenfield from specification
                                 --stdin           Paste multi-line spec (--- to finish)
                                 --desc "text"     Greenfield from description
                                 --features spec   Brownfield extension
                                 --brainstorm      Add 6th dimension: creative features
                                 --auto            Autonomous: skip confirmations, run implement inline
                                 --continue        Resume the newest .docs/<ts>/ run
                                 --force           Skip existing file checks
                                 --workspace <p>   Target workspace
model [list|load|unload|reload|profile|name]  Manage models and providers (LM Studio + opencode-go)
optimize <file|dir> [--apply] [--stdin] [--yes] Find and apply performance/memory optimizations
                                   Shows side-by-side diff with line numbers before applying
                                   --yes: skip the y/N prompt and apply automatically
                                   Mechanical regex_in_loop fixes hoist re.compile() into
                                   descriptive constants (e.g. _FOR_WHILE_RE) and never
                                   touch docstring/comment lines
perf [--detail|--reset|--html]       Command performance dashboard (timing per command)
clear [stats|--force]            Show memory stats, confirm then clear
decide "title" --why "..."       Record a design decision with rationale
     --what "..."                 (tags, files, context tracked)
     --tags t1,t2 --files f1.py
decide list [--tag t] [--search q] Search past decisions
decide show <id>                 Show full decision record
decide check --text "idea"       LLM checks if idea contradicts past decisions
decide resolve <id1> <id2>       LLM resolves contradiction between two decisions
decide extract [--from a.md]     Auto-extract decisions from project analysis
review <refresh|list|show|label|auto|export>  Human gate over failed task traces
     label <task> <bug|regression|noise|ok>   Label a trace disposition
     label <task> auto | auto [<task>]        Agent reviews it (evidence rules)
     export <task>                            Write a diagnosis-pinning pytest file
reconstruct [--start <file>] [--end <file>]   Rebuild files from JSONL trace logs
    [--workspace <path>] [--search <query>]   (replays write/edit ops in timestamp order)
    [--dry-run] [--force]
paste_image [path] [--desc "text" | --prompt "text"]  Paste an image (clipboard or file)
                                                     for vision-capable LLMs
```

Any text not matching a command is sent to the LLM as **natural language**. The agent uses **native OpenAI-format tool calling** — the model is given schemas for `search`, `read`, `list_files`, `write`, `edit`, `run`, `git`, `diff`, `tests`, `fix`, and `analyze`, and must either emit a structured tool call or answer in text:

```
> What safety guards does the implement command use?
  [tool] read(path=agent_core/commands/implement_cmd.py)
  [result] ... (file contents fed back to the model)
The implement command guards against stdlib shadowing, class-name
collisions, and unwired modules (see agent_core/commands/implement_cmd.py).
```

- **Actions, not descriptions**: the API forces the model to actually call a tool — it cannot just say "I will read the file."
- **Visible execution**: every tool call is printed (`[tool] name(args)`) with its result (`[result] ...`).
- **Verified writes**: after `write`/`edit`, a `py_compile` summary is appended so the model reports verified changes.
- **Tool errors feed back**: a failed tool returns its error to the model, which can retry with a different approach.
- **Tool schemas are declared once** in `agent_core/tool_schemas.py` — the schema set sent to the LLM is exactly the set the dispatcher can execute (`NLP_TOOL_NAMES`).
- Conversation history is maintained so follow-up questions build on previous answers.

### Workflow Pipeline

The `workflow` command runs a full analysis-to-implementation pipeline. **Greenfield mode** (`--desc`/`--stdin`/`--from`) now includes a spec analysis step (scope, assumptions, risks, dependencies) before plan generation. All spec/feature text is saved to persistent project files (`project_spec.md`, `project_features.md`) — no temporary files.

**Enhanced greenfield analyze** (for specs referencing agent/self-improvement/security):
- Auto-detects keywords (`agent`, `self-improvement`, `vulnerability`, `safe`, etc.) and scans the **target workspace** for relevant Python files, including them in the analysis so findings cite real code.
- Produces a structured 8-section analysis: SCOPE, ASSUMPTIONS, RISKS, DEPENDENCIES, **THREAT MODEL & ATTACK SURFACE**, **MISSING INFORMATION (BLOCKERS)**, **CLARIFYING QUESTIONS**, **SUCCESS METRICS & OVERSIGHT**.
- Ends with `**BLOCKED:** yes|no` — if blocked and `--force` is not given, the pipeline halts before plan/entities/tasks and prints the questions for the user.
- Runs a self-critique refinement pass and appends findings to `project_analysis.md`.
- **Reasoning leakage stripping**: LLM chain-of-thought, self-correction marks, and XML thinking tags (``, ``) are automatically stripped from all generated output files (analysis, plan, entities, taskplan). Reasoning patterns like "Let's...", "Wait,...", "[Output Generation]", and checkmark markers are removed before writing to disk.
- **Code-claim verification** (all analyze modes): after generation, every file path, symbol name, line number, and code snippet in the analysis is checked against the actual workspace. Unverifiable claims are flagged in an appended `## Verification Report` — nothing is silently trusted.
- Writes a traceability header referencing the workspace path and spec file.

Brownfield mode (`workflow .`) uses multi-dimensional analysis across 5 dimensions:

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

**implement** — File safety (6 layers):
- **Prevention**: LLM prompts instruct the LLM to use sub-package paths (`agent_core/thing.py`) and avoid bare root-level filenames
- **Stdlib shadowing detection**: workflows warn the LLM to avoid stdlib-module directory names (e.g. `logging/`, `json/`, `types/`). The analysis verifier flags shadowed paths as `[UNVERIFIED]` before implementation. The implement command auto-redirects shadowed paths to safe alternatives (`logging` → `logging_utils`).
- **Collision warnings at workflow time**: taskplan generation scans existing class/function names and filenames per directory, warns the LLM to avoid conflicts (e.g. "DO NOT create retry_policy.py if retry.py already exists")
- **Post-write rejection**: every generated file is checked for class-name conflicts with existing code in the same directory. Conflicting files are auto-deleted immediately — no manual cleanup needed.
- **Auto-review after every run**: static checks for class-name conflicts, module collisions, and unwired modules — printed immediately without LLM
- **`--review` flag**: adds LLM deep analysis + offers to delete dangerous files with a y/N prompt
- **`--review` flag**: adds LLM deep analysis + offers to delete dangerous files with a y/N prompt
- **Patch-based fixing**: LLM outputs minimal `[PATCH:]` diffs instead of full-file rewrites. Shows changed lines with +/-, asks y/N before applying. Safety checks: old lines must exist, result must compile. `[FILE:]` still works as fallback.
- **`--modify` mode**: merges generated content into existing compile-OK modules via a reviewed unified diff (diff-apply) — the middle ground between `--keep` (skip) and `--force` (wholesale overwrite). Wholesale rewrites are rejected unless similarity ≥ 0.5; `--allow-rewrite` opts in.
- **Anchored patch fallback**: When LLM ``@@`` line numbers are wrong (a common failure mode), a content-based anchored matcher locates the correct position by matching actual file text in a ±60-line window — absorbing fence-wrapped diffs, fused headers, and ``N |`` numbered-context artifacts.
- **Workspace-agnostic**: path rules detected dynamically from workspace structure (finds `__init__.py` directories). No hardcoded prefixes — works in any project.
- **SOLID enforcement**: implement system prompt enforces SRP. New files capped at 150 lines — LLM splits large concepts. Modifying existing code uses minimal changes. Prefers composition over inheritance.

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
│   │   ├── opencode_provider.py  # Hosted opencode-go provider (direct API + serve)
│   │   ├── provider.py           # LLMProvider protocol
│   │   ├── retry.py              # RetryPolicy
│   │   └── tool_loop.py          # ToolLoopRunner (native tool_calls loop)
│   ├── tool_schemas.py           # NLP tool schemas + names (single source of truth)
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
│       ├── workflow_cmd.py       # Full pipeline
│       ├── reasoning_strip.py    # LLM reasoning token removal
│       ├── decide_cmd.py         # Decision tracking command
│       ├── analysis_verifier.py  # Code-claim verification
│       ├── review_cmd.py         # Human verification gate over failed traces
│       ├── reconstruct_cmd.py    # Rebuild files from JSONL trace logs
│       ├── paste_image_cmd.py    # Paste an image for vision-capable LLMs
│       ├── run_cmd.py            # LLM-free shell command execution
│       └── freshness.py          # Stale-module guard (REPL warning)
├── harnessfix/                   # Self-improvement: traces, HTIR, diagnose, review, history
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

Set environment variables or use `.env` (see `.env.example` for the full list):
```env
AGENT_MODEL=laguna-s-2.1
LMSTUDIO_URL=http://localhost:1234/v1
OPENAI_API_KEY=your-key
AGENT_LLM_PROVIDER=lmstudio    # or opencode (an "opencode-go/..." model prefix wins)
OPENCODE_API_KEY=sk-...        # hosted opencode-go API (direct mode; also read from opencode's auth.json)
OPENCODE_TIMEOUT=600           # API request timeout in seconds
```

## Models

| Model | Description |
|---|---|
| `laguna-s-2.1` | Laguna S 2.1 MoE A8B — fast agentic coding |
| `qwen3.6-27b-mtp` | Qwen 3.6 27B — chat, codegen, large context |
| `qwen/qwen3.8-27b` | Qwen 3.8 27B — chat, codegen, reasoning (safe minimal thinking-disable) |
| `google/gemma-4-31b` | Gemma 4 31B — chat, reasoning, fast generation |
| `opencode-go/deepseek-v4-flash` | Hosted opencode-go default (opencode.ai/zen/go, native tools) |

## Testing

```bash
pytest tests/ -v
```
1210 tests (1210 passed, 2 skipped; ~3.5 min full run — use `--no-cov` for speed).
Plus a targeted entry-point package `agent_core/tests/` (31 tests:
`python -m pytest agent_core/tests -q --no-cov`).

### Recent Additions
- **Stdlib shadowing protection**: three-layer defense across workflow prompts, analysis verification, and implement file generation. Detects directory names that shadow stdlib modules (`logging/`, `json/`, `types/`, `config/`), warns the LLM to avoid them, flags violations in verification reports, and auto-redirects shadowed paths to safe alternatives.
- **Reasoning leakage stripping**: automatically removes LLM chain-of-thought text (`` / `` tags), self-correction markers, "[Output Generation]" badges, "Let's..."/"Wait,..." reasoning lines, and checkmark emoji markers from all workflow-generated output files. Keeps analysis, plan, entities, and taskplan files clean.
- **Config module shadowing resolved**: removed dead `agent_core/config/` directory that shadowed `agent_core/config.py`, fixing `import agent_core.config.schema` failures. Similarly resolved `agent_core/logging/` stdlib shadow.
- Optimize command: mechanical `regex_in_loop` fixes hoist `re.compile()` into descriptive named constants (`_FOR_WHILE_RE`, `_TYPE_EQ_RE`) derived from the pattern, reusing existing identical compiles instead of duplicating them; detectors and mechanical fixers skip docstring lines entirely
- **Decision tracking system**: record, search, and enforce design decisions. `decide` command for manual recording. Auto-extracts decision candidates from `workflow`, `implement`, and `fix` runs. Past decisions are injected as hard constraints into LLM prompts — preventing accidental contradictions. LLM-powered contradiction detection and resolution.
- Patch application tests for strict, anchored, deletion-only, and `N |`-prefix stripping
- LM Studio thinking-disable payload tests across models and server versions
- **`implement --modify`** mode: diff-apply merge of generated content into existing compile-OK modules as a reviewed unified diff (middle ground between `--keep` skip and `--force` overwrite), with wholesale-rewrite rejection unless similarity ≥ 0.5 (`--allow-rewrite`)
