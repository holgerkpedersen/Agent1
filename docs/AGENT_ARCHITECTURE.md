# Agent1 — System Overview

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     agent.py (main)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Agent     │  │  LLMClient   │  │ run_interactive│  │
│  │  ─────────  │  │  ──────────  │  │  ─────────────│  │
│  │ workspace   │  │ chat()       │  │ REPL loop     │  │
│  │ chat_nlp()  │  │ chat_stream()│  │ commands→Reg   │  │
│  │ _execute_   │  │ analyze_code │  │ fallback→NLP   │  │
│  │   nlp_tool()│  └──────┬───────┘  └───────────────┘  │
│  │ read_file() │         │                               │
│  │ _resolve_   │    ┌────▼────────────┐                  │
│  │   nlp_path()│    │ LMStudioProvider│                  │
│  └──────┬──────┘    │  ───────────────│                  │
│         │           │ POST /chat/     │                  │
│         │           │ completions     │                  │
│         │           └────────────────┘                  │
└─────────┼───────────────────────────────────────────────┘
          │
    ┌─────┴──────────────────────────────────┐
    │               agent_core/              │
    │                                        │
    │  ┌─ commands/ ─┐  ┌── llm/ ──┐         │
    │  │ base.py     │  │ lmstudio │         │
    │  │ registry.py │  │ tool_loop│         │
    │  │ paste_cmd   │  │ retry    │         │
    │  │ fix_cmd     │  │ profiles │         │
    │  │ implement_  │  └──────────┘         │
    │  │ workflow_   │                       │
    │  │ ...         │  ┌────────────────┐   │
    │  └─────────────┘  │ file_system.py │   │
    │                   │ file_searcher  │   │
    │  ┌──────────────┐ │ tool_dispatcher│   │
    │  │ constants.py │ │ entities       │   │
    │  │ config.py    │ │ path_utils     │   │
    │  │ patterns.py  │ └────────────────┘   │
    │  └──────────────┘                      │
    └────────────────────────────────────────┘
```

---

## Entry Point: `agent.py`

### `Agent` class (`agent.py:116`)

Central state container. Holds all memory, delegates to LLM, registers tools.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `workspace` | `str` | Default project root (`/c/Dev/Agent1`) |
| `model_name` | `str` | Resolved via `KNOWN_MODELS` map |
| `llm` | `LLMClient` | Thin wrapper around `LMStudioProvider` |
| `fs` | `FileSystem` | Read/write file operations |
| `searcher` | `FileSearcher` | Text search across project files |
| `dispatcher` | `ToolDispatcher` | Structured tool dispatch (10 tool handlers) |
| `_chat_history` | `list[dict]` | NLP conversation context (role/content pairs) |
| `_nlp_workspace` | `str\|None` | Workspace override for NLP tools — set by `paste --workspace` |

### `LLMClient` class (`agent.py:43`)

Adapter wrapping `LMStudioProvider`. Key methods:
- `chat(messages, tools?, max_tokens?)` → sends to LM Studio `/chat/completions`
- `chat_with_continuation(messages)` → auto-resumes truncated responses (up to 3 retries)
- `analyze_code(code)` → delegates to provider

### `run_interactive()` (`agent.py:718`)

The REPL entry point. Two dispatch branches:
1. **Command dispatch** — first word matches a registered command (read/write/search/fix/implement/workflow/...15 total) → `registry.execute(name, args, agent)`
2. **NLP fallback** — no match → `agent.chat_nlp(user_input)`

Commands are registered via `CommandRegistry` and triggered from a hardcoded dispatch list at `agent.py:732`.

### `chat_nlp()` — The NLP ReAct Loop (`agent.py:490`)

Flow:
```
user_input
  │
  ├─ len > 500 or _nlp_workspace set? → skip heuristic, go to ReAct
  └─ else _parse_natural_language() → if tool matched → execute directly
     └─ "unknown" → ReAct loop
                      │
                      ├─ inject system prompt (first call only)
                      ├─ append user message to _chat_history
                      │
                      └─ for _ in range(6):
                           llm.chat(_chat_history[-20:])
                           │
                           ├─ <tool_call> tag found? → _execute_nlp_tool()
                           │   └─ append result to history, loop
                           │
                           └─ no tag → print response, break
                              │
                              └─ loop exhausted → force final synthesis
```

### `_execute_nlp_tool()` (`agent.py:162`)

Parses `<tool_call>` text with `shlex.split(tool_text, posix=False)`, strips `<arg_key>`/`<arg_value>` XML noise, dispatches by command name:

| Tool | Arguments | Action |
|------|-----------|--------|
| `search` | `<query>` | `FileSearcher.search(query, _nlp_workspace \|\| workspace)` |
| `read` | `<path>` | `self.read_file(path)`, returns first 5000 chars |
| `list_files` / `list` | `[path]` (default `.`) | `os.listdir()`, dirs marked `/` |
| `fix` | `<path> --desc "text"` | `FixCommand().execute(args, self)` |

### `_parse_natural_language()` (`agent.py:444`)

Keyword-heuristic parser for short REPL inputs (bypassed for pasted text >500 chars). Matches patterns like:
- `search` + `file` + `for/in/inside` → `search_file`
- `read` + `.py` → `read_file`
- `write` + `to` → `write_file`
- `analyze`/`explain` + file extension → `llm_analyze`

Returns `("unknown", {})` if no pattern matches, triggering the ReAct loop.

---

## LLM Layer: `agent_core/llm/`

### `lmstudio.py` — `LMStudioProvider`

Models as REST API. Key methods:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `chat(messages, tools?, max_tokens?)` | `POST /chat/completions` | OpenAI-compatible chat |
| `chat_stream(messages)` | `POST /chat/completions` (stream) | Real-time token streaming |
| `analyze_code(code)` | `POST /chat/completions` | Code analysis with system prompt |
| `list_models()` | `GET /api/v1/models` | List loaded models |
| `load_model(name)` | `POST /api/v1/models/load` | Load a model into VRAM |
| `unload_model(name)` | `POST /api/v1/models/unload` | Unload a model |

Uses `RetryPolicy` for transient failures (connection errors, rate limits).

### `tool_loop.py` — `ToolLoopRunner`

Reusable, testable tool-orchestration loop (separate from `chat_nlp`). Used by `fixcommand/`. 

Flow: up to 15 iterations of `llm_chat_fn` → parse native `tool_calls` → `execute_tool_fn` → feed results back.

Uses OpenAI-format `tool_calls` (structured JSON), NOT string-based `<tool_call>` tags.

### `model_profiles.py`

Named presets for temperature + max_tokens:
- `fast-codegen` (temp=0.1, tok=16000)
- `deep-analysis` (temp=0.7, tok=16000)
- `precise` (temp=0.3, tok=16000)

Managed via `model profile` commands.

### `retry.py` — `RetryPolicy`

Exponential backoff with jitter for LLM API calls. Handles connection errors, rate limits, server overload.

---

## Command Layer: `agent_core/commands/`

### Contract: `base.py` — `Command` (ABC)

```python
class Command(ABC):
    name: str         # command name in REPL
    help_text: str    # shown in banner
    async execute(args: list[str], agent: Agent) -> bool   # True = continue REPL
```

### Registry: `registry.py` — `CommandRegistry`

Key-value map. `register(cmd)` → `execute(name, args, agent)`. Follows Open/Closed principle.

### Registered Commands (15 total)

| Command | File | LLM Calls | Writes Files? | Use Case |
|---------|------|-----------|---------------|----------|
| `read` | `read_cmd.py` | 0 | No | Read a file |
| `write` | `write_cmd.py` | 0 | Yes | Write content to file |
| `search` | `search_cmd.py` | 0 | No | Search for text |
| `clear` | `clear_cmd.py` | 0 | No | Clear agent memory |
| `model` | `model_cmd.py` | 0 | No | Manage LM Studio models |
| `analyze` | `analyze_cmd.py` | 1 | Yes (.md) | AI analysis of file/code |
| `plan` | `plan_cmd.py` | 1 | Yes (.md) | Generate coding plan |
| `entities` | `entities_cmd.py` | 1 | Yes (.md) | Generate shared entities |
| `taskplan` | `taskplan_cmd.py` | 1 | Yes (.md) | Generate implementation tasks |
| `cleanup` | `cleanup_cmd.py` | 0 | No | Show unreferenced files |
| `implement` | `implement_cmd.py` | 1+/file | Yes (.py) | Batch code generation |
| `fix` | `fix_cmd.py` | 1-3 | Yes (.py) | Fix errors in files |
| `workflow` | `workflow_cmd.py` | 4 | Yes (.md) | Full pipeline: analyze→plan→entities→taskplan |
| `optimize` | `optimize_cmd.py` | 1 | Yes (.py) | Find and apply optimizations |
| `perf` | `perf_cmd.py` | 0 | No | Performance dashboard |
| `paste` | `paste_cmd.py` | 0 (delegates to NLP) | No | Multiline text → ReAct loop |

### `fix_cmd.py` — FixCommand (deep dive)

Three modes, selected by argument presence:

| Mode | Trigger | LLM Calls | How it works |
|------|---------|-----------|-------------|
| Traceback | No `--desc` | 1 | Parse `.py:line` from traceback, read file, send to LLM, apply `[FILE:]` |
| `--desc` on-demand | `--desc` (default) | 1-3 | Keyword-score candidate files, top-5 full source + sigs for rest, `[READ:]` loop |
| `--desc --full` | `--desc --full` | 1 | All project `.py` files (full source for candidates, sigs for others) in one call |

**Fix application:**
- `[PATCH:]` blocks: 3-layer safety (line-match verification → compile check → y/N confirmation)
- `[FILE:]` blocks: write directly, no compile check in on-demand mode

### `implement_cmd.py` — ImplementCommand

Batch code generation from taskplan. Flow:
1. Parse `project_tasks.md` → extract file list and tasks
2. Scan workspace for existing exports → build export map
3. Generate files in batches (one per file or grouped)
4. Verify: compile-check, import-check, name-collision detection
5. Write to disk, log to `CHANGES.md`

States: completed, compile error, missing, rejected (name collision).

### `workflow_cmd.py` — WorkflowCommand

Orchestrated LLM pipeline (4 steps: analyze → plan → entities → taskplan). Each step writes a `project_*.md` file. Steps are skipped if output exists (unless `--force`). Three modes: greenfield (`--stdin`/`--from`/`--desc`), features (`--features`), default (analyze existing).

### `paste_cmd.py` — PasteCommand

Three-line wrapper: reads all stdin until Ctrl+Z, sets `_nlp_workspace`, calls `agent.chat_nlp(text)`. No error parsing, no auto-fix — delegates fully to the ReAct loop.

---

## Core Utilities: `agent_core/`

| Module | Class/Function | Purpose |
|--------|---------------|---------|
| `file_system.py` | `FileSystem` | Read/write/patch files with path safety (isolated to workspace) |
| `file_searcher.py` | `FileSearcher` | Text search via `findstr` (Windows) or `grep` (Unix) with Python fallback |
| `tool_dispatcher.py` | `ToolDispatcher` | Registry-based tool dispatch for structured tool calls |
| `constants.py` | `KNOWN_MODELS`, `DEFAULT_MODEL`, `resolve_model()` | Model name resolution and persistence |
| `path_utils.py` | `to_windows_path()` | Path normalization (Unix→Windows) |
| `entities.py` | `extract_signatures()` | Parse Python source for class/function signatures |
| `context_management.py` | — | Context window management for LLM calls |
| `config.py`, `config/schema.py` | — | Configuration loading and validation |

---

## NLP Flow Summary

```
User types at REPL prompt "> hello"
  │
  ├─ "hello" not in command list → NLP fallback
  │
  └─ agent.chat_nlp("hello")
       │
       ├─ len("hello") < 500, no _nlp_workspace
       ├─ _parse_natural_language("hello") → ("unknown", {})
       │
       └─ ReAct loop:
            │
            ├─ [1st call] inject system prompt with tool instructions
            ├─ append "hello" as user message
            ├─ llm.chat(history[-20:])
            │
            └─ LLM responds: "How can I help you?"
                 │
                 ├─ no <tool_call> tags → print response, break
                 └─ done

User types "> paste --workspace C:\Dev\CandyCrush"
  │
  ├─ matched as paste command
  ├─ PasteCommand.execute() → sys.stdin.read() (block until Ctrl+Z)
  ├─ agent._nlp_workspace = "C:\Dev\CandyCrush"
  ├─ agent.chat_nlp(pasted_text)
  │
  └─ len(pasted_text) > 500 → skip _parse_natural_language
       └─ ReAct loop with _nlp_workspace set
            │
            ├─ LLM: <tool_call>list_files C:\Dev\CandyCrush</tool_call>
            ├─ _execute_nlp_tool: list_files → shows dirs with /, .py files
            ├─ LLM: <tool_call>read C:\Dev\CandyCrush\src\agent1\file.py</tool_call>
            ├─ _execute_nlp_tool: read → returns first 5000 chars
            ├─ LLM: <tool_call>fix C:\Dev\CandyCrush\src\agent1\file.py --desc "syntax error"</tool_call>
            ├─ _execute_nlp_tool: fix → FixCommand().execute()
            └─ ... up to 6 iterations
```

---

## Key Strategies & Patterns

| Pattern | Where | How |
|---------|-------|-----|
| **Command Pattern** | `agent_core/commands/` | Each REPL command is a `Command` subclass, registered with `CommandRegistry`. Open for extension, closed for modification. |
| **ReAct Loop** | `agent.py:490` (chat_nlp) | Reason (LLM generates tool call) → Act (execute tool) → Observe (result fed back). Up to 6 iterations. |
| **Tool Dispatch** | `agent_core/tool_dispatcher.py` + `_execute_nlp_tool()` | Two parallel dispatch systems: structured (`ToolDispatcher` for 10 tools) and string-based (`_execute_nlp_tool` for 4 tools). |
| **Adapter Pattern** | `LLMClient` → `LMStudioProvider` | Thin wrapper providing backward-compatible API over the provider. |
| **Retry Pattern** | `agent_core/llm/retry.py` | Exponential backoff with jitter for transient LLM API failures. |
| **Path Safety** | `_safe_path()` → `_normalize_path()` → `to_windows_path()` | All file operations go through path normalization chain. |
| **Workspace Isolation** | `_nlp_workspace` | `paste --workspace` overrides the default workspace for NLP tools only, cleared automatically after `chat_nlp`. |
| **Heuristic Preprocessing** | `_parse_natural_language()` | Short REPL inputs are keyword-parsed to avoid LLM calls for simple operations. Bypassed for pasted text. |
| **Context Management** | `_chat_history[-20:]` | Only last 20 messages sent to LLM. Tool results truncated to 3000 chars. |

---

## Extension Points

1. **New REPL commands** — subclass `Command`, implement `name`/`help_text`/`execute()`, register in `run_interactive()`
2. **New NLP tools** — add `if cmd == "..."` branch in `_execute_nlp_tool()`, update system prompt
3. **New LLM providers** — implement provider interface in `agent_core/llm/`, swap in `LLMClient.__init__`
4. **New structured tools** — register with `ToolDispatcher` in `_register_tool_handlers()`
5. **New model profiles** — add to `model_profiles.py`, persist in model.json
