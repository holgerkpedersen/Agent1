# Agent1 — User Manual

Run: `python agent.py`

| Command                      | Purpose                             |
| ---------------------------- | ----------------------------------- |
| `read <path>`                | Read a file                         |
| `write <path> <text>`        | Write content to a file             |
| `search <text>`              | Search files for a string           |
| `analyze <target>`           | AI analysis of code                 |
| `plan <analysis> <plan>`     | Generate coding plan                |
| `entities <analysis> <plan>` | Extract shared entities             |
| `taskplan <analysis> <plan>` | Generate task list                  |
| `workflow <target>`          | Full pipeline (analyze → implement) |
| `implement <taskplan>`       | Implement files from task plan      |
| `fix <traceback>`            | Auto-fix from error traceback       |
| `fix <file> --desc "..."`    | Describe an issue, LLM fixes it     |
| `model <name>`               | Switch LLM model                    |
| `model list`                 | List available models               |
| `clear`                      | View and clear memory               |
| `cleanup`                    | Find orphaned files                 |
| `quit` / `exit` / `q`        | Exit the REPL                       |

Any input not matching a command is treated as natural language and sent directly to the LLM.

---

## Quick Examples

### Reading and writing files

```
> read tests/test_tool_router.py
# --- contents of test_tool_router.py ---

> read agent_core/entities.py:85-100
# Lines 85-100 of entities.py

> write notes.txt Started working on the memory module today.
Writing to file: notes.txt
```

### Searching code

```
> search ToolRouter
Found 14 matches in 6 files
agent_core/tool_dispatcher.py:23:    router: ToolRouter
tests/test_tool_router.py:10:    from tool_router import ToolRouter
tool_router.py:129: class ToolRouter:
...
```

### Analyzing code

```
> analyze agent_core/commands/fix_cmd.py
[Sending to LLM for analysis...]
## Analysis: fix_cmd.py
The fix command handles two modes: traceback parsing and
description-based fixes. The dependency tracing via imports
is thorough but could be optimized by caching the import graph...
```

With a specific question:

```
> analyze agent.py --desc "How does the fix command work?"
[LLM reads agent.py and fix_cmd.py, answers with detailed explanation]

> analyze agent_core/commands/implement_cmd.py --desc "What safety guards prevent file collisions?"
[LLM answers about _is_dangerous_filename, auto-repair, etc.]

> analyze agent.py --desc "List every class and its responsibility"
```

### Natural language

```
> What does the ToolRouter class do?

[LLM responds based on project files it has read]

> Show me all places where ConfigurationError is raised

[LLM searches the project and answers]
```

---

## The Workflow Pipeline

### 1. Analyze existing codebase

```
> workflow .
[analyze] Scanning py files...
Workspace: C:\Dev\Agent1
38 Python files found
[analyze] Written to project_analysis.md

Next: implement project_tasks.md ... --workspace C:/Dev/Agent1 --keep
```

Add `--brainstorm` for a 6th dimension (bold creative features):

```
> workflow . --brainstorm
```

### 2. Greenfield: build from a specification

```
> workflow . --from spec.md
[spec] Loaded from spec.md
[plan] Creating plan...
[plan] Written
[entities] Written
[taskplan] Written

Next: implement project_tasks.md ... --workspace C:/Dev/Agent1 --force
```

Inline specification with `--desc`:

```
> workflow . --desc "A CLI tool that converts CSV to JSON with streaming support"
```

### 3. Brownfield: add features to existing code

```
> workflow . --features features.md
[features] Loaded from features.md
[analyze] Scanning existing py files...
[analyze] Written  (existing issues + new feature gaps)
[plan] Appended to project_plan.md
[entities] Appended
[taskplan] Appended

Next: implement project_tasks.md ... --workspace C:/Dev/Agent1 --keep
```

Inline features:

```
> workflow . --features "Add a caching layer for file reads, add rate limiting to LLM calls"
```

### 4. Implement from task plan

```
> implement project_tasks.md --force
[File list from LLM...]
Found 6 files to implement: agent_core/cache.py, ...
[Generating...]
Compiled OK: agent_core/cache.py
Compiled OK: ...
Implementation complete: 6/6 files
```

Flags:
| Flag | Effect |
|---|---|
| `--force` | Overwrite existing files |
| `--keep` | Skip files that already compile OK |
| `--fix` | Retry compilation errors with the LLM |
| `--retry` | Re-generate only files missing from disk (uses cached file list) |
| `--workspace <path>` | Target a different workspace |
| `--workspace <path>` | Target a different workspace |

Example with supporting documents:

```
> implement project_tasks.md project_analysis.md project_plan.md project_entities.md --workspace /c/Dev/MyProject --force
```

Safety: implement automatically refuses to write files that shadow stdlib modules (e.g., `types.py`, `config.py`) or conflict with existing packages. Bare filenames are auto-repaired into sub-packages (e.g., `types.py` → `agent_core/types.py`).

---

## Fixing errors

### From a traceback (paste it)

```
> fix
Paste the full traceback, then press Enter on an empty line:
Traceback (most recent call last):
  File "C:\Dev\Agent1\agent.py", line 528, in run_interactive
    registry.register(ImplementCommand())
TypeError: Can't instantiate abstract class ImplementCommand
                  [press Enter on empty line]

Parsing traceback...
Error in agent.py:528
  TypeError: Can't instantiate abstract class ImplementCommand...
Context (lines 525-531):
    525: registry.register(ReadCommand())
    526: registry.register(WriteCommand())
>>> 528: registry.register(ImplementCommand())

Sending to LLM for fix...
Fixed: agent.py (480 bytes)
Compiled OK!
```

Inline traceback:

```
> fix "Traceback (most recent call last):
  File \"test.py\", line 3, in <module>
    import nonexistent_module
ModuleNotFoundError: No module named 'nonexistent_module'"
```

### Cascade import chains

```
> fix
[Paste traceback...]

Parsing traceback...
Cascade detected! 6 files in trace:
  1. <frozen runpy>:15 -> ROOT
  2. types.py:1
  3. typing.py:27
  4. contextlib.py:7
  5. functools.py:22

  Root cause is in types.py:1, not in functools.py
  Fixing types.py instead...

Shadow warning: local file is conflicting with stdlib module 'types'
The local file C:\Dev\Agent1\types.py shadows 'types' from the Python stdlib.
Fix: rename or move it (e.g. types_defs.py or put it inside a package).
Skipping LLM fix — this is a naming conflict, not a code error.
```

### From a description

```
> fix agent_core/config.py --desc "The ConfigManager.load() method doesn't validate that the loaded file is actually a YAML file before parsing"
```

---

## Model management

```
> model
# Shows all models from LM Studio API with loaded status, sizes, and VRAM usage

  VRAM: 15.8 GB — 1 model(s) loaded

    1. laguna-s-2.1          8B-MoE     4.2 GB   [loaded, current]
    2. qwen3.6-27b-mtp       27B        15.8 GB
    3. google/gemma-4-31b    31B        18.1 GB

  Current model: laguna-s-2.1  (Laguna S 2.1 MoE A8B — agentic coding, thinking)
  3 models available from LM Studio API
```

```
> model list
# Same table without interactive prompt

> model 27b
  Loading qwen3.6-27b-mtp ...
  loaded (9.1s) — qwen3.6-27b-mtp
  Switched: laguna-s-2.1 -> qwen3.6-27b-mtp  (Qwen 3.6 27B — chat, codegen, large context)
```

```
> model load google/gemma-4-31b
  Loading: google/gemma-4-31b ...
  loaded (12.3s) — google/gemma-4-31b
  Switched to: google/gemma-4-31b
```

```
> model unload qwen
  Unloading: qwen3.6-27b-mtp
  unloaded qwen3.6-27b-mtp

> model unload --all
  Unloading all (2 model(s)) ...
    unloaded google/gemma-4-31b
```

```
> model reload
  Agent: laguna-s-2.1  |  LM Studio has: qwen3.6-27b-mtp
  Syncing agent to match LM Studio...
  Done: qwen3.6-27b-mtp
```

Fuzzy matching works with partial names: `model 27b`, `model gemma`, `model laguna`.

---

## Memory management

```
> clear
Agent Memory Status
  Chat history: 12 messages
  Files read: 7 (1 stale)
  Working memory: 3 items
  Semantic index: 142 entries
  Knowledge graph: 8 relationships
Clear all? (y/N): n
Not cleared.

> clear stats
# Same output, but skips the confirmation prompt

> clear --force
# Clears everything without asking
```

---

## Finding unused files

```
> cleanup
Scanning workspace...
Reference graph from project_*.md: 24 referenced files
Unreferenced .py files (candidates for deletion):
  old_module.py
  temp_script.py
  _backup_.py
```

---

## Common patterns

### Full greenfield project

```
> workflow . --desc "A REST API client for GitHub issues with caching"
# generates analysis, plan, entities, taskplan

> implement project_tasks.md --force
# generates all source files
```

### Fix a bug end-to-end

```
> fix
[Paste traceback]
# LLM analyzes, finds root cause, fixes the file

> python agent.py
# Verify the fix works
```

### Add a feature to an existing project

```
> workflow . --features "Add database persistence layer using SQLite"
# analyzes existing code + new feature requirements

> implement project_tasks.md --keep
# generates new files, skips existing ones that compile OK
```

### Explore and understand code

```
> read agent_core/commands/workflow_cmd.py:25-60

> search "from agent_core import"

> What is the relationship between ToolRouter and ToolDispatcher?
# Natural language — LLM reads relevant files and answers

> analyze src/agent1/graph/dependency_visualizer.py
# Full AI analysis of a specific file
```

---

## Tips

- **`quit`** or `Ctrl+C` to exit anytime.
- **Use natural language** for exploration — ask questions about the codebase without memorizing command syntax.
- **The workflow pipeline** handles everything: `workflow . --brainstorm` → review `project_analysis.md` → `implement ...`.
- **Paste full tracebacks** into `fix` — it finds the root cause across import chains, even when the error manifests deep in stdlib.
- **`--keep`** avoids overwriting files that already compile OK — safe for brownfield work.
- **`--force`** regenerates everything — use for greenfield or when you want a clean slate.
