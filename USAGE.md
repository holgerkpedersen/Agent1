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
| `optimize <file>`            | Find speed/memory/quality issues    |
| `perf`                       | Command performance dashboard       |
| `quit` / `exit` / `q`        | Exit the REPL                       |

Any text not matching a command is sent to the LLM as **natural language** with full conversation history. The LLM can explore the codebase using tools:

```
> Make a brainstorm on this repo
  [tool] list_files C:/Dev/Agent1 -> 609 bytes
  [tool] read C:/Dev/Agent1/README.md -> 5000 bytes
  [tool] read C:/Dev/Agent1/agent_core/commands -> 325 bytes
Agent1 is a Python AI agent framework with 14 commands...

> What safety guards does implement use?
  [LLM remembers previous context + reads implement_cmd.py → answers]
```

The LLM can use `<tool_call>search query</tool_call>`, `<tool_call>read path</tool_call>`, and `<tool_call>list_files dir</tool_call>` — up to 5 exploration rounds before producing a final answer.

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
[LLM reads agent.py, follows imports to fix_cmd.py, answers with full analysis]

> analyze agent_core/commands/implement_cmd.py --desc "What safety guards prevent file collisions?"
[LLM answers about _is_dangerous_filename, auto-repair, etc.]
```

Analyze pasted text (no file needed):

```
> analyze --stdin --desc "What approach works best?"
Paste text to analyze. Type --- on its own line when done:
[Paste any multi-line content with as many blank lines as you want]
---
[LLM analyzes the pasted text with your question]
```

Deep analysis (follows imports without a question):

```
> analyze agent.py --deep
# Reads agent.py + all local imports, sends combined analysis
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
[analyze] Analyzing spec...    ← NEW: scope, assumptions, risks, dependencies
[analyze] Written
[plan] Creating plan...        ← now uses spec + analysis as context
[plan] Written
[entities] Written             ← uses spec + analysis + plan
[taskplan] Written             ← uses spec + analysis + plan + entities

Next: implement project_tasks.md ... --workspace C:/Dev/Agent1 --force
```

Inline specification with `--desc`:

```
> workflow . --desc "A CLI tool that converts CSV to JSON with streaming support"
```
Specs are saved to `project_spec.md` (not temp files) for traceability.

Or paste a multi-line spec with `--stdin`:

```
> workflow . --stdin --brainstorm --force
Paste spec or description. Type --- on its own line when done:
Build me a CLI tool that converts CSV to JSON
with streaming support and chunked file handling.
It should handle files up to 10GB.
---
[stdin] 120 chars
[plan] Creating plan...
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
| `--review` | After generation, review new files for bugs + DRY violations |
| `--workspace <path>` | Target a different workspace |

Example with supporting documents:

```
> implement project_tasks.md project_analysis.md project_plan.md project_entities.md --workspace /c/Dev/MyProject --force --review
```

The `--review` flag performs a post-generation audit:

```
  [review] Reviewing 8 new file(s): fixcommand/...
  [review] Duplicate functions across files:
    parse_tool_calls() in: parser/__init__.py, parser/structured.py

  Reviewing fixcommand/core/tools/definitions.py (1200 bytes)...
  Invalid API schema: missing "type":"object" and "required" fields
```

Safety: implement now has 4 layers of protection:

1. **Workflow collision warnings** — taskplan LLM sees existing names per directory before generating
2. **SOLID enforcement** — SRP (Single Responsibility Principle): new files max 150 lines, LLM splits large concepts across multiple focused files. Existing files get minimal changes only.
3. **Post-write rejection** — files with class-name conflicts are auto-deleted immediately
4. **Auto-review** — after every run: class conflicts, module collisions, unwired modules flagged
5. **`--review` flag** — offers to delete dangerous files (y/N) + LLM deep analysis

File discovery now parses the taskplan directly — no LLM invents wrong filenames.

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

### From a description (on-demand)

```
> fix agent_core/config.py --desc "The ConfigManager.load() method doesn't validate YAML files"
```

On-demand mode (default): sends only the top 5 most relevant files by keyword match, plus signatures for the rest. The LLM can request additional files with `[READ: path.py]` and iterates up to 3 rounds.

```
  On-demand: 5 full files + 12 candidate sigs + 86 other sigs (48KB)
  Full source: config.py, fix_cmd.py, implement_cmd.py, workflow_cmd.py, model_cmd.py
  Round 1 (48KB)...
    Read: agent_core/constants.py, agent_core/exceptions.py
  Round 2 (65KB)...
  Fixed: agent_core/config.py (2340 bytes)
```

Use `--full` for the legacy "send everything" mode:

```
> fix agent.py --desc "..." --full
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

**Auto-sync**: Running `model` without args automatically detects if the agent's configured model isn't loaded and switches to what LM Studio actually has in VRAM.

**VRAM-aware**: `model load` tries to load directly. If LM Studio returns a space/memory error, it automatically unloads the current model and retries — no manual unload needed.

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

> analyze agent.py --desc "How does the fix command work?"
# LLM reads agent.py and answers the specific question
```

---

## Optimizing code

```
> optimize agent_core/commands/fix_cmd.py
  Static analysis found 12 issue(s):
  fix_cmd.py:
    line   21: [silent_except] Replace 'pass' with logging or re-raise
    line  134: [regex_in_loop] Move re.compile() to module level
  Sending to LLM for deeper analysis...
  [OPTIMIZE: fix_cmd.py:134] Speed: regex compiled inside loop

> optimize agent_core/ --apply
  # Analyzes all .py files, asks y/N before applying each fix

> optimize agent.py --apply --yes
  # Analyzes and applies all suggestions without asking
```

## Performance dashboard

Every command is automatically timed. View stats anytime:

```
> perf
  Commands: 23  |  Runtime: 342.1s  |  Input: 824 chars  |  LLM calls: ~12
  ────────────────────────────────────────────────────────────────────────
  command      calls    total      avg      max      last
  workflow        2    228.5s    114.3s   198.2s    30.3s
  implement       1     98.4s     98.4s    98.4s    98.4s
  analyze         8      8.2s      1.0s     3.2s     0.8s

> perf --detail
  # Shows every individual command execution with timestamp

> perf --reset
  # Clears all collected stats

> perf --html
  # Exports self-contained HTML dashboard
```

---

## Tips

- **`quit`** or `Ctrl+C` to exit anytime.
- **`analyze --desc`** gives targeted answers — skip the generic review and ask exactly what you want to know.
- **`analyze --stdin`** analyzes any pasted text — feed LLM output back for critique.
- **`fix --desc`** is now on-demand — sends only relevant files, uses `[READ:]` to fetch more as needed.
- **`workflow --stdin`** accepts multi-line specs pasted directly.
- **Use natural language** for exploration — ask questions about the codebase without memorizing command syntax.
- **The workflow pipeline** handles everything: `workflow . --brainstorm` → review `project_analysis.md` → `implement ...`.
- **Paste full tracebacks** into `fix` — it finds the root cause across import chains, even when the error manifests deep in stdlib.
- **`--keep`** avoids overwriting files that already compile OK — safe for brownfield work.
- **`--force`** regenerates everything — use for greenfield or when you want a clean slate.
- **`--retry`** with implement re-generates only files missing from disk after a previous run.
- **`--review`** with implement audits generated code for bugs, duplicates, and DRY violations.
