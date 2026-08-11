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

All workflow-generated files (analysis, plan, entities, taskplan) are **automatically stripped of LLM reasoning tokens** — chain-of-thought text, self-correction notes, "[Output Generation]" markers, and checkmark emoji markers are removed before writing to disk. This keeps the output files clean and parseable by the implementation pipeline.

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

Safety: implement now has 6 layers of protection:

1. **Workflow stdlib shadowing warnings** — taskplan LLM is warned against stdlib-module directory names (`logging/`, `json/`, `types/`)
2. **Analysis verification** — generated plans are checked for stdlib-shadowed paths; violations are flagged as `[UNVERIFIED]` before implementation
3. **Auto-redirect** — during implementation, shadowed paths are redirected to safe alternatives (`logging` → `logging_utils`)
4. **Collision warnings** — taskplan LLM sees existing names per directory before generating
5. **Post-write rejection** — files with class-name conflicts are auto-deleted immediately
6. **Auto-review + `--review`** — offers to delete dangerous files (y/N) + LLM deep analysis

File discovery parses the taskplan directly — no LLM invents wrong filenames. Path rules adapt to any workspace by detecting existing `__init__.py` directories — no hardcoded prefixes.

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

**Stdlib shadowing detection** runs across the entire pipeline:
- `workflow` prompts warn the LLM to avoid directory names matching stdlib modules
- `analysis_verifier` flags shadowed paths in `## Verification Report` before implementation
- `implement` auto-redirects shadowed paths (e.g. `logging/` → `logging_utils/`)
- `fix` catches shadowed local files during traceback cascade analysis

### From a description (on-demand)

```
> fix agent_core/config.py --desc "The ConfigManager.load() method doesn't validate YAML files"
```

On-demand mode (default): sends only the top 5 most relevant files by keyword match, plus signatures for the rest. The LLM can request additional files with `[READ: path.py]` and iterates up to 6 rounds before a synthesis-ultimatum retry.

**Patch-based fixing**: The LLM prefers `[PATCH:]` format — minimal diffs showing only changed lines. Shows the diff with +/- markers, asks `Apply this patch? (y/N)` before writing. Falls back to `[FILE:]` for new files or full rewrites.

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
  Static analysis found 12 issue(s) in 1 file(s):

  fix_cmd.py:
      line   21: [silent_except] Replace 'pass' with a print/log warning — do NOT re-raise
      line  134: [regex_in_loop] Move re.compile() to module level

  Processing 1 file(s) in 1 batch(es)...

  Batch 1: fix_cmd.py
    Estimated tokens: 3240
    Got fixes for 1 file(s)

  Applying fixes...

  [PATCH: fix_cmd.py]
   19 | import os                                     19 | import os
   20 | import sys                                    20 | import sys
   21 |     pass                                 <    21 |     logger.exception("Unhandled exception")
   ...
   (5 lines removed, 3 lines added)

  Apply fix_cmd.py? (12400 → 12380 bytes) (y/N): y
  Applied: fix_cmd.py (12380 bytes)

  Done. Applied 1/1 fix(es).

> optimize agent_core/ --apply
  # Analyzes all .py files, batches them, shows diff before applying

> optimize agent.py --apply --yes
  # Analyzes and applies all suggestions without asking
```

The optimizer uses a two-layer patch system:
1. **Strict** — applies hunks by `@@` line numbers with content verification. Rejects mismatches.
2. **Anchored fallback** — when `@@` numbers are wrong (LLMs commonly get these off), searches a ±60-line window for matching content. Also strips `N |` numbered-context artifacts that models sometimes copy into hunk bodies.

Pure-removal hunks (unused imports, dead assignments) are now accepted — no replacement line required.

`regex_in_loop` findings are fixed mechanically (no LLM needed): the static pattern is
hoisted above the loop into a named compile constant derived from its content, e.g.

```
> optimize agent_core/patterns.py --apply
    Fixed line 77 [regex_in_loop] (mechanical, 25 -> 24 remaining)
  +    _FOR_WHILE_RE = re.compile(r"^\s*(for|while)\s")
  -    if re.match(r"^\s*(for|while)\s", line):
  +    if _FOR_WHILE_RE.match(line):
```

Naming rules:
- The constant name comes from the pattern's words (`(for|while)` → `_FOR_WHILE_RE`, `type(...) == ...` → `_TYPE_EQ_RE`), so the code stays readable.
- An existing identical `re.compile()` in the same scope is **reused** — no duplicate compile line is added.
- A visible name collision gets a numeric suffix (`_FOR_WHILE_RE_2`); patterns with no usable words fall back to `_RE_1`-style names.
- Detectors and mechanical fixers skip docstring lines, so `--apply` never rewrites docstring prose that merely looks like code.
- LLM-reviewed findings that would break real logic (e.g. `while changed:` fixed-point guards) are rejected and left untouched.

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

## Decision Tracking

Every `implement` and `fix` run auto-extracts design decisions and suggests recording them. Decisions are stored in `.decisions.json` and are enforced as constraints on future changes.

### Recording decisions

```
> decide "Use composition over inheritance for security" --why "Injected via __init__" --what "Chose composition to avoid fragile base class issues" --tags architecture,security --files agent_core/security/allowlist.py
  
  Recorded decision #001: Use composition over inheritance for security
```

### Searching decisions

```
> decide list --tag security

  1 decision(s):
  ------------------------------------------------------------
    #001  2026-08-11  Use composition over inheritance for security
           files: agent_core/security/allowlist.py
           tags:  architecture, security

> decide list --search "logging"

> decide list --file agent_core/config.py

> decide show 001
    id: 001
    date: 2026-08-11
    title: Use composition over inheritance for security
    ...
```

### Auto-extraction from workflow/implement/fix

After `workflow` runs, the analysis is scanned for decision candidates:

```
[decide] Extracted 3 decision candidates:
  1. "Resolve module-vs-package conflict in agent_core/config"
  2. "Add deterministic termination to refinement loop"
  3. "Sandbox shell execution via allowlist"

  Record? (1,2/all/N, press Enter to skip): 1,3
  Recorded #002
  Recorded #003
```

After `implement` or `fix` runs, file changes are analyzed for design choices:

```
[decide] Extracted 2 decision candidates from this run:
  1. "Use path_utils.normalize_path for all file operations"
  2. "Inject dependencies via __init__ instead of module globals"

  Record? (1,2/all/N, press Enter to skip): 
```

Press Enter to skip — decisions are optional. They accumulate in `.decisions.json` as you go.

### Contradiction detection

Before recording a new decision, check if it conflicts:

```
> decide check --text "Move schema.py back under agent_core/config/"

  Tag/file overlap detected with: #002

  Checking for contradictions (LLM)...
  CONTRADICTION with Decision #002:
  #002 deleted agent_core/config/ because schema.py was dead code and
  config.py shadows the directory package. Moving schema.py back would
  recreate the module-vs-package conflict. Consider agent_core/schemas/
  as an alternative non-shadowing location.
```

### Enforcement: decisions as constraints

Past decisions are injected as **hard constraints** into every `implement` and `fix` LLM prompt:

```
CRITICAL DESIGN CONSTRAINTS — past decisions that MUST be respected:
  Decision #001 (Use composition over inheritance):
    Chose: Inject via __init__ instead of extending base classes
    Why: Avoids fragile base class issues and circular imports
```

The LLM cannot accidentally undo deliberate choices. To override a decision, use:

```
> decide resolve 001 --text "We now need inheritance because..."
```

### The full lifecycle

```
implement/fix runs ──► decisions auto-extracted ──► .decisions.json
        ▲                                              │
        │                                              ▼
        └──── future runs read as hard constraints ────┘
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
