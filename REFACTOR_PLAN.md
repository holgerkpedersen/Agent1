# Refactoring Plan: `agent.py` → `agent_core/` modules

## Why this plan exists

`agent.py` is ~4177 lines.  Keeping it monolithic makes it hard to test, hard to navigate, and
risky to change.  This plan breaks the work into **14 small, reversible steps** — each one
produces a working, test-passing state before the next step begins.

---

## Ground rules for every step

1. **Write the failing test first** — before moving any code, add or update the test that will
   verify the new module surface.  Run it; watch it fail with a clear message.
2. **Create the new module** with the extracted code.  Write stubs for any names that can't be
   extracted yet (they refer to code that lives in `agent.py` and will move later).
3. **Update `agent.py`** to import from the new module; keep old definitions as aliases during
   the transition so `agent.py` stays importable at every commit.
4. **Run `tests/test_agent_module_surface.py`** — it is the gate for every step.  No step
   merges to `master` while this test is red.
5. **Commit after every step.**  Message format: `refactor(t0N): extract <topic> to agent_core/<module>.py`

---

## Step summary

| Step | What moves | Target module | Approx. lines moved |
|------|-----------|---------------|---------------------|
| t01  | Signal handlers, git branch, interruptible input | `agent_core/signal_handlers.py` | ~130 |
| t02  | Shell detection, command hints, pytest guards, subprocess helpers | `agent_core/shell_utils.py` | ~180 |
| t03  | JSON read/write with quarantine | `agent_core/json_utils.py` | ~45 |
| t04  | Run utilities: kanban bridge, process tree, stream/trace helpers | `agent_core/run_utils.py` | ~100 |
| t05  | Type constants (`_GOAL_RELATED_THRESHOLD`) | `agent_core/types.py` | ~1 |
| t06  | History-compaction pure functions | `agent_core/history_compaction.py` | ~250 *(already exists)* |
| t07  | Bootstrap import block, signal install, environment setup | `agent_core/bootstrap.py` | ~210 *(already exists)* |
| t08  | LLMClient class | `agent_core/llm/client.py` | ~175 |
| t09  | NLP delegate / batch / subagent / MCP tool handlers | `agent_core/nlp_handlers.py` | ~290 |
| t10  | File-tool methods (read, write, edit, patch, search, list, delete, analyze) | `agent_core/file_tool.py` | ~200 |
| t11  | System-prompt builders (semantic memory, habits, file history, goals, decisions, skills) | `agent_core/prompt_builders.py` | ~270 |
| t12  | Tool-loop runner and turn-finish (quality score, trace, metrics) | `agent_core/turn_loop.py` | ~350 |
| t13  | chat_nlp + _finish_turn (main interaction loop) | `agent_core/chat_loop.py` | ~220 |
| t14  | Remaining Agent methods + `main` / `run_interactive` | `agent_core/agent.py` | ~400 |

---

## Step details

---

### Step t01 — Signal handlers → `agent_core/signal_handlers.py`

**Already created** (`agent_core/signal_handlers.py` written in this session).  Still missing:
- `_DANGEROUS_SHELL_PATTERNS` — move to `agent_core/security.py`
- `_ENV_FILE_PATH` — move to `agent_core/config.py`
- `_FULL_SUITE_MARGIN` — move to `agent_core/constants.py`
- `_PYTEST_BARE_TOKENS`, `_PYTEST_FULL_SUITE_TIMEOUT_KEY`, `_PYTEST_LAST_FULL_RUN_KEY`,
  `_PYTEST_SUBSET_FLAGS`, `_PYTEST_VALUE_FLAGS` — move to `agent_core/constants.py`
- `_install_signal_handlers` → `_install_signal_handlers` in this module
- `_signal_break` → already in this module
- `_SHELL_NAME`, `_shutdown_flag` → already in this module (as module-level state)

**TODO**: Sync these constants into their respective modules, update `agent.py` imports.

---

### Step t02 — Shell utilities → `agent_core/shell_utils.py`

**Already created** (`agent_core/shell_utils.py` written in this session).  Contains:
- `detect_shell`, `shell_name_token`, `current_git_branch`
- `unix_command_hint`, `windows_command_hint`, `shell_command_hint`
- `truncate_output`, `run_subprocess_captured`, `shape_run_stderr`
- `blocked_shell_command`, `read_env_value`
- `pytest_full_suite_timeout`, `is_full_pytest_command`, `is_pytest_command`,
  `is_subset_pytest_command`
- `SYSTEM_PROMPT` (stub)

**TODO**: Verify exports, fix `agent.py` to import from here.

---

### Step t03 — JSON utilities → `agent_core/json_utils.py`

**Already created** (`agent_core/json_utils.py` written in this session).  Contains:
- `_read_json_quarantining` (verified imported by `agent.py` at lines 3485, 3559)

**TODO**: Verify re-export in `agent.py`, confirm `_quarantine` is internal-only.

---

### Step t04 — Run utilities → `agent_core/run_utils.py`

**Already created** (`agent_core/run_utils.py` written in this session).  Contains:
- `_start_kanban_inbound` (verified called at line 4166)
- `kill_process_tree` (aliased as `_kill_process_tree` at line 136)
- `resolve_stream_fn`, `resolve_display_mode`
- `final_answer_fallback`
- `turn_quality_score`
- `begin_trace_turn`, `append_turn_log`
- `strip_dynamic_system_blocks`
- `unverified_action_claim`

**TODO**: Verify all call sites in `agent.py` resolve correctly.

---

### Step t05 — Type constants → `agent_core/types.py`

**Already created** (`agent_core/types.py` written in this session).  Contains:
- `_GOAL_RELATED_THRESHOLD` (verified imported at line 154, used at line 3406)

---

### Step t06 — History compaction → `agent_core/history_compaction.py`

**Already exists.**  Verified exports: `_content_tokens`, `_compaction_note_goal`,
`_drop_orphan_tool_messages`, `_first_user_message_text`, `_is_compaction_note`,
`_message_size`, `_project_chat_history`, `_recover_original_goal`,
`_repair_unanswered_tool_calls`, `_strip_image_blocks`, `_strip_turn_wrappers`,
`_strip_tool_args`, `_trim_chat_history`.

No further action needed.

---

### Step t07 — Bootstrap → `agent_core/bootstrap.py`

**Already exists.**  `__all__` needs ~30 names added (see `tests/test_agent_module_surface.py`
`EXPECTED_IMPORTED`).  Run `pytest tests/test_agent_module_surface.py` and fix each missing name
individually.

---

### Step t08 — LLMClient → `agent_core/llm/client.py`

**Target**: Class `LLMClient` (lines 156–331, ~175 lines).

```
agent_core/llm/client.py
├── class LLMClient
│   ├── __init__ (model resolution, api_key)
│   ├── _reconcile_llama_model
│   ├── model_name (property)
│   ├── chat (async)
│   ├── chat_stream (async)
│   ├── chat_with_continuation (async)
│   └── analyze_code (async)
```

`agent.py` keeps `from agent_core.llm.client import LLMClient` as an alias.

---

### Step t09 — NLP handlers → `agent_core/nlp_handlers.py`

**Target**: All `_nlp_*` methods on `Agent` (lines 684–1190, ~290 lines).

```
agent_core/nlp_handlers.py
├── class NlpToolHandler (TypedDict or dataclass)
├── _nlp_tool_handlers (method -> returns dict[str, NlpToolHandler])
├── _nlp_delegate
├── _nlp_delegate_batch
├── _nlp_create_subagent
├── _nlp_run_subagent_task
├── _nlp_mcp_tools
├── _nlp_mcp_call
├── _nlp_definitions
├── _nlp_references
├── _nlp_jev_decide
├── _nlp_speculate
├── _nlp_search
├── _nlp_read
├── _nlp_read_many
├── _nlp_read_skill
├── _nlp_list_files
├── _nlp_fix
├── _nlp_write
├── _nlp_run
├── _nlp_git
├── _nlp_merge
├── _nlp_diff
├── _nlp_edit
├── _nlp_tests
├── _nlp_analyze
├── _nlp_web_search
└── _nlp_get_current_datetime
```

`Agent` gets a `_nlp_handlers: NlpToolHandlers` instance attribute; `_nlp_*` methods delegate
to it.  This keeps the call signatures unchanged for callers.

---

### Step t10 — File tools → `agent_core/file_tool.py`

**Target**: All `_tool_*` and `_nlp_*` file I/O methods on `Agent` (lines 2156–2343, ~200 lines).

```
agent_core/file_tool.py
├── execute_tool (dispatches to named method)
├── _normalize_path
├── _safe_path
├── read_file
├── write_file
├── apply_patch
├── edit_file
├── search_file
├── list_files
├── delete_file
├── analyze_file
├── llm_analyze_file
```

`Agent` gets a `self._file_tool: FileTool` attribute; tool methods become thin wrappers.

---

### Step t11 — Prompt builders → `agent_core/prompt_builders.py`

**Target**: System-prompt builder methods on `Agent` (lines 3170–3473, ~270 lines).

```
agent_core/prompt_builders.py
├── _refresh_system_message
├── _append_user_turn
├── _decision_constraints_block
├── _skill_index_block
├── _skill_hint_block
├── _habits_block
├── _file_history_block
├── _semantic_memory_block
├── _ensure_original_goal
├── _recent_user_texts
├── _prompts_related
├── _suspend_history_for_new_task
├── _original_goal_block
├── _load_chat_history
├── _save_chat_history
├── _load_memory
└── _save_memory
```

These are pure-ish (mostly string-building + light I/O) and easiest to extract after
the tool layer is settled.

---

### Step t12 — Turn loop → `agent_core/turn_loop.py`

**Target**: `_run_chained_tool_loop`, `_finish_turn`, `_report_failover`, `_reset_failover_notice`
(lines 2439–2871, ~350 lines).

```
agent_core/turn_loop.py
├── class TurnLoopRunner
│   ├── __init__(agent, ...)
│   ├── run(user_input, messages, display_mode) -> (final_text, messages, error, runner)
│   ├── _reset_failover_notice
│   ├── _report_failover
│   └── _finish_turn
```

`Agent.chat_nlp` delegates to `TurnLoopRunner`.  All metrics/trace/quality-score logic lives here.

---

### Step t13 — Chat loop → `agent_core/chat_loop.py`

**Target**: `chat_nlp` (lines 2873–3017, ~144 lines) and `_mutating_files_this_turn`,
`_print_self_review_note`, `_persist_plan_answer` (lines 3019–3096, ~77 lines).

```
agent_core/chat_loop.py
├── async def chat_nlp(agent, user_input, images) -> None
├── _mutating_files_this_turn(agent) -> list[str]
├── _print_self_review_note(agent, mutated_files) -> None
└── _persist_plan_answer(agent, plan_text) -> None
```

---

### Step t14 — Agent class body + entry points → `agent_core/agent.py`

**Target**: The `Agent.__init__` body (lines 335–459, ~124 lines), remaining methods, and the
top-level functions at the bottom of `agent.py`.

After all previous steps:
- `Agent` holds only: `__init__` body, `_register_tool_handlers`, `build_transport`,
  `spawn_subagent`, `run_parallel_tasks`, `record_demo_activity`, `get_metrics_collector`,
  `_resolve_nlp_path`, `_execute_tool_call`, `_get_plan_state`, `check_stale_files`,
  `invalidate_stale`, `memory_stats`, `clear_history`, `_record_llm_experience`
- Top-level: `_looks_incomplete`, `_unverified_action_claim`, `_register_commands`,
  `_build_registry`, `_warn_uncommitted`, `_build_chat_prompt`, `run_interactive`, `main`

`agent.py` becomes:
```python
"""Public entry point — thin wrapper over agent_core.agent."""
from agent_core.agent import Agent, main, run_interactive
from agent_core.agent import (
    _looks_incomplete, _unverified_action_claim,
    _register_commands, _build_registry, _warn_uncommitted, _build_chat_prompt,
)
```

---

## What happens to `agent.py`

After step t14, `agent.py` is **≤ 20 lines**: docstring, import from `agent_core.agent`,
re-exports for backward compatibility with any code that does `from agent import X`.

The old `from agent import Agent` import continues to work throughout all 14 steps via aliases.

---

## Test surface

`tests/test_agent_module_surface.py` is the gate for every step.  At each step:
1. Update `EXPECTED_DEFINITIONS` to reflect which names are now in `agent.py` vs. extracted.
2. Update `EXPECTED_IMPORTED` to reflect which names are now imported from new modules.
3. `pytest tests/test_agent_module_surface.py` must pass before committing.

After step t14, `EXPECTED_DEFINITIONS` contains only the 9 names `agent.py` keeps
(`Agent`, `LLMClient`, `main`, `run_interactive`, `_looks_incomplete`, `_unverified_action_claim`,
`_register_commands`, `_build_registry`, `_warn_uncommitted`, `_build_chat_prompt`).

---

## How to run the plan

```powershell
# Verify baseline (step 0)
pytest tests/test_agent_module_surface.py -q

# Run one step at a time:
# 1. Make the code change
# 2. Update the test fixture
# 3. pytest tests/test_agent_module_surface.py -q
# 4. git add -A; git commit -m "refactor(t01): ..."
```

Each step is independent and revertable via `git revert <commit>`.
