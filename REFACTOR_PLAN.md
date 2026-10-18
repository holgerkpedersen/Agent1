# Refactoring Plan: `agent.py` → `agent_core/` modules

## Status as of 2026-10-18

**`agent.py` is 4009 lines**, down from ~4177 at the start. The plan is ~60% complete.

### ✅ Already extracted (8 modules done)

| Module | What it holds | Est. lines moved |
|--------|--------------|-----------------|
| `agent_core/history_compaction.py` | History-compaction pure functions | ~250 |
| `agent_core/llm/client.py` | `LLMClient` class + read_json wrapper | ~175 |
| `agent_core/bootstrap.py` | Bootstrap import block, signal install, environment setup | ~210 |
| `agent_core/runtime_globals.py` | `live()` accessor for cross-module globals | — |
| `agent_core/json_utils.py` | JSON read/write with quarantine | ~45 |
| `agent_core/shell_utils.py` | Shell detection, command hints, subprocess helpers | ~180 |
| `agent_core/signal_handlers.py` | Signal handlers, interruptible input | ~130 |
| `agent_core/types.py` | Type constants (`_GOAL_RELATED_THRESHOLD`) | ~1 |
| `agent_core/run_utils.py` | Kanban bridge, process tree, stream/trace helpers | ~100 |

### ❌ Still to extract (9 modules remaining)

| Step | What moves | Target module | Est. lines |
|------|-----------|---------------|-----------|
| r01  | NLP delegate / batch / subagent / MCP / plan handlers | `agent_core/handlers/nlp_tools.py` | ~290 |
| r02  | Tool-loop runner, turn-finish, quality score, trace | `agent_core/turn_loop.py` | ~350 |
| r03  | File-tool method wrappers (read, write, edit, patch, search, list, delete, analyze) | `agent_core/agent_files.py` | ~200 |
| r04  | File-effect helpers: _resolve_nlp_path, _save_verify_note, _verify_file, _note_effect, _take_trace_effects | `agent_core/agent_effects.py` | ~70 |
| r05  | Plan tool handlers (_nlp_plan_*) | `agent_core/handlers/plan_tools.py` | ~280 |
| r06  | Shell + subprocess helpers (deduped from shell_utils.py), pytest command policy, constants | `agent_core/shell_intel.py`, `agent_core/pytest_cmd_policy.py` | ~150 |
| r07  | System-prompt builders (semantic memory, habits, file history, goals, decisions, skills) | `agent_core/prompts/context_blocks.py` | ~270 |
| r08  | Chat/memory persistence (_load_chat_history, _save_chat_history, _load_memory, _save_memory) | `agent_core/memory/persistence.py` | ~120 |
| r09  | REPL layer (run_interactive, main, command registry, warn_uncommitted) | `agent_core/cli/interactive.py` | ~400 |

### 🔧 Final cleanup (1 step)

| Step | What | Notes |
|------|------|-------|
| r10  | Prune dead imports, run full test suite, update Architecture.md | Gate: `test_agent_module_surface.py` must pass |

---

## Ground rules for every remaining step

1. **Write the failing test first** — before moving any code, add or update `tests/test_agent_module_surface.py` with the new names you expect to find in `agent_core/<module>.py`. Run it; watch it fail.
2. **Create the new module** with the extracted code. Write stubs for any names that can't be extracted yet (they refer to code that still lives in `agent.py` and will move later).
3. **Update `agent.py`** to import from the new module; keep old definitions as aliases during the transition so `agent.py` stays importable at every commit.
4. **Run `tests/test_agent_module_surface.py`** — it is the gate. No step merges to `master` while this test is red.
5. **Commit after every step.** Message format: `refactor(r0N): extract <topic> to agent_core/<module>.py`

---

## Step details

---

### Step r01 — NLP tool handlers → `agent_core/handlers/nlp_tools.py`

**Target**: All `_nlp_*` methods on `Agent` (lines 509–1536, ~290 lines).

```
agent_core/handlers/nlp_tools.py
├── _nlp_tool_handlers (method → returns dict[str, NlpToolHandler])
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
to it. This keeps the call signatures unchanged for callers.

Note: `_nlp_hue_control` stays in `Agent` for now (depends on `self` state tightly).
`_get_plan_state` is plan-tool infrastructure — handled in r05.

**Test**: `EXPECTED_DEFINITIONS` in `test_agent_module_surface.py` gains `_nlp_delegate`, etc.

---

### Step r02 — Tool loop + turn finish → `agent_core/turn_loop.py`

**Target**: `_run_chained_tool_loop`, `_finish_turn`, `_reset_failover_notice`, `_report_failover`,
`_looks_incomplete`, `_unverified_action_claim` (lines 2264–2696, 3652–3702, ~350 lines).

```
agent_core/turn_loop.py
├── _looks_incomplete
├── _unverified_action_claim
├── _reset_failover_notice
├── _report_failover
├── _finish_turn
└── _run_chained_tool_loop
```

`Agent.chat_nlp` stays in `agent.py` but calls into `agent_core/turn_loop` for the loop body.

**Test**: `EXPECTED_DEFINITIONS` gains `_run_chained_tool_loop`, `_finish_turn`, `_looks_incomplete`, `_unverified_action_claim`.

---

### Step r03 — File-tool method wrappers → `agent_core/agent_files.py`

**Target**: `_tool_*` aliases, `execute_tool`, `_normalize_path`, `_safe_path`, and the
public `read_file`, `write_file`, `apply_patch`, `edit_file`, `search_file` methods
(lines 1981–2168, ~200 lines).

```
agent_core/agent_files.py
├── _tool_read_file
├── _tool_write_file
├── _tool_apply_patch
├── _tool_edit_file
├── _tool_search
├── _tool_list_files
├── _tool_delete_file
├── _tool_analyze_file
├── _tool_llm_analyze
├── execute_tool
├── _normalize_path
├── _safe_path
├── read_file
├── write_file
├── apply_patch
├── edit_file
└── search_file
```

`Agent` keeps one-line delegators. `Agent._build_semantic_index` and `_cleanup_semantic_index`
stay on `Agent` (tightly coupled to `Agent._semantic_index` dict).

**Test**: `EXPECTED_DEFINITIONS` gains `execute_tool`, `read_file`, `write_file`, `apply_patch`, `edit_file`, `search_file`.

---

### Step r04 — File-effect helpers → `agent_core/agent_effects.py`

**Target**: `_resolve_nlp_path`, `_save_verify_note`, `_verify_file`, `_note_effect`,
`_take_trace_effects` (lines 389–456, ~70 lines).

```
agent_core/agent_effects.py
├── _resolve_nlp_path
├── _save_verify_note
├── _verify_file
├── _note_effect
└── _take_trace_effects
```

These are helpers that track file-system side effects. They need access to `Agent` state
(e.g. `self._effective_ws_dir`, `self._trace_file`). Pass `Agent` instance or relevant
fields as parameters.

**Test**: `EXPECTED_DEFINITIONS` gains these five names.

---

### Step r05 — Plan tool handlers → `agent_core/handlers/plan_tools.py`

**Target**: `_nlp_plan_status`, `_nlp_plan_start`, `_nlp_plan_step`, `_nlp_plan_finish`,
`_get_plan_state`, `_persist_plan_answer` (lines 1652–1977, ~280 lines).

```
agent_core/handlers/plan_tools.py
├── _get_plan_state
├── _nlp_plan_status
├── _nlp_plan_start
├── _nlp_plan_step
├── _nlp_plan_finish
└── _persist_plan_answer
```

Note: `plan_status`, `plan_start`, `plan_step`, `plan_finish` are top-level functions in
`agent.py` (imported from `agent_core.run_utils`). This module handles the NLP-layer wrapping.

**Test**: `EXPECTED_DEFINITIONS` gains `_nlp_plan_start`, etc.

---

### Step r06 — Shell intel + pytest policy → `agent_core/shell_intel.py` + `agent_core/pytest_cmd_policy.py`

**Target** (from `agent.py` + deduplicating `agent_core/shell_utils.py`):

```
agent_core/shell_intel.py
├── _detect_shell
├── _git_branch
├── _dangerous_shell_patterns (moved from agent.py body)
├── blocked_shell_command
├── read_env_value
└── _ENV_FILE_PATH (two-level __file__ form, routed through live())
```

```
agent_core/pytest_cmd_policy.py
├── _ENV_FILE_PATH
├── _PYTEST_FULL_SUITE_TIMEOUT_KEY
├── _PYTEST_LAST_FULL_RUN_KEY
├── _PYTEST_BARE_TOKENS
├── _PYTEST_SUBSET_FLAGS
├── _PYTEST_VALUE_FLAGS
├── _FULL_SUITE_MARGIN
├── pytest_full_suite_timeout
├── is_full_pytest_command
├── is_pytest_command
├── is_subset_pytest_command
└── _install_signal_handlers
```

**Test**: `EXPECTED_DEFINITIONS` gains `_dangerous_shell_patterns`, `_ENV_FILE_PATH`, etc.

---

### Step r07 — Prompt context blocks → `agent_core/prompts/context_blocks.py`

**Target**: `_refresh_system_message`, `_append_user_turn`, `_decision_constraints_block`,
`_skill_index_block`, `_skill_hint_block`, `_habits_block`, `_file_history_block`,
`_semantic_memory_block`, `_ensure_original_goal`, `_recent_user_texts`,
`_prompts_related`, `_suspend_history_for_new_task`, `_original_goal_block`
(lines 2176–3298, ~270 lines).

```
agent_core/prompts/context_blocks.py
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
└── _original_goal_block
```

**Test**: `EXPECTED_DEFINITIONS` gains `_refresh_system_message`, `_semantic_memory_block`, etc.

---

### Step r08 — Chat/memory persistence → `agent_core/memory/persistence.py`

**Target**: `_load_chat_history`, `_save_chat_history`, `_load_memory`, `_save_memory`,
`check_stale_files`, `invalidate_stale`, `memory_stats`, `clear_history`
(lines 2923–3439, ~120 lines).

```
agent_core/memory/persistence.py
├── check_stale_files
├── invalidate_stale
├── memory_stats
├── clear_history
├── _load_chat_history
├── _save_chat_history
├── _load_memory
└── _save_memory
```

Routes `AGENT_MEMORY_JSON_PATH` and `TURN_LOG_PATH` through `live()`.

**Test**: `EXPECTED_DEFINITIONS` gains `clear_history`, `_load_memory`, etc.

---

### Step r09 — REPL / interactive layer → `agent_core/cli/interactive.py`

**Target**: `_register_commands`, `_build_registry`, `_warn_uncommitted`, `_build_chat_prompt`,
`run_one`, `llm_chat_fn`, `_short`, `run_interactive`, `main` (lines 3730–4009, ~400 lines).

```
agent_core/cli/interactive.py
├── _register_commands
├── _build_registry
├── _warn_uncommitted
├── _build_chat_prompt
├── run_one
├── llm_chat_fn
├── _short
├── run_interactive
└── main
```

The `Agent` class (now ~100 method lines) plus these top-level functions remain in `agent.py`.
`agent.py` becomes a thin facade that imports and re-exports from `agent_core/`.

**Test**: `EXPECTED_DEFINITIONS` gains `run_interactive`, `main`, `_warn_uncommitted`.

---

### Step r10 — Final cleanup

1. Run `python -m pytest tests/test_agent_module_surface.py` — all 6 tests must pass.
2. Run full lint sweep: `ruff check agent.py agent_core/ --fix`.
3. Prune all dead import aliases from `agent.py` (every `Name = agent_core.X` alias that is
   no longer needed because callers now import from `agent_core/` directly).
4. Update `Architecture.md` and any module-map docs to reflect the new layout.
5. One final full test run: `python -m pytest`.
6. Commit with message: `refactor(r10): finish agent.py extraction — thin facade complete`.

---

## Expected final state

```
agent.py                          (~100 lines — thin facade + Agent class skeleton)
agent_core/
  __init__.py
  bootstrap.py                    (~210 lines)
  history_compaction.py           (~250 lines)
  llm/
    __init__.py
    client.py                     (~175 lines)
  json_utils.py                   (~45 lines)
  shell_utils.py                  (~180 lines)
  shell_intel.py                  (~80 lines)
  signal_handlers.py              (~130 lines)
  pytest_cmd_policy.py           (~100 lines)
  run_utils.py                    (~100 lines)
  runtime_globals.py             (live() accessor)
  types.py                        (~1 line)
  agent_files.py                  (~200 lines)
  agent_effects.py                (~70 lines)
  prompts/
    context_blocks.py             (~270 lines)
  memory/
    persistence.py               (~120 lines)
  handlers/
    __init__.py
    nlp_tools.py                  (~290 lines)
    plan_tools.py                 (~280 lines)
  cli/
    __init__.py
    interactive.py               (~400 lines)
```

---

## How to run the gate test

```bash
python -m pytest tests/test_agent_module_surface.py -v
```

All 6 tests must be green before every commit.
