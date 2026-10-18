# agent.py Refactoring Task Plan

**File:** `C:\Dev\Agent1\agent.py` (~5 558 lines)  
**Goal:** Split it into focused, reviewable modules under `agent_core/`.  
**Strategy:** One step per commit. Each step: isolate a chunk, create a new module, re-export from `agent.py`, update `tests/test_agent_module_surface.py` frozen lists. Never break the import surface.

---

## Phase 1 — Bootstrap and Utilities (already partially done)

> **Note:** The `from agent_core.bootstrap import *` block and several helper imports are already in place. Phase 1 cleans up what remains.

| Step | What to move | Target module | What's already there |
|------|-------------|---------------|---------------------|
| **t01** | `_escape_json_string_controls` | `agent_core/json_helpers.py` | `_read_json_quarantining` |
| **t02** | `OPENBLAS_NUM_THREADS = ...`, `sys.stdout.reconfigure(...)`, `sys.stderr.reconfigure(...)` | `agent_core/bootstrap.py` (consolidate) | Star-import already covers bootstrap |
| **t03** | Constants: `_LLM_ERROR_BACKOFF_S`, `_LLM_ERROR_MAX_RETRIES`, `_MAX_RUN_TIMEOUT_S`, `_MAX_CHAINED_RUNS`, `_MAX_ACTIVE_SUBAGENTS`, `_DELEGATE_TIMEOUT_S`, `_MAX_READ_FILES`, `_MAX_READ_LINES`, `_MAX_CONSECUTIVE_READS`, `_ACTION_VERB_RE`, `_INCOMPLETE_MARKERS`, `_COMPLETE_MARKERS`, `_QUESTION_LEAD_RE`, `_CONTEXT_AST_THRESHOLD_KB` | `agent_core/constants.py` | Already has a `constants.py` with some constants |
| **t04** | `_looks_incomplete` | `agent_core/text_policy.py` | — |

---

## Phase 2 — Chat History and Goal Management

| Step | What to move | Target module | What's already there |
|------|-------------|---------------|---------------------|
| **t05** | `ORIGINAL_GOAL_MARKER`, `_strip_dynamic_system_blocks`, `_build_chat_prompt` | `agent_core/chat_prompt.py` | — |
| **t06** | `CHAT_HISTORY_JSON_PATH`, `CHAT_HISTORY_TMP_PATH`, `_save_chat_history`, `_load_chat_history` | `agent_core/file_history.py` | Already exists (`FILE_HISTORY_MARKER`) |
| **t07** | `_HISTORY_CHAR_BUDGET`, `_MAX_CHAT_MESSAGES`, `_trim_chat_history`, `_strip_image_blocks`, `_recover_original_goal`, `_HISTORY_TRIM_GOAL_CAP`, `_HISTORY_TRIM_NOTE`, `_content_tokens`, `_project_chat_history`, `_compaction_note_goal`, `_first_user_message_text`, `_is_compaction_note`, `_drop_orphan_tool_messages` | `agent_core/history_compaction.py` | Already exists |
| **t08** | `_original_goal`, `_ensure_original_goal`, `_clear_history` (Agent methods) | `agent_core/original_goal.py` | — |

---

## Phase 3 — Agent Core: `Agent` Class

| Step | What to move | Target module | What's already there |
|------|-------------|---------------|---------------------|
| **t09** | `Agent.__init__`, `_register_commands`, `_build_registry` | `agent_core/agent_init.py` | — |
| **t10** | `Agent.chat_nlp`, `_run_chained_tool_loop`, `_execute_tool_call`, `_repair_tool_arguments` | `agent_core/agent_chat.py` | — |
| **t11** | `Agent.run_interactive`, `_interruptible_input` | `agent_core/agent_interactive.py` | — |
| **t12** | `Agent.main`, signal handlers (`_signal_break`, `_install_signal_handlers`) | `agent_core/agent_main.py` | — |

---

## Phase 4 — NLP Tool Handlers

| Step | What to move | Target module | What's already there |
|------|-------------|---------------|---------------------|
| **t13** | `_nlp_tool_handlers` dict + all `_nlp_*` bound methods | `agent_core/tool_handlers.py` | — |
| **t14** | Decision ledger: `_append_turn_log`, `_begin_trace_turn`, `_turn_quality_score`, `_finish_turn` | `agent_core/agent_decisions.py` | — |

---

## Phase 5 — Shell and Git Helpers

| Step | What to move | Target module | What's already there |
|------|-------------|---------------|---------------------|
| **t15** | `_detect_shell`, `_shell_name_token`, `_unix_command_hint`, `_windows_command_hint`, `_blocked_shell_command`, `_shell_command_hint` | `agent_core/shell_detection.py` | — |
| **t16** | `_run_subprocess_captured`, `_kill_process_tree`, `_shape_run_stderr` | `agent_core/subprocess_utils.py` | Already exists in `agent_core/subprocess_utils.py` |
| **t17** | `_git_branch`, `_current_git_branch`, `_warn_uncommitted` | `agent_core/git_status.py` | — |

---

## Phase 6 — Entrypoint and Dashboard

| Step | What to move | Target module | What's already there |
|------|-------------|---------------|---------------------|
| **t18** | Dashboard: `_build_dashboard`, `_dashboard_port`, `_start_kanban_inbound` | `agent_dashboard.py` | Already exists |
| **t19** | `main`, `run_interactive` (top-level), `CHAT_INPUT_PROMPT` | `agent_core/entrypoint.py` | — |

---

## Phase 7 — Cleanup

| Step | Action |
|------|--------|
| **t20** | Replace `from agent_core.bootstrap import *` with explicit imports for every name still used by `agent.py`. Remove `# noqa: F401,F403`. |
| **t21** | Run full test suite. Update `tests/test_agent_module_surface.py` frozen lists (`EXPECTED_DEFINITIONS`, `EXPECTED_IMPORTED`) to reflect the final clean surface. All `from agent import _thing` test imports should point to the canonical module (e.g., `from agent_core.history_compaction import _trim_chat_history`). |
| **t22** | Delete `agent.py` and rename the last refactor step to `agent.py`, keeping only re-exports and the `Agent` class stub that wires everything together. |

---

## Rules for every step

1. **Create the new module** with the extracted code.
2. **Add it to `agent_core/__init__.py`** `__all__` if it should be public.
3. **Add a re-export** in `agent.py` so existing `from agent import _x` calls keep working.
4. **Update frozen lists** in `tests/test_agent_module_surface.py` in the same commit.
5. **Run the affected test files** (never the full suite mid-refactor).
6. **Commit message format:** `refactor(tXX): move <names> to agent_core/<module>.py`

---

## Dependency order (no forward references)

```
t01 → t03 → t04 → t02
                ↓
t05 → t06 → t07 → t08
                ↓
t09 → t10 → t11 → t12 → t13 → t14
                                   ↓
t15 → t16 → t17 → t18 → t19 → t20 → t21 → t22
```

> **t02** (bootstrap consolidation) depends on **t01** (json helpers extracted first) so the import order is safe.

---

*This plan was written based on static analysis of `agent.py`. The actual step boundaries may shift as each chunk is extracted and the import graph is verified.*
