# agent.py Refactor Task Plan

Goal: reduce `agent.py` from ~5008 lines to manageable modules (<500 lines each).

---

## What is already done (committed in HEAD)

| Step | What | File | Approx saved |
|------|------|------|-------------|
| t02  | Extract import bootstrap + OPENBLAS_NUM_THREADS + UTF-8 stdout reconfigure | `agent_core/bootstrap.py` | ~100 |
| t03  | Extract history compaction (trim, repair, strip wrappers) | `agent_core/history_compaction.py` | ~461 |

---

## Remaining steps to execute

### t04 — Extract Agent `__init__` groupings
Group `Agent.__init__` attributes into nested dataclasses / sub-objects
(LLMConfig fields → `_llm`, FileSystem fields → `_fs`, etc.).
Target: `agent.py` loses ~80 lines.

### t05 — Extract system-prompt block methods
Move these to `agent_core/system_prompt.py`:
`_decision_constraints_block`, `_skill_index_block`, `_skill_hint_block`,
`_habits_block`, `_file_history_block`, `_semantic_memory_block`,
`_ensure_original_goal`, `_original_goal_block`.
Target: `agent.py` loses ~200 lines.

### t06 — Extract experience / memory methods
Move to `agent_core/experience.py`:
`_record_llm_experience`, `_load_memory`, `_save_memory`,
`_suspend_history_for_new_task`, `_prompts_related`, `_recent_user_texts`.
Target: `agent.py` loses ~180 lines.

### t07 — Extract NLP tool-handler methods
Move all `async def _nlp_*` methods to `agent_core/agent_nlp.py`.
Target: `agent.py` loses ~1500 lines.

### t08 — Extract file-operation methods
Move `tool_read_file`, `tool_write_file`, `tool_apply_patch`, `tool_search`,
`read_file`, `write_file`, `apply_patch`, `search_file` to `agent_core/agent_files.py`.
Target: `agent.py` loses ~250 lines.

### t09 — Extract the tool-loop / display class
Move `chat_nlp`, `_run_chained_tool_loop`, `_finish_turn`, `_report_failover`,
`_reset_failover_notice` to `agent_core/agent_loop.py` as `AgentLoop(agent)`.
Target: `agent.py` loses ~450 lines.

### t10 — Extract top-level utility functions
Group free functions into thematic modules:
- `agent_core/signal_handlers.py` — `_signal_break`, `_signal_break_with_cleanup`, `_install_signal_handlers`
- `agent_core/shell_utils.py` — `_detect_shell`, `_shell_name_token`, `_unix_command_hint`, `_windows_command_hint`, `_shell_command_hint`, `_blocked_shell_command`
- `agent_core/subprocess_utils.py` — `_run_subprocess_captured`, `_shape_run_stderr`, `_kill_process_tree`
- `agent_core/json_utils.py` — `_escape_json_string_controls`, `_repair_tool_arguments`, `_read_json_quarantining`, `_looks_incomplete`
- `agent_core/logging_utils.py` — `_begin_trace_turn`, `_append_turn_log`, `_strip_dynamic_system_blocks`
- `agent_core/cli_utils.py` — `_current_git_branch`, `_git_branch`, `_interruptible_input`, `_warn_uncommitted`
Target: `agent.py` loses ~400 lines.

### t11 — Extract CLI entry point
Move `run_interactive`, `_start_kanban_inbound`, `async def main` to
`agent_core/run_interactive.py`. `agent.py` keeps `if __name__ == "__main__"`
just calling `asyncio.run(main())`.
Target: `agent.py` loses ~150 lines.

### t12 — Extract command registry
Move `_register_commands`, `_build_registry` to `agent_core/commands/__init__.py`.
Target: `agent.py` loses ~80 lines.

### t13 — Extract LLM client wrapper
Move `LLMClient` and `_build_llm_client` to `agent_core/llm/client.py`.
Target: `agent.py` loses ~200 lines.

### t14 — Extract pytest policy
Move pytest-gate logic to `agent_core/pytest_cmd_policy.py`.
Target: `agent.py` loses ~200 lines.

### t15 — Final cleanup pass
After all extractions:
1. Remove stub wrappers that only delegate to extracted modules.
2. Check for circular imports (agent_core → agent → agent_core).
3. Run `python -m pytest tests/ -q`.
4. Verify `import agent` still works.
5. Target: `agent.py` < 600 lines.

---

## Summary of target sizes

| File / Module | Target lines |
|--------------|-------------|
| `agent.py` (entry point + Agent class + stubs) | < 600 |
| `agent_core/bootstrap.py` | ~100 (done) |
| `agent_core/history_compaction.py` | ~461 (done) |
| `agent_core/system_prompt.py` | ~200 (new) |
| `agent_core/experience.py` | ~180 (new) |
| `agent_core/agent_nlp.py` | ~1500 (new) |
| `agent_core/agent_files.py` | ~250 (new) |
| `agent_core/agent_loop.py` | ~450 (new) |
| `agent_core/signal_handlers.py` | ~50 (new) |
| `agent_core/shell_utils.py` | ~80 (new) |
| `agent_core/subprocess_utils.py` | ~60 (new) |
| `agent_core/json_utils.py` | ~50 (new) |
| `agent_core/logging_utils.py` | ~50 (new) |
| `agent_core/cli_utils.py` | ~40 (new) |
| `agent_core/run_interactive.py` | ~150 (new) |
| `agent_core/commands/__init__.py` | ~80 (new) |
| `agent_core/llm/client.py` | ~200 (new) |
| `agent_core/pytest_cmd_policy.py` | ~200 (new) |

---

## Verification gates (after each step)
1. `python -c "import agent"` — must not raise
2. `python -m pytest tests/ -q` — must pass
3. `python -m mypy agent.py` — must be clean (or baseline-acceptable)
4. `agent.py` line count — shrinking monotonically
