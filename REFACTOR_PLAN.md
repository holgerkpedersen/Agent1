# Refactoring Plan: `agent.py` → `agent_core/` modules

## Status as of 2026-10-18

**`agent.py` is 1799 lines.** This plan splits it into focused modules.

**Done:** r01 (NLP tool handlers → `agent_core/handlers/nlp_tools.py`;
`Agent._nlp_*` methods are thin delegates; surface test updated).
r02 (file tool wrappers → `agent_core/agent_files.py`; same delegate pattern).
r03 (file effects helpers → `agent_core/agent_effects.py`).
r04 (turn-loop core → `agent_core/turn_loop.py`; `TraceWriter`/`trace_enabled`/
`ToolLoopRunner`/`_resolve_display_mode`/`record_turn_outcome` resolved lazily
from the `agent` module so test monkeypatching keeps working).

---

## Ground rules for every step

1. **Write the failing test first** — add or update `tests/test_agent_module_surface.py` with the names you expect to find in `agent_core/<module>.py`. Run it; watch it fail.
2. **Create the new module** with the extracted code. Write stubs for any names that still reference code in `agent.py`.
3. **Update `agent.py`** to import from the new module; keep old definitions as aliases during the transition so `agent.py` stays importable at every commit.
4. **Run `tests/test_agent_module_surface.py`** — it is the gate. No step merges while this test is red.
5. **Commit after every step.** Message format: `refactor(r0N): extract <topic> to agent_core/<module>.py`

---

## Remaining extraction steps

### r01 — NLP tool handlers → `agent_core/handlers/nlp_tools.py` (DONE)

Extract all `async def _nlp_*` methods on `Agent` that delegate to sub-systems:
`_nlp_delegate`, `_nlp_delegate_batch`, `_nlp_create_subagent`, `_nlp_run_subagent_task`,
`_nlp_mcp_tools`, `_nlp_mcp_call`, `_nlp_definitions`, `_nlp_references`,
`_nlp_jev_decide`, `_nlp_speculate`, `_nlp_search`, `_nlp_read`, `_nlp_read_many`,
`_read_page`, `_nlp_read_skill`, `_nlp_list_files`, `_nlp_fix`, `_nlp_write`,
`_nlp_run`, `_nlp_git`, `_nlp_merge`, `_nlp_diff`, `_nlp_edit`, `_nlp_tests`,
`_nlp_analyze`, `_nlp_web_search`, `_nlp_get_current_datetime`, `_nlp_hue_control`,
`_nlp_tool_handlers`, `_run_command_quietly`, `_effective_ws_dir`.
Also extract `_nlp_plan_status`, `_nlp_plan_start`, `_nlp_plan_step`, `_nlp_plan_finish`,
`_get_plan_state` (these are plan tool handlers but live between the NLP handlers).

Target: `agent_core/handlers/nlp_tools.py` (~320 lines).

### r02 — File tool wrappers → `agent_core/agent_files.py` (DONE)

Extract `_tool_read_file`, `_tool_write_file`, `_tool_apply_patch`, `_tool_edit_file`,
`_tool_search`, `_tool_list_files`, `_tool_delete_file`, `_tool_analyze_file`,
`_tool_llm_analyze`, `execute_tool`, `_normalize_path`, `_safe_path` (~130 lines).

Target: `agent_core/agent_files.py`.

### r03 — File effects helpers → `agent_core/agent_effects.py` (DONE)

Extract `_resolve_nlp_path`, `_save_verify_note`, `_verify_file`, `_note_effect`,
`_take_trace_effects` (~70 lines).

Target: `agent_core/agent_effects.py`.

### r04 — Tool-loop core → `agent_core/turn_loop.py` (DONE)

Extract `_run_chained_tool_loop`, `_finish_turn`, `_reset_failover_notice`, `_report_failover`,
`_mutating_files_this_turn`, `_print_self_review_note`, `check_stale_files`, `invalidate_stale`
(~400 lines).

Target: `agent_core/turn_loop.py`.

### r05 — Prompt block builders → `agent_core/prompts/context_blocks.py`

Extract `_decision_constraints_block`, `_skill_index_block`, `_skill_hint_block`,
`_habits_block`, `_file_history_block`, `_semantic_memory_block`, `_original_goal_block`,
`_recent_user_texts`, `_prompts_related`, `_ensure_original_goal`, `_suspend_history_for_new_task`
(~350 lines).

Target: `agent_core/prompts/context_blocks.py`.

### r06 — Memory persistence → `agent_core/memory/persistence.py`

Extract `_load_chat_history`, `_save_chat_history`, `_load_memory`, `_save_memory`,
`_persist_plan_answer`, `memory_stats`, `clear_history`, `_record_llm_experience`
(~200 lines).

Target: `agent_core/memory/persistence.py`.

### r07 — REPL / CLI layer → `agent_core/cli/interactive.py`

Extract `run_interactive`, `main`, `_register_commands`, `_build_registry`,
`_warn_uncommitted`, `_build_chat_prompt`, `_looks_incomplete`, `_unverified_action_claim`
(~400 lines).

Target: `agent_core/cli/interactive.py`.

### r08 — Stub Agent → `agent_core/agent.py`

After steps r01–r07, `agent.py` holds only the `Agent` class body: `__init__`, `_register_tool_handlers`,
`build_transport`, `spawn_subagent`, `run_parallel_tasks`, `record_demo_activity`,
`get_metrics_collector`, `set_mode`, `is_plan_mode`, `_refresh_system_message`,
`_append_user_turn`, `chat_nlp`. Rename the class to `AgentCore` and expose it as
`Agent` via a thin alias at the top of `agent_core/__init__.py`.

Target: `agent_core/agent.py`.

### r09 — Final cleanup

- Remove all dead alias imports from `agent.py`.
- Update `agent_core/__init__.py` to expose the clean public API.
- Run full test suite (`pytest tests/ -x`).
- Update `Architecture.md` with the new module map.
- Remove any empty `__all__` stubs.

Target: `agent.py` shrinks to a thin re-export shim.

---

## What NOT to extract (already done)

- `_escape_json_string_controls`, `_repair_tool_arguments` — small helpers, stay in `agent.py`
- `bootstrap.py` — signal install + env setup (already extracted)
- `history_compaction.py` — history compaction pure functions (already extracted)
- `llm/client.py` — LLMClient (already extracted)
- `runtime_globals.py`, `json_utils.py`, `shell_utils.py`, `signal_handlers.py`, `types.py`, `run_utils.py` — already extracted

---

## Estimated final state

| File | Est. lines |
|------|-----------|
| `agent.py` (thin shim) | ~50 |
| `agent_core/agent.py` (AgentCore class) | ~500 |
| `agent_core/handlers/nlp_tools.py` | ~320 |
| `agent_core/agent_files.py` | ~130 |
| `agent_core/agent_effects.py` | ~70 |
| `agent_core/turn_loop.py` | ~400 |
| `agent_core/prompts/context_blocks.py` | ~350 |
| `agent_core/memory/persistence.py` | ~200 |
| `agent_core/cli/interactive.py` | ~400 |
| Other `agent_core/` modules | ~2000 |
