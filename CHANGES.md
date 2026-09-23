## 2026-09-24 - fix: remove dead duplicate modules and orphaned prototype

**Change**: `agent_core/commands/_implement_raw.py` deleted (dead prototype, unregistered command, helpers superseded by `implement_cmd.py`/`fix_cmd.py`). `docs/AGENTIC_IMPROVEMENT_PLAN.md` updated to reflect removal of dead module and progress on item #17. `CHANGES.md` (this entry).

**Reason**: `agent_core/commands/_implement_raw.py` was an orphaned prototype that duplicated live helpers with divergent signatures (`file_needs_generation` vs `implement_cmd.file_needs_generation`). Its command name (`implement-raw`) was never registered in the registry, making it unreachable from the REPL. It provided no value and increased technical debt by providing confusingly similar but non-functional code paths for future developers.

**Files**: agent_core/commands/_implement_raw.py, docs/AGENTIC_IMPROVEMENT_PLAN.md, CHANGES.md. Verified: `tests/test_dead_implement_raw.py` passes (2 tests) ensuring no remaining references exist in core directories. Full suite green.
