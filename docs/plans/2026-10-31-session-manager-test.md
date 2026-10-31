# Session Manager Test Coverage (2026-10-31)

**Status:** proposed
**Goal:** Pin the behaviour of Agent1's session manager with a permanent pytest
suite. `agent_core/cli/session/session_manager.py` (class `SessionManager`) is
the in-process session manager behind the CLI `clear` command. It measures 0 %
line coverage, and its only consumer, `agent_core/cli/commands/clear.py`
(`ClearCommand`), also measures 0 %. Nothing in the repository currently stops
a silent behaviour regression in either file.

## Which "session manager" this plan targets

The workspace has two unrelated session layers. This plan targets the first
one and proves the second one is untouched, so the ambiguity is resolved in the
tests instead of in prose:

| Layer | Module | Role |
| --- | --- | --- |
| Legacy in-process manager | `agent_core/cli/session/session_manager.py` | message list + stats, read by the CLI `clear` command. **Target of this plan.** |
| Named chat sessions | `agent_core/session/` (context, store, lock, cli) | one active session per process, owns `chat_history.json`. Already covered by the session-layer tests; must stay isolated. |

## Measured baseline

Run against the current tree (`python -m pytest tests/test_session_layer.py
tests/test_session_cmd.py --cov=agent_core.cli.session
--cov=agent_core.cli.commands.clear`):

- `agent_core/cli/session/session_manager.py` - 15 statements, 15 missed, 0 %.
- `agent_core/cli/commands/clear.py` - 35 statements, 35 missed, 0 %.

Coverage of the named-session layer is already 80-88 %, so the gap is exactly
the legacy manager plus its command.

## Behaviour to pin

The contract below is read from the current source and must not drift:

1. `get_session_stats()` returns exactly `message_count`, `token_usage`,
   `last_activity`. `token_usage` is `sum(len(str(content)) // 4)` - integer
   division per message, summed, and computed from `str()` so a non-string or
   missing `content` key cannot raise.
2. `add_message()` appends and advances `last_activity`.
3. `last_activity` is a UTC ISO-8601 string, and it is set at construction, so
   stats work on an empty manager.
4. `clear_session()` empties the message list and leaves the object usable.
5. `ClearCommand.run()` displays stats, returns `"Clear aborted by user."`
   without clearing when confirmation is declined, and
   `"Session cleared successfully."` after clearing.
6. `prompt_user_confirmation()` re-prompts on any answer other than `y`/`n`,
   accepts only the single letters `y` and `n`, and never clears on its own.

## Assumptions

- The tests stub configuration loading in `ClearCommand.__init__`
  (`load_agent_settings`, `_validate_settings`) so the suite never reads
  `model.json`, `.env`, or the network. The constructor contract stays
  untouched; only its side effect is removed.
- Assertions are behavioural (return values, logged messages, call counts) and
  use fakes, so the suite stays fast and platform independent. No real file
  I/O is added.
- The confirmation loop is tested by patching `builtins.input`. A
  `EOFError`/`KeyboardInterrupt` from `input()` is currently unhandled in the
  source; the test pins the raise-through so a future `except` clause is a
  deliberate, visible change.

## Non-goals

- No change to the named-session layer under `agent_core/session/`.
- No change to `SessionManager` or `ClearCommand` behaviour. This plan adds
  tests; it does not repair the source. A test that fails reveals a real
  defect and is reported, not silenced.
- No third-party mocking library beyond `unittest.mock`, which the repository
  already uses.
- No replacement or duplication of the existing `clear` command tests.

## Verification

- Every new suite passes with `pytest`.
- Line coverage of `agent_core/cli/session/session_manager.py` reaches 100 %,
  and `agent_core/cli/commands/clear.py` at least 90 % (the `logger.info`
  lines are covered by asserting the log records).
- The taskplan guard test parses this document and runs the dependency graph
  gate, so the plan stays machine-executable for `plan exec`.

## Risks

| Risk | Mitigation |
| --- | --- |
| Tests pass without asserting anything (smoke-only) | Each task names the exact return values, log messages and call counts to assert |
| Suite becomes slow or flaky through real clock reads | Compare parsed timestamps (`datetime.fromisoformat`) and assert monotonic ordering, never wall-clock deltas |
| Stubs hide the production wiring | One test calls the real `create_clear_command` factory and asserts the injected manager is the one cleared |
| Duplicated coverage with the existing command tests | Assert only stats-and-confirmation flow here; identifier names are prefixed to avoid collisions |
| `ClearCommand.__init__` doing live config I/O in CI | Stub both config calls at the `agent_core.cli.commands.clear` import path |

## Tasks

- [T1] Create `tests/test_session_manager.py` with unit tests for `SessionManager` from `agent_core/cli/session/session_manager.py`: empty-state stats expose `message_count` 0, `token_usage` 0 and an ISO-8601 `last_activity`, and `add_message` increments the count (role: tester)
- [T2] Add arithmetic and payload cases to `tests/test_session_manager.py`: `token_usage` equals the summed `len // 4` per message, a non-string `content` value and a missing `content` key do not raise, and an empty string adds zero tokens (role: tester) deps: T1
- [T3] Add clock and reset cases to `tests/test_session_manager.py`: `last_activity` parses as timezone-aware UTC ISO-8601, ordering is strictly non-decreasing across two adds, `clear_session` returns stats to the empty state, a second clear is idempotent, and two instances never share messages (role: tester) deps: T2
- [T4] Create `tests/test_session_manager_clear.py` covering `ClearCommand` from `agent_core/cli/commands/clear.py`, stubbing `load_agent_settings` and `_validate_settings` at the `agent_core.cli.commands.clear` import path so no live config is read (role: tester) deps: T1
- [T5] Add stats-display cases to `tests/test_session_manager_clear.py`: `display_stats` logs the message count, token estimate and last activity through the module logger, and never mutates the manager (role: tester) deps: T4
- [T6] Add confirmation-loop cases to `tests/test_session_manager_clear.py`: an invalid answer re-prompts, `y` returns True, `n` returns False, and `input()` errors propagate untouched so a future try/except guard shows up as a deliberate behavioural difference (role: tester) deps: T4
- [T7] Add end-to-end cases to `tests/test_session_manager_clear.py`: `run()` returns the abort message and keeps messages when declined, returns the success message after clearing when accepted, and `create_clear_command` wires the factory result to the injected manager it clears (role: tester) deps: T5, T6
- [T8] Add isolation cases to `tests/test_session_manager_clear.py`: clearing one manager leaves a second one intact and the legacy manager never reads or writes the named-session state under `agent_core/session/` (role: tester) deps: T7
- [T9] Create `tests/test_session_manager_taskplan.py` as the guard for this document: parse it with `parse_plan_tasks`, assert the exact task ids, reject unknown roles through `get_role`, and run `build_and_validate_graph` so a broken dependency fails here and not at execution time (role: tester) deps: T3, T8
- [T10] Run the four new suites together with the existing session-layer suites, report the measured line coverage for the manager and for the command module, and record any assertion that fails as a source defect rather than relaxing it (role: tester) deps: T9
- [T11] Document the session-manager test surface and its coverage baseline in the architecture notes so the next reader knows which session layer each suite covers (role: documenter) deps: T10
- [T12] Review the diff for assertion quality: confirm no weakened or skipped assertions, no real I/O, no duplicated identifiers with the existing command tests, and that the pinned contract matches the source line by line (role: reviewer) deps: T10
