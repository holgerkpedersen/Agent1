# Taskplan — decompose `agent.py` in small, reviewable steps

**Target:** `C:\Dev\Agent1\agent.py`
**Status:** steps t01–t02 done; this plan covers the remaining **t03–t15**.
**Revision:** re-measured and corrected this session. The previous revision had
(i) overlapping step ranges, (ii) ~47 units in no step at all, (iii) four wrong
patch counts, (iv) no rule for module-level constants, (v) no rule for `logger`.
Every number below was produced by an `ast` walk of the file as it is now.

---

## 1. Measured current state

| Metric | Value | How measured |
| --- | --- | --- |
| `agent.py` lines | **5416** | `len(open("agent.py").read().splitlines())` |
| Classes | **2** — `LLMClient` 202–377 (8 methods), `Agent` 400–3791 | `ast` walk of module body |
| `Agent` methods | **100** | `ast` walk |
| Module-level functions | **54** (1329 body lines) | `ast` walk |
| Module-level constants | **37** (36 `Assign` + 1 `AnnAssign`) | `ast` walk, strict-uppercase names |
| Guard frozensets | `EXPECTED_DEFINITIONS` **98**, `EXPECTED_IMPORTED` **146** | `ast.literal_eval` of the two calls |
| Already extracted | `agent_core/bootstrap.py` (import block + OpenBLAS + UTF-8 stdout) | commit `7dac445` |

> Correction: the previous revision said 40 constants. `DEFAULT_WORKSPACE` (405)
> is a **class attribute of `Agent`**, not a module constant, and `logger` (15) is
> lowercase. The true module-level count is 37.

Committed history for this refactor:

| Commit | Step | Effect |
| --- | --- | --- |
| `b45d5c8` | t01 | baseline recorded + `tests/test_agent_module_surface.py` guard |
| `7dac445` | t02 | imports and bootstrap side effects → `agent_core/bootstrap.py` |

The guard (`tests/test_agent_module_surface.py`, 403 lines) freezes the module
surface in two frozensets and fails loudly when a name moves without an
announced edit:

* `EXPECTED_DEFINITIONS` (98 names) — names `agent.py`'s **own body** must keep defining.
* `EXPECTED_IMPORTED` (146 names) — names reachable through `agent.py`'s **import block**.

Its docstring states the rule for every step:

> When a step moves code out, remove the name from this list in the SAME commit
> and add it to `EXPECTED_IMPORTED` (the re-export).

`test_expected_definitions_are_live` exempts three `None`-initialised
placeholders, so they need no re-export bookkeeping:

```python
exempt = {"_input_executor", "_shutdown_save_memory", "_shutdown_trace_close"}
```

---

## 2. Four hard constraints

1. **`agent.py` is a public import surface.** Measured: **84** files do
   `from agent import Agent`, **5** import `LLMClient`, **17** import a private
   name, and **43** do `import agent`. Every moved name must stay importable from
   `agent` through an explicit re-export.

   > Correction: the previous revision said 83 / 5 / ~30. The `~30` conflated two
   > different populations — 17 files *import* a private name, while 24 files
   > *monkeypatch* one on the `agent` module. Both matter, but they are not the
   > same list.

2. **No new module may `import agent` at module level.** Verified: under
   `agent_core/` (non-test code) there are exactly **two** `agent` imports and
   both are function-scope — `commands/memory_cmd.py:53` (`import agent` inside
   `_memory_db_path`) and `subagent.py:77` (`from agent import …` inside
   `__init__`). The module-level ones are all in `agent_core/tests/`
   (`test_agent_entrypoint.py:13-14`, `test_plan_workflow.py:34`), which are
   tests, not shipped modules. `agent_core.decisions` and `agent_core.subagent`
   already import `agent` at call time; a module-level import back would close
   the cycle.

   > Correction: the previous revision said "all 43 `import agent` sites under
   > `agent_core/`". That conflated the repo-wide count (**43** files, of which
   > **30** are module-level) with the `agent_core/` count (**2** files). The
   > conclusion is unchanged; the evidence was wrong.

3. **Tests monkeypatch configuration on the `agent` module.** `conftest.py` plus
   the test suite do `monkeypatch.setattr(agent, "NAME", …)`. Moving a function
   changes which module a *bare global read* resolves against — see §3.

4. **Every new module needs its own `logger`.** `logger = logging.getLogger(__name__)`
   lives at `agent.py:15` and **31 of the moving units read it as a bare global**
   (7 module-level functions + 24 `Agent` methods).
   The previous revision never mentioned this. A moved function that keeps
   writing `logger.warning(...)` will raise `NameError`. Each new module must
   define its own logger at the top; `logger` itself **stays** in
   `EXPECTED_DEFINITIONS` (it is `D=True`, `I=False`).

---

## 3. The rule that makes every step safe

**A moved function must not read a monkeypatched name as a bare global.**

A bare read resolves against the enclosing module's globals. After the move the
enclosing module is the new one, so a patch on `agent.NAME` is silently ignored
and the test still passes or fails for the wrong reason.

Verified bare-global reads of test-patched names inside `agent.py`. The last
column is the number of **test files that patch that name on `agent`** —
measured by an `ast` walk of all 487 `.py` files (`backups/` excluded) that
resolves each file's local alias of the `agent` module (`import agent`,
`import agent as X`, or a helper such as `agent_module()`) and then counts
`setattr` / `delattr` / `patch.object(<alias>, "NAME", …)` plus dotted
`patch("agent.NAME", …)`:

| Name read as a bare global | Read at line(s) | Enclosing function | Files patching |
| --- | --- | --- | --- |
| `AGENT_MEMORY_JSON_PATH` | 134, 3222, 3624, 3673, 3677, 3749 | `_signal_break_with_cleanup`, `clear_history`, persistence | **17** |
| `CHAT_HISTORY_JSON_PATH` | 3221, 3514, 3553, 3602, 3606 | `clear_history`, `_suspend_history_for_new_task`, persistence | **16** |
| `LLMClient` | 495 | `Agent.__init__` | **9** |
| `TURN_LOG_PATH` | 4219 | `_append_turn_log` | **4** |
| `_MAX_CHAINED_RUNS` | 2686 | `Agent._run_chained_tool_loop` | **3** |
| `_resolve_display_mode` | 2481, 2792, 3007, 3053, 3064, 3081, 3300 | `Agent` (7 sites) | **4** |
| `_start_kanban_inbound` | 5398 | `main` | **3** |
| `run_interactive` | 5409 | `main` | **3** |
| `_pytest_full_suite_timeout` | 1463, 1505, 1711 | `Agent` | **2** |
| `record_turn_outcome` | 2872 | `Agent._finish_turn` | **2** |
| `_CONTEXT_AST_THRESHOLD_KB` | 1330, 1333 | `Agent._read_page` | 1 |
| `_DELEGATE_TIMEOUT_S` | 792, 796, 874, 877 | `Agent` (delegate, parallel) | 1 |
| `_ENV_FILE_PATH` | 4537 | `_read_env_value` | 1 |
| `load_agent_settings` | 215, 558, 4019, 4039 | `LLMClient`, `Agent`, `_resolve_stream_fn`, `_resolve_display_mode` | **3** |
| `resolve_model` | 210, 413 | `LLMClient`, `Agent.__init__` | **3** |
| `_kill_process_tree` | 1494 | `Agent._nlp_run` | 1 |
| `_looks_incomplete` | 2683 | `Agent._run_chained_tool_loop` | 1 |
| `record_full_run_seconds` | 1500, 1518, 1721 | `Agent` | 1 |
| `decisions_as_system_prompt` | 3248 | `Agent._decision_constraints_block` | 1 |
| `recommend_profile` | 2993 | `Agent.chat_nlp` | **1** |
| `ToolLoopRunner` | 2510, 2621, 2816 | `Agent` | **2** |
| `run_dashboard_server` | 5402 | `main` | 1 |
| `start_dashboard_thread` | 5407 | `main` | 1 |
| `_effective_ws_dir` | *not read as a global* | `Agent` method (918–930) | 1 |

> **Patch-count provenance.** The `Files patching` column above was re-measured
> this session by an `ast` walk of all 487 `.py` files (`backups/` excluded);
> nine rows were raised to the values shown (previously `16`/`14`/`6`/`3`/`1`/
> `1`/`2`/`1`/`1`, now `17`/`16`/`9`/`4`/`3`/`3`/`1`/`2`/`2`). Raw **site**
> counts, which matter when a step rewrites a patch site: `AGENT_MEMORY_JSON_PATH`
> 56, `CHAT_HISTORY_JSON_PATH` 44, `_resolve_display_mode` 12, `LLMClient` 10,
> `load_agent_settings` 10, `TURN_LOG_PATH` 7, `_pytest_full_suite_timeout` 7,
> `_MAX_CHAINED_RUNS` 6, `_ENV_FILE_PATH` 6, `ToolLoopRunner` 5, `_start_kanban_inbound`
> 3, `run_interactive` 3, `resolve_model` 3, `record_full_run_seconds` 3,
> `_kill_process_tree` 2, `recommend_profile` 2, and 1 each for the rest.
>
> **Correction (re-verified).** Two rows were wrong in the previous revision and
> are fixed above: `record_turn_outcome` is patched in **2** files / 2 sites (it
> also appears in `tests/test_learning_wiring.py:276`, not just `test_habits.py`),
> and `recommend_profile` is patched in **1** file (`test_learning_wiring.py`)
> with **2** sites — the earlier revision put its *site* count in the *files*
> column. Every other row in both columns re-measures exactly.
>
> `_ENV_FILE_PATH` is patched on `pytest_gate` by a further **4** sites in
> `tests/test_pytest_full_run_gate.py` — a different module, so it does **not**
> count toward the `agent` surface. `_effective_ws_dir` is patched on `agent` at
> `tests/test_pytest_full_run_gate.py:288` (the doc's "918-930" is its *definition*).
>
> The **14** files that patch a *private* name on `agent` are:
> `test_agent_dry_refactor`, `test_agent_main_dispatch`, `test_agent_robustness`,
> `test_dashboard_bridge`, `test_delegate_and_router`, `test_demo_data`,
> `test_kanban_inbound_boot`, `test_learning_wiring`, `test_pytest_elapsed_budget`,
> `test_pytest_full_run_gate`, `test_pytest_timeout_injection`,
> `test_stream_final_answer_wiring`, `test_tool_loop_nlp`,
> `test_unverified_action_claim`.

Three rows are **new** and were missing entirely:

* `clear_history` (3199–3233) reads `CHAT_HISTORY_JSON_PATH` **and**
  `AGENT_MEMORY_JSON_PATH` — the two most-patched names in the suite. No step
  covered this method.
* `_effective_ws_dir` (918–930) is patched on `agent` by
  `tests/test_pytest_full_run_gate.py:288`, so it must stay reachable as a bare
  global even though it is an `Agent` method.
* `_refresh_system_message` (2416–2455) reads `_SYSTEM_PROMPT` (t07's target)
  and `_strip_dynamic_system_blocks` (t04's target); `_append_user_turn`
  (2457–2502) reads `_resolve_display_mode`. Neither was assigned a step.

### Mitigation — one tiny helper, used by all moved code

Created in t03 and reused by every later step:

```python
# agent_core/runtime_globals.py
"""Call-time reads of the `agent` module's globals.

Extracted code lives in agent_core, but tests and conftest.py monkeypatch
configuration on the `agent` module. A bare global read inside a moved
function would resolve against the new module and silently ignore those
patches, so moved code reads patchable configuration through `live()`.
"""

def live(name: str, default: Any = None) -> Any:
    """Read *name* from the `agent` module at CALL time (never at import)."""
    import agent  # function-scope: keeps the import graph acyclic
    return getattr(agent, name, default)
```

Then a moved function writes `continuations < live("_MAX_CHAINED_RUNS", 6)`
instead of `continuations < _MAX_CHAINED_RUNS`. Every existing
`monkeypatch.setattr(agent, "_MAX_CHAINED_RUNS", 0)` keeps working, and **no test
file needs editing**. The patches stay where they are; the code reads them from
where they are.

### The constants rule (new — this was the biggest hole)

A step must also move **the module-level constants its moved code reads**. Those
constants are *not* contiguous: the block at **3823–3931** is read by code that
belongs to **t03, t04, t12, t13 and t14**, so it must be split explicitly.
Measured readers:

| Constant | Line(s) | Read by (step) | Move with |
| --- | --- | --- | --- |
| `_GOAL_TOPIC_STOPWORDS` | 384–391 | `_content_tokens` (t03) | **t03** |
| `_GOAL_RELATED_THRESHOLD` | 397 | `_prompts_related` (t13) | **t13** |
| `_MAX_CHAT_MESSAGES` | 3823 | `_trim_chat_history` (t03) | **t03** |
| `_HISTORY_CHAR_BUDGET` | 3830 | `_trim_chat_history` (t03) | **t03** |
| `_HISTORY_TRIM_NOTE` | 3832–3836 | `_trim_chat_history` (t03) | **t03** |
| `_HISTORY_TRIM_GOAL_CAP` | 3841 | `_trim_chat_history` (t03) | **t03** |
| `_MAX_CHAINED_RUNS` | 3848 | `_run_chained_tool_loop`, `chat_nlp` (t14) | **t14** |
| `_LLM_ERROR_MAX_RETRIES` | 3856 | `chat_nlp` (t14) | **t14** |
| `_LLM_ERROR_BACKOFF_S` | 3857 | `chat_nlp` (t14) | **t14** |
| `_CONTINUE_NOTE_TAG` | 3864 | `_run_chained_tool_loop` (t14) | **t14** |
| `_MAX_RUN_TIMEOUT_S` | 3867 | `_nlp_run` (t12) | **t12** |
| `_MAX_READ_LINES` | 3870 | `_read_page` (t12) | **t12** |
| `_MAX_READ_FILES` | 3875 | `_nlp_read_many` (t12) | **t12** |
| `_CONTEXT_AST_THRESHOLD_KB` | 3880 | `_read_page` (t12) | **t12** |
| `_MAX_CONSECUTIVE_READS` | 3889 | `_read_page` (t12) | **t12** |
| `_READ_LOOP_NOTE` | 3891–3896 | `_read_page` (t12) | **t12** |
| `_MAX_ACTIVE_SUBAGENTS` | 3903 | `_nlp_delegate`, `_nlp_delegate_batch` (t12) | **t12** |
| `_DELEGATE_TIMEOUT_S` | 3904 | `_nlp_delegate`, `run_one` (t12) | **t12** |
| `_CONTINUE_NOTE_TAG_KEY` | 3908 | `_load_chat_history` (**t13**) + t14 | **t13** |
| `_CONTINUE_NOTE` | 3910–3915 | `_run_chained_tool_loop` (t14) | **t14** |
| `_INCOMPLETE_MARKERS` | 3917–3923 | `_looks_incomplete` (t04) | **t04** |
| `_COMPLETE_MARKERS` | 3928–3931 | `_looks_incomplete` (t04) | **t04** |
| `_ACTION_VERB_RE` | 3950 | `_unverified_action_claim` (t04) | **t04** |
| `_QUESTION_LEAD_RE` | 3954–3958 | `_unverified_action_claim` (t04) | **t04** |
| `_VERIFY_NOTE` | 3990–3996 | `_run_chained_tool_loop` (t14) | **t14** |
| `_ENV_FILE_PATH` | 4524 | `_read_env_value` (t06) | **t06** |
| `_PYTEST_FULL_SUITE_TIMEOUT_KEY` | 4525 | `_pytest_full_suite_timeout` (t06) | **t06** |
| `_PYTEST_LAST_FULL_RUN_KEY` | 4526 | `_pytest_full_suite_timeout` (t06) | **t06** |
| `_FULL_SUITE_MARGIN` | 4529 | `_pytest_full_suite_timeout` (t06) | **t06** |
| `_PYTEST_BARE_TOKENS` | 4577–4580 | `_is_full_pytest_command` (t06) | **t06** |
| `_PYTEST_VALUE_FLAGS` | 4588–4595 | `_is_full_pytest_command` (t06) | **t06** |
| `_PYTEST_SUBSET_FLAGS` | 4603–4606 | `_is_full_pytest_command`, `_is_subset_pytest_command` (t06) | **t06** |
| `_DANGEROUS_SHELL_PATTERNS` | 4053 | *nothing* — alias, only re-exported | **stays in agent.py** |
| `_SHUTDOWN_FLAG` | 102 | signals (t09) | **t09** |
| `_SHELL_NAME` | 4302 | `_SYSTEM_PROMPT` at import time | **t07** |
| `CHAT_INPUT_PROMPT` | 5237 | `run_interactive` (t10) | **t10** |

`_CONTINUE_NOTE_TAG_KEY` is the one cross-step constant: t13 moves it (its
`_load_chat_history` reads it), and t14 then reaches it through the re-export
that t13 added. t14 runs after t13, so this ordering works — but only because
t14 is strictly last.

### The root-path rule (new)

Two moved names derive a filesystem path from `__file__`, which resolves against
the **new** module's directory:

* `_ENV_FILE_PATH` (4524) — `os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")`
  → moves in t06.
* `DEFAULT_WORKSPACE` (405) — a class attribute of `Agent`; **stays**, no action.

`agent_core/constants.py` already solves this problem correctly at line 11:

```python
_MODEL_JSON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

So t06 must use the **two-level** form (or reuse an existing constant), not the
one-level form copied verbatim. A verbatim copy would silently point `.env`
lookups at `agent_core/` instead of the repo root — and the test that patches
`_ENV_FILE_PATH` would still pass, because it patches the name rather than
reading the file.

---

## 4. Per-step checklist (definition of done)

Copy this list into each step's commit message. A step is done only when all ten
pass, in this order.

1. **Pure move.** Copy the functions verbatim. No renames, no signature changes,
   no logic edits, no "while I'm here" cleanups. The only permitted edit is
   `live()` substitution (§3) and the `__file__` path form (§3).
2. **Patchable reads go through `live()`.** Check the §3 table for names the
   moved functions read.
3. **Module constants travel with their readers** (§3 constants table).
4. **Add `logger = logging.getLogger(__name__)`** to the new module if any moved
   unit reads `logger` (§2 constraint 4).
5. **Re-export explicitly** in `agent.py`: `from agent_core.<mod> import (…)  # noqa: F401`.
6. **Update the guard in the same commit**: delete each moved name from
   `EXPECTED_DEFINITIONS` and add it to `EXPECTED_IMPORTED`. Both frozensets are
   checked in both directions, so an unannounced add fails too.
7. **Keep the delegators.** See §5.
8. **Run the guard plus the tests that touch the moved names.** At minimum:
   `python -m pytest tests/test_agent_module_surface.py -q` and any test file
   named in the §3 table for a name you moved.
9. **One commit per step** (t12/t13: one commit per cluster).
10. **`agent.py` must shrink.** If a step makes the file longer, code was copied
    instead of moved.

---

## 5. Why delegators stay on `Agent` (the deliberate trade)

Moving an `Agent` method to a module function changes the call syntax. Rather
than rewrite every `self._nlp_read(...)` call site and every dispatch entry, each
moved method leaves a one-line delegator:

```python
def _nlp_read(self, args: dict[str, Any]) -> str:
    return nlp_tools.nlp_read(self, args)
```

Cost: ~200 extra lines in `agent.py`. Benefit: every call site, every
`_nlp_tool_handlers` mapping entry, and every `monkeypatch.setattr(Agent, …)`
target keeps working unchanged. That is what makes each step easy to understand —
and it is why the final `agent.py` is ~1300–1500 lines rather than the 600–900
the earlier draft assumed.

---

## 6. Step ranges (corrected — no overlaps, no orphans)

Every range below is exclusive: a line belongs to exactly one step. Ranges that
the previous revision had overlapping or missing are marked **fixed**.

### Phase A — module-level helpers

**t03 — history compaction** → `agent_core/history_compaction.py` *(deps: t01)*
Functions, 4694–5100 — this range is exact, all 13 functions, no strays:
`_drop_orphan_tool_messages` (4694–4728), `_repair_unanswered_tool_calls`
(4731–4768), `_strip_tool_args` (4771–4804), `_message_size` (4807–4819),
`_content_tokens` (4822–4832), `_compaction_note_goal` (4835–4848),
`_is_compaction_note` (4851–4852), `_strip_turn_wrappers` (4855–4872),
`_recover_original_goal` (4875–4912), `_first_user_message_text` (4915–4942),
`_trim_chat_history` (4945–5037), `_strip_image_blocks` (5040–5070),
`_project_chat_history` (5073–5100).

**This step also creates `agent_core/runtime_globals.py`** (§3).
Constants: `_MAX_CHAT_MESSAGES` (3823), `_HISTORY_CHAR_BUDGET` (3830),
`_HISTORY_TRIM_NOTE` (3832–3836), `_HISTORY_TRIM_GOAL_CAP` (3841),
`_GOAL_TOPIC_STOPWORDS` (384–391).
No `logger` reader, no patched-global read — the safest first step.

**t04 — turn telemetry** → `agent_core/turn_telemetry.py` *(deps: t01)*
Functions: `_looks_incomplete` (3934–3944), `_unverified_action_claim`
(3961–3984), `_resolve_stream_fn` (3999–4030), `_resolve_display_mode`
(4033–4045), `_final_answer_fallback` (4086–4104), `_turn_quality_score`
(4147–4189), `_begin_trace_turn` (4192–4207), `_append_turn_log` (4210–4235),
`_strip_dynamic_system_blocks` (4238–4260).
Constants: `_INCOMPLETE_MARKERS` (3917–3923), `_COMPLETE_MARKERS` (3928–3931),
`_ACTION_VERB_RE` (3950), `_QUESTION_LEAD_RE` (3954–3958).
⚠ **Fixed:** the old range `3934–4260` swept up six functions owned by t05
(`_unix_command_hint` 4056–4066, `_windows_command_hint` 4069–4077,
`_shell_command_hint` 4080–4083, `_kill_process_tree` 4107–4125, `_git_branch`
4128–4144) and the `_DANGEROUS_SHELL_PATTERNS` alias (4053). t04's true ranges
are **3934–4045, 4086–4104, 4147–4260**.
`live()`: `load_agent_settings` (4019, 4039), `TURN_LOG_PATH` (4219),
`_resolve_display_mode` (3 files).
`logger`: `_begin_trace_turn`, `_append_turn_log`.

**t05 — shell and subprocess intel** → `agent_core/shell_intel.py` *(deps: t01)*
Functions: `_current_git_branch` (164–176), `_unix_command_hint` (4056–4066),
`_windows_command_hint` (4069–4077), `_shell_command_hint` (4080–4083),
`_kill_process_tree` (4107–4125), `_git_branch` (4128–4144), `_detect_shell`
(4263–4279), `_shell_name_token` (4282–4297), `_truncate_output` (4433–4448),
`_run_subprocess_captured` (4451–4474), `_shape_run_stderr` (4477–4505),
`_blocked_shell_command` (4508–4520).
⚠ **Fixed:** `_current_git_branch` (164–176) sat inside t09's range `81–199`
as well as t05's list. It is **t05's**; t09 must skip it.
`_DANGEROUS_SHELL_PATTERNS` (4053) **stays in `agent.py`** — it is read by
nothing, only re-exported.
`live()`: `_kill_process_tree` (1 file, read at 1494 by t12's `_nlp_run` — the
call site is t12's problem, not this step's).
`logger`: `_kill_process_tree`.

**t06 — pytest command policy** → `agent_core/pytest_cmd_policy.py` *(deps: t01)*
Functions: `_read_env_value` (4532–4547), `_pytest_full_suite_timeout`
(4550–4573), `_is_full_pytest_command` (4609–4651), `_is_pytest_command`
(4654–4669), `_is_subset_pytest_command` (4672–4691).
Constants: `_ENV_FILE_PATH` (4524), `_PYTEST_FULL_SUITE_TIMEOUT_KEY` (4525),
`_PYTEST_LAST_FULL_RUN_KEY` (4526), `_FULL_SUITE_MARGIN` (4529),
`_PYTEST_BARE_TOKENS` (4577–4580), `_PYTEST_VALUE_FLAGS` (4588–4595),
`_PYTEST_SUBSET_FLAGS` (4603–4606).
⚠ **Root path:** `_ENV_FILE_PATH` must use the two-level `__file__` form (§3).
`live()`: `_ENV_FILE_PATH` (1 file), `_pytest_full_suite_timeout` (2 files, call
sites at 1463/1505/1711 are t12's).

**t07 — system prompt** → `agent_core/prompts/system_prompt.py` *(deps: t05)*
`_SHELL_NAME` (4302, `AnnAssign`), `_SYSTEM_PROMPT` (4309–4430, 122 lines).
Both are in `EXPECTED_DEFINITIONS` (`D=True`, `I=False`) → both move to
`EXPECTED_IMPORTED`.
⚠ `_SYSTEM_PROMPT` is evaluated **at import time** and calls `_detect_shell()`
(4325, 4332) and `_git_branch()` (4360), so the new module must import them from
`shell_intel` — hence the t05 dependency. The rendered prompt must stay
**byte-identical**; `tests/test_harmony_safety_prompt.py`,
`tests/test_speculate_nlp_tool.py` and `tests/test_tool_loop_nlp.py` compare it.

**t08 — `LLMClient`** → `agent_core/llm/client.py` *(deps: t02)*
Class `LLMClient`, 202–377, 8 methods. Patched by **9** test files.
`live()`: `load_agent_settings` (215), `resolve_model` (210).
`logger`: `LLMClient.__init__` (239).

**t09 — signals and interruptible input** → `agent_core/cli/signals.py` *(deps: t02)*
Functions: `_suppress_and_log` (81–91), `_signal_break` (106–112),
`_signal_break_with_cleanup` (122–137), `_install_signal_handlers` (140–161),
`_interruptible_input` (179–199).
⚠ **Fixed:** skip **164–176** (`_current_git_branch` belongs to t05).
Move the mutable globals with the functions that write them —
`_SHUTDOWN_FLAG` (102), `_shutdown_save_memory`, `_shutdown_trace_close`,
`_input_executor` — so every `global` statement keeps targeting one module.
⚠ `_signal_break_with_cleanup` reads `AGENT_MEMORY_JSON_PATH` at 134 → `live()`
(**17** files patch it, the most-patched name in the suite).
The guard exempts the three `None`-initialised placeholders (line 361), so the
surface edit is smaller than it looks.
`logger`: `_suppress_and_log`.

**t10 — REPL / CLI layer** → `agent_core/cli/registry_setup.py` + `agent_core/cli/interactive.py` *(deps: t08, t09)*
`registry_setup.py`: `_register_commands` (5122–5170), `_build_registry` (5173–5177),
`_warn_uncommitted` (5180–5211), `_build_chat_prompt` (5214–5232).
`interactive.py`: `CHAT_INPUT_PROMPT` (5237), `run_interactive` (5240–5370),
`_start_kanban_inbound` (5373–5388), `main` (5391–5409).
⚠ **Keep 5101–5121 untouched** — that is the `from agent_dashboard import (…)`
re-export block. `_shared_metrics_collector` is patched on `agent` by **3** test
files, so it must keep resolving there.
⚠ `main` reads four patched names as bare globals (5398–5409) → `live()`:
`_start_kanban_inbound` (3), `run_interactive` (3), `run_dashboard_server` (1),
`start_dashboard_thread` (1).
`logger`: `_warn_uncommitted`, `_start_kanban_inbound`.
**Keep `if __name__ == "__main__"` behaviour byte-identical** — this is the file
users launch. `main` and `run_interactive` are in `EXPECTED_DEFINITIONS`.

### Phase B — `Agent` decomposition (strictly serial)

**t11 — legacy tool aliases and file operations** → `agent_core/agent_files.py` *(deps: t03–t10)*
Methods 2221–2408: `_tool_read_file`, `_tool_write_file`, `_tool_apply_patch`,
`_tool_edit_file`, `_tool_search`, `_tool_list_files`, `_tool_delete_file`,
`_tool_analyze_file`, `_tool_llm_analyze`, `execute_tool`, `_normalize_path`,
`_safe_path`, `read_file`, `write_file`, `apply_patch`, `edit_file`,
`_build_semantic_index`, `_cleanup_semantic_index`, `search_file`.
**170 method-lines**, not 111 — the previous revision undercounted by 59.
Self-contained; the warm-up for Phase B. Keep the delegators per §5.
`logger`: `_tool_read_file`, `read_file`.

**t11b — file-effect and verify helpers** → `agent_core/agent_effects.py` *(same step, second commit)*
⚠ **New step.** These five methods sat in **no** step in the previous revision:
`_resolve_nlp_path` (629–639), `_save_verify_note` (641–653), `_verify_file`
(655–672), `_note_effect` (674–685), `_take_trace_effects` (687–696).
They are file/trace helpers with no `live()` reads, so they move cleanly.

**t12 — NLP tool handlers** → `agent_core/handlers/nlp_tools.py` *(deps: t11)*
28 methods, 749–1888, **1039 method-lines** — the single biggest win.
⚠ **`_run_command_quietly` (905–916) and `_effective_ws_dir` (918–930) sit inside
the 749–1888 range but are NOT in the cluster list.** Both **stay on `Agent`**:
`_effective_ws_dir` is monkeypatched on `agent` by
`tests/test_pytest_full_run_gate.py:288`, so it must remain a bare-global-
reachable `Agent` method. State this explicitly in the commit message.
**Split into per-cluster commits** (each cluster = one commit, same checklist):

* 12a — `agent_core/tool_call_repair.py`: `_escape_json_string_controls` (18–54),
  `_repair_tool_arguments` (57–77).
* 12b — file/search: `_nlp_read` (1257–1270), `_read_page` (1301–1362),
  `_nlp_read_many` (1272–1299), `_nlp_read_skill` (1364–1389), `_nlp_search`
  (1243–1255), `_nlp_list_files` (1391–1410), `_nlp_definitions` (1079–1092),
  `_nlp_references` (1094–1106).
* 12c — exec: `_nlp_run` (1439–1519), `_nlp_git` (1521–1608), `_nlp_merge`
  (1610–1624), `_nlp_diff` (1626–1642), `_nlp_edit` (1644–1688), `_nlp_write`
  (1428–1437), `_nlp_fix` (1412–1426), `_nlp_tests` (1690–1724).
* 12d — delegation: `_nlp_delegate` (749–825), `_nlp_delegate_batch` (827–903),
  `_nlp_create_subagent` (932–958), `_nlp_run_subagent_task` (960–975).
* 12e — reasoning/misc: `_nlp_jev_decide` (1108–1170), `_nlp_speculate`
  (1172–1241), `_nlp_analyze` (1726–1742), `_nlp_web_search` (1744–1763),
  `_nlp_get_current_datetime` (1765–1776), `_nlp_hue_control` (1778–1888).
* 12f — MCP: `_nlp_mcp_tools` (1024–1038), `_nlp_mcp_call` (1040–1077).

⚠ `_nlp_tool_handlers()` (977–1022) is the wiring point and **stays on `Agent`**;
it must return the same mapping. `_execute_tool_call` (713–747) also stays — it
reads `NLP_TOOL_NAMES`.
Constants that move with this step: `_MAX_RUN_TIMEOUT_S` (3867), `_MAX_READ_LINES`
(3870), `_MAX_READ_FILES` (3875), `_CONTEXT_AST_THRESHOLD_KB` (3880),
`_MAX_CONSECUTIVE_READS` (3889), `_READ_LOOP_NOTE` (3891–3896),
`_MAX_ACTIVE_SUBAGENTS` (3903), `_DELEGATE_TIMEOUT_S` (3904).
⚠ Reads to route through `live()`: `_CONTEXT_AST_THRESHOLD_KB` (1330, 1333),
`_DELEGATE_TIMEOUT_S` (792, 796, 874, 877), `_kill_process_tree` (1494),
`_pytest_full_suite_timeout` (1463, 1505, 1711), `record_full_run_seconds`
(1500, 1518, 1721).
`logger`: `_nlp_delegate`, `_nlp_delegate_batch`, `_nlp_read_skill`.

**t13 — plan tools, prompt blocks, persistence** *(deps: t03, t12)*
Four modules, four commits. Two of them absorb methods the previous revision
left unassigned:

* `agent_core/handlers/plan_tools.py` — 1892–2217: `_get_plan_state`,
  `_nlp_plan_status`, `_nlp_plan_start`, `_nlp_plan_step`, `_nlp_plan_finish`.
* `agent_core/prompts/context_blocks.py` — 3235–3538: `_decision_constraints_block`
  (3235–3253), `_skill_index_block` (3255–3270), `_skill_hint_block` (3272–3306),
  `_habits_block` (3308–3323), `_file_history_block` (3325–3345),
  `_semantic_memory_block` (3347–3386), `_ensure_original_goal` (3388–3436),
  `_recent_user_texts` (3438–3465), `_prompts_related` (3467–3488),
  `_suspend_history_for_new_task` (3490–3524), `_original_goal_block` (3526–3538)
  **plus ⚠ NEW: `_refresh_system_message` (2416–2455) and `_append_user_turn`
  (2457–2502)** — they read `_SYSTEM_PROMPT`, `_strip_dynamic_system_blocks` and
  `_resolve_display_mode`, so they belong here and must use `live()` for
  `_resolve_display_mode` (3 files).
  ⚠ `_decision_constraints_block` reads `decisions_as_system_prompt` at 3248.
* `agent_core/memory/persistence.py` — 3544–3791 plus `_read_json_quarantining`
  (3795–3820) **plus ⚠ NEW: `check_stale_files` (3163–3174), `invalidate_stale`
  (3176–3184), `memory_stats` (3186–3197), `clear_history` (3199–3233)**.
  ⚠ `clear_history` reads `CHAT_HISTORY_JSON_PATH` (3221) **and**
  `AGENT_MEMORY_JSON_PATH` (3222) — the two most-patched names in the suite
  (16 and 17 files). This method was in no step before; route both through
  `live()`.
  ⚠ The heaviest patch surface: `CHAT_HISTORY_JSON_PATH` (5 reads),
  `AGENT_MEMORY_JSON_PATH` (5 reads). **This step is the reason `live()` exists.**
* Constants: `_GOAL_RELATED_THRESHOLD` (397), `_CONTINUE_NOTE_TAG_KEY` (3908 —
  t14 will read it through t13's re-export).
* `logger`: **12 units** — `_decision_constraints_block`, `_skill_index_block`,
  `_skill_hint_block`, `_habits_block`, `_file_history_block`,
  `_semantic_memory_block`, `_original_goal_block`, `_suspend_history_for_new_task`,
  `_save_chat_history`, `_save_memory`, `_record_llm_experience`,
  `_read_json_quarantining`.

**t14 — the turn loop** → `agent_core/turn_loop.py` *(deps: t03–t13)*
Methods 2504–3161: `_run_chained_tool_loop` (2504–2756, 253 lines),
`_reset_failover_notice` (2758–2772), `_report_failover` (2774–2810),
`_finish_turn` (2812–2936, 125 lines), `chat_nlp` (2938–3082, 145 lines),
`_mutating_files_this_turn` (3084–3119), `_print_self_review_note` (3121–3140),
`_persist_plan_answer` (3142–3161).
**Last and most delicate:** `chat_nlp` is the entry point of the whole NLP path
and owns continuation chaining, retry/backoff and the trace writer lifecycle.
Extract only after everything it calls has settled.
Constants that move here: `_MAX_CHAINED_RUNS` (3848), `_LLM_ERROR_MAX_RETRIES`
(3856), `_LLM_ERROR_BACKOFF_S` (3857), `_CONTINUE_NOTE_TAG` (3864),
`_CONTINUE_NOTE` (3910–3915), `_VERIFY_NOTE` (3990–3996).
⚠ Reads to route through `live()`: `_MAX_CHAINED_RUNS` (2686),
`_looks_incomplete` (2683), `record_turn_outcome` (2872), `recommend_profile`
(2993), `_resolve_display_mode` (7 sites), `ToolLoopRunner` (3 sites).
`logger`: `_reset_failover_notice`, `_report_failover`, `_finish_turn`, `chat_nlp`.

### Phase C — consolidation

**t15 — final consolidation** *(deps: t03–t14, role: reviewer + integrator)*
1. Delete imports that became dead in `agent.py`.
2. Run `agent_core/utils/lint_guard.py` if it is wired, plus the project linter.
3. Confirm `agent.py` is a composition root: bootstrap import, re-exports,
   `Agent.__init__` wiring, delegators, entry point.
4. Update the module map in `Architecture.md` / `AGENTS.md` and the step log in
   `docs/refactor_agent_py.md`.
5. Delete scratch files; **one** full `python -m pytest` run; commit.

### Methods that deliberately stay on `Agent`

Listed so no future pass "discovers" them as orphans. `Agent` is 100 methods;
~40 remain after t14, and these are the class itself:

`__init__` (407–524), `_register_tool_handlers` (526–543), `build_transport`
(545–561), `spawn_subagent` (564–580), `run_parallel_tasks` (582–601),
`record_demo_activity` (604–623), `get_metrics_collector` (625–627),
`set_mode` (698–707), `is_plan_mode` (709–711), `_execute_tool_call` (713–747),
`_run_command_quietly` (905–916), `_effective_ws_dir` (918–930),
`_nlp_tool_handlers` (977–1022), plus the one-line delegators t11–t14 add.

---

## 7. Expected shape

| Milestone | `agent.py` lines | After |
| --- | --- | --- |
| Baseline (t01) | 5565 | — |
| After t02 (actual) | **5416** | t02 |
| Module-level helpers gone (~1560 moved out, ~90 re-export lines in) | ≈ 3950 | t10 |
| `Agent` methods extracted (~2600 out, ~200 delegator lines in) | ≈ 1550 | t14 |
| Composition root | ≈ 1300–1500 | t15 |

The last number is larger than the old plan's 600–900 because the delegators stay
on `Agent` by design. That is the deliberate trade: ~200 extra lines in exchange
for every call site, dispatch entry and monkeypatch target staying valid — which
is what makes each step easy to understand.

---

## 8. Ordering and parallelism

* Safe in parallel (independent regions, no shared state): **t03, t04, t05, t06**.
* Safe after t02: **t08, t09**.
* **Serial:** t07 → t10 → t11 → t11b → t12 → t13 → t14. All of Phase B touches
  `Agent` instance state; running two of them at once produces conflicting diffs.
* t12 and t13 are multi-commit by design; each commit still satisfies §4.

---

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A missed re-export breaks a `from agent import X` caller (84 files import `Agent`) | Guard test from t01; update both frozensets in the same commit |
| A monkeypatch silently stops applying after a move | §3 table + `live()` for every listed read |
| A moved unit reads `logger` and raises `NameError` | §2 constraint 4: every new module defines its own logger |
| A constant's readers span several steps (the 3823–3931 block) | §3 constants table: split per step, not as one block |
| `__file__`-derived path silently points at `agent_core/` | §3 root-path rule: two-level form as in `agent_core/constants.py:11` |
| A unit belongs to no step (47 did) | §6 ranges are **verified exhaustive**: of 162 units (module functions + methods), 152 fall in exactly one step, **0** overlap, and the 10 remaining are all named in the §6 "stay on `Agent`" list. Re-checked by script this session. |
| Circular import `agent_core.*` → `agent` | `import agent` only inside `live()`, never at module level |
| Import-time side effects reorder (OpenBLAS, UTF-8 stdout, `_SHELL_NAME`) | Side effects stay in `bootstrap.py`; `_SHELL_NAME` evaluated once |
| A step changes behaviour while "cleaning up" | Pure-move rule in §4 item 1; behaviour-preserving is the acceptance bar |
| Diff too large to review | t12/t13 split per cluster; one concern per commit |
| `agent.py` grows instead of shrinking | §4 item 10: a growing step copied code |

---

## Tasks

- [t03] Extract history compaction (agent.py 4694-5100) plus constants _MAX_CHAT_MESSAGES, _HISTORY_CHAR_BUDGET, _HISTORY_TRIM_NOTE, _HISTORY_TRIM_GOAL_CAP, _GOAL_TOPIC_STOPWORDS to agent_core/history_compaction.py; create agent_core/runtime_globals.py live() accessor; re-export every name and update both guard frozensets (role: implementer)
- [t04] Extract turn telemetry (agent.py 3934-4045, 4086-4104, 4147-4260) plus _INCOMPLETE_MARKERS, _COMPLETE_MARKERS, _ACTION_VERB_RE, _QUESTION_LEAD_RE to agent_core/turn_telemetry.py, routing load_agent_settings, TURN_LOG_PATH and _resolve_display_mode reads through live(); define module logger (role: implementer)
- [t05] Extract shell and subprocess helpers (agent.py 164-176, 4056-4083, 4107-4144, 4263-4297, 4433-4520) to agent_core/shell_intel.py, keeping the _DANGEROUS_SHELL_PATTERNS alias in agent.py; t09 must skip 164-176 (role: implementer)
- [t06] Extract pytest command policy (agent.py 4532-4691 plus _ENV_FILE_PATH, _PYTEST_FULL_SUITE_TIMEOUT_KEY, _PYTEST_LAST_FULL_RUN_KEY, _FULL_SUITE_MARGIN, _PYTEST_BARE_TOKENS, _PYTEST_VALUE_FLAGS, _PYTEST_SUBSET_FLAGS) to agent_core/pytest_cmd_policy.py; rewrite _ENV_FILE_PATH with the two-level __file__ form; route _ENV_FILE_PATH through live() (role: implementer)
- [t07] Extract _SHELL_NAME (4302) and _SYSTEM_PROMPT (4309-4430) to agent_core/prompts/system_prompt.py, importing _detect_shell and _git_branch from shell_intel; keep the rendered prompt byte-identical; re-export both (role: implementer) deps: t05
- [t08] Extract class LLMClient (agent.py 202-377) to agent_core/llm/client.py, routing resolve_model and load_agent_settings through live(), defining a module logger, with re-export (role: implementer)
- [t09] Extract signal handling and interruptible input (agent.py 81-161, 179-199, skipping 164-176) to agent_core/cli/signals.py, moving _SHUTDOWN_FLAG, _shutdown_save_memory, _shutdown_trace_close and _input_executor with their mutators; route AGENT_MEMORY_JSON_PATH through live() (role: implementer)
- [t10] Extract the REPL layer (agent.py 5122-5409) to agent_core/cli/registry_setup.py and agent_core/cli/interactive.py, leaving the 5101-5121 agent_dashboard re-export block in place and keeping __main__ behaviour byte-identical (role: implementer) deps: t08, t09
- [t11] Extract legacy _tool_* aliases and file operations (agent.py 2221-2408, 170 method-lines) to agent_core/agent_files.py, keeping one-line delegators on Agent (role: implementer) deps: t03, t04, t05, t06, t07, t08, t09, t10
- [t11b] Extract the previously unassigned file-effect helpers _resolve_nlp_path, _save_verify_note, _verify_file, _note_effect, _take_trace_effects (agent.py 629-696) to agent_core/agent_effects.py with delegators (role: implementer) deps: t11
- [t12] Extract the _nlp_* tool handlers (agent.py 749-1888, 1039 method-lines) to agent_core/handlers/nlp_tools.py in six per-cluster commits, keeping _nlp_tool_handlers, _execute_tool_call, _run_command_quietly and _effective_ws_dir on Agent; move the eight t12 constants; route patched reads through live() (role: implementer) deps: t11b
- [t13] Extract plan tools to agent_core/handlers/plan_tools.py, prompt context blocks (3235-3538 plus _refresh_system_message and _append_user_turn) to agent_core/prompts/context_blocks.py, and chat/memory persistence (3544-3820 plus check_stale_files, invalidate_stale, memory_stats, clear_history) to agent_core/memory/persistence.py; route both JSON paths through live() (role: implementer) deps: t03, t12
- [t14] Extract the turn loop (agent.py 2504-3161) to agent_core/turn_loop.py, moving _MAX_CHAINED_RUNS, _LLM_ERROR_MAX_RETRIES, _LLM_ERROR_BACKOFF_S, _CONTINUE_NOTE_TAG, _CONTINUE_NOTE and _VERIFY_NOTE, and routing _MAX_CHAINED_RUNS, _looks_incomplete, record_turn_outcome, recommend_profile, _resolve_display_mode and ToolLoopRunner through live() (role: implementer) deps: t03, t04, t05, t06, t07, t08, t09, t10, t11, t11b, t12, t13
- [t15] Final consolidation: prune dead imports, run the lint guard, one full test run, update Architecture.md and AGENTS.md module map and docs/refactor_agent_py.md (role: integrator) deps: t03, t04, t05, t06, t07, t08, t09, t10, t11, t11b, t12, t13, t14
