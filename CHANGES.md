## 2026-09-24 - fix: size the LM Studio socket timeout to the prompt prefill

**Change**: `agent_core/llm/lmstudio.py` — `_scaled_timeout()` now estimates the prompt tokens (`_estimate_prompt_tokens`: ~3.5 chars/token over messages + tool-call arguments + tool schemas) and sets the socket timeout to `max(floor, tokens / rate * 1.5 + 60)`, capped at 3600s, where `rate` is `LMSTUDIO_PREFILL_TOKENS_PER_SEC` (default 15, deliberately pessimistic). The streaming path uses the same sizing. The previous rule added ~1s per 50 KB — roughly 400x too small.

**Reason**: LM Studio emits nothing until prompt processing finishes, so the client's read (socket-inactivity) timeout must exceed the WHOLE prefill. A ~20k-token prefill (a 78-message conversation) was granted ~601s, and LM Studio logged "Client disconnected. Stopping generation..." at ~602s, returning an empty completion. The bytes rule under-sized it ~400x; the token estimate gives ~2060s for that prompt.

**Files**: agent_core/llm/lmstudio.py, tests/test_lmstudio_payload.py. Verified: `test_lmstudio_payload.py` (23), `test_anti_stuck_guard.py` (11), `test_llm_retry_policy.py` green.

## 2026-09-24 - fix: speculate never commits a leaked tool call

**Change**: `agent_core/commands/speculate_cmd.py` — new `_looks_like_tool_call()` detects a model's tool-call syntax leaking through as plain text (`<|tool_call>…`, `call:name{…}`, bare `{"tool_calls": …}`). A branch that ends on such text is failed (`{"error": …}`) so it is never a candidate; the judge rubric now includes the question and scores a non-answer (empty or tool-call-shaped) 0.0 without calling the model; the branch prompt names the read-only tools and asks for a prose answer.

**Reason**: A gemma branch emitted `<|tool_call>call:run{command:"pytest agent_core/tests/"}<tool_call|>` as text; the branch returned it as the "answer", the judge rated it 1.00, and it was COMMITted. The read-only allowlist only guards *structured* calls, so the text path bypassed the whole branch.

**Files**: agent_core/commands/speculate_cmd.py, tests/test_speculate_cmd.py. Verified: `tests/test_speculate_cmd.py` (10 tests) green.

## 2026-09-24 - feat: speculative branches use the agent system prompt + a read-only tool loop

**Change**: `agent_core/commands/speculate_cmd.py` — every `speculate` branch now sends the agent's live system prompt (`_chat_history[0]` after `_refresh_system_message`) as a leading system message and runs a short read-only tool loop (up to 4 model round-trips) with the read-only tool schemas (`search`/`read`/`list_files`/`definitions`/`references`/`web_search`). A hard allowlist (`_BRANCH_TOOLS`) refuses any other tool name before the executor, so parallel branches cannot mutate the workspace. Default branch-dispatch timeout 60s -> 300s.

**Reason**: Branches previously sent only a user message, so they answered as a generic assistant — no persona, no environment awareness, and no way to check anything (e.g. "Is ruff installed?" was REFUSEd because the branch could only guess). The system prompt gives persona/environment; the read-only tool loop lets a branch ground its answer from the workspace without risking concurrent mutations from parallel branches.

**Files**: agent_core/commands/speculate_cmd.py, tests/test_speculate_cmd.py. Verified: `tests/test_speculate_cmd.py` (8 tests) green.

## 2026-09-24 - fix: honour explicit read pages on large .py files

**Change**: `agent.py::_nlp_read` — the AST strategy (return a `definitions` summary for `.py` files over `_CONTEXT_AST_THRESHOLD_KB`, default 50 KB) now fires only for a BARE read (no `offset`/`limit`); an explicit page always returns the requested source lines, and the summary appends a how-to-page note. `agent_core/tool_schemas.py` — the `run.timeout` description now documents that a whole-suite `python -m pytest` run ignores `timeout` and uses `PYTEST_FULL_SUITE_TIMEOUT`.

**Reason**: The AST branch ignored `offset`, so every `read(agent.py, offset=N)` returned the identical definitions list; the model paged to fresh offsets, saw no new content, and looped forever. The model also kept guessing a `timeout` for full pytest runs even though the configured suite budget overrides it.

**Files**: agent.py, agent_core/tool_schemas.py, tests/test_agent_robustness.py. Verified: `tests/test_agent_robustness.py` green.

## 2026-09-24 - fix: remove dead duplicate modules and orphaned prototype

**Change**: `agent_core/commands/_implement_raw.py` deleted (dead prototype, unregistered command, helpers superseded by `implement_cmd.py`/`fix_cmd.py`). `docs/AGENTIC_IMPROVEMENT_PLAN.md` updated to reflect removal of dead module and progress on item #17. `CHANGES.md` (this entry).

**Reason**: `agent_core/commands/_implement_raw.py` was an orphaned prototype that duplicated live helpers with divergent signatures (`file_needs_generation` vs `implement_cmd.file_needs_generation`). Its command name (`implement-raw`) was never registered in the registry, making it unreachable from the REPL. It provided no value and increased technical debt by providing confusingly similar but non-functional code paths for future developers.

**Files**: agent_core/commands/_implement_raw.py, docs/AGENTIC_IMPROVEMENT_PLAN.md, CHANGES.md. Verified: `tests/test_dead_implement_raw.py` passes (2 tests) ensuring no remaining references exist in core directories. Full suite green.
