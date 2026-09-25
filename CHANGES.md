## 2026-09-25 - feat: Jev commands auto-select the Jev model

**Change**: `LMStudioProvider.ensure_model_loaded()` (new) + `JevEngine._ensure_ready()`: before the first request, a Jev engine ensures its dedicated model is loaded in LM Studio — skipping when already loaded, loading it via the management API otherwise, and deferring to the normal request path when LM Studio is unreachable. Runs once per engine, via `asyncio.to_thread` (never blocks the loop); a failed load prints `[jev] could not select model <name>: <reason>` and falls through. Wired for `jev`, `jev_decide` and `speculate --judge jev` (all go through `JevEngine.decide`).

**Reason**: the `model` command loads/unloads models, so switching the main model can evict the small Jev model from VRAM; Jev-specific commands relied on LM Studio JIT / a 400 auto-load round-trip and could fail with an opaque abstention. The dedicated model is now re-selected automatically and deterministically.

**Files**: agent_core/llm/lmstudio.py, agent_core/jev_engine.py, tests/test_lmstudio_payload.py, tests/test_jev_engine.py. Verified live: unloaded the model, ran a Jev decision — it loaded the model (8.4s) and returned `P(yes)=0.87 -> TRUE`; model loaded afterwards.

## 2026-09-25 - feat: Jev telemetry + calibration loop (jev stats)

**Change**: New `harnessfix/jev_telemetry.py` — every `JevEngine.decide` appends one record (id/ts/source/model/kind/mechanism/question/question_hash/probabilities/decision/confidence/threshold/n/abstentions) to `reports/history/jev.jsonl`, the same gitignored tree as the execution ledger. Recording is best-effort (a broken ledger never breaks a decision) and opt-out via `AGENT_NO_JEV_LOG=1`. `record_outcome(id, 'correct'|'incorrect')` labels a decision (atomic rewrite via `.tmp` + `os.replace`); `calibration_bins`/`suggest_threshold`/`summarize`/`format_report` turn labeled records into a reliability report and the threshold that best separates correct from incorrect yesno decisions. `jev stats [--last N] [--json]` is the REPL surface. `speculate --judge jev` also gained the gray-band cascade (`--low`, default 0.3): `P >= threshold` accepts, `P <= low` rejects, only `low < P < threshold` escalates that candidate to the LLM judge (max of the two) — a 1.5B judge sitting just under 0.70 used to REFUSE good answers.

**Reason**: every Jev threshold (0.7 accept / 0.3 reject) was a hardcoded guess and `harnessfix/` had no record of any prediction, so the engine could not be audited or calibrated — and the live margin between a good answer (P=0.78) and a mediocre one (P=0.67) around the 0.70 gate is only 0.11. Measuring outcomes first is the prerequisite for trusting (and tuning) the engine; the cascade is the first consumer of the gray-band idea.

**Files**: harnessfix/jev_telemetry.py (new), agent_core/jev_engine.py, agent_core/commands/jev_cmd.py, agent_core/commands/speculate_cmd.py, agent.py, tests/test_jev_telemetry.py (new), tests/test_jev_cmd.py, tests/test_jev_engine.py, tests/test_speculate_cmd.py. Verified: 97 focused tests green; live loop: 3 real decisions recorded, outcome labeled, `jev stats` reports calibration + suggested threshold.

## 2026-09-25 - feat: Jev decision engine (typed probabilistic answers from a dedicated small model)

**Change**: New `agent_core/jev_engine.py` — a Jev-style decision engine that turns a state plus a TYPED question into a typed, probabilistic answer: `yesno` (P(yes)/P(no) -> TRUE/FALSE/UNKNOWN), `choice` (distribution over fixed options -> argmax/UNDECIDED) and `score` (a 0-100 value plus spread). Two mechanisms: `logprobs` (one call; reads the model's own token mass via the new `LMStudioProvider.chat_logprobs`) and `vote` (N samples; agreement IS the probability), with `auto` trying logprobs then falling back to voting. Provider errors, timeouts, empty output, unparseable tokens and leaked tool calls are ABSTENTIONS (majority -> UNKNOWN), never fabricated votes. It builds its OWN provider from `settings.jev_model` (default `qwen2.5-coder-1.5b-instruct`, catalog `_defaults.jev_model`; env `AGENT_JEV_MODEL`) and never touches the agent's main LLM. New settings: `jev_model`/`jev_provider`/`jev_samples`/`jev_temperature`/`jev_max_tokens`/`jev_timeout` (validated). New REPL `jev` command (`agent_core/commands/jev_cmd.py`), a read-only `jev_decide` NLP tool (added to `PLAN_MODE_TOOLS`), and `speculate --judge llm|jev|both` (Jev scores each candidate; `both` averages).

**Reason**: The `speculate` judge was a single subjective LLM score — it once rated a leaked tool call 1.0 — and there was no way to get calibrated, typed answers to small decision questions. A dedicated 4B-class local model answering short constrained micro-decisions is fast and cheap (Jev's System One role), and agreement across samples/logprobs is a far better confidence signal than one judge.

**Note**: model choice was verified live (2026-09-25). Rejected: `google/gemma-4-e4b` — a THINKING model that emits `reasoning_content` first; client-side reasoning-off knobs (`reasoning:"off"`, `chat_template_kwargs.enable_thinking=false`, `thinking.disabled`, `/no_think`) do NOT suppress it (it only reaches a verdict with `jev_max_tokens` >= 512, and with tools it returns a raw `{"role":"assistant",...,"reasoning_content":...}` tool-call payload, so it is also a bad speculate branch model); `llama3.2-1b` (enumerates options instead of answering); `qwen2.5-0.5b-instruct` and `reactagent-1.5b` (misjudge the "bad answer -> no" case). Chosen: `qwen2.5-coder-1.5b-instruct` — non-thinking, terse, 4/4 on the terse probe and 5/5 parsed on every Jev primitive (yesno/choice/score) through the engine. `jev_max_tokens` still defaults to 512 as a safe upper bound for any thinking fallback model.

**Fix (same day)**: `build_provider(..., single=True)` (new) + `build_jev_engine` uses it. Before, the Jev engine inherited the multi-provider FAILOVER chain (opencode-go/deepseek, openrouter, lmstudio, llama), so a Jev call silently answered with a reasoning/cloud model whenever LM Studio was unreachable — `speculate --judge jev` was not actually judging with the small model. Jev now builds ONE provider pinned to `settings.jev_model` (no failover; unreachable -> abstention, never a different model). `LMStudioProvider._announce_model` is shared by `chat` and `chat_logprobs`, so the Jev logprobs path is visible in the console too.

**Files**: agent_core/jev_engine.py (new), agent_core/commands/jev_cmd.py (new), agent_core/llm/lmstudio.py, agent_core/config.py, agent_core/constants.py, agent_core/tool_schemas.py, agent_core/modes.py, agent_core/commands/speculate_cmd.py, agent.py, agent_core/commands/help_cmd.py, model_catalog.json, tests/test_jev_engine.py, tests/test_jev_cmd.py, tests/test_lmstudio_payload.py, tests/test_speculate_cmd.py. Verified: test_jev_engine (43), test_jev_cmd (16), test_lmstudio_payload (26), test_speculate_cmd (16), test_model_catalog_required (23), test_plan_mode + test_failover_provider (49) green.

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

## 2026-09-25 02:14 — fix --mypy

**Change**: Modified `bridge.py`
**Reason**: mypy error fixes

## 2026-09-25 02:15 — fix --mypy

**Change**: Modified `bridge.py`
**Reason**: mypy error fixes

## 2026-09-25 02:23 — fix --mypy

**Change**: Modified `lmstudio.py`
**Reason**: mypy error fixes
