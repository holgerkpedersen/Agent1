## 2026-09-25 - feat: Jev prompts carry the current local time (time of day)

**Change**: `agent_core/jev_engine.py` — `_now_context()` / `_time_of_day()` prepend a CONTEXT line to every Jev prompt (`_build_messages`): `current local date and time is 2026-09-25 23:52 (Friday, UTC+0200); time of day: night`. The named bucket (morning/afternoon/evening/night) is handed to the model so a 1.5B need not derive it from a 24-hour clock.

**Reason**: `jev yesno "is it evening?"` had no clock in its prompt, so time-dependent questions could only be guessed or abstained; the agent already exposes `get_current_datetime`, so the same fact is now available to every Jev call automatically.

**Files**: agent_core/jev_engine.py, tests/test_jev_engine.py. Verified live at 23:52 local: `is it night?` -> P(yes)=0.76 TRUE, `is it morning?` -> P(yes)=0.22 FALSE, `is it evening?` -> P(yes)=0.54 (borderline, correct for late night). Tests: 51 engine tests green (new: time injected, time-of-day buckets).

## 2026-09-25 - feat: speculate agreement gate, mandatory citations, calibrated threshold

**Change**: repo-question answers now require a `file:line` citation (evidence), and `--agree N` (default 2) requires N independent branches to cite the SAME evidence files before COMMIT (agreement signature = cited basenames; `--agree 1` disables). `--threshold auto` uses the threshold fitted from labeled outcomes (`harnessfix.jev_telemetry.load_suggested_threshold`, minimum 10 labels), and the new `jev label <id> correct|incorrect [--note ...]` command records those outcomes. The header shows `agree=` and the resolved threshold.

**Reason**: the total solution for meaningful repo changes — corroboration (independent branches agreeing on evidence), evidence (mandatory citations), verification (existing guards) and calibration (measured threshold) — so a single confident branch cannot be committed on an unvalidated gate.

**Files**: agent_core/commands/speculate_cmd.py, agent_core/commands/jev_cmd.py, harnessfix/jev_telemetry.py, tests/test_speculate_cmd.py, tests/test_jev_cmd.py. Verified live: `threshold=auto -> 0.5 (calibrated from 12 labeled decisions, accuracy 1.00)`; agreement REFUSE on split evidence; citation REFUSE on uncited repo answers. Tests: 39 speculate + 23 jev-cmd green.

## 2026-09-25 - feat: relevant grounding + cited-line verification in speculate

**Change**: `_question_subject_terms()` extracts distinctive subject terms from a question (path stems + 3+/4+ char non-stopwords, hyphen-split); `_grounded_in_subject()` requires a repo-question branch's executed tool EVIDENCE to mention at least one such term — calling *any* tool no longer satisfies grounding (a `web_search` used to count). `_verify_claims()` now also checks CONTENT: when the answer names a backticked symbol next to a `file:line` citation, that symbol must appear within ±3 lines of the cited line. The branch prompt asks for `path:line` citations.

**Reason**: the 22:04 run answered a repo question wrongly (claimed `--judge jev` uses Jev as a co-thinker on branches and proposed a redundant `--gate` flag) yet passed grounding + claim verification and was scored 0.95. The guards checked form, not meaning; these two deterministic levers close the cheap part of that gap.

**Files**: agent_core/commands/speculate_cmd.py, tests/test_speculate_cmd.py. Verified live: irrelevant tool evidence → REFUSE `without tool evidence about the subject`; a real file cited at a wrong line with a symbol name → REFUSE `speculate_cmd.py:300 does not mention branch_llm` (even with a 0.9 judge score); a grounded, claim-free answer still COMMITs. Tests: 36 speculate tests green.

## 2026-09-25 - fix: --judge jev keeps chat branches; --branch-model jev for pure Jev

**Change**: new `speculate --branch-model chat|jev` (default `chat`). `--judge jev` now scores with the dedicated small model while the reasoning model generates the branches — the smart split (big model thinks, small model decides). `--branch-model jev` runs the branches on the Jev model too (cheap/offline). The header prints `branch_model=chat|jev`; the Jev engine is built when either role needs it and degrades gracefully per role.

**Reason**: making `--judge jev` run the branches on a 1.5B made the small model *think* — it fabricated tool output and could not ground a repo question, so every repo question REFUSEd (correctly, but uselessly). Jev's value is the decision, not generation.

**Files**: agent_core/commands/speculate_cmd.py, tests/test_speculate_cmd.py. Verified: 32 speculate tests green (new: `--judge jev` uses chat branches + Jev judge; pure-Jev via `--branch-model jev`).

## 2026-09-25 - feat: speculate grounding guard + deterministic claim verification

**Change**: `agent_core/commands/speculate_cmd.py` — (1) `_is_repo_question()` detects workspace/code questions (path tokens or repo hints); when true, `require_grounding` (default ON, `--no-grounding` disables) makes every branch call at least one tool before its answer is a candidate — an ungrounded branch is failed with `repo question answered without calling any tool`. (2) `_verify_claims()` checks every `file[:line]` claim in a COMMIT candidate against the real workspace before printing it (missing file, line past EOF; bare basenames are resolved anywhere in the workspace) and REFUSEs with the mismatch list instead of committing. The header now shows `grounding=on|off`.

**Reason**: a live run answered a repo question from memory with a confident but stale/incorrect description of `speculate_cmd.py`, and the same-model LLM judge scored it 1.00 → COMMIT. Ungrounded code claims and nonexistent files must not be committed as verified fact. Both guards are deterministic and cost no judge call.

**Files**: agent_core/commands/speculate_cmd.py, tests/test_speculate_cmd.py. Verified live: `grounding=on` on a repo question → REFUSE (pure-Jev branches did not ground); with `--no-grounding` the answer cited `project/main.py` → REFUSE `unverified file/line claim(s): project/main.py does not exist`. Tests: 30 speculate tests green (new: repo-question detection, grounding required/allowed, verified-claim commit, unverified-claim refusal, claim checker incl. bare basenames).

## 2026-09-25 - fix: speculate rejects fabricated tool output

**Change**: `agent_core/commands/speculate_cmd.py` gains `_looks_like_fabricated_output()` — pseudo-tool XML with attributes (`<definitions path="…">`, `<references symbol="…">`, `<search query="…">`, `<web_search …>`) and placeholder markers (`path/to/`, `example.com`, `your_file`) are non-answers. It is applied to every branch (a fabricating branch is failed, never a candidate) AND in the scorer (score 0.0), so a fabricated answer can never be COMMITted even if a judge rates it highly. The branch prompt now explicitly forbids inventing tool output.

**Reason**: live pure-Jev run (2026-09-25): qwen2.5-coder-1.5b wrote `<definitions path="path/to/jev_integration.py">`, `<references …>`, `<search …>` blocks with placeholder paths, called NO tool, and its own 1.5B judge scored the hallucination 0.83 — it was COMMITted as the answer.

**Files**: agent_core/commands/speculate_cmd.py, tests/test_speculate_cmd.py. Verified: 23 speculate tests green (new: fabrication detector + `test_speculate_rejects_fabricated_branch_answer` + `test_speculate_jev_mode_rejects_fabricated_answer`); live re-run produces a prose answer, no fabricated XML, and the selected chat model is still never called.

## 2026-09-25 - feat: speculate --judge jev runs the whole command on the Jev model

**Change**: `speculate --judge jev` now uses the dedicated Jev model for the BRANCHES as well as the judge — the selected chat model is not touched at all. `branch_llm` is the Jev engine's pinned provider; branch calls pass an explicit `max_tokens=1024` so the Jev profile's 512-token sample cap cannot truncate them; `JevEngine.ensure_ready()` (new public wrapper) selects/loads the small model BEFORE the branches run; the header prints `branch_model=<name>, jev_model=<name>, low=0.3, escalate=off`. The gray-band escalation to the chat-model judge is now opt-in via `--escalate` (default off), and `--judge both` still keeps chat-model branches and averages the two judges.

**Reason**: `--judge jev` still dispatched its three branches on the selected chat model (the 27B), so a Jev-specific command visibly ran on llama/27B and could time out there — the opposite of "the Jev model is used every time, independent of the currently selected model". The user's requirement is now enforced: a Jev command never calls the chat model unless `--escalate` is given.

**Files**: agent_core/commands/speculate_cmd.py, agent_core/jev_engine.py, tests/test_speculate_cmd.py. Verified live: ran `--judge jev` with an agent whose chat model RAISES if called — header `branch_model=qwen2.5-coder-1.5b-instruct, jev_model=qwen2.5-coder-1.5b-instruct, escalate=off`, no exception, `COMMIT score=0.88`. Tests: 20 speculate tests green (new: pure-Jev runs everything on Jev, `--escalate` gray-band, gray-band stays pure without it).

## 2026-09-25 - fix: failover tests must not clobber the real model.json/.env

**Change**: `tests/test_failover_chain.py` and `tests/test_failover_provider.py` gain an autouse `_no_model_persist` fixture that stubs `agent_core.constants.persist_model_choice`; `test_model_helpers.py::test_save_and_load_roundtrip` and `test_opencode_provider.py::test_persist_model_choice_infers_provider` now sandbox `MODEL_JSON_PATH`/cwd instead of touching the real files; new `tests/test_model_state_isolation.py` proves a real failover persists only in the sandbox and leaves the real `model.json` byte-for-byte unchanged.

**Reason**: `FailoverProvider.chat` persists the model that actually answered (so the next turn starts on it). A failover test whose second stub was named `go` therefore rewrote the developer's real `model.json` and `.env` (`AGENT_MODEL=go`), silently changing the session's selected model.

**Files**: tests/test_failover_chain.py, tests/test_failover_provider.py, tests/test_model_helpers.py, tests/test_opencode_provider.py, tests/test_model_state_isolation.py (new). Verified: the previously-polluting batch (119 tests) leaves `AGENT_MODEL`/model.json unchanged; new isolation test green.

## 2026-09-25 - feat: model jev setup, visible judge model, longer speculate timeout

**Change**: `model jev [<name>]` (new subcommand of `model`) shows the dedicated Jev model with its resolved provider and LM Studio status, and `model jev <name>` persists it (`persist_jev_model` -> model.json `jev_model` + .env `AGENT_JEV_MODEL`). `load_agent_settings` now reads model.json's `jev_model` as a tier between env/.env and the catalog default, so a one-time choice survives restarts. `speculate` prints the judge model in its header (`judge=jev, jev_model=<name>, low=0.3`) and its default branch-dispatch timeout is 300s -> 600s.

**Reason**: the Jev model was already independent of the chat model (`build_jev_engine` + `single=True`), but it was invisible and not user-settable. A `speculate --judge jev` run on a local 27B timed out in the BRANCH phase at 300s, so the judge never ran and the only model on screen was the branch model — which reads as "Jev is running on the default model". Setup is now explicit, the judge model is visible before any branch runs, and the default timeout fits a local reasoning model.

**Files**: agent_core/commands/model_cmd.py, agent_core/constants.py, agent_core/config.py, agent_core/commands/speculate_cmd.py, tests/test_model_jev.py (new), tests/test_speculate_cmd.py. Verified: 6 new tests + 19 speculate tests green; `model jev` live shows `qwen2.5-coder-1.5b-instruct (provider=LMStudioProvider), loaded`.

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
