# Plan: Cleaner Agent Output to the End User

## Problem
The REPL output shown to the end user during natural-language (NLP) agent turns is noisy and hard to follow:

1. **Unexplained tool calls** — every invocation is printed as `[tool] name(args)` with no context about *why* it was chosen, so a reader sees a string of bare calls without reasoning.
2. **Cut-off / meaningless result dumps** — each result is truncated to `_RESULT_DISPLAY_LIMIT` (1500 chars) and dumped verbatim (`[result] ...`). For `read`/`search`, the truncation looks "cut off" even though the model receives the full payload; for `run`/`git`/`tests`, raw stdout/stderr is shown with little framing.
3. **No control over verbosity** — there is no existing flag to reduce or suppress tool-call display, so users can't opt into a cleaner view.

## Verified facts (current code)
- Display path: `agent_core/llm/tool_loop.py` → `ToolLoopRunner.run()` prints `[tool] name(args)` and `[result] ...` via plain `print()`. Final answer printed in `agent.py` → `chat_nlp()`.
- `_RESULT_DISPLAY_LIMIT = 1500` bounds what the human sees; the model always gets the full result.
- No existing display/verbosity toggle exists in config.py, constants.py, cli/session, or commands/registry. (Confirmed by search.)

## Goals (ordered)
G1. Explain reasoning — give the user a short natural-language note for each tool call (what/why), so calls aren't bare.
G2. Cleaner result framing — summarize truncated results instead of dumping raw tail; keep full payload for the model unchanged.
G3. Optional less display — let users opt into a compact/summary-only mode that hides or condenses `[tool]`/`[result]` lines.

## Design (minimal, non-breaking)

### Layer A: Reasoning explanation per tool call
- Add an optional `reasoning` field to the assistant message emitted by the LLM wrapper (`agent.py chat_nlp`'s `llm_chat_fn`). The provider already returns JSON; if it contains a top-level reasoning text, surface it.
- In `ToolLoopRunner.run`, when printing `[tool] name(args)`, prepend a 1-line human-readable reason:
  ```
  [reason] read agent.py — to find chat_nlp's entry point (already searched for the symbol)
  [tool]   read(path=agent.py)
  ```
- Reason line is derived from: (a) model-supplied reasoning if present, else (b) a small heuristic built from the prior assistant text + tool name. This keeps output meaningful even when the model omits explicit reasoning.

### Layer B: Result framing / summarization for display only
- Keep `_RESULT_DISPLAY_LIMIT` feeding the *model* unchanged — never truncate what the LLM sees.
- Add a display-only formatter that, instead of dumping `result_str[:1500]`, produces:
  - For read/search/list_files: "N lines returned; showing first/last snippet" with a clear continuation hint (reuse existing `[truncated … offset=]` style).
  - For run/git/diff/tests: one-line summary + truncated stdout tail, labeled clearly.
  - For write/edit/fix/analyze: keep the verification line (`[verify] py_compile ✓`) prominent; fold long LLM output into a short digest.
- Gate this behind a display mode so existing behavior can be preserved as "verbose".

### Layer C: Display-mode toggle (opt-in cleaner view)
- Introduce `AgentDisplayMode` enum: `VERBOSE` (current behavior), `CLEAN` (reason + summarized results, no raw dumps), `QUIET` (only the final answer; tool calls/results hidden).
- Source it from a new env var `AGENT_DISPLAY_MODE=CLEAN|QUIET|VERBOSE` (default VERBOSE = current behavior, so nothing changes for existing users/tests) and/or a REPL command `display <mode>` that persists to `.env`.
- Wire the mode into:
  - `ToolLoopRunner.run` — accept `display_mode`; decide whether to print `[tool]`, `[result]`, and how to format results.
  - `agent.py chat_nlp` — pass the agent's display mode to each chained run; honor QUIET by not printing intermediate narration.

### Layer D: Final answer hygiene (already partially done)
- `chat_nlp` already strips `</?tool_call>`/`</?function_call>` tags and prints only the last non-empty assistant text. Extend it to also fold any stray "I will ..." narration into a short preamble when mode is CLEAN, so the final report reads as one coherent answer (what changed / where / verification evidence).

## Files touched
- `agent_core/llm/tool_loop.py` — display-mode param; reason line + summarized result formatting; `_fmt_args` stays. Tests already assert on *history messages*, not stdout prints, so adding a mode param with default VERBOSE keeps them green.
- `agent.py` — `chat_nlp` reads display mode from env/command and passes it to `ToolLoopRunner`; final-answer cleanup for CLEAN/QUIET; new `display <mode>` command stub (reuse CommandRegistry pattern).
- `agent_core/config.py` — add `display_mode: AgentDisplayMode` field + env mapping (`AGENT_DISPLAY_MODE`).
- `.env.example` — document the new var.
- docs/README/USAGE.md — one short section on the display modes.

## Test strategy (keep existing green, add coverage)
- Existing tests assert message *history* contents and `final_text`, not stdout — so default VERBOSE mode preserves contracts; no test changes required for G1–G3 defaults.
- Add new tests: (a) CLEAN mode prints a `[reason]` line before each `[tool]`; (b) QUIET mode suppresses `[result]/[tool]` stdout while still returning full results to the model + same final_text/history; (c) result framing summarizes read output ("N lines") without losing continuation hint.

## Rollout order
1. Layer C toggle + config wiring (default VERBOSE → zero behavior change).
2. Layer A reasoning line (enabled in CLEAN).
3. Layer B summarized results (enabled in CLEAN; QUIET hides entirely).
4. Layer D final-answer cleanup for CLEAN/QUIET.
5. Docs + `.env.example` + new tests.

## Non-goals / constraints respected
- Never truncate what the LLM receives — display-only summarization only.
- Keep `write`/`edit`/`fix` verification (`[verify] py_compile`) visible in all modes (it's end-user evidence).
- Preserve steering notes' "do not leak into persisted history" invariant (notes stay system messages, stripped before save) regardless of mode.
