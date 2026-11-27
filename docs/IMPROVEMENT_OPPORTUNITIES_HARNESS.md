# Improvement Opportunities — Agent1 Harness (2026)

Card `e50f65c601304c86afaf279e06e0ec2e`
Issue `iss-general-kanban_e50f65c601304c86afaf279e06e0ec2e`
Column: Explorations — "Check for improvement opportunities for Agent1"

Method: `web_search` for harness / loop-engineering material, then every
candidate checked against this repo's real code. Article advice that Agent1
already implements is listed first, so the gap list is the only actionable part.

## Sources consulted

Treated as untrusted external input; used for framing only, not as evidence
about this codebase.

- Agent harness / scaffolding overviews: datacamp, Microsoft Learn
  (agent-framework harness), Wikipedia "Agent harness", amux.io guide,
  datanorth guide, dev.to "build a harness from scratch".
- Loop engineering / agentic design patterns: datasciencedojo "10 loop
  engineering design patterns" (ReAct → circuit breakers), dagworks "Agentic
  Design Pattern #1: Tool Calling", hussamahmed "Agentic Design Patterns",
  Temporal cookbook "durable agentic loop".
- Context management: Microsoft Learn "Compaction", langchain "Context
  Management for Deep Agents", agentnative "Context Compaction Pattern",
  zylos "Agent Context Compaction for Long-Running Sessions", codazz "Context
  Window Management for Long-Running Agents".

## Already implemented — do not re-propose

| Article theme | Where Agent1 already does it |
|---|---|
| Context compaction / token-starved loops | `llm/tool_loop.py:838-874` — in-run compaction, `off`/`note`/`summarize`, char budget derived from the model's REAL window via `char_budget_for_limit` |
| Forced final answer instead of endless looping | `llm/tool_loop.py:2115` `_forced_final_answer`, deadline warning at `:1232-1239`, `termination_reason` in `answer/cap/stuck/no_progress` (`:903-907`) |
| Circuit breaker on repeated identical calls | `_seen_progress_keys` + `no_mutation_limit` / `force_after_no_mutation` (`llm/tool_loop.py:875-880`) |
| Verification before claiming success | `turn_loop.py:385-410` claim-vs-evidence guard; `_action_request_without_evidence` (`:162-180`); Jev delivery check (`:181+`) |
| Fail-fast over silent degradation | `llm/lmstudio.py:950-977` `_classify_tool_path_failure` + pre-check `:1057-1065` |
| Oversized-payload retry bounding | `llm/lmstudio.py:1074-1076` (`> 200_000` bytes → `max_retries=1`) |
| Token accounting | `llm/tool_loop.py:1342` accumulate, emitted at `:1097-1104` |

## Verified gaps (each with evidence)

### 1. Tool-path failure memo dies with the process — HIGH value, LOW risk

`self._tools_unsupported` is a plain dict created at `lmstudio.py:630`. Read by
`_tools_blocked` (`:913-933`), written by `_remember_tools_unsupported`
(`:936-948`), applied at `:1057-1065` and `:950-977`.

It already distinguishes a deterministic peg-grammar failure (permanent,
`None` deadline) from a transient engine crash (TTL via
`TOOLS_UNSUPPORTED_TTL_S`). The logic is right; only the lifetime is wrong.

Effect: within one session the model is remembered and later tool turns fail
fast. In a NEW session the harness re-discovers the crash the expensive way —
a full doomed tool request (prefill included) before failover. That is exactly
the wasted work the fail-fast path exists to avoid, across a process boundary.

Fix sketch: persist the dict next to the session state, honouring the existing
permanent-vs-TTL split. The repo already has the idiom —
`llm/learning.py:93` `_state_path()` → `current_paths().meta_policy_path`, with
an atomic `tmp` + `os.replace` save (`:131-151`, `save_weights`) that never
raises. Reuse that
shape; store a wall-clock deadline (not `time.monotonic`, which is
process-relative) for timed entries and drop expired ones on load.

### 2. The memo is keyed by model name only — MEDIUM value, LOW risk

Key and value are set from `self.model_name` (`:967`) and documented that way
(`:1030-1047`), but the blocking decision is a property of the model's TOOL
GRAMMAR in this engine build, not of a prompt profile. So one bad profile bans
that model for all tool use, including profiles that never hit the crash.

Fix sketch: key on `(model_name, tool_grammar_signature)` where available, or
store the profile alongside the model and treat a hit from a different profile
as advisory-only. Low priority until the same model is used under two profiles.

### 3. No wall-clock cap on a turn — MEDIUM value, MEDIUM risk

`deadline_window` sounds temporal but counts ITERATIONS (`:875`, compared at
`:1233`). The only timer is the per-call provider timeout,
`_MAX_SCALED_TIMEOUT_S = 3600` (`lmstudio.py:341`, `:911`). With
`max_iterations=150` the theoretical worst case for one turn is very large, and
nothing bounds the SUM of iterations.

Fix sketch: an optional per-turn wall-clock budget checked at the top of each
iteration, feeding the same forced-final-answer path as
`deadline_window` so the turn still ends in a text answer. Risk is medium
because a legitimate long refactor must not be cut off — default it off or
generously sized, and log the termination reason as a new value (e.g.
`"time_budget"`) so the outcome stays observable.

### 4. Token usage is measured but never enforced — LOW value now

Counts accumulate at `:1342` and are read once for the end-of-run event
(`:1097-1104`). No per-run token budget steers behaviour. This only matters
once local context/throughput cost becomes a constraint; note it, do not build
it yet.

## Recommended order

1. Gap 1 — certain benefit, small diff, existing persistence idiom to copy.
2. Gap 3 — decide deliberately rather than by accident; touches the loop's exit
   conditions, so it needs its own tests.
3. Gap 2, then Gap 4 — cheap to defer.

Every fix needs a regression test. Gap 1's test must assert that a NEW provider
instance (prefer: new process-level state load) refuses a tool-using call
WITHOUT issuing a request, and that an expired TTL entry is retried.
