---
name: subagent-orchestration
description: Orchestrate work across this workspace's subagents; scale assessment, parallel dispatch of independent domains, self-contained task briefs, review and integration
when_to_use: 2+ independent tasks to fan out, or multi-step jobs that fit delegate / persistent subagents / plan tools
tags: [subagents, orchestration, delegation]
---

# Subagent Orchestration (this workspace)

Provenance: merged from obra/superpowers skills `dispatching-parallel-agents` and shinpr/claude-code-workflows' `subagents-orchestration-guide`, reworked to this repo's actual subagent tools.

## Role: You Are the Orchestrator

You own workflow decisions, routing, progress management and user interaction; named specialists own their assigned deliverables. Invoke a specialist before producing or changing code, tests, config or docs that belong to its role — don't do semantic work inline just because it's "quick".

**Delegation boundary: assign WHAT, not HOW.** Give goal, scope, constraints and expected output. Don't dictate implementation approach unless the constraint is real (e.g. "must stay in the harness layer per decision #014").

## Scale Assessment First — don't orchestrate what one pass can finish

Handoff cost is real: every dispatch costs a self-contained brief plus an integration review.

| Scale | Do this |
|-------|---------|
| Small (one focused change, < ~1 hour of agent work) | Do it inline yourself. No subagents. |
| Medium (one specialist's job + your coordination) | One `delegate` call; you integrate and verify. |
| Large / 3+ independent domains | Parallel dispatch (`delegate_batch` or multiple `delegate` calls in ONE response), or the plan lifecycle for sequential multi-task work. |

## This Workspace's Tool Map

- **`delegate(role, task)`** — one-shot isolated subagent with a restricted toolset; returns its final answer; your context stays small. Roles: planner, implementer, tester, debugger, reviewer, integrator, researcher, security, documenter.
- **`delegate_batch(roles=[...], task=...)`** — the SAME task fanned to up to 3 roles in parallel (independent perspectives on one question, e.g. security + reviewer + planner triage). One merged report back.
- **`create_subagent(name, role)` → `run_subagent_task(name, task)`** — a persistent subagent that keeps its own conversation history across multiple tasks. Use when the same specialist iterates (e.g. one debugger chasing related failures over several rounds). Create once, reuse many times; don't recreate per task.
- **`plan_start` / `plan_step` / `plan_finish`** — multi-task plans with ids and topological dependencies; each step executes in an isolated subagent. Use when the job is a sequence of more than ~2 tasks or spans sessions. `dry_run=true` validates before committing to execution.

## Parallel Dispatch (independent domains only)

Multiple tool calls in ONE response = parallel execution. One per response = sequential. Pick the mode deliberately.

**Dispatch one subagent per independent problem domain when:**
- 3+ test files failing with different root causes;
- multiple subsystems broken independently;
- each problem is understandable without context from the others;
- no shared state between investigations (no two agents editing the same file).

**Don't dispatch in parallel when:** failures are related (fixing one may fix others); you need full-system understanding first; agents would interfere. Then: single subagent for all, or sequential steps via `plan_step`.

## Task Brief Structure — every subagent task is self-contained

Subagents do NOT inherit your session context or history. Construct exactly what they need:

```
<One-sentence goal>

SCOPE: files/subsystem this task owns (e.g. harnessfix/diagnose.py + tests/test_diagnose.py)
GOAL: the concrete outcome ("make these 3 failing tests pass": list them)
CONTEXT: file paths, symbols, decision ids or doc references it must respect; exact current behavior vs desired behavior
CONSTRAINTS: what NOT to touch (e.g. guard-note strings per decision #020); no model/prompt changes as the fix lever (#014)
EXPECTED OUTPUT: summary of findings/changes + evidence to return (test command run, exit status, diff scope). "Done" means X — state it.
```

Good briefs are focused (one problem domain), self-contained (all needed context), specific about output (what to return and what counts as done).

## Review and Integrate (never skip)

When subagents return:

1. Read each summary; treat success reports as UNTRUSTED — verify against the actual `git diff` and a fresh full test run (skill `verification-before-completion`).
2. Check parallel changes don't conflict (same files, same symbols).
3. Run the FULL suite — integration failures appear only at the seams.
4. Integrate all changes; report per-domain outcomes by name.

## Explicit Stop Points (autonomous execution)

Stop and surface to the human when:
- A merge/approval gate requires a human decision (decision #015: post-batch gate, fail-closed in headless mode — never auto-merge repairs).
- Scope changed mid-flight (new requirements arrived) → stop, re-plan, don't silently absorb.
- A subagent loop shows 3+ failed fix attempts for the same issue → stop and question architecture instead of spawning a fourth attempt.

## Rationalizations

| Excuse | Reality |
|--------|---------|
| "One big task is fine" | Big tasks blow the subagent's context and your integration budget; independent domains should be parallel, not stacked. |
| "The subagent needs my session history" | It doesn't — a self-contained brief beats inherited noise, and keeps both contexts small. |
| "I'll integrate without running the full suite" | Seams are exactly where parallel work breaks. Fresh full-suite run is part of integration, not optional. |
