---
name: requesting-code-review
description: Dispatch a reviewer subagent with precisely crafted context before merging or continuing multi-task work; triage findings by severity and push back on wrong ones
when_to_use: after major features, after each task in subagent-driven work, before merge to main
tags: [code-review, subagents]
---

# Requesting Code Review

Provenance: adapted from obra/superpowers (MIT license), dispatch mechanics reworked to this workspace's `delegate` tool.

Dispatch a code reviewer subagent to catch issues before they cascade. The reviewer gets precisely crafted context for evaluation — never your session's history.

**Core principle:** review early, review often.

## When to Request Review

**Mandatory:**
- After each task in subagent-driven development (see skill `subagent-orchestration`).
- After completing a major feature.
- Before merge to main / before the human review gate sees your work.

**Optional but valuable:** when stuck (fresh perspective); before refactoring (baseline check); after fixing a complex bug.

## How to Request

1. **Scope the diff.** Identify what changed: `git status` + `git diff` for uncommitted work, or `git log --oneline -5` / merge-base SHAs for committed work.
2. **Dispatch** with `delegate(role="reviewer", task=...)`. The brief must be self-contained — the reviewer cannot see your session:

   ```
   Review this change in the agent framework repo (branch master).

   WHAT WAS BUILT: <one-paragraph description>
   REQUIREMENTS / PLAN REFERENCE: <what it should do; decision ids or doc path if any>
   SCOPE: files changed — e.g. harnessfix/loop/*.py, tests/test_loop.py
   HOW TO VERIFY: pytest (full suite); key test file(s): ...
   WATCH FOR: correctness regressions, security-policy drift (decision #014: repairs must stay in the harness layer), unverified claims in code comments
   RETURN: findings as Critical / Important / Minor with file:line and a one-line rationale each; overall readiness assessment.
   ```

3. **Act on feedback:**
   - Fix **Critical** issues immediately.
   - Fix **Important** issues before proceeding to the next task/merge.
   - Note **Minor** issues for later (list them in your report).
   - **Push back when the reviewer is wrong** — with technical reasoning and code/tests that prove it; request clarification rather than silently implementing a bad suggestion.

## Rationalizations

| Excuse | Reality |
|--------|---------|
| "I'll just review my own diff inline" | You're the coordinator: reviewing burns the context you need to keep driving the work. The diff and evaluation belong in the reviewer's context; only findings come back to you. |
| "The reviewer needs my whole session history" | Hand it precisely crafted context, never your session. That keeps the review on the work product, not your thought process. |

## Red Flags — Never

Skip review because "it's simple"; ignore Critical issues; proceed with unfixed Important issues; argue with valid technical feedback (push back with evidence instead).
