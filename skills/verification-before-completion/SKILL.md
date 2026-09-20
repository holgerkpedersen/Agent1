---
name: verification-before-completion
description: Evidence before claims; run fresh verification and read the full output before claiming work is complete, fixed or passing
when_to_use: about to claim completion, a fix, green tests, or to commit/merge
tags: [verification, done-gate]
---

# Verification Before Completion

Provenance: adapted from obra/superpowers (MIT license).

## Overview

**Core principle:** Evidence before claims, always.

Violating the letter of this rule is violating its spirit.

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in THIS message, you cannot claim it passes.

## The Gate Function

Before claiming any status or expressing satisfaction:

1. **IDENTIFY** — what command proves this claim?
2. **RUN** — execute the FULL command (fresh, complete).
3. **READ** — full output; check exit code (`run` reports non-zero as `[EXIT CODE: N]`); count failures.
4. **VERIFY** — does the output confirm the claim? NO → state the actual status with evidence. YES → state the claim WITH evidence.
5. **ONLY THEN** make the claim.

Skipping any step is lying, not verifying.

## Common Failures

| Claim | Requires | Not sufficient |
|-------|----------|----------------|
| Tests pass | Test command output: 0 failures | A previous run, "should pass" |
| Linter/type checker clean | Its output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build command exit 0 | Linter passing, logs look good |
| Bug fixed | The original symptom's test passes | Code changed, assumed fixed |
| Regression test works | Red-green verified (revert fix → must fail) | Test passes once |
| Subagent completed its task | VCS diff shows the changes + suite green | The subagent reported "success" |
| Requirements met | Line-by-line checklist against the request | Tests passing |

## Red Flags — STOP

- Using "should", "probably", "seems to".
- Expressing satisfaction before verification ("Great!", "Done!").
- About to commit/push/merge without fresh evidence.
- Trusting a subagent's success report instead of checking the diff and tests yourself.
- Relying on partial verification; thinking "just this once"; being tired and wanting work over.
- ANY wording implying success that hasn't been verified in this message — including paraphrases, not just exact phrases.

## Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification. |
| "I'm confident" | Confidence ≠ evidence. |
| "Linter passed" | Linter ≠ compiler ≠ tests. |
| "Subagent said success" | Verify independently against the diff and a fresh test run. |
| "Partial check is enough" | Partial proves nothing about the rest. |

## Key Patterns

```
✅ [run pytest] [see: 214 passed, 0 failed] → "All tests pass (214/214)."
❌ "Should pass now." / "Looks correct."

✅ Regression test: write → run (pass) → revert fix → run (MUST FAIL) → restore → run (pass).
❌ "I've written a regression test." (without red-green verification)

✅ Subagent reports done → git diff shows the claimed changes → fresh full suite green → report actual state.
❌ Trust the subagent's summary and move on.

✅ Requirements: re-read the request → checklist → verify each item → report gaps or completion.
❌ "Tests pass, so it's complete."
```
