---
name: systematic-debugging
description: Find the root cause before proposing fixes; four-phase investigation for test failures, bugs and unexpected behavior
when_to_use: any bug, test failure or unexpected behavior, before proposing a fix
tags: [debugging, root-cause]
---

# Systematic Debugging

Provenance: adapted from obra/superpowers (MIT license), tooling references reworked to this workspace.

## Overview

**Core principle:** ALWAYS find the root cause before attempting fixes. Symptom fixes are failure.

Violating the letter of this process is violating its spirit.

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose fixes.

## When to Use

Any technical issue: test failures, production bugs, unexpected behavior, performance problems, build failures, integration issues.

**Use this ESPECIALLY when:** under time pressure (emergencies make guessing tempting); "just one quick fix" seems obvious; you've already tried multiple fixes; a previous fix didn't work; you don't fully understand the issue.

Don't skip it because the issue looks simple — simple bugs have root causes too, and rushing guarantees rework.

## The Four Phases (each must complete before the next)

### Phase 1: Root Cause Investigation

BEFORE attempting ANY fix:

1. **Read error messages carefully.** Don't skip past errors or warnings; they often contain the exact solution. Read stack traces completely — line numbers, file paths, error codes.
2. **Reproduce consistently.** Exact steps? Every time? If not reproducible → gather more data, don't guess.
3. **Check recent changes.** `git log` / `git diff`, new dependencies, config and environment differences (`.env`, model catalog, provider payloads).
4. **Gather evidence at component boundaries** — for multi-component paths (NLP tool loop → provider → tools; harnessfix trace → diagnose → repair): before proposing fixes, instrument each boundary (log what enters/exits), run once, find WHERE it breaks, then investigate only that layer. For tool-loop/harness behavior: read the `harnessfix/` tracing JSONL for the failing task first — traces exist precisely so you don't guess.
5. **Trace data flow backward.** Where does the bad value originate? What called this with the bad value? Keep going up until you find the source. Fix at the source, not the symptom.

### Phase 2: Pattern Analysis

Find the pattern before fixing:

1. **Find working examples** — locate similar code in THIS codebase that works (`references` for usage sites, `definitions` to orient inside a file).
2. **Compare against references** — if implementing a known pattern, read the reference implementation completely; don't skim.
3. **Identify differences** between working and broken: list every difference however small; never assume "that can't matter".
4. **Understand dependencies** — what config, environment or state does this code assume?

### Phase 3: Hypothesis and Testing (scientific method)

1. **Form a single hypothesis.** State it clearly: "I think X is the root cause because Y." Write it down; be specific, not vague.
2. **Test minimally.** The SMALLEST possible change to test that hypothesis. One variable at a time — never stack fixes.
3. **Verify before continuing.** Worked → Phase 4. Didn't work → form a NEW hypothesis; do NOT add more fixes on top of the old one.
4. **When you don't know, say so.** "I don't understand X" is a valid status. Research or ask instead of pretending to know.

### Phase 4: Implementation — fix the root cause, not the symptom

1. **Create the failing test case first** (simplest reproduction; automated under `tests/` if possible). It MUST exist before the fix. Use skill `test-driven-development`.
2. **Implement a single fix.** Address the identified root cause; ONE change at a time; no bundled refactors.
3. **Verify the fix:** your test passes, full suite green, original symptom gone. Load skill `verification-before-completion` before claiming success.
4. **If the fix doesn't work — STOP and count.** How many fixes have you tried?
   - Fewer than 3 → back to Phase 1 with the new information.
   - **3 or more failed fixes → question the architecture** (wrong abstraction, wrong layer boundary, dependency pointing the wrong way) and discuss it before attempting a fourth fix. Don't attempt Fix #4 without that discussion.

## Red Flags — STOP

Guessing under time pressure; "just one quick patch" before Phase 1; multiple simultaneous changes in one test run; stacking fixes on top of unfixed hypotheses; declaring victory after a single green check without the full suite; a third failed fix with no architectural question asked.
