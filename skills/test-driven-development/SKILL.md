---
name: test-driven-development
description: Write failing tests before implementation code; red-green-refactor with mandatory verification gates for features, bug fixes and refactors
when_to_use: implementing any feature or bug fix, before writing implementation code
tags: [tdd, testing, pytest]
---

# Test-Driven Development (TDD)

Provenance: adapted from obra/superpowers (MIT license), examples reworked to this repo's Python/pytest stack.

## Overview

Write the test first. Watch it fail. Write minimal code to pass.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

Violating the letter of these rules is violating their spirit.

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? Delete it. Start over — not as "reference", not "adapted" while writing tests, not even looked at. Implement fresh from the tests. Period.

## When to Use

**Always:** new features, bug fixes, refactors, behavior changes.

**Exceptions (ask your human partner first):** throwaway prototypes, generated code, pure configuration files.

Thinking "skip TDD just this once"? Stop. That's rationalization.

## Red-Green-Refactor

### RED — Write the failing test

One minimal pytest case showing what should happen: one behavior, a clear name, real code (no mocks unless unavoidable). Import and exercise the REAL function under `agent_core/`, `harnessfix/` or wherever it lives — never a copy of its logic.

<good>
```python
def test_retry_retries_failed_operations():
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("boom")
        return "ok"

    assert retry_operation(operation) == "ok"
    assert attempts == 3
```
Clear name, tests real behavior, one thing.
</good>

<bad>
```python
def test_retry_works(mocker):
    op = mocker.MagicMock(side_effect=[RuntimeError("x"), RuntimeError("y"), "ok"])
    result = retry_operation(op)
    assert op.call_count == 3
```
Vague name, tests the mock instead of the code.
</bad>

### Verify RED — watch it fail (MANDATORY, never skip)

Run `pytest tests/path/test_x.py::test_name` (the `tests` tool or `run`). Confirm:

- It **fails** (not errors).
- The failure message is the one you expected.
- It fails because the feature is missing — not a typo/import error.

Test passes immediately? You're testing existing behavior — fix the test. Test errors? Fix the error and re-run until it fails for the right reason.

### GREEN — minimal code

Write the simplest code that passes. No options plumbing, no backoff strategies, no "while I'm here" improvements. YAGNI.

### Verify GREEN — watch it pass (MANDATORY)

Run the file, then run the FULL suite (`pytest` bare). Confirm: your test passes, nothing else broke, output is pristine.

**"Other tests" means the project's suite, not just your file.** A green run of the test you wrote is not a green suite — run the whole thing even when the task named only one file. Any failure that run shows (including pre-existing ones) goes in your report by name; a red test you watched scroll past and didn't mention is a report falsified by omission.

### REFACTOR — clean up, after green only

Remove duplication, improve names, extract helpers. Keep the suite green. Don't add behavior. Then: next failing test for the next feature.

## Good Tests

| Quality | Good | Bad |
|---------|------|-----|
| Minimal | One thing; "and" in the name? Split it. | `test_validates_email_and_domain_and_whitespace` |
| Clear | Name describes behavior | `test_1` |
| Shows intent | Demonstrates the desired API | Obscures what code should do |

Rules for every test you write or change:

- Name the production change that would make this test fail — before writing it.
- Assert on real behavior, never on mock internals.
- Keep test-only helpers in `tests/`, out of production classes.
- Understand a dependency's side effects before mocking it.

## This workspace's contract

- Every bug fix ships a **permanent regression test** under `tests/` (pytest).
- Verify fixes against the real code path (import and call the actual function), not a copied simulation of it.
- A failing test is a signal: fix the implementation — never weaken or delete an assertion just to make a test pass. Only change a demonstrably wrong test, and say why in your report.

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Too simple to test" | Simple code breaks; the test takes 30 seconds. |
| "I'll add tests after" | Tests written after pass immediately — which proves nothing, and you never proved they can catch the bug. |
| "Already manually tested" | No record, not re-runnable, easy to forget cases under pressure. "Worked when I tried it" ≠ comprehensive. |
| "Deleting X hours is wasteful" | Sunk cost — that time is spent either way. The real choice: rewrite with TDD (high confidence) vs keep untrusted code (likely bugs). |
| "Need to explore first" | Fine — throw the exploration away, then start with TDD. |
| "Hard to test = design unclear" | Listen to the test: hard to test means hard to use. |
| "TDD will slow me down" | TDD is the pragmatic path: bugs caught before commit, refactors without fear. The shortcut costs a production debug later. |

## Red Flags — STOP and start over

Code written before its test; test passes immediately; can't explain why it failed first; tests deferred to "later"; any flavor of "just this once" or "this is different because...". All of these mean: delete the code, restart with TDD.

## Verification Checklist (before marking work complete)

- [ ] Every new function/method has a test
- [ ] Each test was watched fail first, for the expected reason
- [ ] Minimal code per test; full suite green after refactor
- [ ] Bug fix → permanent regression test committed under `tests/`

Then load skill `verification-before-completion` before claiming done.
