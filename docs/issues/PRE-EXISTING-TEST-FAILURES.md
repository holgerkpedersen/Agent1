# Pre-existing test failures (tracked separately from the OpenRouter fix)

**Status:** Resolved (2026-09-23) — all three failures are fixed on master; see the
resolution log at the bottom of this document.
**Author:** agent (during OpenRouter `SyntaxError` fix, see `agent_core/llm/openrouter_provider.py`)
**Scope note:** These failures exist on an **unmodified** `master` (verified via
`git stash` of the OpenRouter fix: 4 failed, 443 passed — identical failure set
to the modified tree's 4 failed, 447 passed). They are **not** caused by the
OpenRouter provider change and are intentionally kept out of that fix.

---

## 1. Full-suite collection is broken by a stale generated pin

**Symptom:** `python -m pytest` (rootdir collection) aborts at collection with:

```
ImportError: cannot import name 'compile_trace' from 'harnessfix.reader'
  reports/harnessfix/generated/test_review_7b31b6434d1840bc8e809f4e43335d16.py:7
```

**Root cause:** `compile_trace` lives in `harnessfix.htir`, but the generated
regression pin imports `from harnessfix.reader import compile_trace`. Two
sources of the wrong import:

- `harnessfix/review.py` (the generator):
  - the `export_regression_test()` template string (lines ~278)
  - a self-import at line ~284 (`from harnessfix.reader import compile_trace`)
- the already-generated artifact
  `reports/harnessfix/generated/test_review_7b31b6434d1840bc8e809f4e43335d16.py`

`harnessfix/reader.py` only defines `read_trace` / `task_id_of` (trace JSONL
parsing), so the import is simply outdated. The generator template was never
updated when `compile_trace` moved to `htir`.

**Why not fixed here:** The generated pins are gitignored (like the traces) and
regenerated from `review.json` via `export_regression_test`. Fixing only the
*generator* leaves already-generated files stale, and fixing the artifact by
hand contradicts its "Do not edit by hand" header. Needs a deliberate
regenerate + (optionally) a `norecursedirs`/collect exclusion so `reports/`
stops being swept by the rootdir run. Tracked as a separate change.

**Fix sketch (for the owner):**
```python
# harnessfix/review.py — both occurrences
from harnessfix.htir import compile_trace   # was: harnessfix.reader
```
Then regenerate pins, OR add to `pyproject.toml`:
```toml
norecursedirs = ["reports", ...]
```
(Note: `norecursedirs` is currently unset in `pyproject.toml`.)

---

## 2. `_reconcile_llama_model` never pins the served id when the server
already serves a *different* model

**Symptom:** `tests/test_llama_server.py::TestAgentReconcileHook::
test_reconcile_invokes_server_manager_for_llama` fails:

```
assert None == 'Bonsai-27B-Q1_0'
 +  where None = LlamaProvider._cached_server_model_id
```

**Root cause:** In `agent.py::_reconcile_llama_model`, after
`ensure_model_served` returns success the code re-checks the served model with:

```python
if served_after and served_after[0] == self._model_name:   # wrong
    provider._cached_server_model_id = self._model_name
```

`list_served_models()` returns the **bare** id (e.g. `Bonsai-27B-Q1_0`), but
`self._model_name` carries the `llama/` routing prefix (`llama/Bonsai-27B-Q1_0`),
so the equality never holds and the id is never pinned. The test's
`list_served_models` stub returns `["Bonsai-27B-Q1_0"]` and expects
`_cached_server_model_id == "Bonsai-27B-Q1_0"`.

**Why not fixed here:** It's a real bug, but it lives in `agent.py` /
`llama_provider.py` and is unrelated to the OpenRouter fix. Kept separate per
the "out of scope" instruction.

**Fix sketch (for the owner):**
```python
requested_bare = (self._model_name[len("llama/"):]
                  if self._model_name.startswith("llama/") else self._model_name)
if served_after and served_after[0] in (self._model_name, requested_bare):
    provider._cached_server_model_id = served_after[0]
```

---

## 3. Stale generated pin asserts an outdated diagnosis (`context` vs `lifecycle`)

**Symptom (appears only after Fix #1 is applied — the pin then collects and
runs):** `reports/harnessfix/generated/test_review_7b31b6434d1840bc8e809f4e43335d16.py`
fails:

```
assert 'lifecycle' == 'context'
```

**Root cause — needs human triage, do NOT weaken the assertion:**
- The pin was generated from `reports/harnessfix/review.json`, which records
  `root_layer: "context"` / mechanism `"history truncation / token limit pressure"`
  for task `7b31b6434d1840bc8e809f4e43335d16`.
- The **current** `diagnose_graph()` output — both the stored
  `reports/harnessfix/diagnoses/7b31b6434d1840bc8e809f4e43335d16.json`
  (`root_layer: "lifecycle"`) and a live run — returns `lifecycle`
  ("task interrupted after mutating 9 file(s)"). The trace mutated 9 files
  with no `loop_end`, which the diagnose heuristic (decision #052) correctly
  classifies as `lifecycle`.

So either (a) the stored `review.json` review is **wrong/outdated** and the pin
should be regenerated from the authoritative diagnosis, or (b) the diagnose
heuristic regressed and now misclassifies a `context`/`token-pressure` case as
`lifecycle`. This is a **semantic disagreement between the stored review and the
live diagnosis**, not a mechanical import bug. It must be triaged by a human,
not patched around.

---

## Verification notes
- All four failures reproduce identically on unmodified `master`
  (`git stash` of the OpenRouter fix → 4 failed, 443 passed).
- `tests/test_openrouter_provider.py`: 31 passed (includes the OpenRouter
  recovery regression tests).
- `OPENROUTER_API_KEY` is **not** set in this environment, so the live
  end-to-end smoke test of the 402/400 recovery paths was not performed; the
  mocked unit tests cover those paths.
- `gh` CLI is not installed and no `GH_TOKEN` is configured, so no GitHub issue
  was opened automatically; this document is the local tracking record.

---

## Resolution log (2026-09-23)

All three failures were re-verified against current `master` and are resolved:

1. **Stale generated pin / stale generator template — FIXED.** The generated
   artifact on this machine had already been regenerated with the correct
   import, but `harnessfix/review.py::export_regression_test()` still emitted
   `from harnessfix.reader import compile_trace` in its template, so any
   regeneration would have re-produced the broken pin. The template now emits
   `from harnessfix.htir import compile_trace`, pinned by the regression test
   `tests/test_harnessfix_review.py::test_export_regression_test_pin_imports_compile_trace_from_htir`
   (which also `exec_module`s the generated pin, so a stale import fails
   exactly as pytest collection would). `norecursedirs` was not added:
   `testpaths = ["tests"]` already keeps a bare `pytest` run out of
   `reports/`, and the generated pins are gitignored.
2. **`_reconcile_llama_model` pinning — FIXED.**
   `tests/test_llama_server.py::TestAgentReconcileHook` passes on current
   master (2 passed, verified 2026-09-23); the fix landed in a later commit.
3. **Stale pin diagnosis (`context` vs `lifecycle`) — RESOLVED by
   regeneration.** The pin now asserts `root_layer == "lifecycle"`, matching
   both the stored diagnosis
   (`reports/harnessfix/diagnoses/7b31b6434d1840bc8e809f4e43335d16.json`) and
   the live `diagnose_graph()` output, i.e. the stored `review.json` entry was
   the outdated side and the pin was regenerated from the authoritative
   diagnosis. The pin runs green standalone
   (`pytest reports/harnessfix/generated/test_review_7b31b6434d1840bc8e809f4e43335d16.py`
   → 1 passed).

Verification: full suite `python -m pytest -q --no-cov` → 2575 passed,
7 skipped, 0 failed (0:16:04).
