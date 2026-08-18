# Review Guide — the human verification gate

The agent records every tool loop run as a trace (`reports/traces/<task_id>.jsonl`).
Most runs succeed and are never looked at again. The **failed** ones — runs with
tool errors, guard interruptions, or crashes — are surfaced by the `review`
command so *you* decide what each failure means. Your labels (`bug`, `regression`,
`noise`, `ok`) are what the improvement loop consumes: label honestly, and the
agent gets fixed in the right places; label carelessly, and it gets fixed in the
wrong places.

This guide covers the command itself and, more importantly, **how to gather
enough evidence to label a task with confidence**.

---

## 1. Quick start

```
review refresh                          # rebuild the ledger from the trace corpus
review list                             # table of all failed tasks + labels
review show a669a26e4f0c40dd9e2bf20807f3778a
review label a669a26e4f0c40dd9e2bf20807f3778a noise --note "delivered answer despite stuck guard"
review export a669a26e4f0c40dd9e2bf20807f3778a   # after labeling
```

Run `review refresh` **after** any agent session — new traces appear, old ones
drop out. Labels you already assigned are preserved across refreshes.

## 2. Command reference

| Command | What it does |
|---|---|
| `review refresh [--trace-dir dir] [--diags-dir dir]` | Rebuild `reports/harnessfix/review.json` from `reports/traces/` + `reports/harnessfix/diagnoses/`. Only *failed* traces appear. |
| `review list` | Table: task id, disposition, model, diagnosis layer, outcome. |
| `review show <task>` | Full record: prompt, model, profile, outcome, guards, files touched, diagnosis, label. |
| `review label <task> bug\|regression\|noise\|ok [--note "..."]` | Assign a verdict. Persists immediately to the ledger. |
| `review export <task>` | (labeled tasks only) Write a pytest pin asserting this trace keeps diagnosing the same way → `reports/harnessfix/generated/test_review_<task>.py`. |

## 3. Reading a review record

`review show` prints the fields below. Each answers one question:

| Field | Tells you |
|---|---|
| `prompt` | What the user actually asked — sanity-check the rest against this. |
| `model` / `profile` | Which model+profile ran it. Compare apples to apples. |
| `outcome` | How the loop ended: `completed`, `stuck` (repeated identical calls), `cap`/`no_progress` (budget or no-mutation guard), `error`. |
| `guards` | Which lifecycle guards fired (`stuck`, `no_mutation`, `budget_exhausted`…). |
| `affected_files` | **The most useful field.** What the run actually wrote/edited. Empty + guards → it never did anything. |
| `root_layer` / `mechanism` | The diagnosis: which facet failed and why. **Treat as a hypothesis, not a verdict** — verify it against the trace (see traps). |
| `disposition` | Your label. |

> **Pre-#050 traces are not reviewable.** Traces recorded before the metadata
> stamping (decision #050) carry no prompt and no model/profile — they cannot
> be human-judged, so the ledger excludes them entirely (`review refresh`
> drops them). Their historical assessment is documented in
> `docs/PRE050_TRACE_LABELS.md`. Everything `review` shows you is therefore
> reviewable.

## 4. Gathering evidence — three levels

**Level 1 — the record.** `review show <task>`. Fast triage: if `affected_files`
is empty, outcome is `stuck`, and there's no answer, that's already a strong
`bug` signal. If `affected_files` is full of files and the final answer exists,
it's likely `noise` regardless of what the guards say.

**Level 2 — the timeline.**
```
python harnessfix/dashboard.py --traces reports/traces --task <task>
```
Shows the full event stream (tool calls, results, guard notes, final answer),
files touched, and the diagnosis explanation. This is where you see *what the
model was doing*: which tools it called, whether it repeated calls, whether the
guard fired at the very end (usually fine) or early (usually a defect).

**Level 3 — the raw trace.** `reports/traces/<task>.jsonl` (one JSON object per
line). Look at four things:

1. **Head** — the `task_begin` event carries the original prompt (new traces
   only; older ones lack it — use the dashboard or context).
2. **`tool_error` events** — `exception` + `message`: why a tool call failed.
3. **`guard_triggered` events** — the guard's `note` tells the model (and you)
   exactly why the loop stopped, e.g. *"You have now repeated the same tool call
   without making progress…"*.
4. **The last `llm_response` before `loop_end`** — this is the final answer.
   Substantive text here means the task *was delivered* even if the outcome says
   `stuck`/`cap`/`no_progress`.

### Evidence checklist (answer these three)

- **Did it deliver?** Is there a substantive final answer at the end?
- **Did it touch the right things?** Do `affected_files` match what the prompt
  asked for? Were changes actually made, or just reads?
- **Does the mechanism match reality?** Find the actual event that supposedly
  triggered the diagnosis. If you can't find it, the diagnosis is wrong — label
  based on what you *see*, not on the mechanism text.

## 5. Labeling decision guide

| If you see… | Label |
|---|---|
| Repeated identical tool calls, no progress, **no final answer** | `bug` |
| Tool errors (permission, missing file, schema validation) the model never recovered from | `bug` |
| Model ignored explicit instructions (e.g. kept calling tools after being told to stop) | `bug` |
| Final answer delivered despite stuck/cap/no_progress guards | `noise` |
| Provider hiccup: run ended with outcome `error` early, no tool errors of the agent's making | `noise` |
| Model style variant (e.g. "autumn" vs "fall", US vs UK spelling) | `noise` |
| Demo/fixture traces (`demo-*`) | `noise` |
| Failure type you've seen succeed before on the same kind of task | `regression` |
| Nothing wrong — ran as intended, or the failure is fully explained by the environment | `ok` |

### Traps — the three ways to mislabel

1. **Guard ≠ failure.** The stuck/cap guards stop the *loop*, not the answer.
   A 108-iteration run that ends `stuck` but produced a full final answer
   delivered the task → `noise`, not `bug`. (`has_final_answer` now handles
   this automatically — such runs no longer even appear in the ledger.)
2. **Mechanism text ≠ proof.** Diagnosis signatures match event fields; in old
   traces a file that merely *mentions* "truncation" could trigger a context
   diagnosis. Always confirm the mechanism against the actual event.
3. **Don't label before checking the final answer.** The last `llm_response` is
   the ground truth of delivery. Check it first, always.

## 6. Worked examples (real corpus)

**Example A — `a669a26e…` (→ `noise`)**
108 iterations fixing `analysis_verifier.py`, zero tool errors, `stuck` guard
fired on a repeated verification one-liner at the very end, then a complete
final answer. Verdict: the task was delivered; the repeated command cost a few
iterations at most. `noise`, with a note about the wasted iterations.

**Example B — `0f7793…` (→ `bug`)**
204 events, two `no_mutation` guards, no final answer, `affected_files` empty.
The model produced no changes and no answer. Verdict: real defect — the model
failed to make progress on the task.

**Example C — `e0b03b…` (→ `noise` or `bug`, check `termination_reason`)**
Ended with outcome `error` after 3 iterations, no tool errors. If the
termination reason is an LLM/API failure (provider-side), it's `noise`; if the
model itself requested termination, it's `bug`. The note field is where you
record which one it was.

## 7. After labeling

- **`review export <task>`** writes a regression pin that re-diagnoses the
  trace and asserts the same layer+mechanism — future diagnosis changes for
  this trace fail loudly instead of silently reclassifying your labeled task.
- The ledger `reports/harnessfix/review.json` is human-owned and gitignored.
  `bug` + `regression` labels are the backlog for the improvement loop;
  `noise` + `ok` teach it what *not* to chase.
- Per model+profile label stats are the signal for what to improve next — a
  cluster of `stuck`+no-answer `bug`s says the loop needs a better stuck
  response; a cluster of `noise` delivered-runs says the guard is too eager.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Task not in `review list` | Run `review refresh`. Not appearing at all = it's no longer classified as failed (e.g. delivered answer), or the trace is too short/older than the schema. |
| `review export` refuses | Label the task first. |
| Ledger missing/corrupt | `review refresh` rebuilds it from the corpus (labels are lost — they lived in the ledger). |
| Trace file gone | `reports/traces/` is gitignored and prunable; the ledger entry stays until the next refresh drops it. |
