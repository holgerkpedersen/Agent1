---
name: jev-strategy
description: Use the cheap small Jev model (jev_decide) as a typed pre-gate before expensive work — yesno/choice/score judgements with thresholds and rubrics — and know when a tool, not a judge, must answer
when_to_use: bounded judgement needed, second opinion before committing, triage, score against rubric, deciding whether to escalate to speculate
tags: [jev, routing, judgement]
---

# Strategic Jev — cheap typed judgements, escalated only when inconclusive

Provenance: workspace-authored. Grounded in model-routing literature
(cascade / "cheap first, escalate on doubt": FrugalGPT-style cascades report
large cost savings by sending easy queries to a small model and reserving the
frontier model for hard cases) and in LLM-as-judge bias research (position,
sycophancy, verbosity, rubric-compression biases).

## Core Principle

```
SETTLE EACH QUESTION WITH THE CHEAPEST TOOL THAT CAN ACTUALLY ANSWER IT.
```

`jev_decide` is a **decision model, not a writer**: it returns a probability
for a typed question, fast and cheap, on a small model that never sees your
context unless you pass it. Use it to **triage and gate**, then escalate to
heavier machinery (`speculate`, subagents, long reasoning, full analysis)
only when the cheap signal is inconclusive.

The rules below never override: plan-mode gating, verification-before-
completion, or a failing test. Jev is an *opinion*; tests, diffs and tool
output are *evidence*.

## Pick the Right Kind

| kind | question shape | output | use for |
|------|----------------|--------|---------|
| `yesno` | one proposition ("the failure is caused by X") | P(yes)/P(no) + TRUE/FALSE/UNKNOWN | gating: "is it worth escalating?" |
| `choice` | 2–5 concrete, mutually exclusive options | distribution | triage / tie-break |
| `score` | subject + explicit rubric | 0–100 + spread | quality self-check against anchors |

Phrasing rules:

- **yesno = a proposition**, not a question. "This fix addresses the root
  cause" beats "Is this fix OK?".
- **choice options must be self-contained** — the model sees only the option
  strings plus `state`.
- **score rubrics need anchors**: say what 80 means and what 40 means.
  Unanchored rubrics compress into the top of the scale and become theater.
- **One question per call.** Compound questions produce meaningless middles.

## `state` Is the Judge's Entire World

The small model does **not** read this repo. Whatever context you omit is
invisible to it. Put in `state`:

- the concrete evidence: `file:line`, error text, test output, constraints;
- **facts only — no conclusions.** Editorialising in `state` ("clearly the
  bug is in…") invites confirmation bias: the judge agrees with what you
  already believe instead of checking it.

Short and factual beats long and persuasive.

## Thresholds

- Default `threshold=0.7`.
- **Raise to 0.85+** for anything irreversible, security-relevant, or that
  would trigger destructive changes.
- **Lower to ~0.5–0.6** for cheap triage where a wrong yes just costs a look.
- **Act only on TRUE (P ≥ threshold).** Otherwise treat the result as
  *undecided* and say so — never round 0.55 up to "jev approved".

## When to Call — the Escalation Ladder

1. **Gate before expensive work.** One `yesno` can save a `speculate`
   run, a subagent fan-out, or a long analysis chain:
   `P("the failure reproduces deterministically")` → if UNKNOWN, go look;
   if high, launch the heavy tool with a sharpened question.
2. **Tie-break / triage** with `choice` — two designs, three suspect files,
   four plausible root causes. The distribution tells you where to look
   first, not what is true.
3. **Score against a rubric** before claiming quality (review-readiness,
   plan completeness). A low score or **wide spread** = go improve, not go
   claim. This never replaces running the tests.
4. **Second opinion at mid confidence.** When your own answer lands in the
   middle of the plausibility range, ask; disagreement is worth surfacing
   to the user with both sides stated.
5. **Cheap pre-flight on ambiguity.** When a request admits two readings,
   `choice` the readings and ask the user rather than guessing — cheap
   compared to building the wrong thing.

## When NOT to Call

- **Open-ended generation, code, or explanations** — you write those.
- **Anything with verifiable ground truth.** Run the test, the diff,
  `references`, the linter. Asking a judge to predict what a tool can
  measure is prompt theater and can *disagree with reality*.
- **Factual questions about this repo** — `references` / `search` /
  `speculate` (grounded branches) answer those; a judge has no file access.
- **As a substitute for evidence.** "jev said yes" never ships: a completion
  claim still requires fresh verification (see verification-before-completion).

## Bias Countermeasures (judge research → habits)

| Bias | Mitigation |
|------|------------|
| Position (order) bias | For a pairwise judgement, ask twice with options **swapped**; trust only if the distribution agrees both ways. |
| Sycophancy / leading state | `state` = facts and evidence only, no conclusion words, no "obviously". |
| Rubric compression | Anchored rubric with distinct level descriptions; ask for spread and treat wide spread as weak signal. |
| Self-preference | Don't have jev score jev's own draft and call it independent review — use `judge="llm"` or a real reviewer subagent for that. |
| Confidence theater | A high P on an ungrounded question is still ungrounded — judge quality is bounded by `state` quality. |

## Cascading with `speculate`

- `judge="jev"` — fast scoring of branches; fine for triage-speed questions.
- `judge="both"` — average of small + main model; use when stakes are real
  but speed still matters.
- `judge="llm"` (default) — strongest grounding judgement; the default for
  repo questions where branches cite `file:line`.
- `branch_model="jev"` — fastest/cheapest, **weak at grounding repo
  questions**; reserve for non-repo deliberation or pure triage.
- Fewer branches = faster; more branches = stronger corroboration.

General cascade rule: start with the cheapest signal (`jev_decide`), then
`speculate` with grounded branches, then subagents/full analysis — each step
only when the previous one was inconclusive or disagreed with your own read.

## Worked Examples

```text
# Gate before fan-out
jev_decide(kind="yesno",
  question="The failing test is caused by a stale expected value rather than a regression.",
  state="test expects 5; impl returns 5; last change touched parsing (a.py:12).",
  threshold=0.6)
# P >= 0.6 -> update expectation + run; else -> systematic-debugging, not a guess.

# Tie-break with order swap
jev_decide(kind="choice", question="Most likely root cause",
  options=["regex too greedy", "off-by-one in slice", "encoding mismatch"],
  state="<evidence>")
# If the top option is unstable across a swapped-order re-ask, escalate.

# Anchored score
jev_decide(kind="score", question="Review readiness of this diff",
  rubric="80+: every changed line has a test; 50: tests exist for main path only; 20: untested logic",
  state="<diff summary>")
```

## Red Flags — STOP

- Asking jev to *write* something, or to answer what a tool can verify.
- Passing a question that already contains your desired answer.
- Treating a mid-range probability as a decision.
- Using a positive jev result as verification evidence in a completion claim.
- Escalating straight to `speculate`/subagents on a question one cheap
  `yesno` could have triaged (cost) — or conversely asking jev repeatedly
  when the ground truth is one `run` away (theater).
