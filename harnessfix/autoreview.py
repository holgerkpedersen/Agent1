"""Agent auto-review of failed task traces — the capability fallback.

When the user cannot determine a disposition for a failed trace, the agent
reviews it instead: a deterministic rules engine encodes the same evidence
methodology documented in docs/REVIEW_GUIDE.md and validated against the 12
hand-labeled pre-#050 traces in docs/PRE050_TRACE_LABELS.md.

Trust model (in harnessfix/review.py): auto-review records are marked
source="agent"; the human can always override; the agent never overwrites a
human label; the agent cannot assign "ok"/"regression" (no self-certification);
ambiguity resolves to the harsher disposition ("bug") to counter self-serving
bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .htir import TraceGraph

FINAL_ANSWER_MIN_CHARS = 80

_BUG = "bug"
_NOISE = "noise"

#: Dispositions the agent may never assign (self-certification guard).
AGENT_FORBIDDEN: frozenset[str] = frozenset({"ok", "regression"})


@dataclass
class AutoReview:
    disposition: str
    confidence: str  # high | medium | low
    evidence: list[str] = field(default_factory=list)

    @property
    def note(self) -> str:
        return "agent auto-review: " + " | ".join(self.evidence)


def auto_review(graph: TraceGraph) -> AutoReview:
    """Classify a failed trace by evidence rules (highest-priority match
    wins; see docs/REVIEW_GUIDE.md decision table)."""
    evidence: list[str] = []
    steps = graph.steps

    if graph.task_id.startswith("demo-"):
        return AutoReview(_NOISE, "high", ["spec fixture trace"])

    outcome = _loop_outcome(graph)
    has_loop_end = graph.has_loop_end()
    errors = [s for s in steps if s.kind == "tool_error"]
    guard_names = [
        str(s.payload.get("guard", ""))
        for s in steps
        if s.kind == "guard_triggered" and s.payload.get("guard")
    ]
    writes = [
        s
        for s in steps
        if s.kind == "tool_call" and s.payload.get("tool") in ("write", "edit")
    ]
    final_answer = _final_answer_after_guards(graph)

    # Delivered runs: the loop itself ended normally (outcome=completed only
    # happens on the answer path; such traces reach review solely because of
    # a tool_error), or a substantive final answer came AFTER the last guard
    # stop (the loop's "give your final answer now" pattern).
    if outcome == "completed":
        return AutoReview(_NOISE, "high", ["loop_end outcome=completed — task delivered"])
    if has_loop_end and final_answer and len(final_answer) >= FINAL_ANSWER_MIN_CHARS:
        if errors:
            evidence.append(f"tool errors ({len(errors)}) before answer")
        elif guard_names:
            evidence.append(f"guard-terminated ({', '.join(guard_names)}) after answer")
        return AutoReview(_NOISE, "high", evidence or ["delivered final answer"])

    # Interrupted runs (no loop_end, no guard) died externally — not agent
    # behavior, and never with damage: the loop stops tool execution on death.
    if not has_loop_end and not guard_names:
        return AutoReview(
            _NOISE,
            "high",
            [f"interrupted run (no loop_end, {len(steps)} events) — external death"],
        )

    if errors:
        evidence.append(
            f"tool_error {errors[0].payload.get('exception')} — no final answer"
        )
        return AutoReview(_BUG, "high", evidence)

    if "stuck" in guard_names:
        evidence.append("stuck guard — repeated identical call")
        if writes:
            evidence.append(f"{len(writes)} write/edit calls made before the loop")
        return AutoReview(_BUG, "high", evidence)

    if "no_mutation" in guard_names or "circle" in guard_names:
        if writes:
            evidence.append(
                f"{len(writes)} write/edit calls, yet no final answer — work done, "
                "never delivered"
            )
        else:
            evidence.append(
                "no_mutation guard, zero write/edit calls — circling exploration"
            )
        return AutoReview(_BUG, "high", evidence)

    if not has_loop_end:
        # guard fired, then the run died before writing loop_end
        evidence.append("guard fired, run interrupted before loop_end")
        return AutoReview(_BUG, "medium", evidence)

    if outcome == "error":
        evidence.append(
            "loop_end outcome=error with no tool_error/guard — provider or "
            "environment failure"
        )
        return AutoReview(_NOISE, "medium", evidence)

    evidence.append(
        f"unmatched failure signature (outcome={outcome or '?'}, "
        f"guards={guard_names or 'none'}) — harsher default"
    )
    return AutoReview(_BUG, "low", evidence)


def _final_answer_after_guards(graph: TraceGraph) -> str:
    """Last non-empty llm_response text, but only when it came AFTER the
    last guard stop (the loop's "give your final answer now" pattern).

    Mid-work narration must never count as a delivered answer: interrupted
    runs and pre-guard texts are exactly the false-positive trap that
    mislabeled 536848/9acf2d/b5b264 in the pre-#050 corpus."""
    last_guard = -1
    last_answer = ""
    for idx, s in enumerate(graph.steps):
        if s.kind == "guard_triggered":
            last_guard = idx
        elif s.kind == "llm_response":
            text = str(s.payload.get("text", "") or "")
            if text:
                last_answer = text
                last_answer_idx = idx
    if last_answer and last_answer_idx > last_guard:
        return last_answer
    return ""


def _loop_outcome(graph: TraceGraph) -> str:
    for s in reversed(graph.steps):
        if s.kind == "loop_end":
            return str(s.payload.get("outcome", ""))
    return ""