"""Deterministic rubric judge for agentic-benchmark runs.

``agentic_bench.py`` scores every scenario run against a weighted rubric. The
scoring pipeline calls :func:`judge_outcome` **synchronously** (one call per
rubric criterion), so the default judging path is rule-based and fully
deterministic — a clean run (the model echoed the marker verbatim) scores 1.0,
an interrupted one does not. No live model is required to make the benchmark
run; an optional ``llm_chat`` hook can layer semantic LLM judgement on top of
the rules when a caller wants it.

The judge answers one narrow question per call: *does this run's outcome
satisfy this rubric criterion?* It returns a :class:`JudgeVerdict` carrying a
score in [0, 1] and a short human-readable reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["JudgeVerdict", "judge_outcome"]


@dataclass(frozen=True)
class JudgeVerdict:
    """Outcome of judging one rubric criterion against one run.

    ``score`` is in [0, 1] (1.0 = fully satisfied). ``verdict`` is a short
    label — "pass" when the criterion is met, "fail" otherwise. ``reason`` is
    a human-readable line for the report notes.
    """

    score: float
    verdict: str
    reason: str


# A marker token as it appears in rubric text — always back-quoted and always
# starts with "<<".  The same token lives in the prompt (``<<token>>1``), so we
# match on the ``<<token>>`` prefix only; the trailing digits are scenario-
# specific metadata, not part of the identity.
_MARKER_RE = re.compile(r"<<[A-Za-z0-9_.:-]+?>>")

#: Heuristic signals a rubric criterion wants for a "fully satisfied" verdict.
_MARKER_PHRASES: tuple[str, ...] = (
    "verbatim", "final answer", "on its own line", "line of its own",
)


def _extract_markers(text: str) -> list[str]:
    """Return the distinct marker tokens in *text*, in order of appearance."""
    seen: list[str] = []
    for match in _MARKER_RE.finditer(text):
        tok = match.group(0)
        if tok not in seen:
            seen.append(tok)
    return seen


def _norm(value: Any) -> str:
    """Normalise text for substring checks (casefold + collapse whitespace)."""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


# ---------------------------------------------------------------------------
# Rule-based criteria
# ---------------------------------------------------------------------------

def _rule_marker_in_response(criterion: str, response: str) -> JudgeVerdict | None:
    """Marker-quoting criteria: every marker the criterion names must appear.

    Returns ``None`` when the criterion quotes no marker token — the caller
    then falls through to other heuristics. A marker is "quoted" when its exact
    text appears in the response (whitespace-collapsed, case-insensitive).
    """
    markers = _extract_markers(criterion)
    if not markers:
        return None
    norm_resp = _norm(response)
    missing = [m for m in markers if _norm(m) not in norm_resp]
    if not missing:
        return JudgeVerdict(
            1.0, "pass",
            f"all {len(markers)} marker(s) quoted verbatim",
        )
    # Partial credit: a fraction of the named markers made it through.
    frac = (len(markers) - len(missing)) / len(markers)
    return JudgeVerdict(
        round(frac, 4), "fail" if frac < 1.0 else "pass",
        f"{len(missing)} of {len(markers)} marker(s) missing from the response",
    )


def _rule_run_tool_used(criterion: str, tools_used: int) -> JudgeVerdict | None:
    """Criteria about tool usage (the run tool / any tool fired)."""
    low = criterion.lower()
    if "run tool" not in low and "used at least once" not in low:
        return None
    if "at least once" in low:
        ok = tools_used >= 1
        return JudgeVerdict(
            1.0 if ok else 0.0,
            "pass" if ok else "fail",
            f"{tools_used} tool call(s) executed (need ≥ 1)",
        )
    # "run tool used at least once and the loop terminated cleanly" — a clean
    # termination is implied by reaching the judge at all; tool count decides.
    ok = tools_used >= 1
    return JudgeVerdict(
        1.0 if ok else 0.0, "pass" if ok else "fail",
        f"{tools_used} tool call(s) recorded",
    )


def _rule_write_before_read(criterion: str, response: str) -> JudgeVerdict | None:
    """Process-order criteria (write-then-read).

    The judge receives only the final answer text and counts, not the ordered
    tool trace, so a strict ordering check is impossible here. We approximate:
    if the marker file's name appears in the response, the write landed; the
    criterion passes. Absence of any evidence scores 0.5 (partial credit — a
    process error) rather than 0, because we cannot prove ordering from text.
    """
    low = criterion.lower()
    if "before" not in low or ("write" not in low and "read" not in low):
        return None
    # A path mentioned in the criterion (e.g. scratch/probe.txt) is the signal.
    paths = re.findall(r"[A-Za-z0-9_./-]+\.(?:txt|md|py|json|csv|log)", criterion)
    norm_resp = _norm(response)
    present = [p for p in paths if _norm(p) in norm_resp]
    if present:
        return JudgeVerdict(
            1.0, "pass", f"{len(present)} path(s) referenced in the response"
        )
    return JudgeVerdict(
        0.5, "fail",
        "process order not verifiable from text; partial credit awarded",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def judge_outcome(
    *,
    response: str,
    turns: int,
    tools_used: int,
    rubric_text: str,
    model: str = "",
    profile: str = "",
    llm_chat: Callable[[str], Any] | None = None,
) -> JudgeVerdict:
    """Judge one rubric criterion against one run's outcome.

    Args:
        response: the final assistant text of the run.
        turns: number of assistant messages recorded (observational).
        tools_used: number of tool calls executed during the run.
        rubric_text: a single weighted criterion line (the weight has already
            been stripped by ``parse_rubric`` in agentic_bench.py).
        model / profile: display metadata, echoed into the reason for context.
        llm_chat: optional callable ``(prompt) -> str``. When supplied and no
            rule decides the criterion outright, a semantic LLM judgement is
            requested; its score (parsed from the reply as a 0..1 float) is
            used. The rules always run first — they are deterministic.

    Returns:
        :class:`JudgeVerdict` with ``score`` in [0, 1].
    """
    # Rule-based pass: marker-quoting criteria decide on exact token presence.
    verdict = _rule_marker_in_response(rubric_text, response)
    if verdict is not None and verdict.score == 1.0:
        return verdict

    # Process-order heuristics (write-before-read).
    order_verdict = _rule_write_before_read(rubric_text, response)
    if order_verdict is not None:
        return order_verdict

    # Tool-usage heuristics.
    tool_verdict = _rule_run_tool_used(rubric_text, tools_used)
    if tool_verdict is not None:
        return tool_verdict

    # No rule matched — fall through to the optional LLM judge.
    if llm_chat is not None:
        prompt = (
            "You are a strict rubric grader for an agent benchmark. Score how "
            "well the run satisfies the criterion, as a float from 0.0 (not at "
            "all) to 1.0 (fully).\n\n"
            f"Criterion: {rubric_text}\n"
            f"Final answer:\n{response}\n\n"
            f"(Model: {model!r}, profile: {profile!r})\n"
            "Reply with ONLY a number between 0 and 1."
        )
        try:
            raw = llm_chat(prompt)
            num = re.search(r"\d+(?:\.\d+)?", str(raw))
            if num:
                score = max(0.0, min(1.0, float(num.group(0))))
                return JudgeVerdict(
                    round(score, 4), "pass" if score >= 0.8 else "fail",
                    f"llm judgement {score:.2f}",
                )
        except Exception as exc:  # pragma: no cover - fail-open to neutral
            return JudgeVerdict(
                0.5, "fail", f"llm judge unavailable ({type(exc).__name__})"
            )

    # Last resort: a marker-bearing criterion whose markers are absent scores 0;
    # an unrecognised criterion is inconclusive (partial credit).
    if _extract_markers(rubric_text):
        return JudgeVerdict(0.0, "fail", "required markers not quoted in response")
    return JudgeVerdict(0.5, "fail", "inconclusive — no rule matched the criterion")
