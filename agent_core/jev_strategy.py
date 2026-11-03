"""Jev strategy layer — cheap-first cascade decisions on top of the Jev engine.

The Jev engine answers one typed question with a probability distribution.
This module turns those distributions into *actionable decisions* using the
cheap-first strategies from the model-routing literature and the workspace
``jev-strategy`` skill:

* ``cascade`` — JEV-as-a-Judge: *accept when confident, escalate when
  unsure*.  One cheap verdict settles the question; only doubt escalates to
  the expensive machinery (speculate, subagents, long reasoning).
* ``triage`` — position-bias-checked choice: the winner of an order-swap
  re-ask must be stable before it is trusted (judge position-bias
  countermeasure).
* ``verify`` — FrugalGPT-style answer verifier: a cheap yes/no check of a
  draft answer against its task before the expensive model re-does the work.
* ``EscalationLadder`` — the escalation ladder made concrete: run cheap
  rungs first and stop at the first confident verdict.

Every function is async, takes any engine exposing
``async decide(question, state) -> JevResult`` (the ``JevEngine`` contract),
and returns a typed :class:`JevVerdict` (``accept`` / ``escalate``).  Engine
errors and all-abstention results ALWAYS escalate — a strategy never
fabricates a confident answer from a failed or empty signal.

``agent_core`` namespace rule: this module never imports ``agent``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agent_core.jev_engine import (
    KIND_CHOICE,
    KIND_YESNO,
    JevQuestion,
    JevResult,
)

ACTION_ACCEPT = "accept"
ACTION_ESCALATE = "escalate"

STRATEGY_CASCADE = "cascade"
STRATEGY_TRIAGE = "triage"
STRATEGY_VERIFY = "verify"
STRATEGY_LADDER = "ladder"

#: Strategies the ``jev_decide`` tool accepts in its ``strategy`` argument.
VALID_STRATEGIES: frozenset[str] = frozenset(
    {"single", STRATEGY_CASCADE, STRATEGY_TRIAGE, STRATEGY_VERIFY}
)

#: Default maximum sample spread (std dev on the 0-1 scale) for a ``score``
#: verdict to count as confident — a wide spread means the samples disagree.
DEFAULT_SPREAD_MAX = 0.15

#: Default minimum gap between the winning and runner-up option for a
#: ``choice`` verdict to count as confident.
DEFAULT_MARGIN = 0.0


@dataclass
class JevVerdict:
    """A typed, actionable outcome of a Jev strategy.

    ``action`` is ``accept`` (the cheap verdict is settled — use
    ``decision``) or ``escalate`` (the cheap signal was not confident —
    spend the expensive path).  ``probability`` is the relevant probability
    (winning side for yesno/choice, normalised value for score).  ``result``
    carries the raw engine answer for diagnostics.
    """

    strategy: str
    action: str
    decision: str | None = None
    probability: float = 0.0
    reason: str = ""
    result: JevResult | None = None
    elapsed_ms: float = field(default=0.0)

    @property
    def accepted(self) -> bool:
        return self.action == ACTION_ACCEPT

    def to_text(self) -> str:
        """One compact, plain-text line for tool output / the REPL."""
        if self.action == ACTION_ACCEPT:
            return (
                f"[jev-strategy] {self.strategy}: ACCEPT {self.decision} "
                f"(P={self.probability:.2f}) — {self.reason}"
            )
        return (
            f"[jev-strategy] {self.strategy}: ESCALATE "
            f"(P={self.probability:.2f}) — {self.reason}"
        )


def _escalate(
    strategy: str,
    reason: str,
    *,
    result: JevResult | None = None,
    probability: float = 0.0,
    elapsed_ms: float = 0.0,
) -> JevVerdict:
    return JevVerdict(
        strategy=strategy,
        action=ACTION_ESCALATE,
        decision=None,
        probability=probability,
        reason=reason,
        result=result,
        elapsed_ms=elapsed_ms,
    )


def _accept(
    strategy: str,
    decision: str,
    probability: float,
    reason: str,
    *,
    result: JevResult | None = None,
    elapsed_ms: float = 0.0,
) -> JevVerdict:
    return JevVerdict(
        strategy=strategy,
        action=ACTION_ACCEPT,
        decision=decision,
        probability=probability,
        reason=reason,
        result=result,
        elapsed_ms=elapsed_ms,
    )


def _verdict_from_result(
    strategy: str,
    question: JevQuestion,
    result: JevResult,
    *,
    accept_threshold: float,
    margin: float,
    spread_max: float,
    elapsed_ms: float,
) -> JevVerdict:
    """Map one engine result onto an accept/escalate verdict (pure logic)."""
    if question.kind == KIND_YESNO:
        p_yes = result.probabilities.get("yes")
        p_no = result.probabilities.get("no")
        if p_yes is None or p_no is None:
            return _escalate(
                strategy,
                "no usable answer (all samples abstained)",
                result=result,
                elapsed_ms=elapsed_ms,
            )
        if result.error:
            return _escalate(
                strategy,
                f"engine error: {result.error}",
                result=result,
                elapsed_ms=elapsed_ms,
            )
        if p_yes >= accept_threshold:
            return _accept(
                strategy,
                "yes",
                p_yes,
                f"P(yes) {p_yes:.2f} >= {accept_threshold:g}",
                result=result,
                elapsed_ms=elapsed_ms,
            )
        if p_no >= accept_threshold:
            return _accept(
                strategy,
                "no",
                p_no,
                f"P(no) {p_no:.2f} >= {accept_threshold:g}",
                result=result,
                elapsed_ms=elapsed_ms,
            )
        return _escalate(
            strategy,
            f"undecided: P(yes)={p_yes:.2f}, P(no)={p_no:.2f}, "
            f"neither reaches {accept_threshold:g}",
            result=result,
            probability=max(p_yes, p_no),
            elapsed_ms=elapsed_ms,
        )

    if question.kind == KIND_CHOICE:
        probs = result.probabilities
        if not probs or all(v <= 0.0 for v in probs.values()):
            return _escalate(
                strategy,
                "no usable answer (all samples abstained)",
                result=result,
                elapsed_ms=elapsed_ms,
            )
        if result.error:
            return _escalate(
                strategy,
                f"engine error: {result.error}",
                result=result,
                elapsed_ms=elapsed_ms,
            )
        ordered = sorted(
            probs.values(), reverse=True
        )
        top = ordered[0] if ordered else 0.0
        second = ordered[1] if len(ordered) > 1 else 0.0
        if top >= accept_threshold and (top - second) >= margin:
            winner = max(
                result.probabilities, key=lambda o: result.probabilities[o]
            )
            return _accept(
                strategy,
                winner,
                top,
                f"P({winner}) {top:.2f} >= {accept_threshold:g} "
                f"(gap {top - second:.2f} >= {margin:g})",
                result=result,
                elapsed_ms=elapsed_ms,
            )
        detail = f"best P={top:.2f} (threshold {accept_threshold:g}"
        if len(ordered) > 1:
            detail += f", gap {top - second:.2f} < {margin:g}" if (
                top >= accept_threshold
            ) else ""
        return _escalate(
            strategy,
            f"undecided: {detail})",
            result=result,
            probability=top,
            elapsed_ms=elapsed_ms,
        )

    # score
    if result.value is None:
        return _escalate(
            strategy,
            "no usable score (all samples abstained)",
            result=result,
            elapsed_ms=elapsed_ms,
        )
    if result.error:
        return _escalate(
            strategy,
            f"engine error: {result.error}",
            result=result,
            elapsed_ms=elapsed_ms,
        )
    value = result.value
    spread = result.spread
    if value >= accept_threshold:
        if spread is not None and spread > spread_max:
            return _escalate(
                strategy,
                f"score {value:.2f} >= {accept_threshold:g} but spread "
                f"{spread:.2f} > {spread_max:g} (samples disagree)",
                result=result,
                probability=value,
                elapsed_ms=elapsed_ms,
            )
        return _accept(
            strategy,
            f"{value:.2f}",
            value,
            f"score {value:.2f} >= {accept_threshold:g}"
            + (
                f" (spread {spread:.2f} <= {spread_max:g})"
                if spread is not None
                else ""
            ),
            result=result,
            elapsed_ms=elapsed_ms,
        )
    return _escalate(
        strategy,
        f"score {value:.2f} < {accept_threshold:g}",
        result=result,
        probability=value,
        elapsed_ms=elapsed_ms,
    )


async def cascade_decide(
    engine: Any,
    question: JevQuestion,
    state: str = "",
    *,
    accept_threshold: float = 0.7,
    margin: float = DEFAULT_MARGIN,
    spread_max: float = DEFAULT_SPREAD_MAX,
    strategy: str = STRATEGY_CASCADE,
) -> JevVerdict:
    """Ask the cheap Jev model and accept only a confident verdict.

    *yesno*: accept when P(yes) or P(no) reaches ``accept_threshold``.
    *choice*: accept when the winner reaches the threshold AND beats the
    runner-up by at least ``margin``.  *score*: accept when the normalised
    value reaches the threshold and the sample spread is within
    ``spread_max``.  Anything else — including engine errors and
    all-abstention results — escalates.
    """
    started = time.perf_counter()
    result = await engine.decide(question, state)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _verdict_from_result(
        strategy,
        question,
        result,
        accept_threshold=accept_threshold,
        margin=margin,
        spread_max=spread_max,
        elapsed_ms=elapsed_ms,
    )


async def triage(
    engine: Any,
    question_text: str,
    options: list[str] | tuple[str, ...],
    state: str = "",
    *,
    accept_threshold: float = 0.7,
    margin: float = 0.10,
    swap: bool = True,
) -> JevVerdict:
    """Triage among *options* with a position-bias check.

    Asks the cheap model for the best option; when ``swap`` is on, re-asks
    with the option order REVERSED and accepts only when the SAME option
    wins both times (each ask must also clear the threshold and the
    winner/runner-up ``margin``).  A winner that flips with the order is
    position bias, not a signal — the verdict escalates.
    """
    opts = tuple(str(o).strip() for o in options if str(o).strip())
    if len(opts) < 2:
        return _escalate(
            STRATEGY_TRIAGE,
            "triage needs at least two options",
        )
    first = await cascade_decide(
        engine,
        JevQuestion(kind=KIND_CHOICE, text=question_text, options=opts),
        state,
        accept_threshold=accept_threshold,
        margin=margin,
        strategy=STRATEGY_TRIAGE,
    )
    if not first.accepted or not swap:
        return first
    second = await cascade_decide(
        engine,
        JevQuestion(
            kind=KIND_CHOICE, text=question_text, options=tuple(reversed(opts))
        ),
        state,
        accept_threshold=accept_threshold,
        margin=margin,
        strategy=STRATEGY_TRIAGE,
    )
    if second.accepted and second.decision == first.decision:
        return second
    return _escalate(
        STRATEGY_TRIAGE,
        "winner unstable under order swap (position bias suspected): "
        f"first={first.decision!r}, swapped={second.decision!r}",
        result=second.result,
        probability=max(first.probability, second.probability),
        elapsed_ms=first.elapsed_ms + second.elapsed_ms,
    )


_VERIFY_PROPOSITION = (
    "The answer fully and correctly addresses the task, with no material "
    "omission or error."
)


async def verify_answer(
    engine: Any,
    task: str,
    answer: str,
    context: str = "",
    *,
    threshold: float = 0.7,
) -> JevVerdict:
    """Cheap verifier for a draft answer (FrugalGPT-style cascade check).

    Builds a yes/no proposition ("the answer fully and correctly addresses
    the task") judged against ``state`` = task + answer (+ optional
    context).  Accepts only when the cheap model is confident the draft is
    good; otherwise escalates so the expensive path can redo or repair it.
    """
    state = f"TASK:\n{task}\n\nANSWER:\n{answer}"
    if context:
        state += f"\n\nCONTEXT:\n{context}"
    return await cascade_decide(
        engine,
        JevQuestion(kind=KIND_YESNO, text=_VERIFY_PROPOSITION),
        state,
        accept_threshold=threshold,
        strategy=STRATEGY_VERIFY,
    )


#: One ladder rung: a zero-argument async callable returning a JevVerdict.
Rung = Callable[[], Awaitable[JevVerdict]]


class EscalationLadder:
    """Walk cheap rungs in order; stop at the first confident verdict.

    Each rung is a ``(label, async callable)`` pair; the callable runs one
    cheap strategy (e.g. a cascade gate or a triage) and returns its
    :class:`JevVerdict`.  The first ``accept`` ends the ladder.  When every
    rung escalates, the ladder returns one combined ``escalate`` verdict
    whose reason lists each rung's doubt, so the expensive path sees ALL the
    cheap evidence at once.  A rung that raises is treated as an escalation
    (a broken cheap signal must never kill the decision).
    """

    def __init__(self, rungs: list[tuple[str, Rung]]):
        self.rungs = list(rungs)

    async def run(self) -> JevVerdict:
        started = time.perf_counter()
        doubts: list[str] = []
        last_result: JevResult | None = None
        best_p = 0.0
        for label, rung in self.rungs:
            try:
                verdict = await rung()
            except Exception as exc:  # noqa: BLE001 - a rung must not kill the ladder
                doubts.append(f"{label}: error ({exc})")
                continue
            if verdict.accepted:
                verdict.elapsed_ms = (time.perf_counter() - started) * 1000.0
                return verdict
            doubts.append(f"{label}: {verdict.reason}")
            if verdict.result is not None:
                last_result = verdict.result
            best_p = max(best_p, verdict.probability)
        return _escalate(
            STRATEGY_LADDER,
            "all rungs escalated: " + " | ".join(doubts)
            if doubts
            else "empty ladder (no rungs)",
            result=last_result,
            probability=best_p,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


async def run_ladder(rungs: list[tuple[str, Rung]]) -> JevVerdict:
    """Convenience wrapper: build an :class:`EscalationLadder` and run it."""
    return await EscalationLadder(rungs).run()
