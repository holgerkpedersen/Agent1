"""speculate command — probabilistic deliberation at the REPL.

Usage::

    speculate "question" [--branches N] [--threshold X] [--timeout S]

Exposes the phase 3/4 speculative-deliberation pipeline as an interactive
command.  Each branch asks the agent's LLM one independent take on the
question (real ``Orchestrator.dispatch_speculative`` thread pool), a judge
LLM call scores every surviving candidate 0.0-1.0, and
``ProbabilisticOrchestrator.run_speculative`` decides the commitment:

- **COMMIT** — the best candidate's judge score meets ``--threshold``
  (inclusive); the committed answer is printed.
- **REFUSE** — no candidate reached the threshold (or every branch failed);
  nothing is presented as the answer.

Threshold is validated *before* any branch is dispatched, matching
``ProbabilisticOrchestrator.run_speculative``'s ValueError contract.

REPL convention: input arrives via ``shlex.split(posix=False)``, so quoted
questions keep their literal quotes — stripped here with ``.strip('"')``
(same as ``multillm``/``analyze``/``fix``).
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

_FIRST_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_score(reply: str) -> float:
    """Extract a 0.0-1.0 score from a judge reply; unparseable -> 0.0."""
    match = _FIRST_NUMBER_RE.search(str(reply))
    if not match:
        return 0.0
    try:
        return max(0.0, min(1.0, float(match.group())))
    except ValueError:
        return 0.0


class SpeculateCommand(Command):
    """Run speculative LLM branches and probabilistically commit or refuse."""

    @property
    def name(self) -> str:
        return "speculate"

    @property
    def help_text(self) -> str:
        return (
            'speculate "question" [--branches N] [--threshold X] '
            "[--timeout S] - ask the LLM N independent speculative branches "
            "in parallel, judge-score each answer, and COMMIT only when the "
            "best score meets the threshold (otherwise REFUSE)"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        from agent_core.orchestrator_probabilistic import (
            Decision,
            ProbabilisticOrchestrator,
        )
        from agent_core.swarm_orchestrator import Orchestrator

        parts = list(args)
        num_branches = 3
        threshold = 0.7
        timeout = 60.0

        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "--branches" and i + 1 < len(parts):
                try:
                    num_branches = max(1, int(parts[i + 1]))
                except ValueError:
                    self.error("--branches expects a number.")
                    return True
                i += 2
                continue
            if p == "--threshold" and i + 1 < len(parts):
                try:
                    threshold = float(parts[i + 1])
                except ValueError:
                    self.error("--threshold expects a number between 0 and 1.")
                    return True
                if not 0.0 <= threshold <= 1.0:
                    self.error("threshold must be within [0.0, 1.0]")
                    return True
                i += 2
                continue
            if p == "--timeout" and i + 1 < len(parts):
                try:
                    timeout = max(1.0, float(parts[i + 1]))
                except ValueError:
                    self.error("--timeout expects a number of seconds.")
                    return True
                i += 2
                continue
            i += 1

        skip_values = {"--branches", "--threshold", "--timeout"}
        question_parts: list[str] = []
        for j, p in enumerate(parts):
            if p in skip_values and j + 1 < len(parts):
                continue
            if j > 0 and parts[j - 1] in skip_values:
                continue
            if p.startswith("--"):
                continue
            question_parts.append(p)
        question = " ".join(question_parts).strip().strip('"')
        if not question:
            self.error('Usage: speculate "question" [--branches N] [--threshold X]')
            return True

        llm = agent.llm

        def reasoning_func(branch_id: int, context: Any) -> dict[str, Any]:
            """One speculative branch: an independent LLM take on the question.

            Runs on the orchestrator's thread pool (no event loop there), so
            each branch drives the async client with its own asyncio.run.
            """
            messages = [{"role": "user", "content": (
                f"Speculative branch {branch_id}. {question}"
            )}]
            answer = str(asyncio.run(llm.chat(messages)))
            return {"branch": branch_id, "answer": answer}

        def scorer(result: dict[str, Any]) -> float:
            """Judge-score one candidate 0.0-1.0 via an LLM call."""
            answer = str(result.get("answer", ""))
            prompt = (
                "Quality judge: rate how good this answer is on a scale from "
                "0.0 to 1.0. Reply with only the number.\n"
                f"Answer: {answer}"
            )
            reply = asyncio.run(llm.chat([{"role": "user", "content": prompt}]))
            return _parse_score(str(reply))

        print(f"\n  [speculate] {num_branches} branch(es), threshold={threshold:g}")

        def deliberate() -> Any:
            with Orchestrator(agents=[], max_workers=max(4, num_branches)) as base:
                orch = ProbabilisticOrchestrator(base_orchestrator=base)
                return orch.run_speculative(
                    reasoning_func=reasoning_func,
                    context=question,
                    threshold=threshold,
                    scorer=scorer,
                    num_branches=num_branches,
                    timeout=timeout,
                )

        try:
            # Run off the event loop: wait_for_completion blocks, and the
            # scorer's asyncio.run needs a thread with no running loop.
            decision = await asyncio.to_thread(deliberate)
        except ValueError as exc:
            self.error(str(exc))
            return True
        except RuntimeError as exc:
            self.error(str(exc))
            return True

        if decision.kind == Decision.COMMIT and decision.best is not None:
            print(f"  [speculate] COMMIT score={decision.best.score:.2f} "
                  f"threshold={threshold:g}")
            answer = ""
            best = decision.best.result
            if isinstance(best, dict):
                answer = str(best.get("answer", best))
            else:
                answer = str(best)
            for line in answer.splitlines() or [answer]:
                print(f"  {line}")
        else:
            print(f"  [speculate] REFUSE - no candidate reached "
                  f"threshold {threshold:g}")
        return True
