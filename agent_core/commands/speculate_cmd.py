"""speculate command — probabilistic deliberation at the REPL.

Usage::

    speculate "question" [--branches N] [--threshold X] [--timeout S]

Exposes the phase 3/4 speculative-deliberation pipeline as an interactive
command.  Each branch asks the agent's LLM one independent take on the
question (real ``Orchestrator.dispatch_speculative`` thread pool), carrying
the agent's own system prompt so every branch reasons as the agent
(persona + environment) instead of as a generic assistant, and may run a
short READ-ONLY tool loop (search/read/list_files/definitions/references/
web_search) to ground its answer — mutating tools are refused.  A judge LLM
call scores every surviving candidate 0.0-1.0, and
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
import json
import re
from typing import TYPE_CHECKING, Any

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

_FIRST_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")

#: Tools a speculative branch may execute — the verified read-only set
#: (filesystem inspection + web search).  Enforced as a hard allowlist, so a
#: hallucinated ``run``/``write``/``edit`` can never mutate the workspace even
#: though the branches run in parallel.
_BRANCH_TOOLS: frozenset[str] = frozenset({
    "search", "read", "list_files", "definitions", "references", "web_search",
})

#: Max model round-trips per branch before a final, tools-withheld answer.
_BRANCH_MAX_ITERS = 4


def _parse_score(reply: str) -> float:
    """Extract a 0.0-1.0 score from a judge reply; unparseable -> 0.0."""
    match = _FIRST_NUMBER_RE.search(str(reply))
    if not match:
        return 0.0
    try:
        return max(0.0, min(1.0, float(match.group())))
    except ValueError:
        return 0.0


def _split_tool_reply(reply: str) -> tuple[str, list[dict[str, Any]]] | None:
    """(content, tool_calls) when *reply* is a tool-call payload, else None.

    Providers serialise a native tool call as
    ``{"content": ..., "tool_calls": [...]}``; a plain-text answer parses to
    no ``tool_calls`` and is returned as-is by the caller.
    """
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    calls = [c for c in (data.get("tool_calls") or []) if isinstance(c, dict)]
    if not calls:
        return None
    return str(data.get("content") or ""), calls


def _branch_tool_schemas() -> list[dict[str, Any]]:
    """The read-only tool schemas advertised to every branch."""
    from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

    return [
        s for s in NLP_TOOL_SCHEMAS
        if s.get("function", {}).get("name") in _BRANCH_TOOLS
    ]


#: Textual markers of a model's TOOL-CALL syntax leaking through as plain
#: text instead of a structured ``tool_calls`` payload — e.g. gemma's
#: ``<|tool_call>call:run{command:"…"}<tool_call|>``.  Such a reply is not an
#: answer and must never be committed (it once scored 1.0 and COMMITted).
_TOOL_CALL_MARKERS = (
    "<|tool_call", "<tool_call", "<|tool_response", "<tool_response",
)

#: Inline call syntax without angle brackets: ``call:name{...}`` / ``call_name(...)``.
_INLINE_TOOL_CALL_RE = re.compile(r"^call[:_]\w+\s*[{(]", re.IGNORECASE)


def _looks_like_tool_call(text: str) -> bool:
    """True when *text* is a raw tool call rather than a prose answer."""
    stripped = text.strip()
    if not stripped:
        return False
    low = stripped.lower()
    if any(marker in low for marker in _TOOL_CALL_MARKERS):
        return True
    if _INLINE_TOOL_CALL_RE.match(stripped):
        return True
    # A bare JSON tool-call object that slipped past the JSON fast-path.
    if stripped.startswith("{") and '"tool_calls"' in stripped:
        return True
    if '"function"' in stripped and '"arguments"' in stripped:
        return True
    return False


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
        # Branch-dispatch wait.  Each branch carries the agent's full system
        # prompt AND may make several read-only tool round-trips, so a hosted
        # model needs real room; 60s aborted the whole deliberation.  Override
        # with --timeout.
        timeout = 300.0

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

        # Every branch must see the agent's REAL system prompt — persona,
        # environment and tool inventory — or it answers as a generic assistant
        # that claims it "can't access your system".  Built once here, on the
        # REPL thread: the branches run on the orchestrator pool and must not
        # touch agent state concurrently.  Defensive: a stand-in agent without
        # the method/history yields an empty prompt (behaviour unchanged).
        system_prompt = ""
        try:
            agent._refresh_system_message()
            system_prompt = str(agent._chat_history[0].get("content") or "")
        except Exception:
            system_prompt = ""
        branch_tools = _branch_tool_schemas()
        executor = getattr(agent, "_execute_tool_call", None)

        def _call_branch_tool(name: str, args: dict[str, Any]) -> str:
            """Run ONE branch tool call, gated to the read-only allowlist."""
            if name not in _BRANCH_TOOLS:
                return (
                    f"Tool '{name}' is not available to speculative branches "
                    f"(read-only: {', '.join(sorted(_BRANCH_TOOLS))})."
                )
            if executor is None:
                return f"Tool '{name}' is unavailable in this context."
            try:
                return str(asyncio.run(executor(name, args)))
            except Exception as exc:  # noqa: BLE001 - a bad call must not kill the branch
                return f"{name} error: {exc}"

        def reasoning_func(branch_id: int, context: Any) -> dict[str, Any]:
            """One speculative branch: an independent, READ-ONLY agentic take.

            Runs on the orchestrator's thread pool (no event loop there), so
            each branch drives the async client with its own asyncio.run.  The
            branch answers AS THE AGENT (its system prompt) and may call the
            read-only tools to ground itself; a hallucinated mutating tool is
            refused by the allowlist, so parallel branches cannot change the
            workspace.
            """
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": (
                f"Speculative branch {branch_id}. {question}\n\n"
                "Answer the question directly in prose — that text is the "
                "final answer shown to the user, so never reply with a tool "
                "call. You may call ONLY these read-only tools if you must "
                "check the workspace: "
                f"{', '.join(sorted(_BRANCH_TOOLS))}. Do not call any other tool."
            )})
            answer = ""
            for _ in range(_BRANCH_MAX_ITERS):
                reply = str(asyncio.run(llm.chat(messages, tools=branch_tools)))
                split = _split_tool_reply(reply)
                if split is None:
                    answer = reply
                    break
                content, calls = split
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls,
                })
                for call in calls:
                    fn = call.get("function") if isinstance(call, dict) else {}
                    fn = fn if isinstance(fn, dict) else {}
                    name = str(fn.get("name") or "")
                    try:
                        targs = json.loads(fn.get("arguments") or "{}")
                        if not isinstance(targs, dict):
                            targs = {}
                    except (json.JSONDecodeError, TypeError):
                        targs = {}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "content": _call_branch_tool(name, targs),
                    })
            else:
                # Still calling tools at the cap: force a final text answer
                # with tools withheld so the branch always returns prose.
                answer = str(asyncio.run(llm.chat(messages)))
            if _looks_like_tool_call(answer):
                return {
                    "branch": branch_id,
                    "error": "branch emitted a raw tool call instead of an answer",
                }
            return {"branch": branch_id, "answer": answer}

        def scorer(result: dict[str, Any]) -> float:
            """Judge-score one candidate 0.0-1.0.

            A non-answer (empty, or a raw tool call) scores 0.0 WITHOUT asking
            the judge — the judge once rated a leaked tool call 1.0 and it was
            committed.
            """
            answer = str(result.get("answer", "")).strip()
            if not answer or _looks_like_tool_call(answer):
                return 0.0
            prompt = (
                "Quality judge: score how well the ANSWER answers the QUESTION, "
                "0.0 to 1.0. 1.0 = directly and correctly answers it; 0.0 = "
                "off-topic, empty, evasive, or a tool call / raw command "
                "instead of an answer. Reply with ONLY the number.\n"
                f"QUESTION: {question}\n"
                f"ANSWER: {answer}"
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
