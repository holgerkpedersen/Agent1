"""Parallel multi-LLM dispatch — fire simultaneous calls to DIFFERENT LLMs.

The agent already routes per-model to different providers
(:func:`~agent_core.llm.provider.build_provider`: ``opencode-go/...`` prefixes
→ the hosted opencode-go API, LM Studio prefixes → the local LM Studio
server).  This module is the missing piece: it builds ONE provider instance
per requested model and fires all ``chat`` calls concurrently with
``asyncio.gather``, so local LM Studio models and hosted opencode models
answer the SAME prompt in parallel.

Usage::

    from agent_core.llm.parallel import run_parallel

    results = await run_parallel(
        [{"role": "user", "content": "Review this code"}],
        models=["laguna-s-2.1", "opencode-go/deepseek-v4-flash"],
        settings=load_agent_settings(),
    )
    for r in results:
        print(r.model, r.ok, r.text[:200])

Tools: pass ``tools=NLP_TOOL_SCHEMAS`` plus an ``execute_tool_fn`` and each
model runs the SAME :class:`~agent_core.llm.tool_loop.ToolLoopRunner` the
agent uses — the model can read/search/list files, run tests, etc. instead
of answering from the prompt alone.  Each model gets its OWN loop instance
(no cross-model tool-state contamination), all running concurrently.

Design notes
------------
* **One provider per model** — a single ``LMStudioProvider`` can only hold one
  ``model_name`` and a single opencode ``_session_id``; sharing one instance
  across models would serialize (and cross-contaminate) the calls.
* **Per-model roles** — pass ``roles={model: system_prompt}`` and each model
  gets its OWN system message prepended to the shared prompt, so different
  LLMs can play different expert roles (reviewer, auditor, ...) while
  answering the same question.
* **Error isolation** — providers return ``[Error: ...]`` strings and never
  raise; ``return_exceptions=True`` additionally isolates an unexpected
  exception so one dead server cannot abort the other models' answers.
* **Consensus wiring** — the dormant :class:`ConsensusVoter`
  (:mod:`agent_core.llm.orchestrator`) and :class:`RefinementVoter`
  (:mod:`agent_core.llm.refinement_voter`) finally get a real producer:
  each model's verdict is recorded under a ``template_id`` and the quorum
  gate is exposed via :meth:`ParallelRun.quorum_reached`.
* **Metrics** — each result carries the provider's
  ``last_response_metrics`` (token/latency accounting, plan ARCH item 17).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from .orchestrator import ConsensusVoter
from .refinement_voter import RefinementVoter
from .provider import ResponseMetrics, build_provider, get_last_metrics, provider_for

#: Text prefix every provider returns on failure (never raises).
_ERROR_PREFIXES = ("[Error", "[LM Studio")


@dataclass
class ParallelResult:
    """Outcome of one model's parallel chat call."""

    model: str
    provider: str  # "lmstudio" | "opencode"
    text: str
    ok: bool  # False when the call failed (error string or exception)
    metrics: ResponseMetrics | None = None
    exception: str = ""

    @property
    def error(self) -> str:
        """Human-readable failure text, or '' when the call succeeded."""
        if self.ok:
            return ""
        return self.exception or self.text[:400]


@dataclass
class ParallelRun:
    """One parallel dispatch: the results plus the consensus ledger.

    Every model's verdict is recorded into a :class:`RefinementVoter`
    (which delegates the quorum gate to a :class:`ConsensusVoter`) under
    ``template_id``, so the dormant consensus machinery is driven by real
    multi-model votes.
    """

    template_id: str
    results: list[ParallelResult] = field(default_factory=list)
    voter: RefinementVoter = field(default_factory=RefinementVoter)

    @property
    def ok_results(self) -> list[ParallelResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed_results(self) -> list[ParallelResult]:
        return [r for r in self.results if not r.ok]

    def agree(self, verdict: bool) -> None:
        """Record ``verdict`` (approve=True / reject=False) from every model."""
        for r in self.ok_results:
            self.voter.collect_vote(
                self.template_id, verdict and not _looks_negative(r.text)
            )

    def quorum_reached(self, quorum_threshold: float = 0.5) -> bool:
        """True when ``yes / total >= quorum_threshold`` for the recorded votes.

        Computed directly from the vote ledger (the dormant
        ``RefinementVoter.decide()`` path never feeds its ``ConsensusVoter``,
        so delegating there would always return False — this is the same
        approval-ratio semantics :class:`ConsensusVoter.tally_votes` uses).
        """
        votes = self.voter.vote_status(self.template_id)
        total = int(votes["total"])
        if total == 0:
            return False
        return int(votes["yes"]) / total >= quorum_threshold

    def consensus(self, quorum_threshold: float = 0.5) -> str:
        """One-line consensus summary: agree / disagree / insufficient votes."""
        votes = self.voter.vote_status(self.template_id)
        if votes["total"] == 0:
            return "no usable model answers — no consensus"
        if self.quorum_reached(quorum_threshold):
            return (f"consensus APPROVE ({votes['yes']}/{votes['total']} "
                    f"above {quorum_threshold:.0%} quorum)")
        return (f"no consensus ({votes['yes']}/{votes['total']} approve, "
                f"quorum {quorum_threshold:.0%})")


def _looks_negative(text: str) -> bool:
    """Heuristic: does the answer reject the proposition (no/bug/invalid)?"""
    low = text.strip().lower()
    if not low:
        return True
    first = low[:120]
    return any(
        first.startswith(prefix)
        for prefix in ("no", "nope", "not ok", "invalid", "bug", "fails", "reject")
    )


def _error_or_exception(result: Any) -> tuple[str, str]:
    """Split a gather outcome into ``(text, exception)``.

    Providers return error strings; ``return_exceptions=True`` wraps an
    unexpected exception in the result slot.  Both are mapped to a
    ``ParallelResult`` with ``ok=False``.
    """
    if isinstance(result, BaseException):
        return "", f"{type(result).__name__}: {result}"
    text = str(result or "")
    return text, ""


def _make_tool_llm_chat_fn(provider: Any, max_tokens: int | None,
                           disable_thinking: bool) -> Callable[
    [list[dict[str, Any]], list[dict[str, Any]]],
    Awaitable[tuple[str, list[dict[str, Any]]]],
]:
    """Build the ``llm_chat_fn`` the ToolLoopRunner needs for one provider.

    Mirrors the agent's own chat wrapper (agent.py): call the provider with
    tools, turn a JSON ``tool_calls`` response back into an assistant message
    so the loop can execute the calls, and treat provider error strings as
    empty so the loop's forced-synthesis retry kicks in.
    """
    async def llm_chat_fn(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        raw = await provider.chat(
            messages, tools=tools, max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        if raw.strip() == "(no output)":
            raw = ""
        if raw.startswith(_ERROR_PREFIXES):
            return raw, messages
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            parsed.pop("role", None)
            updated = list(messages)
            updated.append({
                "role": "assistant",
                "content": parsed.get("content") or "",
                **parsed,
            })
            return str(parsed.get("content") or ""), updated
        updated = list(messages)
        updated.append({"role": "assistant", "content": raw})
        return raw, updated

    return llm_chat_fn


async def run_parallel(
    messages: Sequence[dict[str, Any]],
    models: Sequence[str],
    settings: Any,
    *,
    max_tokens: int | None = None,
    disable_thinking: bool = True,
    template_id: str = "parallel",
    concurrency: int | None = None,
    roles: dict[str, str] | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    execute_tool_fn: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
    max_tool_iterations: int = 40,
) -> ParallelRun:
    """Fire ``chat(messages)`` on every *model* simultaneously.

    Args:
        messages: The shared prompt (same for every model — that is the point
            of a parallel comparison; the per-model *roles* system message is
            prepended to it).
        models: Model names exactly as ``build_provider`` resolves them
            (``opencode-go/...`` → opencode provider; LM Studio prefixes →
            local LM Studio).  At least two models are required for a
            meaningful parallel run.
        settings: ``AgentSettings`` (from :func:`load_agent_settings`) used
            by ``build_provider`` for provider construction.
        max_tokens: Optional output cap forwarded to every ``chat`` call.
        disable_thinking: Forwarded to every ``chat`` call — reasoning models
            otherwise burn their budget on ``reasoning_content``.
        template_id: Key under which the votes are recorded in the consensus
            ledger (surfaces in ``ParallelRun.consensus()``).
        concurrency: Optional cap on simultaneous in-flight calls
            (``asyncio.Semaphore``).  None = all at once.
        roles: Optional ``{model_name: system_prompt}`` map — each model gets
            its OWN system message prepended to the shared *messages*, so
            different LLMs can play different expert roles (code reviewer,
            security auditor, ...) while answering the same question.
        tools: Optional OpenAI-format tool schemas (e.g.
            ``NLP_TOOL_SCHEMAS``).  When given together with
            ``execute_tool_fn``, every model runs through the same
            :class:`~agent_core.llm.tool_loop.ToolLoopRunner` the agent uses
            — it can actually READ files, search, list directories, run
            tests, etc. instead of only answering from the prompt.
        execute_tool_fn: ``async (name, args) -> str`` — the tool executor
            (the agent's ``_execute_tool_call``).  Required when *tools* is
            given.
        max_tool_iterations: Per-model tool-loop iteration cap (default 40 —
            enough for a focused review; each model runs its OWN loop so the
            cap is per model, not shared).

    Returns:
        A :class:`ParallelRun` with one :class:`ParallelResult` per model,
        in the same order as *models*.

    Raises:
        ValueError: when fewer than two models are given, a model cannot
            be resolved to a provider, or *tools* is given without
            ``execute_tool_fn``.
    """
    if len(models) < 2:
        raise ValueError("run_parallel needs at least two models")
    if tools and execute_tool_fn is None:
        raise ValueError("tools requires execute_tool_fn")
    providers = []
    for m in models:
        try:
            providers.append(build_provider(settings, str(m)))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"cannot build provider for {m!r}: {exc}") from exc

    roles = roles or {}

    async def _call(provider: Any, model_name: str) -> Any:
        model_messages = list(messages)
        role = roles.get(model_name)
        if role:
            model_messages.insert(0, {"role": "system", "content": role})
        try:
            if tools and execute_tool_fn is not None:
                from .tool_loop import ToolLoopRunner

                loop = ToolLoopRunner(
                    max_iterations=max_tool_iterations,
                    display_mode="quiet",
                )
                final_text, _ = await loop.run(
                    messages=model_messages,
                    llm_chat_fn=_make_tool_llm_chat_fn(
                        provider, max_tokens, disable_thinking,
                    ),
                    execute_tool_fn=execute_tool_fn,
                    tools=list(tools),
                )
                return final_text
            return await provider.chat(
                model_messages,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
        except BaseException as exc:  # return_exceptions safety net
            return exc

    if concurrency is not None and concurrency >= 1:
        sem = asyncio.Semaphore(int(concurrency))

        async def _limited(provider: Any, model_name: str) -> Any:
            async with sem:
                return await _call(provider, model_name)

        coros = [_limited(p, str(m)) for p, m in zip(providers, models)]
    else:
        coros = [_call(p, str(m)) for p, m in zip(providers, models)]

    gathered = await asyncio.gather(*coros, return_exceptions=True)

    run = ParallelRun(template_id=template_id)
    for model, provider, outcome in zip(models, providers, gathered):
        text, exception = _error_or_exception(outcome)
        ok = not exception and not text.startswith(_ERROR_PREFIXES)
        run.results.append(
            ParallelResult(
                model=str(model),
                provider=provider_for(str(model), "lmstudio"),
                text=text,
                ok=ok,
                metrics=get_last_metrics(provider),
                exception=exception,
            )
        )
    return run


def summarize(run: ParallelRun) -> str:
    """Human-readable summary of a parallel run (REPL display)."""
    lines = [f"Parallel run '{run.template_id}': {len(run.results)} model(s)"]
    for r in run.results:
        status = "ok" if r.ok else "ERROR"
        metrics = ""
        if r.metrics is not None and r.metrics.total_tokens:
            metrics = f"  ({r.metrics.total_tokens} tok, {r.metrics.latency_ms:.0f} ms)"
        lines.append(f"  [{status}] {r.model} ({r.provider}){metrics}")
        if not r.ok:
            lines.append(f"      {r.error[:200]}")
    lines.append(f"  {run.consensus()}")
    return "\n".join(lines)


__all__ = [
    "ParallelResult",
    "ParallelRun",
    "run_parallel",
    "summarize",
]