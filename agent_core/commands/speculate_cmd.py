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
import os
import re
from pathlib import Path
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

#: Output cap per branch call.  Passed explicitly so a small Jev branch model
#: is not limited by the Jev profile's per-sample cap (512).
_BRANCH_MAX_TOKENS = 1024


def _is_provider_error(text: str) -> bool:
    """True when *text* is a provider/transport error, not a real answer.

    ``LMStudioProvider.chat`` swallows HTTP failures and returns
    ``"[Error: HTTP Error 400: …]"`` as plain text.  Without this guard the
    judge (or ``_parse_score``'s first-number regex) treats the ``400`` as a
    score of ``1.0`` and COMMITs the error message as the answer — observed
    live against a llama-server with no model loaded (WSL Ubuntu 24.04).
    """
    stripped = str(text).strip()
    if not stripped:
        return False
    return stripped.startswith("[Error:") or stripped.startswith("[Error: HTTP")


def _parse_score(reply: str) -> float:
    """Extract a 0.0-1.0 score from a judge reply; unparseable -> 0.0.

    A provider error string (``[Error: HTTP Error 400: …]``) is NOT a judge
    reply — its first number is the HTTP status code, which clamps to 1.0.
    Such text scores 0.0 so a failed branch can never be committed.
    """
    if _is_provider_error(reply):
        return 0.0
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


#: Pseudo-tool XML a weak branch model writes INSTEAD of calling the tool.
#: Observed live (qwen2.5-coder-1.5b, pure `--judge jev`): the branch emitted
#: ``<definitions path="path/to/jev_integration.py">``, ``<references ...>``,
#: ``<search ...>`` and ``<web_search ...>`` blocks with placeholder paths, did
#: NOT call a single tool, and the 1.5B judge rated its own hallucination 0.83
#: — it was COMMITted.  Such text is a non-answer, never a candidate.
_FAKE_TOOL_XML_RE = re.compile(
    r"<\s*/?\s*(?:definitions|references|search|web_search|read|read_skill|"
    r"list_files|diff|tests|file|result|tool_result)\b[^>]*\b"
    r"(?:path|query|symbol|file|max_results|line)\s*=",
    re.IGNORECASE,
)

#: Placeholder markers that betray fabricated (not real) tool output.
_PLACEHOLDER_MARKERS = ("path/to/", "example.com", "your_file", "your/path")


def _looks_like_fabricated_output(text: str) -> bool:
    """True when *text* fabricates tool output instead of answering.

    A weak model mimics the tool XML it saw in the system prompt with
    placeholder paths and never calls the tool; that is a non-answer, not a
    grounding.  Applied to every branch (and to the scorer), so a fabricated
    answer can never be committed even if a judge rates it highly.
    """
    if not text:
        return False
    if _FAKE_TOOL_XML_RE.search(text):
        return True
    low = text.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


#: Hints that a question is about THIS repo/codebase (so a branch must ground
#: its answer with a tool call instead of answering from memory).
_REPO_QUESTION_HINTS = (
    "repo", "repository", "workspace", "codebase", "this project",
    "agent.py", "agent1", "jev", "speculate", "harnessfix", "implement",
    "fix command", "pytest", "mypy", "ruff", "module", "function",
    "class", "file", "code",
)
_CLAIM_PATH_RE = re.compile(
    r"(?<![\w/\\])([A-Za-z0-9_][\w./\\-]*\.(?:py|md|json|toml|txt|cfg|yaml|yml|ini))"
    r"(?::(\d+))?"
)


def _is_repo_question(text: str) -> bool:
    """True when *question* is about this workspace/codebase."""
    low = str(text).lower()
    if _CLAIM_PATH_RE.search(text):
        return True
    return any(hint in low for hint in _REPO_QUESTION_HINTS)


def _verify_claims(answer: str, workspace: str) -> list[str]:
    """Check every ``file[:line]`` claim in *answer* against the workspace.

    Returns human-readable mismatches (empty = every claim checks out).  A
    missing file or a line past EOF means the answer describes code that is not
    there — it must not be COMMITted as verified fact.  Deterministic and
    free: no judge involved.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for match in _CLAIM_PATH_RE.finditer(str(answer)):
        rel, line = match.group(1), match.group(2)
        if rel in seen:
            continue
        seen.add(rel)
        candidate = Path(rel)
        if not candidate.is_absolute():
            candidate = Path(workspace) / rel
        if (
            not candidate.is_file()
            and "/" not in rel
            and "\\" not in rel
        ):
            # A bare basename ("speculate_cmd.py:324"): resolve it anywhere in
            # the workspace instead of falsely calling it missing.
            try:
                matches = list(Path(workspace).rglob(rel))
            except OSError:
                matches = []
            if len(matches) == 1:
                candidate = matches[0]
        if not candidate.is_file():
            problems.append(f"{rel} does not exist")
            continue
        if line:
            try:
                with candidate.open(encoding="utf-8", errors="replace") as handle:
                    count = sum(1 for _ in handle)
            except OSError:
                continue
            if int(line) > count:
                problems.append(f"{rel}:{line} is past EOF ({count} lines)")
    return problems


class SpeculateCommand(Command):
    """Run speculative LLM branches and probabilistically commit or refuse."""

    @property
    def name(self) -> str:
        return "speculate"

    @property
    def help_text(self) -> str:
        return (
            'speculate "question" [--branches N] [--threshold X] [--low X] '
            "[--branch-model chat|jev] [--escalate] [--no-grounding] "
            "[--timeout S] [--judge llm|jev|both] - ask the LLM N independent "
            "speculative branches in parallel, judge-score each answer, and "
            "COMMIT only when the best score meets the threshold (otherwise "
            "REFUSE); repo/code questions require a tool call per branch and "
            "file:line claims are verified against the workspace before COMMIT; "
            "--judge jev scores with the dedicated small Jev model (the "
            "reasoning model still generates the branches); --branch-model jev "
            "generates the branches on the Jev model too (cheap/offline, for "
            "general questions); --escalate opts into gray-band escalation to "
            "the chat-model judge; both averages the two"
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
        #: Gray-band floor for the Jev cascade: a Jev score at/below this is a
        #: confident reject (no LLM call); a score between --low and --threshold
        #: is uncertain and escalates ONE candidate to the LLM judge.
        low = 0.3
        judge_mode = "llm"
        #: Opt-in gray-band escalation to the chat model's LLM judge.  OFF by
        #: default: a Jev command must not touch the selected chat model, so
        #: `--judge jev` is fully independent unless the user asks to escalate.
        escalate = False
        #: Which model GENERATES the branches: the selected chat model
        #: (default — branches are the "thinking" half) or the dedicated Jev
        #: model (`--branch-model jev`, cheap/offline deliberation).  Jev's real
        #: value is the JUDGE: a small model decides, the reasoning model
        #: thinks.  A 1.5B branch cannot ground a repo question.
        branch_model = "chat"
        #: Repo/code questions require every branch to call at least one tool
        #: before its answer is a candidate (ungrounded code claims were
        #: COMMITted with a perfect judge score).  `--no-grounding` disables.
        no_grounding = False
        # Branch-dispatch wait.  Each branch carries the agent's full system
        # prompt AND may make several read-only tool round-trips, so a local
        # reasoning model needs real room; 60s/300s aborted whole
        # deliberations (a 3-branch 27B run timed out at 300s).  Override with
        # --timeout.
        timeout = 600.0

        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "--judge" and i + 1 < len(parts):
                judge_mode = parts[i + 1].strip('"').lower()
                if judge_mode not in ("llm", "jev", "both"):
                    self.error("--judge must be one of: llm, jev, both")
                    return True
                i += 2
                continue
            if p == "--escalate":
                escalate = True
                i += 1
                continue
            if p == "--no-grounding":
                no_grounding = True
                i += 1
                continue
            if p == "--branch-model" and i + 1 < len(parts):
                branch_model = parts[i + 1].strip('"').lower()
                if branch_model not in ("chat", "jev"):
                    self.error("--branch-model must be one of: chat, jev")
                    return True
                i += 2
                continue
            if p == "--low" and i + 1 < len(parts):
                try:
                    low = float(parts[i + 1])
                except ValueError:
                    self.error("--low expects a number between 0 and 1.")
                    return True
                if not 0.0 <= low <= 1.0:
                    self.error("--low must be within [0.0, 1.0]")
                    return True
                i += 2
                continue
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

        if low > threshold:
            self.error("--low must be <= --threshold")
            return True

        skip_values = {
            "--branches", "--threshold", "--timeout", "--judge", "--low",
            "--branch-model",
        }
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

        # A question about this repo/codebase must be grounded: a branch that
        # answers from memory is not a candidate (it produced confident,
        # wrong file/line claims that a same-model judge scored 1.00).
        require_grounding = (not no_grounding) and _is_repo_question(question)

        # Reasons a branch was disqualified (ungrounded / fabricated / leaked
        # tool call).  Surfaced on REFUSE so the user sees WHY nothing was
        # eligible instead of a bare "no candidate reached threshold".
        branch_errors: list[str] = []

        def _fail(branch_id: int, reason: str) -> dict[str, Any]:
            branch_errors.append(reason)
            return {"branch": branch_id, "error": reason}

        llm = agent.llm

        # The Jev engine is the DEDICATED small model.  Jev's value is the
        # DECISION: `--judge jev` scores with it while the reasoning model
        # generates the branches.  `--branch-model jev` also generates the
        # branches with it (cheap/offline, but a 1.5B cannot ground a repo
        # question).  A build failure degrades gracefully in either role.
        need_jev = judge_mode in ("jev", "both") or branch_model == "jev"
        jev_engine: Any = None
        if need_jev:
            try:
                from agent_core.jev_engine import build_jev_engine

                jev_engine = build_jev_engine(
                    source="speculate",
                    log_workspace=getattr(agent, "workspace", None),
                )
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                if judge_mode in ("jev", "both"):
                    print(
                        "  [speculate] Jev judge unavailable "
                        f"({exc}); using LLM judge"
                    )
                    judge_mode = "llm"
                if branch_model == "jev":
                    print(
                        "  [speculate] Jev branches unavailable "
                        f"({exc}); using chat branches"
                    )
                    branch_model = "chat"

        # Branch model: the chat model by default; the Jev provider only when
        # explicitly requested.
        branch_llm: Any = llm
        if branch_model == "jev" and jev_engine is not None:
            branch_llm = getattr(jev_engine, "provider", None) or llm
            # Select/load the small model BEFORE the branches run.
            try:
                await jev_engine.ensure_ready()
            except Exception:  # noqa: BLE001 - request path reports failures
                pass

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
                f"{', '.join(sorted(_BRANCH_TOOLS))}. Do not call any other tool. "
                "NEVER invent tool output: do not write XML/tags like "
                "<definitions ...> or <search ...>, and no placeholder paths "
                "like path/to/x.py — if you need a fact, actually call the tool."
            )})
            answer = ""
            tool_calls_made = 0
            for _ in range(_BRANCH_MAX_ITERS):
                reply = str(asyncio.run(
                    branch_llm.chat(
                        messages, tools=branch_tools,
                        max_tokens=_BRANCH_MAX_TOKENS,
                    )
                ))
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
                    tool_calls_made += 1
            else:
                # Still calling tools at the cap: force a final text answer
                # with tools withheld so the branch always returns prose.
                answer = str(asyncio.run(
                    branch_llm.chat(messages, max_tokens=_BRANCH_MAX_TOKENS)
                ))
            if _looks_like_tool_call(answer):
                return _fail(
                    branch_id, "emitted a raw tool call instead of an answer",
                )
            if _looks_like_fabricated_output(answer):
                return _fail(
                    branch_id, "fabricated tool output instead of answering",
                )
            if require_grounding and tool_calls_made == 0:
                return _fail(
                    branch_id,
                    "repo question answered without calling any tool "
                    "(ungrounded code claims are not candidates)",
                )
            return {"branch": branch_id, "answer": answer}

        # Optional Jev judge — the dedicated small model scores each candidate
        # with a typed yes/no ("is this answer correct and complete?"), giving a
        # probability instead of a single subjective LLM score.  The engine was
        # built above (and, in pure `--judge jev` mode, already supplies the
        # branch provider).

        def _jev_judge(answer: str) -> float | None:
            """P(yes) that *answer* correctly+completely answers the question."""
            from agent_core.jev_engine import JevQuestion

            question_obj = JevQuestion(
                kind="yesno",
                text=(
                    "The QUESTION below is answered correctly and completely "
                    "by the ANSWER.\n"
                    f"QUESTION: {question}\nANSWER: {answer}"
                ),
            )
            try:
                result = asyncio.run(jev_engine.decide(question_obj))
            except Exception:  # noqa: BLE001 - treat as unscored
                return None
            if result.decision == "UNKNOWN":
                return None
            return float(result.probabilities.get("yes", 0.0))

        def _llm_judge(answer: str) -> float:
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

        def scorer(result: dict[str, Any]) -> float:
            """Score one candidate 0.0-1.0 with the configured judge(s).

            A non-answer (empty, or a raw tool call) scores 0.0 WITHOUT asking
            any judge — the LLM judge once rated a leaked tool call 1.0 and it
            was committed.  ``judge_mode`` selects the LLM judge, the small Jev
            model, or the average of both (whichever is available).
            """
            answer = str(result.get("answer", "")).strip()
            if (
                not answer
                or _looks_like_tool_call(answer)
                or _looks_like_fabricated_output(answer)
                or _is_provider_error(answer)
            ):
                return 0.0
            llm_score: float | None = None
            if judge_mode in ("llm", "both"):
                llm_score = _llm_judge(answer)
            if judge_mode == "llm":
                return llm_score if llm_score is not None else 0.0
            jev_score = _jev_judge(answer)
            if judge_mode == "jev":
                if jev_score is None:
                    return 0.0
                # Optional cascade (--escalate): a confident Jev verdict costs
                # one small-model call; only the gray band (low < P < threshold)
                # escalates THIS candidate to the chat model's LLM judge.  OFF
                # by default so `--judge jev` never touches the selected model.
                if escalate and low < jev_score < threshold:
                    llm_score = _llm_judge(answer)
                    if llm_score is not None:
                        return max(jev_score, llm_score)
                return jev_score
            # both — average whatever is available
            if jev_score is None:
                return llm_score if llm_score is not None else 0.0
            if llm_score is None:
                return jev_score
            return (jev_score + llm_score) / 2.0

        judge_note = f"judge={judge_mode}"
        if judge_mode in ("jev", "both") and jev_engine is not None:
            # Make the DEDICATED Jev model visible up front: it is independent
            # of the branch model, and printing it here proves the judge is not
            # running on the agent's chat model.
            judge_note += (
                f", jev_model={getattr(jev_engine, 'model_name', '?')}"
            )
        if branch_llm is llm:
            judge_note += ", branch_model=chat"
        else:
            judge_note += (
                f", branch_model={getattr(branch_llm, 'model_name', 'jev')}"
            )
        if judge_mode == "jev":
            judge_note += (
                f", low={low:g}, escalate={'on' if escalate else 'off'}"
            )
        judge_note += f", grounding={'on' if require_grounding else 'off'}"
        print(
            f"\n  [speculate] {num_branches} branch(es), threshold={threshold:g}, "
            f"timeout={timeout:g}s, {judge_note}"
        )

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
            best = decision.best.result
            if isinstance(best, dict):
                answer = str(best.get("answer", best))
            else:
                answer = str(best)
            # Verify file[:line] claims against the REAL workspace before
            # committing: an answer describing code that is not there is not a
            # verified fact, however confidently it was judged (deterministic,
            # free — no judge call).
            workspace = str(getattr(agent, "workspace", "") or os.getcwd())
            problems = _verify_claims(answer, workspace)
            if problems:
                print("  [speculate] REFUSE - unverified file/line claim(s):")
                for problem in problems[:5]:
                    print(f"    - {problem}")
                return True
            print(f"  [speculate] COMMIT score={decision.best.score:.2f} "
                  f"threshold={threshold:g}")
            for line in answer.splitlines() or [answer]:
                print(f"  {line}")
        else:
            print(f"  [speculate] REFUSE - no candidate reached "
                  f"threshold {threshold:g}")
            if branch_errors:
                print("  [speculate] branch failures (deterministic guards):")
                for reason in sorted(set(branch_errors))[:3]:
                    print(f"    - {reason}")
        return True
