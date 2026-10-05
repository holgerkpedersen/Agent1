#!/usr/bin/env python3
"""Agentic benchmark scenarios: multi-step tasks scored against rubrics.

Roadmap item 12: the harness-fix gates need a richer, more realistic task
set than ``benchmark.py``'s single-shot prompts — bounded multi-step agent
runs driven through the REAL main tool loop (``ToolLoopRunner.run`` via the
thin :func:`run_tool_loop` wrapper below) on the real main-loop tool surface
(``NLP_TOOL_SCHEMAS`` minus delegation tools), scored per run against a
scenario rubric. Results are appended to ``reports/benchmarks/agentic_bench.json``
in the exact shape ``harnessfix/gates.py:run_agentic_gate`` reads
(``{"models": [{"display_name", "runs": [{"score"}]}``).

This module owns no loop of its own: it builds one fresh Agent per scenario,
hands its own ``llm.chat`` (the real ``LLMClient``, wired to the transport
factory the caller injects) to ``run_tool_loop``, and lets the MAIN-loop
runner drive every step. Scenarios never delegate — the whole point is to
measure how well the main loop completes a multi-step task on its own.

Usage:
    python agentic_bench.py                       # all scenarios, default model
    python agentic_bench.py --model llama/x      # one model, all scenarios
    python agentic_bench.py --scenario shell     # filter by category id
    python agentic_bench.py --out PATH          # report file (default: canonical)
    python agentic_bench.py --trend m1,m2        # print trend table and exit

The same scenarios gate the HarnessFix loop: ``harnessfix/gates.py`` calls
:func:`run_agentic_gate`, which ``harnessfix/loop.py::run_loop`` samples
before/after a repair (see harnessfix/gates.py).  Every scenario runs inside a
throwaway sandbox workspace (see :func:`sandbox_workspace`) — a benchmark must
never mutate the tree it is measuring.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agent import Agent, LLMClient
from agent_core.constants import REPORTS_DIR
from agent_core.llm.tool_loop import ToolLoopRunner
from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

if TYPE_CHECKING:  # import cycle: harnessfix.gate reads this module back
    from harnessfix.judge import JudgeVerdict

__all__ = [
    "MARKER_TOKEN", "PASS_THRESHOLD", "RunOutcome", "Scenario",
    "ScenarioResult", "TransportFactory", "build_main_surface", "main",
    "make_agent", "parse_args", "print_trend", "run_scenarios",
    "run_tool_loop", "run_agentic_gate", "load_report_snapshots",
    "save_report", "SCENARIOS", "DEFAULT_REPETITIONS",
]


#: Unique token embedded in every scenario prompt. The injected fake
#: transport echoes it back inside its final reply, so a clean run (the
#: model answered with the marker verbatim) scores 1.0 and an interrupted
#: run does not — see FakeTransport in tests/test_agentic_bench.py.
MARKER_TOKEN = "<<agentic-bench-marker>>"

#: Weighted rubric score a single run needs to count as passed (0..1).
PASS_THRESHOLD = 0.8

#: Runs per scenario by default; scores are averaged over these repetitions.
DEFAULT_REPETITIONS = 3

#: Per-run iteration cap for gate runs (matches the CLI default).
_GATE_MAX_ITERATIONS = 150

#: Tool names that delegate work to other agents. The benchmark measures how
#: the MAIN loop completes a task on its own, so these are never advertised
#: to a scenario run and must never be executed by one.
DELEGATION_TOOLS: frozenset[str] = frozenset({
    "delegate", "delegate_batch", "create_subagent", "run_subagent_task",
})


def report_path() -> Path:
    """Return the agentic_bench.json report path in the canonical location.

    ``REPORTS_DIR`` is a plain string (os.path.join) in agent_core.constants,
    so wrap it in Path before using the ``/`` join operator.
    """
    return Path(REPORTS_DIR) / "benchmarks" / "agentic_bench.json"


@runtime_checkable
class TransportFactory(Protocol):
    """Builds the transport an Agent's ``LLMClient`` talks to."""

    def __call__(self, agent: Agent) -> Any: ...


# ---------------------------------------------------------------------------
# Scenario bank
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """One multi-step benchmark task with a weighted rubric to score it by."""

    id: str          # stable slug (scn-<category>-<n>) used in CLI filters
    category: str    # shell, file_edit, git, mcp, multi_tool, ... (lower-case)
    style: str       # instruction style this scenario exercises
    prompt: str      # task text handed to the agent (contains MARKER_TOKEN)
    expected_outcome: str   # what a clean run must show (names the marker)
    rubric: tuple[str, ...]  # weighted criteria; each ends with (weight=N)


@dataclass(frozen=True)
class RunOutcome:
    """What one bounded agent run produced — inputs to ``score_run``."""

    response: str          # final assistant text the loop returned
    turns: int             # assistant messages the loop recorded
    tools_used: int        # tool calls executed during the run


@dataclass(frozen=True)
class ScenarioResult:
    """Scored outcome of one scenario, averaged over its repetitions.

    One result per scenario per batch (``len(results) == len(SCENARIOS)``).
    ``repetitions`` carries every raw per-repetition :class:`RunOutcome`.
    """

    scenario: Scenario
    model: str
    profile: str
    repeat_index: int      # 0-based index among the repetitions (aggregate = 0)
    response: str          # first repetition's final answer (representative)
    turns: int             # assistant messages recorded by the loop
    tools_used: int        # tool calls executed during the run
    latency_ms: float      # wall-clock ms for the whole scenario
    passed: bool           # mean score >= PASS_THRESHOLD
    score: float           # weighted rubric score in [0, 1], averaged over reps
    notes: list[str] = field(default_factory=list)  # per-criterion verdicts
    repetitions: list[dict[str, Any]] = field(default_factory=list)


def _build_bank() -> tuple[Scenario, ...]:
    """Return the benchmark scenario bank (the shipped set)."""
    return (
        Scenario(
            id="scn-shell-1", category="shell", style="plain",
            prompt=f"Use the run tool to print exactly `{MARKER_TOKEN}1` and "
                   "nothing else. This is a single-step task: finish fast.",
            expected_outcome=f"The final answer quotes `{MARKER_TOKEN}1` verbatim.",
            rubric=(f"The assistant's final answer quotes `{MARKER_TOKEN}1` "
                    f"verbatim {_w(3)}",
                    f"the run tool was used at least once and the loop "
                    f"terminated cleanly {_w(2)}"),
        ),
        Scenario(
            id="scn-file-edit-1", category="file_edit", style="plain",
            prompt=f"Create scratch/probe.txt containing exactly `{MARKER_TOKEN}2` "
                   "followed by a newline, then print the file's contents. "
                   "Single-step.",
            expected_outcome=(f"The tool sequence creates `scratch/probe.txt`, writes "
                             f"`{MARKER_TOKEN}2` and a single trailing newline to it, "
                             "then reads the file back; the final answer quotes the "
                             "file contents verbatim."),
            rubric=(f"the final answer quotes `{MARKER_TOKEN}2` verbatim {_w(5)}",
                    "a write/write-style tool created scratch/probe.txt before "
                    f"reading it (reading first is a process error) {_w(3)}"),
        ),
        Scenario(
            id="scn-file-edit-2", category="file_edit", style="verbose",
            prompt=f"Create scratch/notes.md, write exactly `{MARKER_TOKEN}3` "
                   "into it and read it back. This is a single-step task: "
                   "finish as fast as possible.",
            expected_outcome=(f"`scratch/notes.md` exists with content "
                             f"`{MARKER_TOKEN}3`; the final answer quotes the file "
                             "contents verbatim."),
            rubric=(f"the final answer quotes `{MARKER_TOKEN}3` verbatim {_w(5)}",
                    "a write/write-style tool created scratch/notes.md before "
                    f"reading it back; reading first is a process error {_w(3)}"),
        ),
        Scenario(
            id="scn-git-1", category="git", style="verbose",
            prompt=(f"Commit the staged changes on the current branch with message "
                   f"`{MARKER_TOKEN}4` and push. Single-step."),
            expected_outcome=(f"the commit subject contains `{MARKER_TOKEN}4`, "
                             "it lands on the current checked-out branch, the "
                             "working tree is clean afterwards; HEAD is not moved "
                             "and no new branch is created."),
            rubric=("a git tool stages all changes with a bare `-A` (not `git "
                    f"add .`) first {_w(2)}",
                    f"the commit subject contains `{MARKER_TOKEN}4` verbatim {_w(5)}"),
        ),
        Scenario(
            id="scn-mcp-1", category="mcp", style="verbose",
            prompt=(f"Call the `read` tool on scratch/harnessfix/README.md with a "
                   f"Python one-liner, print its output verbatim, then answer "
                   f"with `{MARKER_TOKEN}5` on a line of its own. Single-step."),
            expected_outcome=(f"The final answer ends with a line containing "
                             f"exactly `{MARKER_TOKEN}5` after the tool output is "
                             "echoed."),
            rubric=(f"the final answer contains `{MARKER_TOKEN}5` on a line of its "
                    f"own {_w(5)}",
                    "the `read` tool was invoked on "
                    "scratch/harnessfix/README.md before the final answer "
                    f"{_w(3)}"),
        ),
        Scenario(
            id="scn-multi-tool-1", category="multi_tool", style="verbose",
            prompt=(f"Create scratch/a.txt with content `{MARKER_TOKEN}6a`, create "
                   f"scratch/b.txt with content `{MARKER_TOKEN}6b`, read both files "
                   f"back, then answer with a line containing `{MARKER_TOKEN}6a` "
                   f"then a line containing `{MARKER_TOKEN}6b`."),
            expected_outcome=(f"The final answer contains `{MARKER_TOKEN}6a` on one "
                             f"line and `{MARKER_TOKEN}6b` on another, in that order."),
            rubric=(f"the final answer contains `{MARKER_TOKEN}6a` on its own line "
                    f"{_w(3)}",
                    f"the final answer contains `{MARKER_TOKEN}6b` on its own line "
                    f"{_w(4)}"),
        ),
    )


def _w(n: int) -> str:
    """Rubric suffix marking a criterion's weight (weight=N)."""
    return f"(weight={n})"


SCENARIOS: tuple[Scenario, ...] = ()

# The scenarios above are illustrative placeholders; the full bank lives in
# this file so tests can import it. See _build_bank() for the real set.
SCENARIOS = _build_bank()


# ---------------------------------------------------------------------------
# Runner: main loop + main-loop tool surface, injected transport factory
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def sandbox_workspace() -> Iterator[Path]:
    """Yield a throwaway workspace for scenarios to run in.

    Scenarios execute REAL main-loop tools (``run``, ``write``, ``edit``,
    ``git`` …) through the real dispatcher, and one scenario commits and
    pushes.  Running them in the live checkout would let a benchmark mutate —
    and publish — the tree it is measuring: ``scratch/*`` litter in the repo,
    a spurious commit, and a ``git push`` to ``origin``.  The project already
    enforces this for the harness itself (see
    ``tests/test_autonomous_sandbox.py``); the benchmark must hold the same
    line.

    The sandbox is a temp dir with its own ``git init`` so the git scenario
    can stage/commit without touching the real repository.  It is *not* a
    clone: scenarios only need a scratch area plus a commit-able repo, and a
    clone of this checkout would be slow and would carry its ``origin``.
    """
    tmp = Path(tempfile.mkdtemp(prefix="agentic-bench-sbx-"))
    try:
        # Best-effort local repo so scn-git-1 has something to commit into;
        # a failure here (no git on PATH) must not abort the whole gate —
        # the other scenarios do not need it.
        for argv in (
            ["git", "init"],
            ["git", "config", "user.email", "bench@localhost"],
            ["git", "config", "user.name", "agentic-bench"],
        ):
            try:
                subprocess.run(
                    argv, cwd=tmp, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=30,
                )
            except (OSError, subprocess.SubprocessError):
                break
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_agent(workspace: str, model_name: str) -> Agent:
    """Build a fresh main-loop Agent for one scenario turn.

    The gate passes the same ``model`` it passes to ``run_benchmark_gate``,
    so gate runs land under the same display_name key (the plain model name)
    the gates read back from the report file.  *workspace* should be a
    sandbox (see :func:`sandbox_workspace`), never the live checkout.
    """
    return Agent(workspace=workspace, model_name=model_name)


def build_main_surface(parent: Agent) -> list[dict[str, Any]]:
    """Return the MAIN-loop tool surface for one turn.

    The full ``NLP_TOOL_SCHEMAS`` list minus the delegation tools: scenarios
    must run on the main loop, so the runner never hands a scenario the
    delegate/plan/subagent tools. Fresh dicts per call — the schemas are
    shared module constants and runners must not mutate them.
    """
    return [dict(s) for s in NLP_TOOL_SCHEMAS
            if s.get("function", {}).get("name") not in DELEGATION_TOOLS]


async def run_tool_loop(
    transport: LLMClient, messages: list[dict[str, Any]],
    tool_defs: list[dict[str, Any]],
    *, repetitions: int = 1, max_iterations: int = _GATE_MAX_ITERATIONS,
) -> list[RunOutcome]:
    """Drive *messages* through the real main-loop runner, one outcome per rep.

    Thin public wrapper over :meth:`ToolLoopRunner.run`: it builds the loop
    (fresh instance per repetition so no loop state leaks across runs),
    adapts the transport's ``chat`` and the agent's dispatcher to the
    signatures the runner expects, and converts each ``(response, messages)``
    pair into a :class:`RunOutcome`. Returns one outcome per repetition.
    """
    outcomes: list[RunOutcome] = []

    async def llm_chat_fn(
        msgs: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """One model turn against the injected transport."""
        raw = await transport.chat(msgs, tools=tools)
        # The provider contract (see LLMClient.chat / provider.py docstrings):
        # chat returns plain assistant TEXT on a final answer and only a JSON
        # object when tool_calls are present.  Parse defensively: try JSON,
        # accept an object; anything else is the final text as-is.
        parsed: Any = None
        if isinstance(raw, dict):
            parsed = raw
        elif isinstance(raw, str):
            try:
                obj = json.loads(raw)
            except (ValueError, TypeError):
                obj = None
            if isinstance(obj, dict):
                parsed = obj
        if parsed is not None:
            calls = parsed.get("tool_calls")
            if isinstance(calls, list) and calls:
                # The runner reads the batch as a LIST under the "tool_calls"
                # key (tool_loop.py: ToolLoopRunner.run) — the same shape the
                # real providers emit (see lemonade_provider / opencode_provider
                # _encode_tool_response). Spreading calls[0] flat instead put
                # id/type/function on the message, left "tool_calls" absent,
                # and silently disabled every tool call a scenario made.
                updated = [*msgs, {
                    "role": "assistant",
                    "content": str(parsed.get("content") or ""),
                    "tool_calls": calls,
                }]
                return str(parsed.get("content") or ""), updated
            content = parsed.get("content")
            text = str(content) if content is not None else raw
            return text, [*msgs, {"role": "assistant", "content": text}]
        return raw, [*msgs, {"role": "assistant", "content": raw}]

    async def execute_tool(name: str, args: dict[str, Any]) -> str:
        """Execute one main-loop tool call through the real dispatcher.

        Delegation tools are refused outright: a scenario must be completed
        by the MAIN loop, never farmed out to subagents.
        """
        if name.lower() in DELEGATION_TOOLS:
            return "refused: delegation tools are not available in benchmark runs"
        return await transport._agent._execute_tool_call(name.lower(), args)

    for _rep in range(max(1, int(repetitions))):
        runner = ToolLoopRunner(max_iterations=max_iterations)
        response, final_messages = await runner.run(
            list(messages), llm_chat_fn, execute_tool, tools=list(tool_defs),
        )
        turns = sum(1 for m in final_messages if m.get("role") == "assistant")
        tools_used = sum(
            len(m.get("tool_calls") or [])
            for m in final_messages
            if isinstance(m, dict) and m.get("role") == "assistant"
        )
        outcomes.append(
            RunOutcome(
                response=str(response), turns=turns, tools_used=tools_used
            )
        )

    return outcomes


async def run_scenarios(
    agent: Agent,
    model: str = "",
    profile: str = "",
    repetitions: int = DEFAULT_REPETITIONS,
    *,
    max_iterations: int = _GATE_MAX_ITERATIONS,
) -> list[ScenarioResult]:
    """Run every scenario *repetitions* times on the MAIN loop and score it.

    Builds a fresh Agent per repetition through ``make_agent`` (the injected
    factory), wires its ``LLMClient`` to a transport built by the injected
    factory, then hands the prompt, the main-loop tool surface and the
    client to :func:`run_tool_loop` — one call per scenario. Returns ONE
    aggregated result per scenario: scores are averaged over the repetitions.

    Scenarios run against *agent.workspace*: pass a sandbox (see
    :func:`sandbox_workspace`) so real tool calls cannot mutate a live
    checkout.  The offline tests pass their own tmp workspace directly.
    """
    reps = max(1, int(repetitions))
    results: list[ScenarioResult] = []

    for scenario in SCENARIOS:
        prompt_text = (f"{scenario.prompt}\n(Report your result in text; "
                       f"{MARKER_TOKEN} must appear verbatim.)")
        started = time.perf_counter()
        outcomes: list[RunOutcome] = []
        for _rep in range(reps):
            turn_agent = make_agent(agent.workspace, model)
            # Patching build_transport (tests) swaps a fake transport into
            # the turn agent's LLMClient — exactly what run_tool_loop uses.
            transport: Any = turn_agent.llm
            transport._agent = turn_agent
            outcomes.extend(await run_tool_loop(
                transport, [{"role": "user", "content": prompt_text}],
                build_main_surface(agent), repetitions=1, max_iterations=max_iterations,
            ))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        scores: list[float] = []
        rep_rows: list[dict[str, Any]] = []
        notes: list[str] = []
        for idx, outcome in enumerate(outcomes):
            score, note = score_run(outcome, scenario.rubric, model, profile)
            scores.append(score)
            notes.append(f"rep {idx}: {note}")
            rep_rows.append({"repeat_index": idx, "response": outcome.response,
                             "turns": outcome.turns, "tools_used": outcome.tools_used,
                             "score": round(score, 4)})
        mean = sum(scores) / len(scores) if scores else 0.0
        first = outcomes[0] if outcomes else RunOutcome("", 0, 0)
        results.append(ScenarioResult(
            scenario=scenario, model=model, profile=profile, repeat_index=0,
            response=first.response, turns=first.turns, tools_used=first.tools_used,
            latency_ms=round(elapsed_ms, 1), passed=mean >= PASS_THRESHOLD,
            score=round(mean, 4), notes=list(notes), repetitions=list(rep_rows),
        ))
        print(f"[agentic-bench] scenario {scenario.id} done ({reps} reps)")

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

WEIGHT_RE = re.compile(r"\(weight=(\d+)\)$")


def parse_rubric(text: str) -> tuple[str, int]:
    """Split a rubric line into (criterion text, integer weight)."""
    match = WEIGHT_RE.search(text.strip())
    weight = int(match.group(1)) if match else 1
    return WEIGHT_RE.sub("", text).strip(), weight


def score_run(
    outcome: RunOutcome, rubric: tuple[str, ...], model: str, profile: str
) -> tuple[float, list[str]]:
    """Judge every criterion of *rubric* against one run's outcome.

    The weighted mean is normalized by the TOTAL weight, not by the number of
    criteria, so the score stays in [0, 1] and ``PASS_THRESHOLD`` means what
    it says.  Dividing a weight-summed total by ``len(rubric)`` inflated every
    score above 1.0 (scn-shell-1, weights 3+2, scored a *perfect* run 2.5),
    which made the pass threshold vacuous and let ``should_accept_agentic``
    compare incomparable magnitudes.
    """
    total = 0.0
    weight_sum = 0
    notes: list[str] = []
    for text in rubric:
        criterion, weight = parse_rubric(text)
        verdict = judge(
            outcome.response, outcome.turns, outcome.tools_used,
            criterion, model, profile,
        )
        notes.append(f"{verdict.verdict}: {verdict.reason}")
        total += weight * verdict.score
        weight_sum += weight
    if weight_sum <= 0:
        return 0.0, notes
    return total / weight_sum, notes


def judge(
    response: str, turns: int, tools_used: int, rubric_text: str,
    model: str, profile: str,
) -> JudgeVerdict:
    """Judge one rubric line against one run's outcome."""
    from harnessfix.judge import judge_outcome

    return judge_outcome(
        response=response, turns=turns, tools_used=tools_used,
        rubric_text=rubric_text, model=model, profile=profile,
    )


# ---------------------------------------------------------------------------
# Report file (consumed by harnessfix/gates.py:run_agentic_gate)
# ---------------------------------------------------------------------------

def save_report(
    results: list[ScenarioResult], out_path: str | Path | None = None, *,
    model: str = "", profile: str = "", reps: int = DEFAULT_REPETITIONS,
) -> Path:
    """Write the report file for *results* and return its path.

    The file is rebuilt from this batch — one entry per model/profile key,
    runs sorted by scenario id; ``overall_accuracy`` is the mean score in
    percent, which is what ``gates.py:run_agentic_gate`` reads back. The
    first positional argument after *results* is the output path (defaults
    to :func:`report_path`).
    """
    path = Path(out_path) if out_path else report_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _display(m: str, p: str) -> str:
        return f"{m} [{p}]" if p else m

    rows: list[dict[str, Any]] = []
    for res in results:
        rows.append({
            "id": f"{res.scenario.id}#{res.repeat_index}",
            "category": res.scenario.category,
            "style": res.scenario.style,
            "prompt": res.scenario.prompt,
            "expected_outcome": res.scenario.expected_outcome,
            "rubric": list(res.scenario.rubric),
            "model": res.model,
            "profile": res.profile,
            "repeat_index": res.repeat_index,
            "response": res.response,
            "turns": res.turns,
            "tools_used": res.tools_used,
            "latency_ms": res.latency_ms,
            "passed": res.passed,
            "score": res.score,
        })
    rows.sort(key=lambda r: (r["id"], r["model"], r["profile"]))

    scores = [float(r.get("score", 0.0)) for r in rows]
    overall = round(sum(scores) / len(scores) * 100.0, 2) if scores else 0.0

    data: dict[str, Any] = {"models": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("models"), list):
                data = loaded
        except (json.JSONDecodeError, OSError):
            pass

    display = _display(model, profile)
    entry: dict[str, Any] | None = None
    for candidate in data["models"]:
        cand_display = _display(
            candidate.get("model", ""), candidate.get("profile", "")
        )
        if cand_display == display:
            entry = candidate
            break
    if entry is None:
        entry = {"model": model, "profile": profile, "repetitions": reps,
                 "runs": [], "display_name": display}
        data["models"].append(entry)
    entry["repetitions"] = reps
    entry["runs"] = rows
    entry["overall_accuracy"] = overall
    data.setdefault("meta", {})["last_run"] = (
        datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"))

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))  # atomic: readers never see a partial file
    return path


def load_report_snapshots(path: str | None = None) -> list[dict[str, Any]]:
    """Load the report snapshots from ``agentic_bench.json`` (empty if absent)."""
    p = Path(path) if path else report_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    models = data.get("models") if isinstance(data, dict) else None
    return [e for e in models or [] if isinstance(e, dict)]


def print_trend(model: str, path: str | None = None) -> list[dict[str, Any]]:
    """Print the per-run score history for *model* and return the snapshots."""
    snapshots = load_report_snapshots(path)
    rows = [e for e in snapshots if e.get("display_name") == model]
    if not rows:
        print(f"[agentic-bench] no runs recorded for {model!r} yet")
        return snapshots
    for entry in sorted(rows, key=lambda e: str(e["display_name"])):
        scores = [float(r.get("score", 0.0)) for r in entry.get("runs", [])]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{entry['display_name']} [{len(scores)} runs] mean={avg:.1%}")
    return snapshots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the benchmark CLI (list / trend / single-scenario flags)."""
    parser = argparse.ArgumentParser(
        prog="agentic_bench.py", description=__doc__.splitlines()[0].strip(),
    )
    parser.add_argument("--model", default="", metavar="MODEL",
                        help="run every scenario on this model name")
    parser.add_argument(
        "--scenario", default="", metavar="ID",
        help="limit to scenarios whose id/category contains this text "
             f"({', '.join(s.id for s in SCENARIOS)})",
    )
    parser.add_argument(
        "--repetitions", type=int, default=DEFAULT_REPETITIONS, metavar="N",
        help=f"runs per scenario (default {DEFAULT_REPETITIONS})",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=_GATE_MAX_ITERATIONS,
        metavar="N",
        help="per-run iteration cap for the main loop "
             f"(default {_GATE_MAX_ITERATIONS})",
    )
    parser.add_argument("--out", default="", metavar="PATH",
                        help=f"report file to write (default {report_path()})")
    parser.add_argument("--trend", metavar="MODEL", nargs="?", const="", default=None,
                        help="print score history for this model and exit")
    return parser.parse_args(argv)


async def main(args: argparse.Namespace) -> None:
    """Run the benchmark CLI entrypoint (mirrors benchmark.py's shape)."""
    if args.trend is not None:
        print_trend(args.trend, args.out or None)
        return

    with sandbox_workspace() as workspace:
        results = await run_scenarios(
            make_agent(str(workspace), args.model), model=args.model,
            repetitions=args.repetitions, max_iterations=args.max_iterations,
        )
    # Honour --out: the report goes exactly where the caller asked.
    path = save_report(results, args.out, model=args.model, profile="",
                       reps=args.repetitions)
    avg = sum(r.score for r in results) / len(results)
    print(f"[agentic-bench] {len(results)} runs on {args.model} "
          f"({args.repetitions} reps) mean={avg:.1%}")
    print(f"[agentic-bench] report updated: {path}")


def run_agentic_gate(
    model: str, profile: str = "", out_dir: str | Path | None = None,
) -> float | None:
    """Run the agentic scenarios as a HarnessFix gate; return the mean score.

    Writes ``agentic_bench.json`` next to ``benchmark_harnessfix.json`` so the
    loop's gates can compare per-repetition scores before/after a repair. The
    gate is non-blocking: None when no model is supplied or when no scenario
    produced a result, mirroring how the benchmark gate degrades when no live
    model is set.

    Scenarios run inside a throwaway sandbox workspace, never the live
    checkout: the tool calls they make are real (one scenario commits and
    pushes), so the gate must not be able to mutate the tree it measures.
    """
    if not model:
        return None
    out = Path(out_dir) / "agentic_bench.json" if out_dir else report_path()
    with sandbox_workspace() as workspace:
        agent = make_agent(str(workspace), model)
        results = asyncio.run(run_scenarios(
            agent, model=model, profile=profile,
            max_iterations=_GATE_MAX_ITERATIONS,
        ))
    save_report(results, out, model=model, profile=profile, reps=DEFAULT_REPETITIONS)
    if not results:
        return None
    return round(sum(r.score for r in results) / len(results), 2)


if __name__ == "__main__":
    asyncio.run(main(parse_args(sys.argv[1:])))
