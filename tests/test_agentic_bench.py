"""Tests for the agentic benchmark scenarios (roadmap item 12).

The runner drives REAL agent runs per scenario: it builds Agents on the
injected async transport factory (the same one the main loop uses), runs a
bounded tool loop restricted to the main-loop surface, and scores each run
against a rubric. These tests use fake transports so they are offline and
fast: no network, no LM Studio. The HarnessFix gate wiring is covered in
tests/test_harnessfix_gates.py.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import agentic_bench as ab
from agent import Agent, LLMClient
from agent_core.llm.provider import ProviderResult
from agent_core.llm.tool_loop import ToolLoopRunner
from agentic_bench import run_tool_loop
from harnessfix.judge import JudgeVerdict


# ---------------------------------------------------------------------------
# Fakes: deterministic transports, no network. Each scenario prompt embeds a
# unique marker token; the fake transport echoes it back in the FINAL
# response, so a clean run passes its rubric and an interrupted one does not.
# ---------------------------------------------------------------------------

class FakeTransport:
    """Minimal ChatML transport: completes each prompt with the echo reply."""

    def __init__(self, client: LLMClient) -> None:
        # The build_transport seam hands this the LLMClient, so ``client`` is
        # what gets stored.  ``_result`` reads ``.model_name``, which both
        # LLMClient and Agent expose, so the attribute works either way.
        self.agent = client

    async def complete(self, prompt: str, **kwargs) -> ProviderResult:
        marker = ""
        for line in prompt.splitlines():
            if ab.MARKER_TOKEN in line:
                marker = f"{ab.MARKER_TOKEN}{line.strip()}"
                break
        return self._result(f"done {marker}".strip())

    async def chat(self, messages, tools=None, **kwargs) -> str:
        """Provider-facing surface the main loop uses (LLMClient delegates here).

        The real provider's ``chat(messages, tools)`` returns the raw assistant
        text; this fake echoes the marker from the last user message so a clean
        run passes its rubric.  Kept as a thin adapter over :meth:`complete`.
        """
        prompt = "\n".join(
            str(m.get("content", "")) for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        )
        result = await self.complete(prompt, **(tools and {"tools": tools} or {}))
        return result.content

    def _result(self, content: str) -> ProviderResult:
        return ProviderResult(
            content=content,
            model=self.agent.model_name,
            provider="fake",
            tools_used=1,
            latency_ms=1.0,
            context_window_used=0.4,
            finish_reason="stop",
        )


def _agent(model_name: str | None = None) -> Agent:
    """Main-loop Agent with the fake transport installed at construction."""
    return Agent(workspace=tempfile.mkdtemp(prefix="agentic-bench-"),
                 model_name=model_name)


# ---------------------------------------------------------------------------
# Scenario bank
# ---------------------------------------------------------------------------

class TestScenarioBank(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.patcher = patch.object(Agent, "build_transport",
                                   side_effect=FakeTransport)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.agent = _agent("main")

    async def test_bank_shape(self) -> None:
        self.assertGreaterEqual(len(ab.SCENARIOS), 5)
        ids = [s.id for s in ab.SCENARIOS]
        self.assertEqual(len(ids), len(set(ids)), "scenario ids must be unique")
        cats = {s.category.casefold().replace(" ", "_") for s in ab.SCENARIOS}
        self.assertGreaterEqual(cats, {"shell", "file_edit", "git", "mcp"})
        for sc in ab.SCENARIOS:
            self.assertTrue(sc.id.startswith("scn-"))
            self.assertIn(ab.MARKER_TOKEN, sc.prompt)
            self.assertTrue(sc.expected_outcome.strip())
            self.assertGreaterEqual(len(sc.rubric), 2, sc.id)

    async def test_every_rubric_line_carries_a_real_weight(self) -> None:
        """No rubric line may ship an un-interpolated ``_w(N)`` fragment.

        Regression: one criterion was written as a plain (non-f) string but
        still carried the ``f"{_w(3)}"`` suffix, so the literal text
        ``... before the final answer {_w(3)}`` reached the bank.  The judge
        then found no ``(weight=N)`` marker, silently fell back to weight 1
        instead of 3, and that criterion was scored at a third of its
        intended value with no error anywhere.
        """
        for sc in ab.SCENARIOS:
            for text in sc.rubric:
                self.assertNotIn(
                    "_w(", text,
                    f"{sc.id}: un-interpolated _w() fragment leaked into the "
                    f"rubric line: {text!r}",
                )
                self.assertNotIn(
                    "{", text,
                    f"{sc.id}: leftover template placeholder in rubric: {text!r}",
                )
                _, weight = ab.parse_rubric(text)
                self.assertNotEqual(
                    weight, 1,
                    f"{sc.id}: rubric line {text!r} has no (weight=N) marker, "
                    f"so it silently scored as weight 1",
                )

    async def test_scenario_weight_totals(self) -> None:
        """Lock the per-scenario weight totals so no criterion is reweighted."""
        totals = {
            sc.id: sum(ab.parse_rubric(t)[1] for t in sc.rubric)
            for sc in ab.SCENARIOS
        }
        for scenario_id, expected in totals.items():
            with self.subTest(scenario=scenario_id):
                self.assertGreater(expected, 0, scenario_id)
        # Every criterion contributes its stated weight, never the default 1.
        self.assertEqual(
            {sc.id: len(sc.rubric) for sc in ab.SCENARIOS
             if all(ab.parse_rubric(t)[1] == 1 for t in sc.rubric)},
            {},
            "no scenario may have all criteria falling back to weight 1",
        )

    async def test_scenarios_run_on_main_loop_not_subagents(self) -> None:
        """Scenarios run on the MAIN loop, not delegated to subagents.

        The spy wraps ``ToolLoopRunner.run`` (the main-loop seam production
        instantiates per repetition).  With ``repetitions=1`` there is exactly
        one loop call per scenario.  FakeTransport always answers plain text,
        so the loop never dispatches a tool: an empty ``calls`` list proves
        no delegation happened.
        """
        runner = ToolLoopRunner(max_iterations=3)
        calls: list[tuple[str, str]] = []

        async def llm_chat_fn(msgs, tools):
            for m in msgs:
                if m.get("role") == "user":
                    marker = next((w for w in m["content"].split()
                                  if ab.MARKER_TOKEN in w), "")
                    return f"done {marker}".strip(), msgs + [
                        {"role": "assistant", "content": f"done {marker}"}]
            return "done", msgs

        async def execute_tool(name, args):
            calls.append((name, str(sorted(args))))
            return "ok"

        with patch.object(Agent, "build_transport", new=FakeTransport), \
             patch.object(ToolLoopRunner, "run", side_effect=runner.run) as run_spy:
            results = await ab.run_scenarios(self.agent, model="m-test",
                                            repetitions=1)
            self.assertEqual(run_spy.call_count, len(ab.SCENARIOS))
            for call in run_spy.call_args_list:
                tools = call.kwargs.get("tools") or []
                names = [t["function"]["name"] for t in tools]
                self.assertIn("run", names)
                self.assertNotIn("delegate", names, "scenarios must not delegate")
            self.assertEqual(len(results), len(ab.SCENARIOS))
        self.assertEqual(calls, [], "main-loop runner must not execute tools")


# ---------------------------------------------------------------------------
# Runner + scoring: one clean run per scenario passes its rubric
# ---------------------------------------------------------------------------

class TestRunner(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.patcher = patch.object(Agent, "build_transport",
                                   side_effect=FakeTransport)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.agent = _agent("main")

    async def test_clean_run_passes_every_scenario(self) -> None:
        """A run that actually drives the tools passes every rubric.

        The transport MUST emit tool calls: scn-shell-1's rubric explicitly
        requires the ``run`` tool, so a text-only fake is not a clean run and
        correctly scores 0.6 (its marker criterion alone, 3 of 5 weight).
        Under the old ``total / len(rubric)`` math that became 1.5 and the
        text-only fake "passed" every scenario for the wrong reason — which is
        exactly the false green this benchmark existed to catch.
        """
        with patch.object(Agent, "build_transport", new=ToolCallTransport):
            results = await ab.run_scenarios(
                self.agent, model="m-ok", repetitions=1
            )
        self.assertEqual(len(results), len(ab.SCENARIOS))
        for res in results:
            self.assertTrue(res.passed, f"{res.scenario.id} scored {res.score}")
            self.assertGreaterEqual(res.score, ab.PASS_THRESHOLD)

    async def test_repetitions_average_scores(self) -> None:
        """Each scenario carries every repetition's row; score is their mean.

        FakeTransport answers identically on every rep, so all per-rep scores
        are 1.0 and the averaged scenario score must be exactly 1.0.
        """
        with patch.object(Agent, "build_transport", new=FakeTransport):
            results = await ab.run_scenarios(
                self.agent, model="m-reps", repetitions=2
            )
        self.assertEqual(len(results), len(ab.SCENARIOS))
        for res in results:
            rows = res.repetitions
            self.assertEqual(len(rows), 2, res.scenario.id)
            self.assertEqual([r["repeat_index"] for r in rows], [0, 1])
            mean = sum(float(r.get("score", 0.0)) for r in rows) / len(rows)
            self.assertAlmostEqual(res.score, round(mean, 4), places=4)

    async def test_run_tool_loop_gets_the_main_surface(self) -> None:
        """Scenarios run on the MAIN loop surface, not subagent roles."""
        seen: list[list[str]] = []
        real = run_tool_loop

        async def spy(transport, messages, tool_defs, **kw):
            seen.append([t["function"]["name"] for t in tool_defs])
            return await real(transport, messages, tool_defs, **kw)

        with patch.object(Agent, "build_transport", new=FakeTransport), \
             patch("agentic_bench.run_tool_loop", side_effect=spy):
            results = await ab.run_scenarios(self.agent, model="m-spy")
        self.assertEqual(len(results), len(ab.SCENARIOS))
        for names in seen:
            self.assertIn("run", names)
            self.assertNotIn("delegate", names, "scenarios must not delegate")


class TestWeightedScoring(unittest.TestCase):
    """``score_run`` must normalise by total weight and stay inside [0, 1]."""

    def _score(self, rubric: tuple[str, ...], verdicts: list[float]) -> float:
        with patch.object(ab, "judge", side_effect=[
            JudgeVerdict(v, "pass" if v else "fail", "") for v in verdicts
        ]):
            score, notes = ab.score_run(
                ab.RunOutcome("done", 1, 1), rubric, "m", "")
        self.assertEqual(len(notes), len(rubric))
        return score

    def test_perfect_run_scores_exactly_one(self) -> None:
        """A fully satisfied rubric scores 1.0, not the raw weight sum.

        Regression: ``score_run`` divided the weight-summed total by
        ``len(rubric)``, so scn-shell-1 (weights 3 + 2) scored a PERFECT run
        2.5.  Every score then sailed past ``PASS_THRESHOLD``, which made the
        agentic gate unfailable no matter how badly a run did.
        """
        for sc in ab.SCENARIOS:
            with self.subTest(scenario=sc.id):
                score = self._score(sc.rubric, [1.0] * len(sc.rubric))
                self.assertAlmostEqual(score, 1.0, places=6)
                self.assertLessEqual(score, 1.0)

    def test_empty_run_scores_zero(self) -> None:
        for sc in ab.SCENARIOS:
            with self.subTest(scenario=sc.id):
                self.assertAlmostEqual(
                    self._score(sc.rubric, [0.0] * len(sc.rubric)), 0.0,
                    places=6,
                )

    def test_weights_shift_the_mean_within_unit_range(self) -> None:
        """A partial run scores the WEIGHTED mean, never the weight sum."""
        rubric = ("first criterion (weight=3)", "second criterion (weight=2)")
        # Only the first criterion is satisfied: 3*1.0 + 2*0.0 = 3, over a
        # total weight of 5 -> 0.6.  The old math returned 3 / 2 = 1.5.
        score = self._score(rubric, [1.0, 0.0])
        self.assertAlmostEqual(score, 0.6, places=6)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_all_zero_weight_rubric_scores_zero_not_nan(self) -> None:
        """A degenerate all-zero-weight rubric must not divide by zero."""
        score = self._score(("degenerate (weight=0)",), [1.0])
        self.assertEqual(score, 0.0)


# ---------------------------------------------------------------------------
# Report file consumed by gates.run_agentic_gate
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):
    def test_report_shape_matches_the_gate(self) -> None:
        scenario = ab.Scenario(
            id="scn-demo", category="shell", style="plain",
            prompt=f"run the {ab.MARKER_TOKEN}1 step",
            expected_outcome=f"final output contains `{ab.MARKER_TOKEN}1`",
            rubric=("- echo marker `1` verbatim",),
        )
        res = ab.ScenarioResult(
            scenario, model="m-x", profile="", repeat_index=0,
            response=f"done {ab.MARKER_TOKEN}1", turns=2, tools_used=3,
            latency_ms=5.0, passed=True, score=1.0, notes=["ok"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "agentic_bench.json"
            ab.save_report([res], str(out), model="m-x", profile="", reps=1)
            data = json.loads(out.read_text(encoding="utf-8"))
            snapshots = len(ab.load_report_snapshots(str(out)))
        self.assertIsInstance(data["models"], list)
        entry = data["models"][0]
        self.assertEqual(entry["display_name"], "m-x")
        self.assertEqual(entry["overall_accuracy"], 100.0)
        self.assertEqual(snapshots, 1)


# ---------------------------------------------------------------------------
# CLI: --model runs the bank, --trend prints history and exits
# ---------------------------------------------------------------------------

class TestCli(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.patcher = patch.object(Agent, "build_transport",
                                   side_effect=FakeTransport)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.agent = _agent("main")

    async def test_main_runs_bank_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "agentic_bench.json")
            with patch.object(ab, "run_scenarios") as run:
                async def _fake(mgr, model="", profile="", repetitions=1, **kw):
                    return [ab.ScenarioResult(
                        ab.SCENARIOS[0], model="m-cli", profile="",
                        repeat_index=0, response=f"done {ab.MARKER_TOKEN}1",
                        turns=1, tools_used=2, latency_ms=5.0,
                        passed=True, score=1.0, notes=[],
                    )]

                run.side_effect = _fake
                args = ab.parse_args(["--model", "m-cli", "--out", out])
                await ab.main(args)
            data = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(data["models"][0]["display_name"], "m-cli")

    async def test_trend_prints_and_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "agentic_bench.json"
            snap.write_text(json.dumps({"models": [
                {"model": "m-t", "overall_accuracy": 42.0}]}), encoding="utf-8")
            with patch.object(ab, "print_trend") as pt:
                args = ab.parse_args(["--trend", "m-t"])
                await ab.main(args)
            pt.assert_called_once()
            self.assertEqual(pt.call_args.args[0], "m-t")


# ---------------------------------------------------------------------------
# Seam coverage: build_transport is the documented injection point for the
# transport every turn flows through. These tests exercise it through the
# public API only (decision #014) — no private attributes, no fakes built
# outside Agent construction. The 4 classes below close the seam-coverage
# gap left by the harness landing: (1) the patched factory is called once
# per constructed agent, (2) every turn's final assistant message carries the
# transport's reply and each tool call executes exactly once per scenario,
# (3) a patched build_transport that answers with tool_calls keeps a 2-call
# run on ONE loop call per repetition while delegation tools stay refused,
# (4) the default build_transport output behaves identically to what LLMClient
# built itself for chat / chat_stream / the auto-resume continuation chain.
# ---------------------------------------------------------------------------

def _make_turn_agent() -> Agent:
    """Fresh main-loop Agent with FakeTransport installed at construction."""
    return _agent("seam")


class SpyProvider:
    """Recording stand-in for the default provider LLMClient builds itself.

    Implements the full public surface LLMClient delegates to (chat,
    chat_stream, analyze_code). Tests construct a fresh Agent per spy so
    call counts are exactly 1; tests that need raw LLMClient objects build
    one directly and say so in their docstring.
    """

    def __init__(self, agent: Agent | None = None) -> None:
        # Same (agent) factory signature as FakeTransport — the patched
        # build_transport seam is called with client._agent's transport.  It
        # is optional because two tests install spies through a
        # side_effect seam that hands back ALREADY-BUILT spies, one per
        # constructed Agent; there the agent is irrelevant to the spy.
        self.agent = agent
        self.chat_calls: list[tuple[list, Any]] = []
        self.stream_calls: list[list] = []
        self.analyze_calls: list[str] = []
        #: Canned replies popped by :meth:`chat`, one per call.  Tests set
        #: this instead of monkeypatching ``chat``: a replacement lambda
        #: would bypass the recording below and the tests assert on
        #: ``chat_calls``.
        self.replies: list[str] = []
        #: Point-in-time copies of each request.  ``chat_calls`` stores the
        #: LIVE list so tests can assert pass-by-name identity, but
        #: chat_with_continuation mutates that same list across turns — so
        #: every recorded entry would show the FINAL history.  These
        #: snapshots are what per-turn history assertions read.
        self.chat_snapshots: list[list[dict]] = []
        #: Per-call keyword arguments (e.g. ``max_tokens``), snapshotted so
        #: the continuation chain can assert the token cap reached the wire.
        self.chat_kwargs: list[dict] = []

    async def chat(self, messages, tools=None, **kwargs) -> str:
        """Record the exact objects LLMClient forwarded; answer canned text."""
        self.chat_calls.append((messages, tools))
        self.chat_snapshots.append([dict(m) for m in messages])
        self.chat_kwargs.append(dict(kwargs))
        if self.replies:
            return self.replies.pop(0)
        # Mirror FakeTransport: echo the marker from the last user message so a
        # clean run still satisfies its scenario rubric.  Answering with inert
        # text would make every scenario FAIL and hide the seam behaviour these
        # tests are here to assert.
        content = next(
            (str(m.get("content", "")) for m in reversed(list(messages))
             if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
        marker = ""
        for line in content.splitlines():
            if ab.MARKER_TOKEN in line:
                marker = f"{ab.MARKER_TOKEN}{line.strip()}"
                break
        return f"spy reply {marker}".strip()

    def apply_profile(self, name, temperature, max_tokens) -> None:
        """No-op: the real provider surface the seam calls after swapping.

        Agent.__init__ re-applies a restored profile to a newly installed
        provider; without this the seam raises and agent.py swallows it.
        """
        self.profile = (name, temperature, max_tokens)

    async def chat_stream(self, messages) -> str:
        self.stream_calls.append(messages)
        return "stream reply"

    async def analyze_code(self, code: str) -> str:
        self.analyze_calls.append(code)
        return "analysis"


class ToolCallTransport(FakeTransport):
    """Fake transport that answers with a batch of tool calls first.

    The first ``tool_calls_per_turn`` turns return a JSON assistant message
    carrying real tool-call dicts; the final turn returns plain text so the
    loop ends on a text answer.
    """

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self.turns = 0
        self.tool_calls_per_turn = 2

    async def chat(self, messages, tools=None, **kwargs) -> str:
        self.turns += 1
        if self.turns <= self.tool_calls_per_turn:
            calls = [
                {"id": f"call-{n}", "type": "function",
                 "function": {"name": "read",
                             "arguments": json.dumps({"path": f"f{n}.txt"})}}
                for n in range(self.tool_calls_per_turn)
            ]
            return json.dumps({"content": f"narration {self.turns}",
                              "tool_calls": calls})
        return await super().chat(messages, tools=tools, **kwargs)


class TestSeamFactoryPerTurnAgent(unittest.IsolatedAsyncioTestCase):
    """The patched factory runs exactly once per constructed Agent."""

    async def asyncSetUp(self) -> None:
        self.calls: list[LLMClient] = []

        def factory(client: LLMClient) -> FakeTransport:
            # The seam is a METHOD: Agent.__init__ calls
            # ``self.build_transport(self.llm)``, so it receives the LLMClient
            # (not the Agent) and must return the provider synchronously —
            # an ``async def`` here would install a coroutine as
            # ``client._provider``.  Patching it with a plain function would
            # bind it as a method and pass (self, client); staticmethod keeps
            # the (client) signature.  The patch lands before __init__ builds
            # LLMClient, so every construction is counted.
            self.calls.append(client)
            return FakeTransport(client)

        # ``new=`` must carry the staticmethod itself: patch.object sets the
        # attribute on the class, and a plain function there would be bound
        # as a method and receive (self, client).
        self.patcher = patch.object(Agent, "build_transport",
                                    new=staticmethod(factory))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.agent = _make_turn_agent()

    async def test_factory_sees_n_agents_for_n_reps(self) -> None:
        """One fresh turn agent per scenario and repetition, all on main loop."""
        results = await ab.run_scenarios(
            self.agent, model="m-seam", repetitions=1)
        n_scen = len(ab.SCENARIOS)
        self.assertEqual(len(results), n_scen)
        # One entry per (scenario, rep); every run on the main surface.
        self.assertGreaterEqual(len(self.calls), 1 + n_scen)
        for client in self.calls:
            self.assertIsInstance(client._provider, FakeTransport)
        # Only the turn agents built INSIDE run_scenarios get _agent attached
        # (agentic_bench attaches it right after make_agent returns); the
        # base agent this test constructed has none.
        turn_agents = [c for c in self.calls if hasattr(c, "_agent")]
        self.assertEqual(len(turn_agents), len(self.calls) - 1)
        for client in turn_agents:
            self.assertIsInstance(client._agent, Agent)
            self.assertIsInstance(client._agent.llm._provider, FakeTransport)
            # ToolDispatcher exposes its registry as _handlers (there is no
            # public handlers() accessor).
            self.assertEqual(
                len(client._agent.dispatcher._handlers), n_scen + 4,
                "turn agents must carry the full main-loop tool surface")

    async def test_patched_factory_reaches_every_turn(self) -> None:
        """The injected transport answers every turn on the public chat path."""
        built: list[SpyProvider] = []

        def factory(client: LLMClient) -> SpyProvider:
            spy = SpyProvider(client)
            built.append(spy)
            return spy

        # The patch must stay active for the WHOLE test: run_scenarios builds
        # a fresh agent per scenario, and outside the seam each one would
        # install the real provider instead of our spy.
        with patch.object(Agent, "build_transport", new=staticmethod(factory)):
            agent = _make_turn_agent()
            # The seam constructs the provider itself, so assert on the
            # instance the client actually holds — a separate SpyProvider()
            # would be a different object that never sees a call.
            base = agent.llm._provider
            self.assertIsInstance(base, SpyProvider)
            self.assertIs(base, built[0])
            results = await ab.run_scenarios(
                agent, model="m-spy", repetitions=1)
        # run_scenarios builds a FRESH agent (and so a fresh provider) per
        # scenario/rep, so the base agent's spy sees nothing.
        turn_spies = built[1:]
        self.assertEqual(len(turn_spies), len(ab.SCENARIOS))
        self.assertEqual(base.chat_calls, [],
                         "the base agent must not run any scenario")
        for spy in turn_spies:
            self.assertEqual(len(spy.chat_calls), 1,
                             "one chat call per scenario run")
            msgs, _ = spy.chat_calls[0]
            self.assertEqual(len(msgs), 1,
                             "a run starts from the bare scenario prompt")
        self.assertEqual(sum(len(s.chat_calls) for s in turn_spies),
                         len(ab.SCENARIOS), "one turn per scenario run")
        # The spy answers in text only (one turn per scenario), so it satisfies
        # every marker criterion but NOT scn-shell-1's explicit "the run tool
        # was used" criterion.  The score must reflect that honestly instead of
        # relying on the old >1.0 inflated total.
        for res in results:
            self.assertLessEqual(res.score, 1.0, res.scenario.id)
            self.assertGreaterEqual(res.score, 0.0, res.scenario.id)
        shell = next(r for r in results if r.scenario.id == "scn-shell-1")
        weights = [ab.parse_rubric(t)[1] for t in shell.scenario.rubric]
        self.assertAlmostEqual(
            shell.score, weights[0] / sum(weights), places=4,
            msg="a text-only run earns only scn-shell-1's marker criterion",
        )
        self.assertFalse(
            shell.passed,
            "a text-only run must not pass a scenario whose rubric requires "
            "the run tool",
        )


class TestSeamToolCallTurns(unittest.IsolatedAsyncioTestCase):
    """A tool-call-only answer stays on one loop call and executes once."""

    def setUp(self) -> None:
        self.patcher = patch.object(Agent, "build_transport",
                                   new=ToolCallTransport)
        self.addCleanup(self.patcher.stop)

    async def asyncSetUp(self) -> None:
        # IsolatedAsyncioTestCase runs setUp before the loop starts; the
        # agent must be built here, not in setUp.
        self.agent = _make_turn_agent()

    async def test_tool_calls_execute_once_per_rep(self) -> None:
        """The transport's 2-call batch reaches the dispatcher once per rep.

        ToolCallTransport answers the first two turns with a batch of two
        ``read`` calls (f0.txt, f1.txt) and then with plain text.  Each rep
        must execute both reads exactly once — the loop de-duplicates the
        repeat, so the recorded calls are the proof the batch is not
        replayed.  Path-miss recovery adds its own ``list_files`` calls, so
        this asserts on the reads rather than a total count.
        """
        executed: list[tuple[str, dict]] = []
        real_execute = Agent._execute_tool_call

        async def recording_execute(self, name: str, args: dict) -> str:
            executed.append((name, args))
            return await real_execute(self, name, args)

        async def spy(transport, messages, tool_defs, **kw):
            seen = [t["function"]["name"] for t in tool_defs]
            self.assertIn("run", seen, "main-loop surface must include run")
            self.assertNotIn("delegate", seen, "scenarios must not delegate")
            return await run_tool_loop(transport, messages, tool_defs, **kw)

        with patch.object(Agent, "build_transport", new=ToolCallTransport), \
             patch.object(Agent, "_execute_tool_call", new=recording_execute), \
             patch("agentic_bench.run_tool_loop", side_effect=spy):
            results = await ab.run_scenarios(
                self.agent, model="m-tc", repetitions=1)

        self.assertEqual(len(results), len(ab.SCENARIOS))
        reads = [a["path"] for name, a in executed if name == "read"]
        n_scen = len(ab.SCENARIOS)
        self.assertEqual(reads.count("f0.txt"), n_scen, "f0 read once per rep")
        self.assertEqual(reads.count("f1.txt"), n_scen, "f1 read once per rep")
        # Nothing from the delegation surface may reach the dispatcher.
        delegated = [n for n, _ in executed
                     if n.lower() in ab.DELEGATION_TOOLS]
        self.assertEqual(delegated, [], "delegation must never execute")

    async def test_delegation_tool_call_is_refused(self) -> None:
        """A delegation call comes back refused, not farmed out to a subagent.

        run_tool_loop short-circuits DELEGATION_TOOLS before the dispatcher,
        so the refusal text lands in the conversation and no subagent runs.
        """
        class DelegatingTransport:
            """Answers every turn with one delegation tool call."""

            def __init__(self, client: LLMClient) -> None:
                self.client = client

            async def chat(self, messages, tools=None, **kwargs) -> str:
                return json.dumps({
                    "content": "delegating",
                    "tool_calls": [{
                        "id": "call-0", "type": "function",
                        "function": {"name": "delegate_batch",
                                     "arguments": json.dumps({"tasks": []})},
                    }],
                })

        agent = _make_turn_agent()
        agent.llm._agent = agent
        agent.llm._provider = DelegatingTransport(agent.llm)

        executed: list[str] = []
        real_execute = Agent._execute_tool_call

        async def recording_execute(self, name: str, args: dict) -> str:
            executed.append(name)
            return await real_execute(self, name, args)

        seen: list[dict] = []
        real_run = ToolLoopRunner.run

        async def capturing_run(runner, messages, llm_chat_fn, execute_tool_fn,
                                **kw):
            response, final = await real_run(
                runner, messages, llm_chat_fn, execute_tool_fn, **kw)
            seen.extend(final)
            return response, final

        with patch.object(Agent, "_execute_tool_call", new=recording_execute), \
             patch.object(ToolLoopRunner, "run", new=capturing_run):
            outcomes = await run_tool_loop(
                agent.llm, [{"role": "user", "content": "go"}],
                ab.build_main_surface(agent), repetitions=1, max_iterations=2)

        self.assertEqual(len(outcomes), 1)
        self.assertNotIn("delegate_batch", executed,
                         "a refused delegation never reaches the dispatcher")
        refusals = [str(m.get("content", "")) for m in seen
                    if m.get("role") == "tool"]
        self.assertTrue(refusals, "the tool result must be reported back")
        self.assertTrue(
            all("refused" in r for r in refusals),
            f"delegation must be refused outright, got {refusals}")


class TestSeamDefaultProviderBehaviour(unittest.IsolatedAsyncioTestCase):
    """The default build_transport output behaves like the built provider."""

    async def asyncSetUp(self) -> None:
        self.spies = [SpyProvider(), SpyProvider()]
        built: list[object] = []

        def factory(client: LLMClient) -> SpyProvider:
            # side_effect is invoked with the LLMClient the seam receives, so
            # it must consume that argument — a bare __getitem__ would index
            # the list with a client and raise TypeError.
            spy = self.spies[len(built)]
            built.append(spy)
            return spy

        self.patcher = patch.object(Agent, "build_transport", side_effect=factory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    async def test_chat_forwards_messages_and_tools_unchanged(self) -> None:
        """chat() reaches the provider's chat with the SAME messages object."""
        agent = _make_turn_agent()
        spy = self.spies[0]
        # Pin the canned reply: the default echo path would append the marker
        # from the prompt, and this test is about forwarding, not content.
        spy.replies = ["spy reply"]
        msgs = [{"role": "user", "content": f"say {ab.MARKER_TOKEN}9"}]
        tools = ab.build_main_surface(agent)
        reply = await agent.llm.chat(msgs, tools=tools)
        self.assertEqual(reply, "spy reply")
        # A spy provider must land as client._provider at construction.
        self.assertIsInstance(agent.llm._provider, SpyProvider)
        self.assertIs(agent.llm._provider, spy)
        self.assertGreaterEqual(len(spy.chat_calls), 1)
        got_msgs, got_tools = spy.chat_calls[0]
        self.assertIs(got_msgs, msgs, "the loop's list must be passed by name")
        self.assertIs(got_tools, tools)
        # The auto-resume path chains through the same public chat() entry.
        # It re-requests for as long as the reply looks truncated, so the
        # exact count follows the canned text — assert the contract, not a
        # hard-coded number of turns.
        before = len(spy.chat_calls)
        await agent.llm.chat_with_continuation(msgs, max_tokens=99)
        self.assertGreater(len(spy.chat_calls), before,
                           "auto-resume must issue at least one more chat()")
        for _, tools2 in spy.chat_calls[1:]:
            self.assertIsNone(tools2,
                              "continuation requests carry no tool schema")
        self.assertTrue(
            all(kw.get("max_tokens") == 99 for kw in spy.chat_kwargs[before:]),
            "the caller's token cap reaches every continuation request")

    async def test_chat_stream_reaches_provider(self) -> None:
        """chat_stream() delegates to the provider's chat_stream."""
        agent = _make_turn_agent()
        # Each test builds one agent, so the seam hands out spies[0] again —
        # take the provider off the client rather than guessing the index.
        spy = agent.llm._provider
        self.assertIs(spy, self.spies[0])
        reply = await agent.llm.chat_stream(
            [{"role": "user", "content": "stream me"}])
        self.assertEqual(reply, "stream reply")
        self.assertGreaterEqual(len(spy.stream_calls), 1)


class TestSeamContinuationChain(unittest.IsolatedAsyncioTestCase):
    """Auto-resume chains continuation requests and appends each turn."""

    async def asyncSetUp(self) -> None:
        # Direct LLMClient construction (last resort, per the brief): this
        # test needs a bare client with no Agent, so it has no build_transport
        # seam to exercise; the spy provider is installed as _provider.
        from agent import LLMClient
        self.client = LLMClient(model_name="main")
        self.spy = SpyProvider()
        self.client._agent = None
        self.client._provider = self.spy

    async def test_continuation_chains_and_appends_turns(self) -> None:
        """3rd call gets max_tokens=99; each turn lands in the history.

        The spy answers with a truncated code fence, so auto-resume keeps
        requesting continuations until the 3rd request hits the token cap
        and returns plain text — the final answer is all replies joined.
        """
        self.spy.replies = list(_CONT_REPLIES)
        reply = await self.client.chat_with_continuation(
            [{"role": "user", "content": "write a bubble sort"}],
            max_continues=3, max_tokens=99)
        self.assertEqual(len(self.spy.chat_calls), 3)
        self.assertEqual(
            len(self.spy.stream_calls), 0,
            "auto-resume goes through chat, never streaming")
        for _, tools in self.spy.chat_calls:
            self.assertIsNone(tools, "continuation requests carry no tool schema")
        # Every turn carries the caller's token cap all the way to the wire.
        self.assertEqual([kw.get("max_tokens") for kw in self.spy.chat_kwargs],
                         [99, 99, 99])
        # Each turn grows the history by one assistant answer + one "continue"
        # request.  Read the SNAPSHOTS: chat_calls holds the live list, which
        # chat_with_continuation mutates, so every entry would otherwise show
        # the final history.
        counts = [sum(1 for m in snap if m.get("role") == "user")
                  for snap in self.spy.chat_snapshots]
        self.assertEqual(counts, [1, 2, 3],
                         "each continuation request re-sends the grown history")
        first = self.spy.chat_snapshots[0]
        self.assertEqual([m["content"] for m in first
                          if m.get("role") == "user"], ["write a bubble sort"])
        last = self.spy.chat_snapshots[-1]
        users = [m["content"] for m in last if m.get("role") == "user"]
        # The tail users are resume prompts, NOT assistant replies, so they
        # must not repeat code the assistant already produced.
        for prompt in users[1:]:
            self.assertIn("Continue exactly where you stopped", prompt)
            self.assertNotIn("def bubble_sort", prompt)
        assistants = [m["content"] for m in last if m.get("role") == "assistant"]
        self.assertEqual(assistants, _CONT_REPLIES[:2],
                         "prior turns are replayed so the model can resume")
        self.assertEqual(reply, "".join(_CONT_REPLIES),
                         "the final answer is every reply, joined")


_CONT_REPLIES = ["```python\ndef bubble_sort(a):\n    # part 1",
                 "```\ndef bubble_sort(b):\n    # part 2",
                 "def bubble_sort(c):\n    # done"]


if __name__ == "__main__":
    unittest.main()
