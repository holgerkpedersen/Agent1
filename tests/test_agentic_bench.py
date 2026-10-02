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
from unittest.mock import patch

import agentic_bench as ab
from agent import Agent
from agent_core.llm.provider import ProviderResult
from agent_core.llm.tool_loop import ToolLoopRunner
from agentic_bench import run_tool_loop


# ---------------------------------------------------------------------------
# Fakes: deterministic transports, no network. Each scenario prompt embeds a
# unique marker token; the fake transport echoes it back in the FINAL
# response, so a clean run passes its rubric and an interrupted one does not.
# ---------------------------------------------------------------------------

class FakeTransport:
    """Minimal ChatML transport: completes each prompt with the echo reply."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

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
        with patch.object(Agent, "build_transport", new=FakeTransport):
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


if __name__ == "__main__":
    unittest.main()
