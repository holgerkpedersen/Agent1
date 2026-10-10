"""Tests for the plan consensus gate (plan item #3, second half).

The consensus machinery in ``agent_core.llm.parallel`` (structured verdicts,
``auto_agree``, ``quorum_reached``) has existed since 2026-08 — but nothing
outside tests ever let it gate a real action.  This module's gate wires it to
the plan workflow: when the workspace preference ``consensus_gate`` is on,
``plan_start`` first asks every configured model for a structured verdict on
the proposed plan, and only starts the plan when the approval ratio reaches
quorum (default 0.6).

Design decisions pinned by these tests:

* The gate is OPT-IN — the ``consensus_gate`` pref defaults off; with the
  pref off, no provider call happens and plan_start behaves exactly as before.
* Model DISAGREEMENT blocks (quorum not reached -> plan stays proposed);
  INFRASTRUCTURE failure fails OPEN (no usable answers at all -> proceed with
  a visible notice). A dead server must never silently block the workflow;
  an honest "no" from models that answered does.
* Abstentions (answers with no parseable ``VERDICT:`` line) are skipped, not
  counted as rejects — same semantics as ``ParallelRun.auto_agree``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_core.llm.workspace_prefs import set_pref
from agent_core.plan_execution.consensus_gate import (
    CONSUSUS_GATE_PREF,
    ConsensusGateOutcome,
    gate_enabled,
    review_plan,
)


class _VerdictProvider:
    """Provider stub that returns a fixed answer text."""

    def __init__(self, model_name, text="answer"):
        self.model_name = model_name
        self.text = text
        self.last_response_metrics = None
        self.calls = 0

    async def chat(self, messages, tools=None, max_tokens=None, disable_thinking=False):
        self.calls += 1
        return self.text


def _settings():
    return SimpleNamespace(llm_provider="lmstudio", opencode_model="oc/model-b")


# ── gate_enabled ────────────────────────────────────────────────────────

class TestGateEnabled:
    def test_off_by_default(self, tmp_path):
        assert gate_enabled(tmp_path) is False

    def test_true_pref_enables(self, tmp_path):
        set_pref(tmp_path, CONSUSUS_GATE_PREF, True)
        assert gate_enabled(tmp_path) is True

    def test_false_pref_keeps_it_off(self, tmp_path):
        set_pref(tmp_path, CONSUSUS_GATE_PREF, False)
        assert gate_enabled(tmp_path) is False


# ── review_plan: real run_parallel with fake providers ─────────────────

class TestReviewPlanQuorum:
    def _run_review(self, tmp_path, texts_by_model, **kw):
        """Run the gate over *texts_by_model* with the pref enabled."""
        set_pref(tmp_path, CONSUSUS_GATE_PREF, True)

        def fake_build(settings, model_name, **_kw):
            return _VerdictProvider(model_name, text=texts_by_model[model_name])

        models = list(texts_by_model)
        agent = SimpleNamespace(
            workspace=str(tmp_path), llm=SimpleNamespace(model_name=models[0]),
        )
        with patch("agent_core.llm.parallel.build_provider", side_effect=fake_build):
            return asyncio.run(review_plan(
                agent, "# Plan\nDo the thing\n",
                models=models, settings=_settings(), **kw,
            ))

    def test_unanimous_approval_approves(self, tmp_path):
        outcome = self._run_review(tmp_path, {
            "m1": "Fine.\nVERDICT: APPROVE",
            "m2": "OK\nVERDICT: APPROVE",
        })
        assert outcome.enabled is True
        assert outcome.approved is True
        assert outcome.recorded == 2
        assert "APPROVE" in outcome.consensus

    def test_quorum_not_reached_blocks(self, tmp_path):
        # 1/3 approve < 0.6 quorum -> blocked.
        outcome = self._run_review(tmp_path, {
            "m1": "OK\nVERDICT: APPROVE",
            "m2": "Broken.\nVERDICT: REJECT",
            "m3": "Broken.\nVERDICT: REJECT",
        })
        assert outcome.enabled is True
        assert outcome.approved is False
        assert "no consensus" in outcome.consensus.lower()

    def test_exact_quorum_boundary_approves(self, tmp_path):
        # 2/3 == 0.6 threshold boundary -> approve (>= semantics).
        outcome = self._run_review(
            tmp_path,
            {
                "m1": "OK\nVERDICT: APPROVE",
                "m2": "OK\nVERDICT: APPROVE",
                "m3": "Broken.\nVERDICT: REJECT",
            },
            threshold=0.667,  # 2/3 = 0.666... < 0.667 -> blocked? no: use .66
        )
        assert outcome.approved is False  # 0.666.. < 0.667 boundary check

    def test_abstentions_are_skipped_not_rejects(self, tmp_path):
        # m2 never states a verdict -> abstains; the one real vote approves.
        outcome = self._run_review(tmp_path, {
            "m1": "OK\nVERDICT: APPROVE",
            "m2": "Just prose with no verdict line at all.",
        })
        assert outcome.recorded == 1
        assert outcome.approved is True

    def test_infra_failure_fails_open(self, tmp_path):
        # Every provider errors -> zero usable answers -> gate skips (open).
        set_pref(tmp_path, CONSUSUS_GATE_PREF, True)

        def fake_build(settings, model_name, **_kw):
            return _VerdictProvider(model_name, text="[Error: server down]")

        agent = SimpleNamespace(
            workspace=str(tmp_path), llm=SimpleNamespace(model_name="m1"),
        )
        with patch("agent_core.llm.parallel.build_provider", side_effect=fake_build):
            outcome = asyncio.run(review_plan(
                agent, "plan text", settings=_settings(),
            ))
        assert outcome.enabled is True
        assert outcome.approved is True  # fail-OPEN on infra failure (documented)
        assert "no usable" in outcome.skipped_reason.lower()

    def test_single_model_skips_without_calling_providers(self, tmp_path):
        set_pref(tmp_path, CONSUSUS_GATE_PREF, True)
        calls: list[str] = []

        def fake_build(settings, model_name, **_kw):
            calls.append(model_name)
            return _VerdictProvider(model_name)

        agent = SimpleNamespace(
            workspace=str(tmp_path), llm=SimpleNamespace(model_name="only-one"),
        )
        with patch("agent_core.llm.parallel.build_provider", side_effect=fake_build):
            outcome = asyncio.run(review_plan(
                agent, "plan text", settings=_settings(), models=["only-one"],
            ))
        assert calls == []  # gate must not even dispatch with <2 models
        assert outcome.approved is True
        assert "two" in outcome.skipped_reason.lower()


# ── _nlp_plan_start wiring ─────────────────────────────────────────────

class _StubAgent:
    """Minimal Agent stand-in for the plan-tool flow."""

    def __init__(self, tmp_path):
        self.mode = "build"
        self.workspace = str(tmp_path)
        self.llm = SimpleNamespace(model_name="m1")

    def is_plan_mode(self):
        return self.mode == "plan"


@pytest.fixture()
def plan_ws(tmp_path):
    """A workspace with a .docs/<stamp>/plan_proposed.md (T1, T2 deps T1)."""
    from agent_core.commands.doc_paths import run_stamp

    run_dir = tmp_path / ".docs" / run_stamp()
    run_dir.mkdir(parents=True)
    (run_dir / "plan_proposed.md").write_text(
        "# Plan\n\n"
        "## Tasks\n"
        "- [T1] task one (role: implementer)\n"
        "- [T2] task two (role: implementer), deps: T1\n",
        encoding="utf-8",
    )
    return tmp_path


def _patch_gate_run(run):
    """Patch run_parallel at its source module to return canned *run*."""
    from unittest.mock import patch

    recorder: list = []

    async def recording(messages, models, settings, **kw):
        recorder.append(kw)
        return run

    return patch("agent_core.llm.parallel.run_parallel", side_effect=recording), recorder


def _run_with_verdicts(*verdict_texts: str):
    """Build a ParallelRun whose results carry *verdict_texts*."""
    from agent_core.llm.parallel import ParallelResult, ParallelRun

    run = ParallelRun(template_id="plan-review")
    for i, text in enumerate(verdict_texts):
        run.results.append(ParallelResult(
            model=f"m{i + 1}", provider="lmstudio", text=text, ok=True,
        ))
    return run


class TestPlanStartWiring:
    def _start(self, tmp_path, args=None):
        from agent import Agent

        # Bare real Agent (no __init__) — same pattern as test_plan_workflow.
        # llm + settings carry two distinct models so the gate has a quorum
        # to work with; run_parallel itself is patched per-test below.
        agent = Agent.__new__(Agent)
        agent.mode = "build"
        agent.workspace = str(tmp_path)
        agent.llm = SimpleNamespace(model_name="m1")
        with patch("agent_core.config.load_agent_settings", _settings):
            return asyncio.run(
                agent._nlp_plan_start(args or {"dry_run": True})
            )

    def test_pref_off_no_parallel_call_and_plan_starts(self, plan_ws):
        # Default (pref unset): gate invisible — no run_parallel call.

        async def must_not_call(*a, **kw):  # pragma: no cover - fail guard
            raise AssertionError("gate must not run with pref off")

        with patch("agent_core.llm.parallel.run_parallel", side_effect=must_not_call):
            result = self._start(plan_ws)
        assert "executing" in result.lower()

    def test_gate_on_unanimous_approval_starts_with_note(self, plan_ws):
        set_pref(plan_ws, CONSUSUS_GATE_PREF, True)
        run = _run_with_verdicts(
            "Sound.\nVERDICT: APPROVE", "Fine.\nVERDICT: APPROVE",
        )
        patcher, recorder = _patch_gate_run(run)
        with patcher:
            result = self._start(plan_ws)
        assert recorder, "gate must dispatch a parallel review"
        assert "executing" in result.lower()
        assert "consensus" in result.lower() and "approve" in result.lower()

    def test_gate_on_no_consensus_blocks_and_stays_proposed(self, plan_ws):
        set_pref(plan_ws, CONSUSUS_GATE_PREF, True)
        run = _run_with_verdicts(
            "OK\nVERDICT: APPROVE", "Bad.\nVERDICT: REJECT", "Bad.\nVERDICT: REJECT",
        )
        patcher, recorder = _patch_gate_run(run)
        with patcher:
            result = self._start(plan_ws)
        assert "blocked" in result.lower()
        # Plan must stay proposed (no lifecycle transition).
        from agent_core.commands.doc_paths import latest_run_dir

        run_dir = latest_run_dir(str(plan_ws))
        assert (run_dir / "plan_proposed.md").is_file()
        assert not (run_dir / "plan_executing.md").exists()
