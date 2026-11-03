"""Regression tests for the Jev strategy layer (card afad5797aaf744689773fe710247d730).

The strategy layer applies the researched cheap-first cascade patterns
(JEV-as-a-Judge "accept when confident, escalate when unsure",
position-bias-checked triage, FrugalGPT-style answer verification, and the
escalation ladder) on top of the Jev engine, so Agent1's decision processing
can settle bounded questions with the cheap model and escalate only on doubt.

No real LLM: the engine is a fake object exposing ``async decide`` that
returns canned ``JevResult`` instances (same contract as
``JevEngine.decide``).  The tool-level tests monkeypatch ``build_jev_engine``
exactly like ``tests/test_jev_cmd.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core.jev_engine import JevQuestion, JevResult
from agent_core.jev_strategy import (
    ACTION_ACCEPT,
    ACTION_ESCALATE,
    EscalationLadder,
    JevVerdict,
    cascade_decide,
    run_ladder,
    triage,
    verify_answer,
)


# ---------------------------------------------------------------------------
#  Fakes
# ---------------------------------------------------------------------------


class FakeEngine:
    """Returns a canned JevResult for every decide() call, in order."""

    def __init__(self, results: list[JevResult]):
        self.results = list(results)
        self.calls: list[tuple[JevQuestion, str]] = []

    async def decide(self, question: JevQuestion, state: str = "") -> JevResult:
        self.calls.append((question, state))
        if len(self.calls) > len(self.results):
            raise AssertionError("FakeEngine ran out of canned results")
        return self.results[len(self.calls) - 1]


def yesno(p_yes: float, *, threshold: float = 0.7, error: str = "") -> JevResult:
    return JevResult(
        kind="yesno",
        model="fake-jev",
        mechanism="vote",
        probabilities={"yes": p_yes, "no": 1.0 - p_yes},
        decision="TRUE" if p_yes >= threshold else "FALSE",
        confidence=max(p_yes, 1.0 - p_yes),
        threshold=threshold,
        n=5,
        abstentions=0,
        error=error,
    )


def choice(
    probs: dict[str, float],
    *,
    options: tuple[str, ...],
    threshold: float = 0.7,
) -> JevResult:
    ordered = {opt: probs.get(opt, 0.0) for opt in options}
    top = max(ordered, key=lambda o: ordered[o])
    return JevResult(
        kind="choice",
        model="fake-jev",
        mechanism="vote",
        probabilities=ordered,
        decision=top if ordered[top] >= threshold else "UNDECIDED",
        confidence=ordered[top],
        threshold=threshold,
        n=5,
        abstentions=0,
    )


def score(value: float, spread: float, *, threshold: float = 0.7) -> JevResult:
    return JevResult(
        kind="score",
        model="fake-jev",
        mechanism="vote",
        decision=f"{value:.2f}",
        confidence=value,
        threshold=threshold,
        n=5,
        abstentions=0,
        value=value,
        spread=spread,
    )


def abstained(kind: str = "yesno") -> JevResult:
    return JevResult(
        kind=kind,
        model="fake-jev",
        mechanism="vote",
        decision="UNKNOWN",
        confidence=0.0,
        n=5,
        abstentions=5,
        error="no parseable answer (all abstentions)",
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
#  cascade_decide — accept when confident, escalate when unsure
# ---------------------------------------------------------------------------


class TestCascadeYesno:
    def test_confident_yes_is_accepted(self):
        engine = FakeEngine([yesno(0.86)])
        verdict = run(cascade_decide(
            engine, JevQuestion(kind="yesno", text="The fix addresses the root cause."),
        ))
        assert verdict.action == ACTION_ACCEPT
        assert verdict.decision == "yes"
        assert verdict.probability == pytest.approx(0.86)
        assert verdict.strategy == "cascade"

    def test_confident_no_is_accepted(self):
        engine = FakeEngine([yesno(0.12)])
        verdict = run(cascade_decide(
            engine, JevQuestion(kind="yesno", text="The failure is environmental."),
        ))
        assert verdict.action == ACTION_ACCEPT
        assert verdict.decision == "no"
        assert verdict.probability == pytest.approx(0.88)

    def test_mid_range_probabilities_escalate(self):
        engine = FakeEngine([yesno(0.55)])
        verdict = run(cascade_decide(
            engine, JevQuestion(kind="yesno", text="The bug is in the parser."),
        ))
        assert verdict.action == ACTION_ESCALATE
        assert verdict.decision is None

    def test_threshold_is_respected(self):
        # 0.65 < 0.7 default -> escalate, even though it is the winning side.
        engine = FakeEngine([yesno(0.65)])
        verdict = run(cascade_decide(
            engine, JevQuestion(kind="yesno", text="It is worth escalating."),
        ))
        assert verdict.action == ACTION_ESCALATE

    def test_lower_threshold_accepts_weaker_signal(self):
        engine = FakeEngine([yesno(0.65)])
        verdict = run(cascade_decide(
            engine,
            JevQuestion(kind="yesno", text="It is worth escalating."),
            accept_threshold=0.6,
        ))
        assert verdict.action == ACTION_ACCEPT
        assert verdict.decision == "yes"

    def test_all_abstentions_escalate_never_fabricate(self):
        engine = FakeEngine([abstained("yesno")])
        verdict = run(cascade_decide(
            engine, JevQuestion(kind="yesno", text="Anything."),
        ))
        assert verdict.action == ACTION_ESCALATE
        reason = verdict.reason.lower()
        assert "abstain" in reason or "no usable" in reason

    def test_engine_error_escalates(self):
        engine = FakeEngine([yesno(0.9, error="provider timeout")])
        verdict = run(cascade_decide(
            engine, JevQuestion(kind="yesno", text="Anything."),
        ))
        assert verdict.action == ACTION_ESCALATE
        assert "provider timeout" in verdict.reason


class TestCascadeChoice:
    def test_clear_winner_is_accepted(self):
        probs = {"a": 0.70, "b": 0.20, "c": 0.10}
        engine = FakeEngine([choice(probs, options=("a", "b", "c"))])
        verdict = run(cascade_decide(
            engine,
            JevQuestion(kind="choice", text="Most likely root cause",
                        options=("a", "b", "c")),
        ))
        assert verdict.action == ACTION_ACCEPT
        assert verdict.decision == "a"

    def test_close_race_escalates(self):
        probs = {"a": 0.45, "b": 0.40, "c": 0.15}
        engine = FakeEngine([choice(probs, options=("a", "b", "c"))])
        verdict = run(cascade_decide(
            engine,
            JevQuestion(kind="choice", text="Best target", options=("a", "b", "c")),
        ))
        assert verdict.action == ACTION_ESCALATE

    def test_winner_below_threshold_escalates(self):
        probs = {"a": 0.60, "b": 0.25, "c": 0.15}
        engine = FakeEngine([choice(probs, options=("a", "b", "c"))])
        verdict = run(cascade_decide(
            engine,
            JevQuestion(kind="choice", text="Best target", options=("a", "b", "c")),
        ))
        assert verdict.action == ACTION_ESCALATE


class TestCascadeScore:
    def _question(self):
        return JevQuestion(kind="score", text="Review readiness",
                           rubric="80+: tested; 40: untested")

    def test_high_tight_score_is_accepted(self):
        engine = FakeEngine([score(0.84, 0.05)])
        verdict = run(cascade_decide(engine, self._question()))
        assert verdict.action == ACTION_ACCEPT
        assert verdict.probability == pytest.approx(0.84)

    def test_low_score_escalates(self):
        engine = FakeEngine([score(0.42, 0.05)])
        verdict = run(cascade_decide(engine, self._question()))
        assert verdict.action == ACTION_ESCALATE

    def test_wide_spread_escalates(self):
        # Mean is above threshold but the samples disagree -> weak signal.
        engine = FakeEngine([score(0.78, 0.30)])
        verdict = run(cascade_decide(engine, self._question()))
        assert verdict.action == ACTION_ESCALATE
        assert "spread" in verdict.reason.lower()

    def test_no_value_escalates(self):
        engine = FakeEngine([abstained("score")])
        verdict = run(cascade_decide(engine, self._question()))
        assert verdict.action == ACTION_ESCALATE


# ---------------------------------------------------------------------------
#  triage — position-bias-checked choice (order swap)
# ---------------------------------------------------------------------------


class TestTriage:
    def test_stable_winner_across_swap_is_accepted(self):
        # First ask (a, b, c): a wins. Swapped ask (c, b, a): a still wins.
        engine = FakeEngine([
            choice({"a": 0.72, "b": 0.18, "c": 0.10}, options=("a", "b", "c")),
            choice({"a": 0.70, "b": 0.20, "c": 0.10}, options=("c", "b", "a")),
        ])
        verdict = run(triage(engine, "Most likely root cause", ["a", "b", "c"]))
        assert verdict.action == ACTION_ACCEPT
        assert verdict.decision == "a"
        assert verdict.strategy == "triage"
        assert len(engine.calls) == 2

    def test_unstable_winner_escalates(self):
        # First ask: a wins (clears threshold). Swapped order: c wins
        # -> position bias suspected.
        engine = FakeEngine([
            choice({"a": 0.72, "b": 0.10, "c": 0.18}, options=("a", "b", "c")),
            choice({"a": 0.18, "b": 0.10, "c": 0.72}, options=("c", "b", "a")),
        ])
        verdict = run(triage(engine, "Most likely root cause", ["a", "b", "c"]))
        assert verdict.action == ACTION_ESCALATE
        reason = verdict.reason.lower()
        assert "swap" in reason or "bias" in reason

    def test_first_ask_escalates_without_swap_call(self):
        engine = FakeEngine([
            choice({"a": 0.45, "b": 0.40, "c": 0.15}, options=("a", "b", "c")),
        ])
        verdict = run(triage(engine, "Best target", ["a", "b", "c"]))
        assert verdict.action == ACTION_ESCALATE
        assert len(engine.calls) == 1  # no wasted second call

    def test_swap_disabled_uses_single_ask(self):
        engine = FakeEngine([
            choice({"a": 0.72, "b": 0.18, "c": 0.10}, options=("a", "b", "c")),
        ])
        verdict = run(triage(engine, "Best target", ["a", "b", "c"], swap=False))
        assert verdict.action == ACTION_ACCEPT
        assert len(engine.calls) == 1

    def test_margin_is_enforced_on_both_asks(self):
        # First ask: a wins by 0.52 (>= 0.10). Swapped: a wins by 0.05 (< 0.10).
        engine = FakeEngine([
            choice({"a": 0.72, "b": 0.18, "c": 0.10}, options=("a", "b", "c")),
            choice({"a": 0.55, "b": 0.35, "c": 0.10}, options=("c", "b", "a")),
        ])
        verdict = run(triage(engine, "Best target", ["a", "b", "c"], margin=0.10))
        assert verdict.action == ACTION_ESCALATE


# ---------------------------------------------------------------------------
#  verify_answer — cheap verifier before trusting a draft (FrugalGPT pattern)
# ---------------------------------------------------------------------------


class TestVerifyAnswer:
    def test_confident_verification_accepts(self):
        engine = FakeEngine([yesno(0.90)])
        verdict = run(verify_answer(
            engine,
            "Fix the off-by-one in slice",
            "Changed line 42 to end-1 and added a test.",
        ))
        assert verdict.action == ACTION_ACCEPT
        assert verdict.decision == "yes"
        assert verdict.strategy == "verify"
        # The proposition and the task/answer state are handed to the judge.
        question, state = engine.calls[0]
        assert question.kind == "yesno"
        assert "task" in question.text.lower()
        assert "Fix the off-by-one" in state
        assert "end-1" in state

    def test_unsure_verification_escalates(self):
        engine = FakeEngine([yesno(0.50)])
        verdict = run(verify_answer(engine, "Do X", "Maybe I did X."))
        assert verdict.action == ACTION_ESCALATE

    def test_context_is_appended_to_state(self):
        engine = FakeEngine([yesno(0.95)])
        run(verify_answer(engine, "Do X", "I did X.", context="user asked twice"))
        question, state = engine.calls[0]
        assert "user asked twice" in state


# ---------------------------------------------------------------------------
#  EscalationLadder — cheap rungs first, stop at the first confident verdict
# ---------------------------------------------------------------------------


class TestEscalationLadder:
    def test_stops_at_first_accept(self):
        calls: list[str] = []

        def make(label: str, verdict: JevVerdict):
            async def rung() -> JevVerdict:
                calls.append(label)
                return verdict
            return rung

        v1 = JevVerdict(strategy="cascade", action=ACTION_ESCALATE,
                        probability=0.5, reason="mid range")
        v2 = JevVerdict(strategy="triage", action=ACTION_ACCEPT, decision="b",
                        probability=0.8, reason="stable winner")
        ladder = EscalationLadder([
            ("cheap", make("cheap", v1)),
            ("deeper", make("deeper", v2)),
        ])
        verdict = run(ladder.run())
        assert verdict.action == ACTION_ACCEPT
        assert verdict.decision == "b"
        assert calls == ["cheap", "deeper"]  # never reaches a third rung

    def test_all_escalations_produce_combined_escalate(self):
        v1 = JevVerdict(strategy="cascade", action=ACTION_ESCALATE,
                        probability=0.5, reason="mid range")
        v2 = JevVerdict(strategy="verify", action=ACTION_ESCALATE,
                        probability=0.4, reason="unsure")

        async def g1() -> JevVerdict:
            return v1

        async def g2() -> JevVerdict:
            return v2

        ladder = EscalationLadder([("g1", g1), ("g2", g2)])
        verdict = run(ladder.run())
        assert verdict.action == ACTION_ESCALATE
        assert verdict.strategy == "ladder"
        assert "g1" in verdict.reason and "g2" in verdict.reason

    def test_empty_ladder_escalates(self):
        verdict = run(EscalationLadder([]).run())
        assert verdict.action == ACTION_ESCALATE


def test_run_ladder_helper_delegates_to_ladder():
    calls: list[str] = []

    async def rung() -> JevVerdict:
        calls.append("only")
        return JevVerdict(strategy="only", action=ACTION_ACCEPT,
                          decision="ok", probability=0.9, reason="fine")

    verdict = run(run_ladder([("only", rung)]))
    assert verdict.action == ACTION_ACCEPT
    assert calls == ["only"]


# ---------------------------------------------------------------------------
#  JevVerdict.to_text — the compact, actionable form the tool returns
# ---------------------------------------------------------------------------


class TestVerdictText:
    def test_accept_text_names_action_decision_and_probability(self):
        v = JevVerdict(strategy="cascade", action=ACTION_ACCEPT, decision="yes",
                       probability=0.86, reason="P(yes) 0.86 >= 0.70",
                       result=yesno(0.86))
        text = v.to_text()
        assert "ACCEPT" in text
        assert "yes" in text
        assert "0.86" in text

    def test_escalate_text_names_strategy_and_reason(self):
        v = JevVerdict(strategy="triage", action=ACTION_ESCALATE, decision=None,
                       probability=0.45, reason="winner unstable under order swap")
        text = v.to_text()
        assert "ESCALATE" in text
        assert "triage" in text
        assert "unstable" in text


# ---------------------------------------------------------------------------
#  Tool wiring — jev_decide strategy= applies the strategies for the main model
# ---------------------------------------------------------------------------


class _RecordingEngine:
    """Stands in for JevEngine inside the NLP tool (no real LLM)."""

    def __init__(self, results: list[JevResult] | None = None):
        self.results = list(results or [])
        self.questions: list[JevQuestion] = []
        self.states: list[str] = []

    async def decide(self, question: JevQuestion, state: str = "") -> JevResult:
        self.questions.append(question)
        self.states.append(state)
        if not self.results:
            return yesno(0.8)
        return self.results.pop(0)


def _patch_engine(monkeypatch, module, engine: _RecordingEngine):
    # The tool imports build_jev_engine LOCALLY from agent_core.jev_engine,
    # so the patch must land on the source module (not on ``module``) for it
    # to take effect.  ``module`` is kept only to name the consumer under
    # test at the call site.
    import agent_core.jev_engine as jev_engine_mod

    monkeypatch.setattr(
        jev_engine_mod, "build_jev_engine", lambda **kw: engine
    )


class _FakeAgent:
    workspace = None


@pytest.mark.asyncio
class TestJevDecideToolStrategy:
    async def test_default_single_keeps_current_behavior(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        engine = _RecordingEngine([yesno(0.8)])
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "yesno", "question": "The fix addresses the root cause.",
        })
        assert "[jev]" in out  # the classic summary, not a strategy verdict
        assert "P(yes)=0.80" in out

    async def test_cascade_returns_actionable_verdict(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        engine = _RecordingEngine([yesno(0.86)])
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "yesno", "question": "The fix addresses the root cause.",
            "strategy": "cascade",
        })
        assert "ACCEPT" in out
        assert "yes" in out
        assert "0.86" in out

    async def test_cascade_escalates_on_doubt(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        engine = _RecordingEngine([yesno(0.55)])
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "yesno", "question": "The bug is in the parser.",
            "strategy": "cascade",
        })
        assert "ESCALATE" in out

    async def test_triage_requires_options(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        engine = _RecordingEngine()
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "choice", "question": "Best target", "strategy": "triage",
        })
        assert "Error" in out
        assert len(engine.questions) == 0  # no engine call was made

    async def test_triage_applies_swap(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        engine = _RecordingEngine([
            choice({"a": 0.72, "b": 0.18, "c": 0.10}, options=("a", "b", "c")),
            choice({"a": 0.70, "b": 0.20, "c": 0.10}, options=("c", "b", "a")),
        ])
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "choice", "question": "Best target",
            "options": ["a", "b", "c"], "strategy": "triage",
        })
        assert "ACCEPT" in out
        assert len(engine.questions) == 2  # the order-swap re-ask happened

    async def test_verify_uses_task_and_answer(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        engine = _RecordingEngine([yesno(0.9)])
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "yesno", "question": "Fix the off-by-one in slice",
            "state": "Changed line 42 to end-1.",
            "strategy": "verify",
        })
        assert "ACCEPT" in out
        question, state = engine.questions[0], engine.states[0]
        assert question.kind == "yesno"
        assert "Fix the off-by-one" in state
        assert "end-1" in state

    async def test_unknown_strategy_is_an_error(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        engine = _RecordingEngine()
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "yesno", "question": "Anything.", "strategy": "bogus",
        })
        assert "Error" in out
        assert len(engine.questions) == 0

    async def test_threshold_flows_into_the_strategy(self, monkeypatch):
        from agent_core.handlers import nlp_tools

        # 0.65 would escalate at the default 0.7 but accept at threshold 0.6.
        engine = _RecordingEngine([yesno(0.65)])
        _patch_engine(monkeypatch, nlp_tools, engine)
        out = await nlp_tools._nlp_jev_decide(_FakeAgent(), {
            "kind": "yesno", "question": "Worth it?",
            "strategy": "cascade", "threshold": 0.6,
        })
        assert "ACCEPT" in out


# ---------------------------------------------------------------------------
#  Schema + system prompt — the main model must be TOLD the strategies exist
# ---------------------------------------------------------------------------


def test_jev_decide_schema_advertises_the_strategy_param():
    from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

    schemas = {t["function"]["name"]: t["function"] for t in NLP_TOOL_SCHEMAS}
    props = schemas["jev_decide"]["parameters"]["properties"]
    assert "strategy" in props
    assert set(props["strategy"]["enum"]) == {
        "single", "cascade", "triage", "verify"}
    desc = props["strategy"]["description"].lower()
    assert "escalate" in desc
    # kind+question stay the only required params (backwards compatible).
    assert schemas["jev_decide"]["parameters"]["required"] == ["kind", "question"]


def test_system_prompt_teaches_the_escalation_strategy():
    from agent_core.agent import _SYSTEM_PROMPT

    prompt = _SYSTEM_PROMPT
    assert "strategy='cascade'" in prompt or "strategy=cascade" in prompt
    assert "escalate" in prompt.lower()
