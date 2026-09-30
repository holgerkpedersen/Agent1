"""Jev telemetry — decision ledger, outcome labeling and calibration.

The telemetry is the measurement half of the Jev integration: without it the
accept/reject thresholds are unvalidated guesses.  These tests cover the
ledger I/O, best-effort semantics, the automatic recording from
``JevEngine.decide`` and the calibration math (predicted-vs-observed +
suggested threshold).
"""

import asyncio

import pytest

import agent as agent_mod
from agent_core.jev_engine import JevEngine, JevQuestion, JevResult
from harnessfix.jev_telemetry import (
    CORRECT,
    INCORRECT,
    calibration_bins,
    format_report,
    load_decisions,
    load_suggested_threshold,
    log_path,
    question_hash,
    record_decision,
    record_outcome,
    summarize,
    suggest_threshold,
)


def _question(text="is it true?"):
    return JevQuestion(kind="yesno", text=text)


def _result(p_yes=0.9, decision="TRUE"):
    return JevResult(
        kind="yesno", model="qwen2.5-coder-1.5b-instruct", mechanism="vote",
        probabilities={"yes": p_yes, "no": 1.0 - p_yes}, decision=decision,
        confidence=max(p_yes, 1.0 - p_yes), threshold=0.7, n=5, abstentions=0,
    )


class _VoteProvider:
    model_name = "fake-small"

    def __init__(self, answer="yes"):
        self.answer = answer

    def apply_profile(self, *args):
        pass

    async def chat(self, messages, **kwargs):
        return self.answer


class TestLedger:
    def test_question_hash_is_stable(self):
        assert question_hash("abc") == question_hash("  abc  ")
        assert question_hash("abc") != question_hash("abd")

    def test_record_decision_appends(self, tmp_path):
        decision_id = record_decision(
            _result(), _question(), workspace=str(tmp_path), source="tool",
        )
        assert decision_id
        records = load_decisions(workspace=str(tmp_path))
        assert len(records) == 1
        record = records[0]
        assert record["id"] == decision_id
        assert record["kind"] == "yesno"
        assert record["source"] == "tool"
        assert record["probabilities"]["yes"] == pytest.approx(0.9)
        assert record["outcome"] is None
        assert record["question_hash"] == question_hash("is it true?")

    def test_record_decision_opt_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_NO_JEV_LOG", "1")
        assert record_decision(_result(), _question(), workspace=str(tmp_path)) is None
        assert not log_path(str(tmp_path)).exists()

    def test_load_skips_malformed_and_honours_last(self, tmp_path):
        path = log_path(str(tmp_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"id": "a"}\nnot json\n{"id": "b"}\n{"id": "c"}\n',
            encoding="utf-8",
        )
        ids = [r["id"] for r in load_decisions(workspace=str(tmp_path))]
        assert ids == ["a", "b", "c"]
        last_two = [
            r["id"] for r in load_decisions(workspace=str(tmp_path), last=2)
        ]
        assert last_two == ["b", "c"]

    def test_record_outcome_labels_atomically(self, tmp_path):
        decision_id = record_decision(_result(), _question(), workspace=str(tmp_path))
        assert record_outcome(
            decision_id, CORRECT, workspace=str(tmp_path), note="review",
        ) is True
        record = load_decisions(workspace=str(tmp_path))[0]
        assert record["outcome"] == CORRECT
        assert record["outcome_note"] == "review"
        assert record["outcome_ts"]
        # Unknown id -> False, invalid outcome -> ValueError.
        assert record_outcome("nope", CORRECT, workspace=str(tmp_path)) is False
        with pytest.raises(ValueError):
            record_outcome(decision_id, "maybe", workspace=str(tmp_path))

    def test_record_outcome_without_ledger_is_false(self, tmp_path):
        assert record_outcome("x", CORRECT, workspace=str(tmp_path)) is False


class TestCalibration:
    def _records(self, tmp_path):
        pairs = [(0.9, CORRECT), (0.8, CORRECT), (0.2, INCORRECT), (0.6, INCORRECT)]
        for p_yes, outcome in pairs:
            decision_id = record_decision(
                _result(p_yes=p_yes), _question(), workspace=str(tmp_path),
            )
            record_outcome(decision_id, outcome, workspace=str(tmp_path))
        return load_decisions(workspace=str(tmp_path))

    def test_calibration_bins(self, tmp_path):
        bins = calibration_bins(self._records(tmp_path))
        by_bin = {entry["bin"]: entry for entry in bins}
        assert by_bin["0.8-0.9"]["observed"] == 1.0
        assert by_bin["0.6-0.7"]["observed"] == 0.0
        assert by_bin["0.2-0.3"]["observed"] == 0.0
        assert by_bin["0.9-1.0"]["predicted"] == pytest.approx(0.9)

    def test_suggested_threshold_separates(self, tmp_path):
        suggestion = suggest_threshold(self._records(tmp_path))
        assert suggestion is not None
        assert suggestion["accuracy"] == 1.0
        assert suggestion["n"] == 4
        assert suggestion["threshold"] == pytest.approx(0.65)

    def test_suggest_returns_none_without_labels(self, tmp_path):
        record_decision(_result(), _question(), workspace=str(tmp_path))
        assert suggest_threshold(load_decisions(workspace=str(tmp_path))) is None

    def test_summarize_counts_and_report(self, tmp_path):
        report = summarize(self._records(tmp_path))
        assert report["total"] == 4
        assert report["labeled"] == 4
        assert report["by_kind"]["yesno"] == 4
        text = format_report(report)
        assert "calibration" in text
        assert "suggested yesno threshold" in text

    def test_report_without_labels_explains_labeling(self, tmp_path):
        record_decision(_result(), _question(), workspace=str(tmp_path))
        text = format_report(summarize(load_decisions(workspace=str(tmp_path))))
        assert "record_outcome" in text


class TestEngineRecordsAutomatically:
    def test_decide_appends_a_record(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_NO_JEV_LOG", raising=False)
        engine = JevEngine(
            _VoteProvider("yes"), model_name="fake-small", samples=3,
            mechanism="vote", log_workspace=str(tmp_path), source="tool",
        )
        result = asyncio.run(engine.decide(_question()))
        assert result.decision == "TRUE"
        records = load_decisions(workspace=str(tmp_path))
        assert len(records) == 1
        assert records[0]["source"] == "tool"
        assert records[0]["mechanism"] == "vote"

    def test_telemetry_failure_never_breaks_decide(self, monkeypatch):
        """A broken telemetry import/write must not fail the decision."""
        import harnessfix.jev_telemetry as telemetry

        def boom(*args, **kwargs):
            raise RuntimeError("ledger exploded")

        monkeypatch.setattr(telemetry, "record_decision", boom)
        monkeypatch.delenv("AGENT_NO_JEV_LOG", raising=False)
        engine = JevEngine(
            _VoteProvider("no"), model_name="fake-small", samples=2,
            mechanism="vote",
        )
        result = asyncio.run(engine.decide(_question()))
        assert result.decision == "FALSE"


def _label_count(workspace, count, monkeypatch):
    """Record *count* labeled yesno decisions into *workspace*'s ledger."""
    monkeypatch.delenv("AGENT_NO_JEV_LOG", raising=False)
    pairs = [(0.9, CORRECT), (0.8, CORRECT), (0.2, INCORRECT), (0.6, INCORRECT)]
    for i in range(count):
        p_yes, outcome = pairs[i % len(pairs)]
        decision_id = record_decision(
            _result(p_yes=p_yes), _question(), workspace=str(workspace),
        )
        record_outcome(decision_id, outcome, workspace=str(workspace))


class TestThresholdSuggestionGating:
    """B4: a threshold suggestion needs >= 10 labeled samples (min_samples)."""

    def test_nine_labeled_samples_yet_none(self, tmp_path, monkeypatch):
        _label_count(tmp_path, 9, monkeypatch)
        # The calculator itself already has data...
        assert suggest_threshold(load_decisions(workspace=str(tmp_path))) is not None
        # ...but the gated consumer stays silent below min_samples=10.
        assert load_suggested_threshold(workspace=str(tmp_path)) is None

    def test_tenth_labeled_sample_enables_suggestion(self, tmp_path, monkeypatch):
        _label_count(tmp_path, 9, monkeypatch)
        assert load_suggested_threshold(workspace=str(tmp_path)) is None
        _label_count(tmp_path, 1, monkeypatch)
        suggestion = load_suggested_threshold(workspace=str(tmp_path))
        assert suggestion is not None
        assert suggestion["n"] >= 10
        assert 0.0 < float(suggestion["threshold"]) <= 1.0

    def test_min_samples_argument_is_respected(self, tmp_path, monkeypatch):
        _label_count(tmp_path, 10, monkeypatch)
        assert load_suggested_threshold(workspace=str(tmp_path)) is not None
        assert load_suggested_threshold(
            workspace=str(tmp_path), min_samples=20
        ) is None


class TestJevDecideThresholdFallback:
    """agent._nlp_jev_decide: calibrated threshold, else the 0.7 default."""

    @staticmethod
    def _make_agent(workspace, monkeypatch, tmp_path):
        monkeypatch.setattr(
            agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
        )
        monkeypatch.setattr(
            agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
        )
        return agent_mod.Agent(workspace=str(workspace))

    @staticmethod
    def _patch_engine(monkeypatch, captured):
        import agent_core.jev_engine as jev_engine

        class _FakeResult:
            @staticmethod
            def summary():
                return "TRUE"

        class _FakeEngine:
            async def decide(self, question, state=""):
                captured["question"] = question
                return _FakeResult()

        def _fake_build(**kwargs):
            captured.update(kwargs)
            return _FakeEngine()

        monkeypatch.setattr(jev_engine, "build_jev_engine", _fake_build)

    def test_falls_back_to_point_seven_below_min_samples(
        self, tmp_path, monkeypatch
    ):
        ws = tmp_path / "ws"
        ws.mkdir()
        _label_count(ws, 9, monkeypatch)
        bot = self._make_agent(ws, monkeypatch, tmp_path)
        captured: dict = {}
        self._patch_engine(monkeypatch, captured)

        answer = asyncio.run(bot._nlp_jev_decide({"question": "is it true?"}))

        assert answer == "TRUE"
        assert captured["threshold"] == pytest.approx(0.7), (
            "with < 10 labeled samples the tool must fall back to 0.7"
        )

    def test_applies_calibrated_threshold_at_ten_samples(
        self, tmp_path, monkeypatch
    ):
        ws = tmp_path / "ws"
        ws.mkdir()
        _label_count(ws, 10, monkeypatch)
        bot = self._make_agent(ws, monkeypatch, tmp_path)
        captured: dict = {}
        self._patch_engine(monkeypatch, captured)

        answer = asyncio.run(bot._nlp_jev_decide({"question": "is it true?"}))

        assert answer == "TRUE"
        expected = load_suggested_threshold(workspace=str(ws))
        assert expected is not None, "sanity: 10 labels must yield a suggestion"
        assert captured["threshold"] == pytest.approx(float(expected["threshold"]))
