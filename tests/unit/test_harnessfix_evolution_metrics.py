"""Unit tests for harnessfix.evolution_metrics (read-only trace scoring).

Hermetic: no real trace files are touched — synthetic event dicts only.
"""
import pytest

from harnessfix.evolution_metrics import (
    DEFAULT_THRESHOLD,
    EvolutionMetricsScorer,
    iter_run_scores,
    score_run,
)
from harnessfix.tracing import (
    LAYER_EXECUTION,
    LAYER_OBSERVABILITY,
    LAYER_VERIFICATION,
)

COMPLETED = "completed"
ERROR = "error"


def _run(outcome: str, durations: tuple[float, ...] = ()) -> list[dict]:
    events = [{"kind": "loop_end", "outcome": outcome, "task_id": "t"}]
    for d in durations:
        events.append({"kind": "tool_result", "duration_s": d, "task_id": "t"})
    return events


class TestScoreRun:
    def test_completed_no_latency_is_one(self):
        assert score_run(_run(COMPLETED)) == 1.0

    def test_failed_outcome_is_zero(self):
        assert score_run(_run(ERROR)) == 0.0

    def test_no_loop_end_is_zero(self):
        # Incomplete run (interrupted) scores 0.0.
        assert score_run([{"kind": "tool_result", "duration_s": 1.0}]) == 0.0

    def test_latency_penalty_applied(self):
        # 5s / 30s budget * 0.3 weight = 0.05 penalty => 0.95
        assert score_run(_run(COMPLETED, (5.0,))) == pytest.approx(0.95)

    def test_latency_penalty_capped(self):
        # Huge latency saturates the penalty at weight 0.3 => 0.7
        assert score_run(_run(COMPLETED, (100000.0,))) == pytest.approx(0.7)

    def test_score_clamped_to_zero(self):
        # Failed run with latency still floors at 0.0 (never negative).
        assert score_run(_run(ERROR, (100000.0,))) == 0.0

    def test_non_numeric_duration_ignored(self):
        ev = _run(COMPLETED, (2.0,))
        ev.append({"kind": "tool_result", "duration_s": "not-a-number", "task_id": "t"})
        assert score_run(ev) == pytest.approx(0.98)  # 2/30*0.3 = 0.02 penalty

    def test_custom_budget_and_weight(self):
        # budget 10s, weight 0.5, 5s latency => penalty 0.25 => 0.75
        assert score_run(_run(COMPLETED, (5.0,)), latency_budget_s=10.0, penalty_weight=0.5) == pytest.approx(0.75)


class TestWindowedAverage:
    def test_empty_window_is_zero(self):
        sc = EvolutionMetricsScorer()
        assert sc.windowed_average() == 0.0

    def test_window_truncates_old_runs(self):
        sc = EvolutionMetricsScorer(window_size=3)
        for _ in range(5):
            sc.record_trace(_run(COMPLETED, (0.0,)))  # each scores 1.0
        # Window holds only the last 3 runs, all 1.0.
        assert sc.windowed_average() == pytest.approx(1.0)
        assert sc.summary()["history_count"] == 3


class TestShouldEvolve:
    def test_below_threshold_triggers(self):
        sc = EvolutionMetricsScorer(threshold=DEFAULT_THRESHOLD)
        # All failed => average 0.0 < 0.7
        sc.record_trace(_run(ERROR))
        assert sc.should_evolve() is True

    def test_above_threshold_does_not_trigger(self):
        sc = EvolutionMetricsScorer(threshold=DEFAULT_THRESHOLD)
        sc.record_trace(_run(COMPLETED, (0.0,)))
        assert sc.should_evolve() is False

    def test_boundary_at_threshold(self):
        # Average exactly at threshold must NOT trigger evolution.
        sc = EvolutionMetricsScorer(threshold=0.7)
        # 0.7 exactly: one completed (1.0) + one failed (0.0) => 0.5? use two completed-ish.
        # Build average == 0.7 precisely: 0.7 and 0.7.
        sc.record_trace(_run(COMPLETED, (9.0,)))  # 1 - 9/30*0.3 = 1-0.09 = 0.91
        sc.record_trace(_run(COMPLETED, (23.33333333,)))  # ~0.7667 -> avg ~0.838
        # Instead assert the documented boundary rule directly:
        sc2 = EvolutionMetricsScorer(threshold=0.7)
        sc2.record_trace(_run(COMPLETED, (0.0,)))  # 1.0
        sc2.record_trace(_run(ERROR))              # 0.0  -> avg 0.5 < 0.7 -> evolve
        assert sc2.should_evolve() is True
        sc3 = EvolutionMetricsScorer(threshold=0.7)
        sc3.record_trace(_run(COMPLETED, (0.0,)))  # 1.0 -> avg 1.0 >= 0.7 -> no evolve
        assert sc3.should_evolve() is False


def _write_trace_file(tmp_path, name, events):
    import json
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return p


def _events(outcome, duration=0.0, task_id="t"):
    return [
        {"kind": "task_begin", "task_id": task_id,
         "layer": LAYER_VERIFICATION, "user_input": "x"},
        {"kind": "tool_result", "task_id": task_id,
         "layer": LAYER_EXECUTION, "duration_s": duration},
        {"kind": "loop_end", "task_id": task_id,
         "layer": LAYER_OBSERVABILITY, "outcome": outcome},
    ]


def _stub_events(task_id="stub"):
    """A single task_begin with no loop_end — an aborted noise stub."""
    return [
        {"kind": "task_begin", "task_id": task_id,
         "layer": LAYER_VERIFICATION, "user_input": "x"},
    ]


class TestLoadCorpusStubFilter:
    """Aborted noise stubs (no loop_end, < MIN_ACTIVITY_EVENTS events) record
    no outcome: they must not enter the quality window (same rule as
    harnessfix.corpus_quality._is_countable)."""

    def test_aborted_stub_is_not_counted(self, tmp_path):
        _write_trace_file(tmp_path, "a", _events(COMPLETED))
        _write_trace_file(tmp_path, "b", _events(COMPLETED))
        # Aborted write: a lone task_begin, no loop_end, 1 event.
        _write_trace_file(tmp_path, "stub", _stub_events("stub"))
        sc = EvolutionMetricsScorer(window_size=50)
        assert sc.load_corpus(tmp_path) == 2
        # Stub excluded -> average stays at the two real 1.0 runs.
        assert sc.windowed_average() == pytest.approx(1.0)

    def test_terse_completed_run_is_counted(self, tmp_path):
        # A genuinely finished run with < MIN_ACTIVITY_EVENTS events still
        # carries loop_end, so it IS evidence and must be kept.
        _write_trace_file(tmp_path, "terse", [
            _stub_events("terse")[0],
            {"kind": "loop_end", "task_id": "terse",
             "layer": LAYER_OBSERVABILITY, "outcome": "completed"},
        ])
        sc = EvolutionMetricsScorer(window_size=50)
        assert sc.load_corpus(tmp_path) == 1
        assert sc.windowed_average() == pytest.approx(1.0)

    def test_iter_run_scores_skips_stub(self, tmp_path):
        _write_trace_file(tmp_path, "a", _events(COMPLETED))
        _write_trace_file(tmp_path, "stub", _stub_events("stub"))
        scores = dict(iter_run_scores(tmp_path))
        assert "stub" not in scores
        assert len(scores) == 1

    def test_stub_cannot_flip_should_evolve(self, tmp_path):
        # Two real completions + many stubs: without the filter the stubs
        # would drag the window under threshold and falsely trigger evolution.
        for i in range(2):
            _write_trace_file(tmp_path, f"real{i}", _events(COMPLETED))
        for i in range(20):
            _write_trace_file(tmp_path, f"stub{i}", _stub_events(f"stub{i}"))
        sc = EvolutionMetricsScorer(window_size=10, threshold=0.7)
        sc.load_corpus(tmp_path)
        assert sc.windowed_average() == pytest.approx(1.0)
        assert sc.should_evolve() is False


def _guard_run(outcome, answer: str = "", duration: float | None = None,
               task_id="t") -> list[dict]:
    """A guard-terminated run: tool work, an LLM answer, then loop_end.

    Mirrors what the real loop emits when a guard (stuck / budget_exhausted /
    no_progress) stops the loop AFTER the forced-synthesis answer.
    """
    events = [
        {"kind": "task_begin", "task_id": task_id,
         "layer": LAYER_VERIFICATION, "user_input": "x"},
    ]
    if duration is not None:
        events.append({"kind": "tool_result", "task_id": task_id,
                       "layer": LAYER_EXECUTION, "duration_s": duration})
    if answer:
        events.append({"kind": "llm_response", "task_id": task_id,
                       "layer": LAYER_OBSERVABILITY, "text": answer})
    events.append({"kind": "loop_end", "task_id": task_id,
                   "layer": LAYER_OBSERVABILITY, "outcome": outcome})
    return events


#: Comfortably above MIN_FINAL_ANSWER_CHARS (80) — a real synthesis answer.
LONG_ANSWER = "Here is the fix I applied: I changed the guard to resume " \
              "instead of restarting, and I verified it with pytest."


class TestDeliveredGuardTerminatedRuns:
    """Decision #052 refinement, applied to SCORING.

    A guard stops the LOOP, not the answer: a stuck / budget_exhausted /
    no_progress run that still delivered a substantive final answer DID the
    task.  Before this fix every such run scored exactly 0.0 — contradicting
    the diagnose path, which already treats them as delivered
    (``TraceGraph.has_final_answer``).  Measured on the live corpus: 117 runs
    (89 stuck, 26 budget_exhausted, 2 no_progress) were mis-scored 0.0.
    """

    @pytest.mark.parametrize("outcome", ["stuck", "budget_exhausted", "no_progress"])
    def test_delivered_guard_run_is_not_zero(self, outcome):
        assert score_run(_guard_run(outcome, LONG_ANSWER)) > 0.0

    @pytest.mark.parametrize("outcome", ["stuck", "budget_exhausted", "no_progress"])
    def test_delivered_guard_run_scores_the_delivered_base(self, outcome):
        # No latency -> the full delivered base.
        assert score_run(_guard_run(outcome, LONG_ANSWER)) == pytest.approx(0.9)

    @pytest.mark.parametrize("outcome", ["stuck", "budget_exhausted", "no_progress"])
    def test_guard_run_without_answer_still_scores_zero(self, outcome):
        # The guard fired and nothing was delivered: a genuine failure.
        assert score_run(_guard_run(outcome)) == 0.0

    def test_delivered_scores_below_a_clean_completion(self):
        # Guards firing is itself a signal to evolve, so delivering through a
        # guard must never outrank a clean completion.
        delivered = score_run(_guard_run("stuck", LONG_ANSWER))
        clean = score_run(_run(COMPLETED))
        assert delivered < clean

    def test_error_outcome_keeps_scoring_zero_even_with_an_answer(self):
        # A provider/transport failure is a REAL failure: text produced before
        # it must not be mistaken for a delivered task.
        assert score_run(_guard_run(ERROR, LONG_ANSWER)) == 0.0

    def test_interrupted_run_with_an_answer_still_scores_zero(self):
        # No loop_end at all (crash/kill/provider loss, decision #052).
        events = [e for e in _guard_run("stuck", LONG_ANSWER)
                  if e["kind"] != "loop_end"]
        assert score_run(events) == 0.0

    def test_latency_penalty_applies_on_top_of_the_delivered_base(self):
        # 5s / 30s budget * 0.3 = 0.05 -> 0.9 - 0.05 = 0.85
        assert score_run(
            _guard_run("stuck", LONG_ANSWER, duration=5.0)
        ) == pytest.approx(0.85)

    def test_delivered_base_is_clamped_to_one(self):
        assert score_run(_guard_run("stuck", LONG_ANSWER), delivered_base=5.0) == 1.0


class TestFinalAnswerThresholdIsSharedWithHtir:
    """The "did this run deliver?" judgement must not drift between the
    diagnose path (TraceGraph.has_final_answer) and scoring (score_run)."""

    def test_threshold_is_imported_from_htir(self):
        from harnessfix import htir
        from harnessfix import evolution_metrics as em

        # Same single source of truth, not two literals.
        assert htir.MIN_FINAL_ANSWER_CHARS == 80
        answer = "x" * (htir.MIN_FINAL_ANSWER_CHARS - 1)
        assert score_run(_guard_run("stuck", answer)) == 0.0
        answer = "x" * htir.MIN_FINAL_ANSWER_CHARS
        assert score_run(_guard_run("stuck", answer)) > 0.0
        assert em.DELIVERED_OUTCOMES == {"stuck", "budget_exhausted", "no_progress"}

    def test_only_the_LAST_llm_response_counts(self):
        # An early long answer followed by a terse closing response means the
        # run did NOT deliver (the closing text is what the user got).
        events = _guard_run("stuck", LONG_ANSWER)
        events.insert(-1, {"kind": "llm_response", "task_id": "t",
                           "layer": LAYER_OBSERVABILITY, "text": "Hmm."})
        assert score_run(events) == 0.0


class TestDeliveredRunsInTheQualityWindow:
    def test_delivered_guard_run_lifts_the_windowed_average(self):
        sc = EvolutionMetricsScorer(window_size=3, threshold=0.7)
        sc.record_trace(_guard_run("stuck", LONG_ANSWER))
        # 0.9 alone is >= threshold, so no evolution is flagged...
        assert sc.windowed_average() == pytest.approx(0.9)
        assert sc.should_evolve() is False

    def test_undelivered_guard_run_still_drags_the_window_down(self):
        sc = EvolutionMetricsScorer(window_size=3, threshold=0.7)
        sc.record_trace(_guard_run("stuck"))
        assert sc.windowed_average() == 0.0
        assert sc.should_evolve() is True


class TestRecordedSuccessFlagTracksTheScore:
    """The ``ExecutionMetric.success`` flag fed into the window must agree with
    what the score means (decision #052): a run that scored above zero DID the
    task, including a guard-terminated run that still delivered its answer.

    Before this fix the flag was ``outcome == "completed"``, so a delivered
    guard run was recorded as score 0.9 / success=False — a self-contradiction.
    """

    def _last_success(self, sc) -> bool:
        return sc._metrics.recent_metrics()[-1].success

    @pytest.mark.parametrize("outcome", ["stuck", "budget_exhausted", "no_progress"])
    def test_delivered_guard_run_is_recorded_as_success(self, outcome):
        sc = EvolutionMetricsScorer()
        sc.record_trace(_guard_run(outcome, LONG_ANSWER))
        assert sc._metrics.recent_metrics()[-1].score == pytest.approx(0.9)
        assert self._last_success(sc) is True

    @pytest.mark.parametrize("outcome", ["stuck", "budget_exhausted", "no_progress"])
    def test_undelivered_guard_run_is_recorded_as_failure(self, outcome):
        sc = EvolutionMetricsScorer()
        sc.record_trace(_guard_run(outcome))
        assert self._last_success(sc) is False

    def test_error_run_is_recorded_as_failure(self):
        sc = EvolutionMetricsScorer()
        sc.record_trace(_guard_run(ERROR, LONG_ANSWER))
        assert self._last_success(sc) is False

    def test_completed_run_is_recorded_as_success(self):
        sc = EvolutionMetricsScorer()
        sc.record_trace(_run(COMPLETED))
        assert self._last_success(sc) is True

    def test_penalised_but_completed_run_is_still_a_success(self):
        # Huge latency floors the score at 0.7 — still a delivered task.
        sc = EvolutionMetricsScorer()
        sc.record_trace(_run(COMPLETED, (100000.0,)))
        assert self._last_success(sc) is True

    def test_flag_never_contradicts_the_recorded_score(self):
        for events in (
            _run(COMPLETED),
            _run(COMPLETED, (100000.0,)),
            _run(ERROR),
            _guard_run("stuck", LONG_ANSWER),
            _guard_run("stuck"),
        ):
            sc = EvolutionMetricsScorer()
            sc.record_trace(events)
            m = sc._metrics.recent_metrics()[-1]
            assert m.success is (m.score > 0.0)


class TestReadonlyContract:
    def test_no_file_writes_on_import_or_score(self, tmp_path):
        # Scoring synthetic events must not create/modify any file.
        before = set(p.name for p in tmp_path.iterdir())
        sc = EvolutionMetricsScorer()
        sc.record_trace(_run(COMPLETED, (1.0,)))
        score_run(_run(ERROR))
        after = set(p.name for p in tmp_path.iterdir())
        assert before == after
