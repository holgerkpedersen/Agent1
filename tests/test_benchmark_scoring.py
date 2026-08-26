"""Tests for the tightened benchmark scorer (plan FIX item 21):
exact/word-bounded matching instead of lenient substring/keyword overlap,
plus the optional LLM-judge fallback."""

from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, patch

from benchmark import (
    BenchmarkError,
    ModelAPIError,
    accuracy_delta,
    answers_match,
    judge_answer,
    load_models_json,
    load_report_snapshots,
    model_accuracy_history,
    run_benchmark,
    run_category,
    score_instruction_following,
    score_question,
    trend_summary,
)


def _snap(model: str, acc: float | None) -> dict:
    return {"model": model, "overall_accuracy": acc}


class TestModelApiFailure:
    """API failures are unavailable, not wrong: they must not count against
    accuracy or flip correct repetitions to incorrect (regression)."""

    def test_single_failure_is_unscored_not_incorrect(self) -> None:
        async def _run() -> None:
            with patch(
                "benchmark.query_model",
                new=AsyncMock(side_effect=ModelAPIError("api down")),
            ):
                cat = await run_category("m1", "coding", [("What is 2+2?", "4")])
            r = cat.results[0]
            assert r.correct is None
            assert r.score == 0.0
            assert "api down" in r.response
            assert cat.scoreable_count == 0
            assert cat.accuracy is None

        asyncio.run(_run())

    def test_failure_does_not_distort_mixed_run(self) -> None:
        async def _run() -> None:
            with patch(
                "benchmark.query_model",
                new=AsyncMock(
                    side_effect=[("4", 10.0, 5), ModelAPIError("down")]
                ),
            ):
                cat = await run_category(
                    "m1",
                    "coding",
                    [("What is 2+2?", "4"), ("What is the capital of France?", "Paris")],
                )
            assert cat.results[0].correct is True
            assert cat.results[1].correct is None
            assert cat.accuracy == 100.0

        asyncio.run(_run())

    def test_repetition_merge_true_with_failure_stays_correct(self) -> None:
        async def _run() -> None:
            with patch(
                "benchmark.query_model",
                new=AsyncMock(side_effect=[("4", 10.0, 5), ModelAPIError("down")]),
            ):
                res = await run_benchmark(
                    ["m1"], {"coding": [("What is 2+2?", "4")]}, repetitions=2
                )
            r = res[0].categories["coding"].results[0]
            assert r.correct is True
            assert res[0].overall_accuracy == 100.0

        asyncio.run(_run())

    def test_repetition_merge_all_failures_unscored(self) -> None:
        async def _run() -> None:
            with patch(
                "benchmark.query_model",
                new=AsyncMock(
                    side_effect=[ModelAPIError("down"), ModelAPIError("down")]
                ),
            ):
                res = await run_benchmark(
                    ["m1"], {"coding": [("What is 2+2?", "4")]}, repetitions=2
                )
            r = res[0].categories["coding"].results[0]
            assert r.correct is None
            assert res[0].overall_accuracy is None

        asyncio.run(_run())


class TestTrendHelpers:
    def test_load_models_json_missing(self, tmp_path):
        assert load_models_json(str(tmp_path / "nope.json")) == {}

    def test_load_models_json_parses(self, tmp_path):
        p = tmp_path / "models.json"
        p.write_text('{"m1": {"overall_accuracy": 80.0}}', encoding="utf-8")
        assert load_models_json(str(p))["m1"]["overall_accuracy"] == 80.0

    def test_snapshot_loading_sorted(self, tmp_path):
        (tmp_path / "benchmark_20260701_100000.json").write_text(
            '{"models": [{"model": "m1", "overall_accuracy": 60.0}]}', encoding="utf-8"
        )
        (tmp_path / "benchmark_20260702_100000.json").write_text(
            '{"models": [{"model": "m1", "overall_accuracy": 70.0}, {"model": "m2", "overall_accuracy": 90.0}]}',
            encoding="utf-8",
        )
        snaps = load_report_snapshots(str(tmp_path))
        assert [m["model"] for _, m in snaps] == ["m1", "m1", "m2"]

    def test_history_with_gaps(self):
        snaps = [
            ("s1", _snap("m1", 60.0)),
            ("s2", _snap("m1", None)),
            ("s3", _snap("m1", 70.0)),
        ]
        history = model_accuracy_history("m1", snaps)
        assert [acc for _, acc in history] == [60.0, None, 70.0]

    def test_delta_latest_vs_previous(self):
        snaps = [
            ("s1", _snap("m1", 60.0)),
            ("s2", _snap("m1", 75.0)),
        ]
        assert accuracy_delta("m1", snaps) == 15.0

    def test_delta_needs_two_scored_runs(self):
        snaps = [("s1", _snap("m1", 60.0)), ("s2", _snap("m1", None))]
        assert accuracy_delta("m1", snaps) is None

    def test_trend_summary_direction_and_confidence(self):
        snaps = [
            ("s1", _snap("m1", 50.0)),
            ("s2", _snap("m1", 60.0)),
            ("s3", _snap("m1", 70.0)),
        ]
        summary = trend_summary("m1", snaps)
        assert summary["direction"] == "improved"
        assert summary["delta"] == 10.0
        assert summary["confidence"] is True

    def test_trend_summary_regression_low_confidence(self):
        snaps = [
            ("s1", _snap("m1", 80.0)),
            ("s2", _snap("m1", 70.0)),
        ]
        summary = trend_summary("m1", snaps)
        assert summary["direction"] == "regressed"
        assert summary["confidence"] is False


class TestAnswersMatchTightened:
    def test_exact_match(self):
        assert answers_match("Paris", "paris")

    def test_word_bounded_expected_inside_response(self):
        assert answers_match("The answer is 42.", "42")

    def test_no_raw_substring_false_positive(self):
        assert not answers_match("142", "42")
        assert not answers_match("retry the test", "true")
        assert not answers_match("retro", "true")

    def test_no_reverse_containment_credit(self):
        assert not answers_match("42", "The answer is 42.")

    def test_numeric_tolerance(self):
        assert answers_match("3.2", "3")
        assert answers_match("1000", "1,000")

    def test_fraction_equivalence(self):
        assert answers_match("0.5", "1/2")

    def test_keyword_overlap_tightened(self):
        # One keyword out of eight must NOT match anymore.
        assert not answers_match("Paris", "Paris is the capital of France in Europe")
        # Substantial overlap (7/8 words) still matches.
        assert answers_match(
            "Paris is the capital of France Europe",
            "Paris is the capital of France in Europe",
        )

    def test_padded_response_not_credited(self):
        # Same expected words but buried in a much longer response.
        assert not answers_match(
            "The quick brown fox jumps over the lazy dog near the Paris "
            "riverbank on a sunny Sunday afternoon while birds sing",
            "Paris is the capital",
        )


class TestScoreInstructionFollowingTightened:
    def test_nothing_else_exact_echo(self):
        score = score_instruction_following(
            'Respond with "true" and nothing else.', "true"
        )
        assert score == 1.0

    def test_nothing_else_extra_words_penalized(self):
        score = score_instruction_following(
            'Respond with "true" and nothing else.', "true and false"
        )
        assert score < 1.0

    def test_nothing_else_punctuation_tolerant(self):
        score = score_instruction_following(
            'Respond with "true" and nothing else.', "true."
        )
        assert score == 1.0

    def test_exactly_three_words(self):
        assert score_instruction_following(
            "Respond with exactly 3 words.", "one two three"
        ) == 1.0
        assert score_instruction_following(
            "Respond with exactly 3 words.", "one two"
        ) < 1.0

    def test_all_lowercase(self):
        assert score_instruction_following(
            "Write your answer in only lowercase.", "hello world"
        ) == 1.0
        assert score_instruction_following(
            "Write your answer in only lowercase.", "Hello World"
        ) < 1.0


class TestJudgeFallback:
    async def _judge(self, response, expected):
        return "yes they match" if "yes" in expected else "no"

    def test_judge_overrides_heuristic_false(self):
        async def go():
            correct, _ = await score_question(
                "q", "anything", "yes please", "knowledge", judge_fn=self._judge
            )
            return correct

        assert asyncio.run(go()) is True

    def test_heuristic_true_never_consults_judge(self):
        calls = []

        async def judge(response, expected):
            calls.append(1)
            return "no"

        async def go():
            return await score_question(
                "q", "Paris", "paris", "knowledge", judge_fn=judge
            )

        correct, _ = asyncio.run(go())
        assert correct is True
        assert calls == []

    def test_judge_absent_uses_heuristics(self):
        async def go():
            return await score_question("q", "Paris", "paris", "knowledge")

        correct, _ = asyncio.run(go())
        assert correct is True

    def test_judge_answer_parses_verdicts(self):
        async def judge(response, expected):
            return "No, they differ."

        async def go():
            return await judge_answer(judge, "a", "b")

        assert asyncio.run(go()) is False

    def test_judge_answer_ambiguous_returns_none(self):
        async def judge(response, expected):
            return "maybe"

        async def go():
            return await judge_answer(judge, "a", "b")

        assert asyncio.run(go()) is None

    def test_judge_answer_exception_returns_none(self):
        async def judge(response, expected):
            raise RuntimeError("judge down")

        async def go():
            return await judge_answer(judge, "a", "b")

        assert asyncio.run(go()) is None


class TestLooksIncompleteTuning:
    def test_completion_overrides_weak_marker(self):
        from agent import _looks_incomplete
        # "remaining" is a weak marker, but the answer is clearly finished.
        assert not _looks_incomplete("The remaining issue is fixed — all done.")
        assert not _looks_incomplete("Budget spent, task is complete.")

    def test_weak_marker_without_completion_still_incomplete(self):
        from agent import _looks_incomplete
        assert _looks_incomplete("The tool budget is exhausted, I still need more calls.")
        assert _looks_incomplete("I could not complete the analysis yet — would need more budget.")


class TestProbeEndpoint:
    """The fail-fast probe must abort quickly when no usable LM Studio
    endpoint/model is present, so the autonomous gate never stalls for the
    full 1800s timeout grinding through ~125 failing questions."""

    def test_no_model_loaded_raises_benchmark_error(self, monkeypatch):
        import urllib.error

        class _FakeResp:
            def read(self):
                return (b'{"error":{"message":"No models loaded. Please load a '
                        b'model in the developer page.","type":"invalid_request_error"}}')

            def close(self):
                return None

            @property
            def code(self):
                return 400

        def _urlopen(req, timeout=5.0):
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, _FakeResp())

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        from benchmark import _probe_endpoint
        try:
            _probe_endpoint("http://localhost:1234/v1/chat/completions", timeout=1.0)
            assert False, "expected BenchmarkError"
        except BenchmarkError as exc:
            assert "no model" in str(exc).lower()

    def test_unreachable_endpoint_raises_benchmark_error(self, monkeypatch):
        import urllib.error

        def _urlopen(req, timeout=5.0):
            raise urllib.error.URLError("connection refused")

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        from benchmark import _probe_endpoint
        try:
            _probe_endpoint("http://localhost:1234/v1/chat/completions", timeout=1.0)
            assert False, "expected BenchmarkError"
        except BenchmarkError as exc:
            assert "unreachable" in str(exc).lower()

    def test_valid_chat_response_passes(self, monkeypatch):
        import urllib.request

        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"hi"}}]}'

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=5.0: _Ctx())
        from benchmark import _probe_endpoint
        # Should not raise.
        _probe_endpoint("http://localhost:1234/v1/chat/completions", timeout=1.0)
