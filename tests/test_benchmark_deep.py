"""Regression tests for the deep-analysis benchmark run (--deep).

The deep run exists to measure a model the way the agent drives it: the
deep-analysis profile (temperature 0.7, large token budget) plus a second
self-review attempt on reasoning/math questions.  These tests pin that
behaviour on the REAL code path (benchmark.evaluate_question / verify_question
/ run_category / main), so a refactor that silently drops the review pass, the
temperature or the profile key fails here.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from unittest.mock import AsyncMock, patch

import pytest

import benchmark
from benchmark import (
    ModelAPIError,
    evaluate_question,
    parse_args,
    resolve_bench_profile,
    resolve_server_model_id,
    run_category,
    verify_question,
)


# ---------------------------------------------------------------------------
# Profile / CLI surface
# ---------------------------------------------------------------------------


class TestProfileResolution:
    def test_deep_analysis_profile_uses_analysis_temperature(self) -> None:
        settings = resolve_bench_profile("deep-analysis")
        assert settings["temperature"] == 0.7
        assert settings["verify_round"] is True
        assert settings["max_tokens"] > 2048

    def test_standard_profile_stays_single_shot(self) -> None:
        settings = resolve_bench_profile("standard")
        assert settings["temperature"] == 0.1
        assert settings["verify_round"] is False
        assert settings["max_tokens"] == 2048

    def test_unknown_profile_tag_falls_back_to_standard(self) -> None:
        # A custom tag is only a report key, never a crash.
        assert resolve_bench_profile("friday-1") == benchmark.PROFILES["standard"]


class TestDeepMaxTokensOverride:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(benchmark.DEEP_MAX_TOKENS_ENV, "4096")
        assert benchmark.deep_max_tokens(16384) == 4096

    def test_invalid_env_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(benchmark.DEEP_MAX_TOKENS_ENV, "lots")
        assert benchmark.deep_max_tokens(16384) == 16384

    def test_non_positive_env_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(benchmark.DEEP_MAX_TOKENS_ENV, "0")
        assert benchmark.deep_max_tokens(16384) == 16384


class TestCliSurface:
    def test_deep_flag_parses(self) -> None:
        args = parse_args(
            ["--deep", "--provider", "llama", "--model", "llama/deepseek-v4.1-flash"]
        )
        assert args.deep is True
        assert args.provider == "llama"
        assert args.max_tokens is None  # profile decides
        assert args.temperature is None

    def test_plain_run_has_no_deep_flags(self) -> None:
        args = parse_args(["--model", "m1"])
        assert args.deep is False
        assert args.verify_round is False


# ---------------------------------------------------------------------------
# One question, graded
# ---------------------------------------------------------------------------


class TestEvaluateQuestion:
    def test_api_failure_is_unscoreable_not_wrong(self) -> None:
        async def _run() -> benchmark.QuestionOutcome:
            with patch(
                "benchmark.query_model",
                new=AsyncMock(side_effect=ModelAPIError("boom")),
            ):
                return await evaluate_question("m", "q", "4", "math")

        outcome = asyncio.run(_run())
        assert outcome.error
        assert outcome.correct is None
        assert outcome.answered is False

    def test_temperature_is_forwarded(self) -> None:
        mock = AsyncMock(return_value=("4", 10.0, 5))

        async def _run() -> None:
            with patch("benchmark.query_model", new=mock):
                await evaluate_question("m", "2+2?", "4", "math", temperature=0.7)

        asyncio.run(_run())
        assert mock.await_args.kwargs["temperature"] == 0.7
        assert mock.await_args.kwargs["max_tokens"] == 2048


class TestFinalAnswerExtraction:
    def test_labelled_answer_beats_reasoning_text(self) -> None:
        review = "Let me check: 2+2 is definitely 4.\nANSWER: 4"
        assert benchmark.extract_final_answer(review) == "4"
        assert benchmark.answers_match(review, "4") is True

    def test_last_label_wins(self) -> None:
        review = "ANSWER: 5\nOn reflection that was wrong.\nANSWER: 4"
        assert benchmark.answers_match(review, "4") is True

    def test_unlabelled_response_is_graded_as_a_whole(self) -> None:
        assert benchmark.answers_match("The answer is 42.", "42") is True
        assert benchmark.answers_match(None, "42") is False

    def test_empty_label_falls_back_to_full_response(self) -> None:
        assert benchmark.extract_final_answer("ANSWER:   ") == "ANSWER:   "


# ---------------------------------------------------------------------------
# The deep second pass
# ---------------------------------------------------------------------------


class TestVerifyQuestion:
    def test_self_review_can_correct_a_wrong_first_answer(self) -> None:
        mock = AsyncMock(return_value=("ANSWER: 4", 5.0, 11))

        async def _run() -> benchmark.QuestionOutcome:
            first = benchmark.QuestionOutcome(
                response="I think it is 5.", correct=False, score=0.0,
                latency_ms=7.0, tokens_used=13,
            )
            with patch("benchmark.query_model", new=mock):
                return await verify_question(
                    "m", "What is 2+2?", "4", "math", first, temperature=0.7
                )

        second = asyncio.run(_run())
        assert second.correct is True
        # Cost of a deep question is both attempts.
        assert second.latency_ms == 12.0
        assert second.tokens_used == 24
        # The review prompt must carry the model's own previous answer.
        assert "I think it is 5." in mock.await_args.args[1]

    def test_failed_review_keeps_the_first_verdict(self) -> None:
        async def _run() -> benchmark.QuestionOutcome:
            first = benchmark.QuestionOutcome(
                response="4", correct=True, score=1.0,
                latency_ms=7.0, tokens_used=13,
            )
            with patch(
                "benchmark.query_model",
                new=AsyncMock(side_effect=ModelAPIError("down")),
            ):
                return await verify_question("m", "2+2?", "4", "math", first)

        kept = asyncio.run(_run())
        assert kept is not None
        assert kept.correct is True
        assert kept.latency_ms == 7.0


class TestRunCategoryVerifyRound:
    def test_verify_round_asks_reasoning_twice(self) -> None:
        mock = AsyncMock(return_value=("4", 1.0, 2))

        async def _run() -> benchmark.CategoryResult:
            with patch("benchmark.query_model", new=mock):
                return await run_category(
                    "m", "reasoning", [("What is 2+2?", "4")],
                    verify_round=True, temperature=0.7,
                )

        cat = asyncio.run(_run())
        assert mock.await_count == 2
        assert cat.results[0].correct is True

    def test_non_verified_categories_are_asked_once(self) -> None:
        mock = AsyncMock(return_value=("banana", 1.0, 2))

        async def _run() -> None:
            with patch("benchmark.query_model", new=mock):
                await run_category(
                    "m", "instruction_following",
                    [("Respond only with the word 'banana'.", "banana")],
                    verify_round=True,
                )

        asyncio.run(_run())
        assert mock.await_count == 1

    def test_standard_run_asks_once(self) -> None:
        mock = AsyncMock(return_value=("4", 1.0, 2))

        async def _run() -> None:
            with patch("benchmark.query_model", new=mock):
                await run_category(
                    "m", "reasoning", [("What is 2+2?", "4")], verify_round=False
                )

        asyncio.run(_run())
        assert mock.await_count == 1

    def test_deep_run_keeps_long_answers(self) -> None:
        long_answer = "4" * 3000

        async def _run() -> benchmark.CategoryResult:
            with patch(
                "benchmark.query_model",
                new=AsyncMock(return_value=(long_answer, 1.0, 2)),
            ):
                return await run_category(
                    "m", "math", [("2+2?", "4")],
                    verify_round=False,
                    response_chars=benchmark.REPORT_ANSWER_CHARS,
                )

        cat = asyncio.run(_run())
        assert len(cat.results[0].response) == 3000


# ---------------------------------------------------------------------------
# llama-server model id translation
# ---------------------------------------------------------------------------


class _FakeModelsResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeModelsResponse":
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class TestResolveServerModelId:
    URL = "http://127.0.0.1:8080/v1/chat/completions"

    def test_llama_provider_uses_the_served_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=3.0: _FakeModelsResponse(
                {"data": [{"id": "deepseek-v4.1-flash-ista-2.00bpw"}]}
            ),
        )
        resolved = resolve_server_model_id(
            "llama", self.URL, "llama/deepseek-v4.1-flash"
        )
        assert resolved == "deepseek-v4.1-flash-ista-2.00bpw"

    def test_matching_id_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=3.0: _FakeModelsResponse(
                {"data": [{"id": "qwen3-8b"}]}
            ),
        )
        assert resolve_server_model_id("llama", self.URL, "qwen3-8b") == "qwen3-8b"

    def test_lmstudio_model_name_is_untouched(self) -> None:
        # Only the single-model llama provider needs the translation.
        assert (
            resolve_server_model_id("lmstudio", self.URL, "qwen3-8b") == "qwen3-8b"
        )

    def test_unreachable_server_keeps_requested_name(self) -> None:
        def _boom(req, timeout=3.0):
            raise OSError("no route")

        with patch("urllib.request.urlopen", new=_boom):
            assert (
                resolve_server_model_id("llama", self.URL, "llama/m")
                == "llama/m"
            )


# ---------------------------------------------------------------------------
# End-to-end: the exact command line from the request
# ---------------------------------------------------------------------------


class TestDeepRunEndToEnd:
    def test_deep_llama_run_writes_a_deep_profile_report(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(benchmark.DEEP_MAX_TOKENS_ENV, raising=False)
        calls: list[dict] = []

        async def _fake_query(model, prompt, **kwargs):
            calls.append({"model": model, **kwargs})
            return ("4", 1.0, 3)

        monkeypatch.setattr(benchmark.sys, "argv", ["benchmark.py"])
        monkeypatch.setattr(benchmark, "parse_args", lambda: parse_args([
            # fmt: skip
            "--model", "llama/deepseek-v4.1-flash",
            "--provider", "llama",
            "--deep",
            "--categories", "math",
        ]))
        monkeypatch.setattr(benchmark, "_probe_endpoint", lambda *a, **k: None)
        monkeypatch.setattr(
            benchmark, "BASE_URL", "http://127.0.0.1:8080/v1/chat/completions"
        )
        monkeypatch.setattr(
            benchmark, "resolve_server_model_id", lambda p, u, m: "served-model"
        )
        monkeypatch.setattr(benchmark, "query_model", _fake_query)

        asyncio.run(benchmark.main())

        assert calls, "no question was asked"
        # Deep profile drives the request shape.
        assert calls[0]["temperature"] == 0.7
        assert calls[0]["max_tokens"] == 16384
        assert calls[0]["model"] == "served-model"
        # math is verified, so every question was asked twice.
        assert len(calls) % 2 == 0

        models_json = json.loads((tmp_path / "reports" / "models.json").read_text())
        assert "served-model|deep-analysis" in models_json
        entry = models_json["served-model|deep-analysis"]
        assert entry["profile"] == "deep-analysis"

        snapshots = list((tmp_path / "reports").glob("benchmark_*.json"))
        assert len(snapshots) == 1, "--deep must leave one trend snapshot"
        payload = json.loads(snapshots[0].read_text())
        assert payload["models"][0]["display_name"] == "served-model|deep-analysis"
