"""Tests for benchmark entrypoint behavior around API failures and categories."""

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from benchmark import (
    BenchmarkError,
    CategoryResult,
    ModelAPIError,
    ModelResult,
    QuestionResult,
    answers_match,
    normalize_answer,
    save_json_report,
)


def _run_categories_parallel(categories: list[Any]) -> list[Any]:
    """Run category callbacks concurrently and preserve input order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(categories)) as pool:
        futures = [pool.submit(lambda c: c, category) for category in categories]
        return [future.result() for future in futures]


def test_model_api_error_is_benchmark_error() -> None:
    assert issubclass(ModelAPIError, BenchmarkError)


def test_api_failure_is_isolated_from_answer_matching() -> None:
    failure = "ModelAPIError: upstream unavailable"
    assert answers_match(failure, "ok") is False
    assert normalize_answer(failure) == "modelapierror upstream unavailable"


def test_parallel_category_execution_preserves_order_and_isolates_failures() -> None:
    assert _run_categories_parallel([3, 1, 2]) == [3, 1, 2]

    def execute_category(category: str) -> str:
        if category == "middle":
            raise ModelAPIError(f"ModelAPIError: {category} unavailable")
        return f"done:{category}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(execute_category, category): category
            for category in ("first", "middle", "last")
        }
        outcomes: dict[str, Any] = {}
        for future in concurrent.futures.as_completed(futures):
            category = futures[future]
            try:
                outcomes[category] = future.result()
            except ModelAPIError as exc:
                outcomes[category] = exc

    assert outcomes["first"] == "done:first"
    assert isinstance(outcomes["middle"], ModelAPIError)
    assert outcomes["last"] == "done:last"


def test_benchmark_error_report_is_json_serializable(tmp_path: Path) -> None:
    question = QuestionResult(
        question_idx=0,
        prompt="Say ok",
        response="ModelAPIError: rate limited",
        correct=None,
        score=0.0,
        latency_ms=12.0,
    )
    result = ModelResult(model="test-model")
    result.categories["code"] = CategoryResult(category="code", results=[question])

    report_path = tmp_path / "report.json"
    save_json_report([result], str(report_path))
    data = json.loads(report_path.read_text(encoding="utf-8"))

    entry = data["models"][0]
    assert entry["model"] == "test-model"
    assert entry["categories"]["code"]["correct_count"] == 0
    assert entry["overall_accuracy"] is None
    assert "error" in question.response.lower()
    assert result.total_questions == 1


def test_benchmark_helpers_are_stable_for_entrypoint_reports() -> None:
    assert normalize_answer("  Hello, World! ") == "hello world"
    assert answers_match("Hello, World!", "hello world")
    assert answers_match("Say ok", "ok")
