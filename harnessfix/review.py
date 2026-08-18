"""Human review of failed task traces — the verification gate.

Builds a per-task review ledger from the trace corpus + diagnoses, lets the
user label each task (bug / regression / noise / ok), and exports labeled
tasks as diagnosis-pinning regression tests.

The ledger lives at ``reports/harnessfix/review.json`` (gitignored with the
rest of reports/): it is local, human-owned state, not repo test data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .corpus import _is_failed_trace
from .diagnose import Diagnosis, diagnose_graph
from .htir import TraceGraph, compile_trace
from .reader import TraceValidationError

DISPOSITIONS: tuple[str, ...] = ("bug", "regression", "noise", "ok")
REVIEWS_RELPATH = "reports/harnessfix/review.json"
EXPORT_DIR = "reports/harnessfix/generated"

_PROMPT_KEY = "user_input"
_MODEL_KEY = "model"
_PROFILE_KEY = "profile"


@dataclass
class ReviewRecord:
    task_id: str
    prompt: str = ""
    model: str = ""
    profile: str = ""
    outcome: str = ""
    guards: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    root_layer: str = ""
    mechanism: str = ""
    disposition: str = "unreviewed"
    note: str = ""
    review_date: str = ""

    def is_labeled(self) -> bool:
        return self.disposition in DISPOSITIONS


def load_reviews(path: Path) -> dict[str, ReviewRecord]:
    """Load the review ledger (missing/corrupt file = empty ledger)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    reviews: dict[str, ReviewRecord] = {}
    for item in data:
        if not isinstance(item, dict) or not item.get("task_id"):
            continue
        reviews[item["task_id"]] = ReviewRecord(**{
            k: v for k, v in item.items() if k in ReviewRecord.__dataclass_fields__
        })
    return reviews


def save_reviews(reviews: dict[str, ReviewRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(r) for r in sorted(reviews.values(), key=lambda r: r.task_id)]
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_reviews(
    trace_dir: Path, diagnoses_dir: Path | None = None
) -> dict[str, ReviewRecord]:
    """Rebuild the ledger from the trace corpus + persisted diagnoses.

    Only FAILED traces (tool errors / guards / interrupted runs) are
    reviewed; completed successes are not defects to review.
    """
    from .corpus import collect_traces

    reviews: dict[str, ReviewRecord] = {}
    for path in collect_traces(trace_dir):
        try:
            graph = compile_trace(path)
        except TraceValidationError:
            continue
        if not _is_failed_trace(graph):
            continue
        task_id = path.stem
        diagnosis: Diagnosis | None = None
        if diagnoses_dir is not None:
            diag_path = diagnoses_dir / f"{task_id}.json"
            if diag_path.is_file():
                try:
                    diagnosis = Diagnosis.model_validate_json(
                        diag_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    diagnosis = None
        if diagnosis is None:
            diagnosis = diagnose_graph(graph)

        prompt, model, profile = _trace_context(graph)
        guards = [
            str(s.payload["guard"])
            for s in graph.steps
            if s.kind == "guard_triggered" and s.payload.get("guard")
        ]
        outcome = next(
            (
                str(s.payload.get("outcome", ""))
                for s in reversed(graph.steps)
                if s.kind == "loop_end"
            ),
            "",
        )
        reviews[task_id] = ReviewRecord(
            task_id=task_id,
            prompt=prompt,
            model=model,
            profile=profile,
            outcome=outcome,
            guards=guards,
            affected_files=graph.affected_files(),
            root_layer=diagnosis.root_layer,
            mechanism=diagnosis.mechanism,
        )
    return reviews


def _trace_context(graph: TraceGraph) -> tuple[str, str, str]:
    prompt = model = profile = ""
    for s in graph.steps:
        if s.kind == "task_begin":
            prompt = str(s.payload.get(_PROMPT_KEY, ""))
            model = str(s.payload.get(_MODEL_KEY, ""))
            profile = str(s.payload.get(_PROFILE_KEY, ""))
            break
    if not model:
        for s in graph.steps:
            model = str(s.payload.get(_MODEL_KEY, ""))
            profile = str(s.payload.get(_PROFILE_KEY, ""))
            if model:
                break
    return prompt, model, profile


def label_review(
    reviews: dict[str, ReviewRecord],
    task_id: str,
    disposition: str,
    note: str = "",
) -> ReviewRecord:
    if disposition not in DISPOSITIONS:
        raise ValueError(f"disposition must be one of {DISPOSITIONS}")
    record = reviews.get(task_id)
    if record is None:
        raise KeyError(f"no review record for task {task_id}")
    record.disposition = disposition
    record.note = note
    record.review_date = datetime.now().isoformat(timespec="seconds")
    return record


def review_table(reviews: dict[str, ReviewRecord]) -> str:
    """Compact table of every review record for the REPL."""
    if not reviews:
        return "No reviews yet — run `review refresh` first."
    header = (
        f"{'task_id':<36} {'disp':<11} {'model':<20} {'layer':<24} outcome"
    )
    lines = [header, "-" * len(header)]
    for rec in sorted(reviews.values(), key=lambda r: r.task_id):
        lines.append(
            f"{rec.task_id:<36} {(rec.disposition or '-'):<11} "
            f"{(rec.model or '-'):<20} {(rec.root_layer or '-'):<24} "
            f"{rec.outcome or '-'}"
        )
    return "\n".join(lines)


def export_regression_test(
    record: ReviewRecord,
    trace_path: Path,
    out_dir: Path,
    mechanism_key: str | None = None,
) -> Path:
    """Write a pytest file pinning the diagnosis of a labeled task.

    The pin asserts the diagnosis layer and a distinctive mechanism
    substring, so any future change in diagnosis behaviour for this trace
    fails loudly at the human gate.  Exported files live under
    reports/harnessfix/generated/ (gitignored like the traces themselves).
    """
    mechanism = mechanism_key or record.mechanism
    key = mechanism[:80]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"test_review_{record.task_id}.py"
    safe_trace = str(trace_path.resolve()).replace("\\", "\\\\")
    out.write_text(
        f'''"""Generated regression pin for task {record.task_id}
(disposition: {record.disposition}). Do not edit by hand."""

from pathlib import Path

from harnessfix.diagnose import diagnose_graph
from harnessfix.reader import compile_trace

TRACE = Path(r"{safe_trace}")


def test_review_pin():
    graph = compile_trace(TRACE)
    diag = diagnose_graph(graph)
    assert diag.root_layer == "{record.root_layer}"
    assert {key!r} in diag.mechanism
''',
        encoding="utf-8",
    )
    return out
