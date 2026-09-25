"""Jev decision telemetry — predictions + outcomes for threshold calibration.

Every :meth:`agent_core.jev_engine.JevEngine.decide` appends one record to
``reports/history/jev.jsonl`` (same ``reports/history`` tree as the execution
ledger, gitignored).  A record carries the typed decision, the probability
distribution, the threshold and the mechanism, so the harness can later:

* **measure** how often the small model is right (accuracy per primitive);
* **calibrate** the hardcoded accept/reject thresholds from real outcomes
  instead of guessing them (the ``jev stats`` report suggests a threshold);
* **audit** which model/mechanism actually answered (the failover-drift class
  of bug is visible in the data).

Design rules
------------
* **Best-effort**: telemetry must never break or delay a decision — every
  writer swallows ``OSError`` and returns ``None``/``False``.
* **Opt-out**: ``AGENT_NO_JEV_LOG=1`` disables recording entirely (tests use
  it so fake providers never pollute the real ledger).
* **Append-only, atomic updates**: decisions are appended; labeling an outcome
  rewrites the file through a ``.tmp`` + ``os.replace`` so a crash cannot
  corrupt the ledger.
* Outcomes are ``correct`` / ``incorrect`` (the truth value a caller knows
  afterwards, e.g. a human review disposition or a later gate result).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .history import HISTORY_SUBDIR

#: Ledger filename inside ``reports/history/``.
JEV_FILE = "jev.jsonl"
#: Env opt-out (same shape as ``AGENT_NO_TRACE``).
DISABLE_ENV = "AGENT_NO_JEV_LOG"

CORRECT = "correct"
INCORRECT = "incorrect"
OUTCOME_VALUES: tuple[str, ...] = (CORRECT, INCORRECT)


def logging_disabled() -> bool:
    """True when ``AGENT_NO_JEV_LOG`` opts out of telemetry."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def question_hash(text: str) -> str:
    """Stable short hash of a question (groups repeated questions)."""
    return hashlib.sha256(str(text).strip().encode("utf-8")).hexdigest()[:16]


def resolve_root(workspace: str | None = None) -> str:
    """Repo root for the ledger: *workspace* when given, else walk up to the
    nearest ``reports/`` ancestor (``harnessfix.history.history_root``)."""
    if workspace:
        return str(workspace)
    try:
        from .history import history_root

        root = history_root(os.getcwd())
        if root:
            return root
    except Exception:  # noqa: BLE001 - root resolution must never raise
        pass
    return os.getcwd()


def log_path(workspace: str | None = None) -> Path:
    """Path of the Jev decision ledger."""
    return Path(resolve_root(workspace)) / "reports" / HISTORY_SUBDIR / JEV_FILE


def record_decision(
    result: Any,
    question: Any,
    *,
    workspace: str | None = None,
    source: str = "engine",
) -> str | None:
    """Append one decision record; returns its id (or None when skipped).

    *result* is a :class:`agent_core.jev_engine.JevResult`, *question* a
    :class:`agent_core.jev_engine.JevQuestion`.  Never raises.
    """
    if logging_disabled():
        return None
    record = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "source": str(source),
        "model": str(getattr(result, "model", "")),
        "kind": str(getattr(result, "kind", "")),
        "mechanism": str(getattr(result, "mechanism", "")),
        "question": str(getattr(question, "text", ""))[:500],
        "question_hash": question_hash(getattr(question, "text", "")),
        "probabilities": {
            str(k): round(float(v), 4)
            for k, v in (getattr(result, "probabilities", {}) or {}).items()
        },
        "decision": str(getattr(result, "decision", "")),
        "confidence": round(float(getattr(result, "confidence", 0.0) or 0.0), 4),
        "threshold": float(getattr(result, "threshold", 0.7) or 0.7),
        "n": int(getattr(result, "n", 0) or 0),
        "abstentions": int(getattr(result, "abstentions", 0) or 0),
        "outcome": None,
        "outcome_ts": None,
    }
    try:
        path = log_path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(record["id"])
    except OSError:
        return None


def load_decisions(
    *, workspace: str | None = None, last: int | None = None,
) -> list[dict[str, Any]]:
    """Read the ledger (oldest first); unreadable/malformed lines are skipped."""
    path = log_path(workspace)
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    if last is not None and last > 0:
        return records[-last:]
    return records


def record_outcome(
    decision_id: str,
    outcome: str,
    *,
    workspace: str | None = None,
    note: str = "",
) -> bool:
    """Label one decision ``correct``/``incorrect``; returns True when found.

    Rewrites the ledger atomically (``.tmp`` + ``os.replace``).  Never raises
    on I/O; raises ``ValueError`` for an invalid *outcome* (programming error).
    """
    normalized = str(outcome).strip().lower()
    if normalized not in OUTCOME_VALUES:
        raise ValueError(
            f"outcome must be one of {', '.join(OUTCOME_VALUES)}, got {outcome!r}"
        )
    path = log_path(workspace)
    if not path.is_file():
        return False
    records = load_decisions(workspace=workspace)
    found = False
    for record in records:
        if record.get("id") == decision_id:
            record["outcome"] = normalized
            record["outcome_ts"] = time.time()
            if note:
                record["outcome_note"] = str(note)[:300]
            found = True
    if not found:
        return False
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except OSError:
        return False
    return True


def _p_yes(record: dict[str, Any]) -> float | None:
    probabilities = record.get("probabilities") or {}
    value = probabilities.get("yes")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calibration_bins(
    records: list[dict[str, Any]], width: float = 0.1,
) -> list[dict[str, Any]]:
    """Predicted-vs-observed bins for labeled yesno decisions.

    A well-calibrated engine has ``observed ~= predicted`` in every bin; a
    systematic gap means the threshold (and the confidence itself) needs
    adjustment.
    """
    buckets: dict[float, dict[str, float]] = {}
    for record in records:
        if record.get("kind") != "yesno":
            continue
        if record.get("outcome") not in OUTCOME_VALUES:
            continue
        p_yes = _p_yes(record)
        if p_yes is None:
            continue
        index = min(int(p_yes / width + 1e-9), int(1.0 / width) - 1)
        entry = buckets.setdefault(
            round(index * width, 2), {"n": 0.0, "predicted": 0.0, "correct": 0.0}
        )
        entry["n"] += 1
        entry["predicted"] += p_yes
        entry["correct"] += 1 if record.get("outcome") == CORRECT else 0
    report = []
    for low in sorted(buckets):
        entry = buckets[low]
        n = int(entry["n"])
        report.append({
            "bin": f"{low:.1f}-{low + width:.1f}",
            "n": n,
            "predicted": entry["predicted"] / n,
            "observed": entry["correct"] / n,
        })
    return report


def suggest_threshold(
    records: list[dict[str, Any]],
    kind: str = "yesno",
    grid: list[float] | None = None,
) -> dict[str, Any] | None:
    """Threshold maximizing accuracy on labeled records, or None.

    This is the calibration output: instead of trusting the hardcoded 0.7,
    the harness can read the threshold that actually separated correct from
    incorrect decisions in its own history.
    """
    thresholds = grid or [round(i / 100, 2) for i in range(50, 96, 5)]
    labeled: list[tuple[float, bool]] = []
    for record in records:
        if record.get("kind") != kind:
            continue
        if record.get("outcome") not in OUTCOME_VALUES:
            continue
        p_yes = _p_yes(record)
        if p_yes is None:
            continue
        labeled.append((p_yes, record.get("outcome") == CORRECT))
    if not labeled:
        return None
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        correct = sum(
            1 for p_yes, truth in labeled if (p_yes >= threshold) == truth
        )
        accuracy = correct / len(labeled)
        if best is None or accuracy > best["accuracy"]:
            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "n": len(labeled),
                "base_rate": sum(1 for _, truth in labeled if truth) / len(labeled),
            }
    return best


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a ledger slice into a calibration report."""
    total = len(records)
    kinds = Counter(str(r.get("kind", "")) for r in records)
    models = Counter(str(r.get("model", "")) for r in records)
    mechanisms = Counter(str(r.get("mechanism", "")) for r in records)
    sources = Counter(str(r.get("source", "")) for r in records)
    decisions = Counter(str(r.get("decision", "")) for r in records)
    abstentions = sum(int(r.get("abstentions", 0) or 0) for r in records)
    samples = sum(int(r.get("n", 0) or 0) for r in records)
    labeled = sum(1 for r in records if r.get("outcome") in OUTCOME_VALUES)
    return {
        "total": total,
        "labeled": labeled,
        "by_kind": dict(kinds),
        "by_model": dict(models),
        "by_mechanism": dict(mechanisms),
        "by_source": dict(sources),
        "decisions": dict(decisions),
        "abstention_rate": (abstentions / samples) if samples else 0.0,
        "calibration": calibration_bins(records),
        "suggested_threshold": suggest_threshold(records),
    }


def format_report(report: dict[str, Any]) -> str:
    """Plain-text rendering for the ``jev stats`` REPL command."""
    lines = [
        f"[jev] decisions={report['total']} labeled={report['labeled']} "
        f"abstention_rate={report['abstention_rate']:.1%}",
    ]
    for key in ("by_kind", "by_mechanism", "by_source", "by_model", "decisions"):
        data = report.get(key) or {}
        if data:
            rendered = " ".join(
                f"{name}={count}" for name, count in sorted(data.items())
            )
            lines.append(f"[jev] {key}: {rendered}")
    calibration = report.get("calibration") or []
    if calibration:
        lines.append("[jev] calibration (yesno predicted -> observed):")
        for entry in calibration:
            lines.append(
                f"[jev]   {entry['bin']}  n={entry['n']:<4} "
                f"predicted={entry['predicted']:.2f} observed={entry['observed']:.2f}"
            )
    suggestion = report.get("suggested_threshold")
    if suggestion:
        lines.append(
            f"[jev] suggested yesno threshold={suggestion['threshold']:.2f} "
            f"(accuracy {suggestion['accuracy']:.2f}, n={suggestion['n']}, "
            f"base_rate {suggestion['base_rate']:.2f})"
        )
    else:
        lines.append(
            "[jev] no labeled outcomes yet - label decisions with "
            "record_outcome(id, 'correct'|'incorrect') to calibrate."
        )
    return "\n".join(lines)


__all__: list[str] = [
    "CORRECT",
    "DISABLE_ENV",
    "INCORRECT",
    "JEV_FILE",
    "calibration_bins",
    "format_report",
    "load_decisions",
    "log_path",
    "logging_disabled",
    "question_hash",
    "record_decision",
    "record_outcome",
    "resolve_root",
    "suggest_threshold",
    "summarize",
]
