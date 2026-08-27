"""Phase 4 - offline, harness-centric quality signal for the HarnessFix loop.

The LLM benchmark (``benchmark.py``) measures *model* accuracy and needs a
live endpoint, so it is noisy and offline-incompatible.  This module derives
an acceptance signal from the **trace corpus itself** — the same evidence the
repairs are diagnosed from — so the autonomous loop can gate on harness/
agent behavior deterministically and without a model.

The signal is structural, not end-to-end: it validates that a repair is
*targeted* (it does not increase the incidence of the failure mechanism it
claims to fix) and that it does not *regress* the corpus-wide run-success
rate.  It deliberately does NOT claim to measure new improvement versus a
live model run — the LLM benchmark remains the only genuine e2e signal and
stays an optional cross-check.
"""
from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .corpus import _is_failed_trace, collect_traces, diagnose_corpus
from .diagnose import Diagnosis
from .reader import TraceValidationError
from .htir import compile_trace


class CorpusQuality(BaseModel):
    """Offline quality snapshot of a trace corpus.

    ``success_rate`` is the fraction of traces whose run completed
    (``loop_end.outcome == "completed"``), in [0, 1].  ``mechanism_counts``
    and ``layer_counts`` are the diagnosis frequencies across the *failed*
    traces, keyed by mechanism / layer.  ``total`` is the number of readable
    traces in the corpus.
    """

    success_rate: float
    mechanism_counts: dict[str, int]
    layer_counts: dict[str, int]
    total: int

    def mechanism_count(self, mechanism: str) -> int:
        return self.mechanism_counts.get(mechanism, 0)


def _is_completed(graph: "Any") -> bool:
    """True iff the trace's run completed (loop_end outcome == completed)."""
    for s in graph.steps:
        if s.kind == "loop_end":
            return str(s.payload.get("outcome", "completed")) == "completed"
    return False


def corpus_quality(trace_dir: Path | str) -> CorpusQuality:
    """Compute the offline quality snapshot of *trace_dir*.

    Reads every ``.jsonl`` trace, scores run completion, and diagnoses the
    failed ones (reusing :func:`harnessfix.corpus.diagnose_corpus`, which
    applies the same ``_is_failed_trace`` filter as the loop).  A corrupt/
    unreadable trace is skipped, never fatal.  An empty corpus yields
    ``success_rate=0.0`` and zero mechanism/layer counts (an empty corpus
    cannot evidence an improvement, so it should not be used as a baseline).

    Note: the corpus is a *static* record of past runs, so this snapshot is
    the same before and after a source-code repair.  Its role as a gate is
    therefore **target alignment**, not pre/post delta: a repair whose layer
    does not appear in the observed failures is not addressing what the
    corpus actually shows, and is rejected as off-target.  The success_rate
    field is retained for completeness/future drift checks.
    """
    trace_dir = Path(trace_dir)
    traces = collect_traces(trace_dir)
    total = 0
    completed = 0
    mechanism_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()

    for path in traces:
        try:
            graph = compile_trace(path)
        except TraceValidationError:
            continue
        total += 1
        if _is_completed(graph):
            completed += 1
        # Diagnose the failed traces for mechanism/layer evidence.  The
        # per-task diagnosis JSONs are written to a throwaway temp dir so we
        # never pollute the corpus directory.
        if _is_failed_trace(graph):
            with tempfile.TemporaryDirectory() as tmp:
                diagnoses: list[Diagnosis] = diagnose_corpus([path], Path(tmp))
            for d in diagnoses:
                mechanism_counts[d.mechanism] += 1
                layer_counts[d.root_layer] += 1

    success_rate = (completed / total) if total else 0.0
    return CorpusQuality(
        success_rate=success_rate,
        mechanism_counts=dict(mechanism_counts),
        layer_counts=dict(layer_counts),
        total=total,
    )
