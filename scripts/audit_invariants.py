"""Audit the workspace against the AGENTS.md invariants.

Usage:
    python scripts/audit_invariants.py [--strict]

Checks:
  1. Git cleanliness (invariant 4 — commit after every session).  WARN by
     default (mid-session work is normal); ERROR with --strict (expected
     to be run post-commit, e.g. in the verification gate).
  2. Decision files: .decisions.json must exist; agent_memory.json and
     chat_history.json must exist or be absent TOGETHER (invariant 7).
  3. Phantom modules: files named in the latest .docs/<ts>/ task/analysis
     documents that do not exist on disk (invariant 5 — planned modules
     must be checked against workspace reality).
  4. Trace health: report/harnessfix summary + trace corpus counts and
     interrupted-run ratio (decision #052).
  5. backups/ exists (implement's pre-run safety copies).
  6. Emoji policy (decision #079): no emojis/pictographs in repo text files
     (`agent_core/text_policy.py`); monochrome CLI glyphs are allowed.

Exit code 0 = all good; 1 = errors found (warnings never fail the audit).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_core.text_policy import scan_tree, summarize_findings  # noqa: E402

_DECISION_FILE = ROOT / ".decisions.json"
_MEMORY_FILES = ("agent_memory.json", "chat_history.json")


def _git_status_porcelain() -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        return proc.stdout if proc.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _latest_docs_dir() -> Path | None:
    dirs = sorted(
        (d for d in (ROOT / ".docs").glob("*") if d.is_dir()),
        key=lambda d: d.name,
    )
    return dirs[-1] if dirs else None


def _phantom_modules() -> list[str]:
    docs = _latest_docs_dir()
    if docs is None:
        return []
    names: set[str] = set()
    for doc in docs.glob("project_*.md"):
        try:
            text = doc.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for span in re.findall(r"`([^`]+)`", text):
            if not span.endswith(".py"):
                continue
            candidate = Path(span)
            if not candidate.is_absolute() and "\\" not in span and "/" not in span:
                candidate = ROOT / span
            names.add(span)
    missing = sorted(
        n for n in names
        if not (ROOT / n.replace("/", "\\")).exists()
        and not (ROOT / n).exists()
    )
    return missing


def _trace_health() -> tuple[int, int, str | None]:
    traces_dir = ROOT / "reports" / "traces"
    if not traces_dir.is_dir():
        return 0, 0, None
    from harnessfix.corpus import collect_traces
    from harnessfix.htir import compile_trace
    from harnessfix.reader import TraceValidationError

    traces = collect_traces(traces_dir)
    interrupted = 0
    for path in traces:
        try:
            graph = compile_trace(path)
        except TraceValidationError:
            continue
        if not graph.has_loop_end() and len(graph.steps) >= 3:
            interrupted += 1
    summary_path = ROOT / "reports" / "harnessfix" / "summary.json"
    verdict = None
    if summary_path.is_file():
        try:
            verdict = json.loads(
                summary_path.read_text(encoding="utf-8")
            ).get("verdict")
        except (OSError, json.JSONDecodeError):
            verdict = None
    return len(traces), interrupted, verdict


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    errors: list[str] = []
    warnings: list[str] = []

    dirty = _git_status_porcelain()
    if dirty:
        msg = (
            f"git tree has {len([l for l in dirty.splitlines() if l.strip()])} "
            "uncommitted change(s) — invariant 4 (commit after every session)"
        )
        (errors if strict else warnings).append(msg)

    if not _DECISION_FILE.is_file():
        errors.append(".decisions.json is missing")
    else:
        try:
            ledger = json.loads(_DECISION_FILE.read_text(encoding="utf-8"))
            if not isinstance(ledger, list) or not ledger:
                warnings.append(".decisions.json is empty or not a list")
        except json.JSONDecodeError:
            errors.append(".decisions.json is not valid JSON")

    mem_present = {f: (ROOT / f).is_file() for f in _MEMORY_FILES}
    if mem_present[_MEMORY_FILES[0]] != mem_present[_MEMORY_FILES[1]]:
        errors.append(
            "memory files must exist or be absent TOGETHER (invariant 7): "
            + ", ".join(
                f"{f}={v}" for f, v in mem_present.items()
            )
        )

    phantoms = _phantom_modules()
    if phantoms:
        docs_dir = _latest_docs_dir()
        docs_name = docs_dir.name if docs_dir is not None else "<no .docs dir>"
        warnings.append(
            f"phantom module(s) named in {docs_name} but not on disk "
            f"(invariant 5): {', '.join(phantoms[:5])}"
        )

    n_traces, n_interrupted, verdict = _trace_health()
    warnings.append(
        f"trace corpus: {n_traces} trace(s), {n_interrupted} interrupted "
        f"(no loop_end); last loop verdict: {verdict}"
    )

    if not (ROOT / "backups").is_dir():
        warnings.append("backups/ does not exist (implement pre-run copies)")

    emoji_findings = scan_tree(ROOT)
    if emoji_findings:
        errors.append(
            f"emoji/pictograph symbols in {len(emoji_findings)} repo file(s) "
            f"(decision #079 — no emojis in files): "
            f"{summarize_findings(emoji_findings)}"
        )

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if not errors and not warnings:
        print("Audit OK.")
    if errors:
        print("\nFAIL: fix the errors above, then re-run.")
        return 1
    print("\nOK (warnings are informational).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())