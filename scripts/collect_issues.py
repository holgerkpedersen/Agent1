#!/usr/bin/env python3
"""Seed ``.issues.json`` from the repo-wide detectors (idempotent).

Runs the duplication + best-effort-except detectors over every ``.py`` file
(excluding runtime/generated/vendored dirs) and adds any un-tracked finding as
an issue. Re-running never overwrites a human-set status or autonomy_level, so
it is safe to run in pre-commit and in CI.

Usage:
    python scripts/collect_issues.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harnessfix import issue_loop  # noqa: E402


def main() -> int:
    added = issue_loop.collect_issues()
    if added:
        print(f"[collect_issues] seeded {added} new issue(s) into .issues.json")
    else:
        print("[collect_issues] no new issues; ledger already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
