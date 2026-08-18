"""Review command — the human verification gate over failed task traces.

Usage:
    review refresh [--trace-dir dir] [--diags-dir dir]
    review list
    review show <task>
    review label <task> <bug|regression|noise|ok> [--note "..."]
    review export <task>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

from harnessfix.review import (
    EXPORT_DIR,
    REVIEWS_RELPATH,
    build_reviews,
    export_regression_test,
    label_review,
    load_reviews,
    review_table,
    save_reviews,
)

_REVIEW_HELP = """review — Human gate over failed task traces (verification gate)

  review refresh [--trace-dir dir] [--diags-dir dir]
      Rebuild the review ledger from the trace corpus

  review list
      Table of every reviewed task and its label

  review show <task>
      Full record: prompt, model, effects, diagnosis, outcome

  review label <task> <bug|regression|noise|ok> [--note "..."]
      Classify a task for the improvement loop

  review export <task>
      Write a diagnosis-pinning regression test for a labeled task"""


class ReviewCommand(Command):
    @property
    def name(self) -> str:
        return "review"

    @property
    def help_text(self) -> str:
        return _REVIEW_HELP

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            self.error("Usage: review <subcommand> [...]")
            print(_REVIEW_HELP)
            return True

        sub = args[0].lower()
        if sub == "refresh":
            return await self._cmd_refresh(args[1:], agent)
        elif sub == "list":
            return await self._cmd_list(args[1:], agent)
        elif sub == "show":
            return await self._cmd_show(args[1:], agent)
        elif sub == "label":
            return await self._cmd_label(args[1:], agent)
        elif sub == "export":
            return await self._cmd_export(args[1:], agent)
        else:
            self.error(f"Unknown review subcommand: {sub}")
            return True

    # ── refresh ─────────────────────────────────────────────────────────

    async def _cmd_refresh(self, args: list[str], agent: "Agent") -> bool:
        trace_dir = _flag_path(args, "--trace-dir", agent, "reports/traces")
        diags_dir = _flag_path(args, "--diags-dir", agent, "reports/harnessfix/diagnoses")

        reviews = build_reviews(trace_dir, diags_dir)
        existing = load_reviews(self._reviews_path(agent))
        for task_id, rec in reviews.items():
            prev = existing.get(task_id)
            if prev is not None and prev.is_labeled():
                rec.disposition = prev.disposition
                rec.note = prev.note
                rec.review_date = prev.review_date
        save_reviews(reviews, self._reviews_path(agent))
        labeled = sum(1 for r in reviews.values() if r.is_labeled())
        print(f"Reviewed {len(reviews)} failed task(s) "
              f"({labeled} labeled) from {trace_dir}")
        return True

    # ── list ────────────────────────────────────────────────────────────

    async def _cmd_list(self, args: list[str], agent: "Agent") -> bool:
        reviews = load_reviews(self._reviews_path(agent))
        print(review_table(reviews))
        return True

    # ── show ────────────────────────────────────────────────────────────

    async def _cmd_show(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            self.error("Usage: review show <task>")
            return True
        reviews = load_reviews(self._reviews_path(agent))
        rec = reviews.get(args[0])
        if rec is None:
            print(f"No review record for task {args[0]} (run `review refresh`).")
            return True
        for key in ("task_id", "prompt", "model", "profile", "outcome",
                    "guards", "affected_files", "root_layer", "mechanism",
                    "disposition", "note", "review_date"):
            val = getattr(rec, key)
            if isinstance(val, list):
                val = ", ".join(val)
            print(f"  {key}: {val if val else '-'}")
        return True

    # ── label ───────────────────────────────────────────────────────────

    async def _cmd_label(self, args: list[str], agent: "Agent") -> bool:
        if len(args) < 2:
            self.error("Usage: review label <task> <bug|regression|noise|ok> [--note \"...\"]")
            return True
        task_id, disposition = args[0], args[1]
        note = ""
        if "--note" in args:
            idx = args.index("--note")
            if idx + 1 < len(args):
                note = args[idx + 1]
        reviews = load_reviews(self._reviews_path(agent))
        try:
            label_review(reviews, task_id, disposition, note=note)
        except (KeyError, ValueError) as exc:
            self.error(str(exc))
            return True
        save_reviews(reviews, self._reviews_path(agent))
        print(f"Labeled {task_id} as {disposition}")
        return True

    # ── export ──────────────────────────────────────────────────────────

    async def _cmd_export(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            self.error("Usage: review export <task>")
            return True
        task_id = args[0]
        reviews = load_reviews(self._reviews_path(agent))
        rec = reviews.get(task_id)
        if rec is None:
            print(f"No review record for task {task_id}.")
            return True
        if not rec.is_labeled():
            print(f"Label {task_id} before exporting its regression pin.")
            return True
        trace = Path(agent.workspace) / "reports" / "traces" / f"{task_id}.jsonl"
        if not trace.is_file():
            self.error(f"Trace file not found: {trace}")
            return True
        out_dir = Path(agent.workspace) / EXPORT_DIR
        out = export_regression_test(rec, trace, out_dir)
        print(f"Exported {out}")
        return True

    # ── helpers ─────────────────────────────────────────────────────────

    def _reviews_path(self, agent: "Agent") -> Path:
        return Path(agent.workspace) / REVIEWS_RELPATH


def _flag_path(args: list[str], flag: str, agent: "Agent", default: str) -> Path:
    path = default
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            path = args[idx + 1]
    return Path(agent.workspace) / path
