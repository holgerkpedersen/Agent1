"""Issue command — triage and track systematic, autonomous-handled work items.

Mirrors ``review``: a human-facing ledger over ``.issues.json``. Mutating
subcommands (add/resolve/promote) are blocked in plan mode, consistent with the
read-only session contract.

Usage:
    issue add <category> <location> [--title "t"] [--approach "a"]
        [--level N] [--severity S]
    issue list
    issue show <id>
    issue resolve <id> [resolved|deferred|wontfix] [--note "..."]
    issue promote <id> <0|1|2>
    issue autonomy
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import Command
from agent_core.modes import is_plan_mode
from harnessfix import issues as issue_store

if TYPE_CHECKING:
    from agent import Agent

_ISSUE_HELP = """issue - Track systematic, autonomous-handled work items (.issues.json)

  issue add <category> <location> [--title "t"] [--approach "a"]
      [--level N] [--severity S]
      File a new issue (location is file:line; id is derived, so re-adding
      is idempotent). Default autonomy_level=1 (agent may attempt if gates pass).

  issue list
      Table of every issue: id, category, status, level, locations.

  issue show <id>
      Full record.

  issue resolve <id> [resolved|deferred|wontfix] [--note "..."]
      Mark an issue done/shelved (human decision).

  issue promote <id> <0|1|2>
      Raise/lower the autonomy_level. 0 = human-only, 1 = auto-safe,
      2 = benchmark-required (explicit promotion).

  issue autonomy
      Show the current AGENT_AUTONOMY_LEVEL cap (env, default 1)."""


def _autonomy_level() -> int:
    raw = os.environ.get("AGENT_AUTONOMY_LEVEL", "").strip()
    if raw.isdigit():
        return int(raw)
    return issue_store.DEFAULT_AUTONOMY_LEVEL


class IssueCommand(Command):
    @property
    def name(self) -> str:
        return "issue"

    @property
    def help_text(self) -> str:
        return _ISSUE_HELP

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            print(_ISSUE_HELP)
            return True

        sub = args[0].lower()
        if sub == "add":
            return self._cmd_add(args[1:], agent)
        if sub == "list":
            return self._cmd_list([])
        if sub == "show":
            return self._cmd_show(args[1:])
        if sub == "resolve":
            return self._cmd_resolve(args[1:], agent)
        if sub == "promote":
            return self._cmd_promote(args[1:], agent)
        if sub == "autonomy":
            print(f"AGENT_AUTONOMY_LEVEL = {_autonomy_level()}")
            return True
        self.error(f"Unknown issue subcommand: {sub}")
        return True

    # ── add ──────────────────────────────────────────────────────────────

    def _cmd_add(self, args: list[str], agent: "Agent") -> bool:
        if is_plan_mode(agent.mode):
            self.error("issue add is blocked in plan mode (read-only session).")
            return True
        if len(args) < 2:
            self.error("Usage: issue add <category> <location> [--title \"...\"] "
                       "[--approach \"...\"] [--level N] [--severity S]")
            return True
        category, location = args[0], args[1]
        title = category
        approach = ""
        level = issue_store.DEFAULT_AUTONOMY_LEVEL
        severity = "low"
        if "--title" in args:
            i = args.index("--title")
            if i + 1 < len(args):
                title = args[i + 1]
        if "--approach" in args:
            i = args.index("--approach")
            if i + 1 < len(args):
                approach = args[i + 1]
        if "--severity" in args:
            i = args.index("--severity")
            if i + 1 < len(args):
                severity = args[i + 1]
        if "--level" in args:
            i = args.index("--level")
            if i + 1 < len(args) and args[i + 1].isdigit():
                level = int(args[i + 1])
        issues = issue_store.load_issues()
        issue = issue_store.make_issue(
            category, title, [location], severity=severity,
            suggested_approach=approach, autonomy_level=level,
        )
        if issue_store.upsert(issues, issue):
            issue_store.save_issues(issues)
            print(f"Added {issue['id']} (autonomy_level={level})")
        else:
            print(f"Already tracked: {issue['id']}")
        return True

    # ── list ─────────────────────────────────────────────────────────────

    def _cmd_list(self, args: list[str]) -> bool:
        issues = issue_store.load_issues()
        if not issues:
            print("No issues in .issues.json")
            return True
        print(f"{'id':42} {'category':18} {'status':10} {'lvl':>3}  locations")
        for it in issues:
            loc = ";".join(it.get("locations", []))
            print(f"{it['id']:42} {it['category']:18} {it['status']:10} "
                  f"{it['autonomy_level']:>3}  {loc}")
        open_cnt = len(issue_store.open_issues(issues, max_level=_autonomy_level()))
        print(f"\n{len(issues)} total; {open_cnt} open at or below "
              f"AGENT_AUTONOMY_LEVEL={_autonomy_level()}")
        return True

    # ── show ────────────────────────────────────────────────────────────

    def _cmd_show(self, args: list[str]) -> bool:
        if not args:
            self.error("Usage: issue show <id>")
            return True
        issues = issue_store.load_issues()
        it = issue_store.find_by_id(issues, args[0])
        if it is None:
            print(f"No issue {args[0]}")
            return True
        for key in ("id", "title", "category", "severity", "locations",
                    "status", "autonomy_level", "evidence", "suggested_approach",
                    "decision_ref", "repair_ref", "created_at", "resolved_at"):
            print(f"  {key}: {it.get(key)}")
        return True

    # ── resolve ──────────────────────────────────────────────────────────

    def _cmd_resolve(self, args: list[str], agent: "Agent") -> bool:
        if is_plan_mode(agent.mode):
            self.error("issue resolve is blocked in plan mode (read-only session).")
            return True
        if not args:
            self.error(
                "Usage: issue resolve <id> [resolved|deferred|wontfix] [--note \"...\"]"
            )
            return True
        issue_id = args[0]
        disposition = "resolved"
        if len(args) > 1 and not args[1].startswith("--"):
            disposition = args[1]
        note = ""
        if "--note" in args:
            i = args.index("--note")
            if i + 1 < len(args):
                note = args[i + 1]
        issues = issue_store.load_issues()
        if issue_store.resolve(issues, issue_id, disposition, note=note):
            issue_store.save_issues(issues)
            print(f"Resolved {issue_id} as {disposition}")
        else:
            self.error(f"No issue {issue_id}")
        return True

    # ── promote ──────────────────────────────────────────────────────────

    def _cmd_promote(self, args: list[str], agent: "Agent") -> bool:
        if is_plan_mode(agent.mode):
            self.error("issue promote is blocked in plan mode (read-only session).")
            return True
        if len(args) < 2 or not args[1].isdigit():
            self.error("Usage: issue promote <id> <0|1|2>")
            return True
        issues = issue_store.load_issues()
        ok, msg = issue_store.promote(issues, args[0], int(args[1]))
        if ok:
            issue_store.save_issues(issues)
        print(msg)
        return True
