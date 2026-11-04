"""Kanban REPL command — one card, both repos, and the keys between them.

Thin CLI over :mod:`harnessfix.kanban_inspect`, which owns the reading. The
command exists so "what does card X say / where is it" is one keystroke in the
REPL instead of a manual read of ``.issues.json``, ``.card_id_map.json`` and
``board-<id>.json``.

Subcommands:
    kanban status
    kanban card <card-id | issue-id | card title> [--full]
    kanban queue [--limit N]
    kanban board [board-id]
    kanban keys
    kanban sync-descriptions [--dry-run]   (the only mutating subcommand)

Mutating subcommands are blocked in plan mode, matching ``issue add/resolve``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .base import Command
from agent_core.modes import is_plan_mode
from harnessfix import kanban_inspect as ki

if TYPE_CHECKING:
    from agent import Agent


_KANBAN_HELP = """kanban - Inspect the Kanban ↔ Agent1 bridge (one card, both sides)

  kanban status
      Sync switches, active board, id-map health, queue depths + dead letters.

  kanban card <key> [--full]
      Resolve ANY key — a Kanban card id, an Agent1 issue id, a card title,
      or a unique title prefix — and show both sides: the ledger record, the
      board card (column, status, description), how the link was found, any
      queued messages about it, and the composed full description.

  kanban queue [--limit N]
      Unprocessed messages in BOTH directions, each resolved to its issue id.

  kanban board [board-id]
      Frames with titles, the issue status each column maps to, and the cards
      (default: the board the Kanban app has open).

  kanban keys
      Every key that travels between the repos and what it means.

  kanban sync-descriptions [--dry-run]
      Copy Kanban card descriptions into issues whose fields are empty
      (kanban→agent1 used to drop the body). Dry-run by default is NOT the
      case: pass --dry-run to preview."""


def _show(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


class KanbanCommand(Command):
    @property
    def name(self) -> str:
        return "kanban"

    @property
    def help_text(self) -> str:
        return _KANBAN_HELP

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            print(_KANBAN_HELP)
            return True

        sub = args[0].lower()
        rest = args[1:]

        if sub == "status":
            _show(ki.status_summary())
        elif sub == "keys":
            _show(ki.key_anatomy())
        elif sub == "card":
            self._cmd_card(rest)
        elif sub == "queue":
            limit = 10
            if "--limit" in rest:
                i = rest.index("--limit")
                if i + 1 < len(rest) and rest[i + 1].isdigit():
                    limit = int(rest[i + 1])
            _show(ki.queue_status(limit=limit))
        elif sub == "board":
            board_id = rest[0] if rest and not rest[0].startswith("--") else None
            _show(ki.board_view(board_id))
        elif sub == "sync-descriptions":
            if is_plan_mode(agent.mode):
                self.error(
                    "kanban sync-descriptions is blocked in plan mode "
                    "(read-only session)."
                )
                return True
            dry = "--dry-run" in rest
            results = ki.sync_descriptions(dry_run=dry)
            _show(
                {
                    "dry_run": dry,
                    "updated": sum(
                        1 for r in results if r["action"] != "skipped"
                    ),
                    "results": results,
                }
            )
        elif sub in ("help", "-h", "--help"):
            print(_KANBAN_HELP)
        else:
            self.error(f"Unknown kanban subcommand: {sub}")
        return True

    # ── card ─────────────────────────────────────────────────────────────

    def _cmd_card(self, args: list[str]) -> None:
        if not args:
            self.error("Usage: kanban card <card-id | issue-id | card title>")
            return
        key = " ".join(
            a for a in args if not a.startswith("--")
        ).strip()
        full = "--full" in args
        view = ki.issue_view(key)
        link = view.get("link", {})
        if not link.get("card_id") and not link.get("issue_id"):
            print(f"No bridge key matches {key!r}")
            for warning in link.get("warnings", []):
                print(f"  - {warning}")
            return

        issue = view.get("issue")
        card = view.get("card")
        composed = ki.composed_description(issue or {}, card)

        print(f"key          : {link.get('input')} ({link.get('kind')})")
        for hop in link.get("hops", []):
            print(f"  link       : {hop}")
        for warning in link.get("warnings", []):
            print(f"  WARNING    : {warning}")

        if issue:
            print(f"issue        : {issue.get('id')}")
            print(f"  title      : {issue.get('title')}")
            print(f"  status     : {issue.get('status')}")
            print(
                f"  severity   : {issue.get('severity')}  "
                f"level={issue.get('autonomy_level')}"
            )
            print(f"  locations  : {'; '.join(issue.get('locations') or [])}")
        if card:
            print(f"card         : {card.get('id')}")
            print(f"  title      : {card.get('title')}")
            print(
                f"  column     : {card.get('frame_title')} "
                f"→ issue status {ki.COLUMN_TO_ISSUE.get(card.get('frame_title', ''))}"
            )
            print(f"  card status: {card.get('status')}")
        if composed:
            print("description  :")
            for line in composed.splitlines():
                print(f"  | {line}")

        queue = view.get("queue") or []
        if queue:
            print(f"queued about this key: {len(queue)}")
            for item in queue:
                msg = item["message"]
                print(
                    f"  [{item['direction']}] seq={msg.get('seq')} "
                    f"op={msg.get('op')} retry={msg.get('retry_count', 0)}"
                )

        if full:
            _show(view)
