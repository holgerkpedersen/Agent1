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
      queued messages about it, the composed full description, and the
      work log (every Agent1 work record linked to that card).

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
        elif sub == "claim":
            self._cmd_claim(rest, agent)
        elif sub == "takeover":
            self._cmd_takeover(rest, agent)
        elif sub == "release":
            self._cmd_release(rest, agent)
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

        work_log = view.get("work_log") or []
        if work_log:
            print(f"work log     : {len(work_log)} record(s) for this card")
            for rec in work_log:
                print(
                    f"  [{rec.get('created_at')}] {rec.get('issue_id')} "
                    f"{rec.get('status')} — {rec.get('title')}"
                )
                if rec.get("resolved_at"):
                    print(f"      resolved : {rec['resolved_at']}")
                evidence = str(rec.get("evidence") or "").strip()
                for line in evidence.splitlines():
                    if line.strip():
                        print(f"      evidence | {line}")
                approach = str(rec.get("suggested_approach") or "").strip()
                for line in approach.splitlines():
                    if line.strip():
                        print(f"      approach | {line}")
        else:
            print("work log     : no Agent1 work records linked to this card")

        if full:
            _show(view)

    # ── claim / takeover / release ───────────────────────────────────────

    def _card_id_for_key(self, key: str) -> str | None:
        """Resolve any bridge key (issue id, card id, title) to a card id."""
        view = ki.issue_view(key)
        return (view.get("link") or {}).get("card_id")

    def _cmd_claim(self, args: list[str], agent: "Agent") -> None:
        if is_plan_mode(agent.mode):
            self.error("kanban claim is blocked in plan mode (read-only session).")
            return
        from harnessfix import issues as issue_store

        if not args:
            self.error(
                "Usage: kanban claim <issue-id | card-id | card title>\n"
                "       kanban claim --list"
            )
            return
        if args[0] == "--list":
            from agent_core.session import identity

            claims = identity._load_claims(identity.claims_path())
            if not claims:
                print("No card claims in", identity.claims_path())
                return
            for cid, entry in claims.items():
                live = identity.claim_owner(cid) is not None
                print(
                    f"  {cid}  owner={entry.get('owner')}  "
                    f"claimed_at={entry.get('claimed_at')}  "
                    f"{'LIVE' if live else 'STALE/remote'}"
                )
            return
        key = " ".join(args)
        card_id = self._card_id_for_key(key)
        if not card_id:
            self.error(f"No bridge key matches {key!r} — nothing to claim")
            return
        acquired, record = issue_store._claim_for_card(
            card_id, {"id": key}
        )
        if acquired:
            _show({"acquired": True, "card_id": card_id, "claim": record})
        else:
            print(
                f"Card {card_id} is claimed by session "
                f"'{record.get('owner')}' (claimed_at "
                f"{record.get('claimed_at')}).\n"
                f"Use 'kanban takeover {card_id}' to take it over explicitly."
            )

    def _cmd_takeover(self, args: list[str], agent: "Agent") -> None:
        if is_plan_mode(agent.mode):
            self.error(
                "kanban takeover is blocked in plan mode (read-only session)."
            )
            return
        from agent_core.session import identity
        from harnessfix import issues as issue_store

        if not args:
            self.error("Usage: kanban takeover <card-id | issue-id | card title>")
            return
        key = " ".join(args)
        card_id = self._card_id_for_key(key)
        if not card_id:
            self.error(f"No bridge key matches {key!r} — nothing to take over")
            return
        previous = identity.claim_owner(card_id)
        acquired, record = identity.claim_card(card_id, force=True)
        if not acquired:
            self.error(f"Takeover failed for card {card_id}")
            return
        kb = issue_store._get_kanban_bridge()
        if kb is not None:
            try:
                kb.enqueue(
                    {
                        "op": "card_claimed",
                        "source_id": card_id,
                        "payload": {
                            "card_id": card_id,
                            "owner": record.get("owner"),
                            "label": record.get("label"),
                            "claimed_at": record.get("claimed_at"),
                            "previous_owner": (previous or {}).get("owner"),
                            "system": "harnessfix",
                        },
                    }
                )
            except Exception:  # noqa: BLE001 — notification is best-effort
                pass
        print(
            f"Took over card {card_id}"
            + (f" (was claimed by '{previous.get('owner')}')" if previous else "")
        )

    def _cmd_release(self, args: list[str], agent: "Agent") -> None:
        if is_plan_mode(agent.mode):
            self.error("kanban release is blocked in plan mode (read-only session).")
            return
        from agent_core.session import identity

        if not args:
            self.error("Usage: kanban release <card-id | issue-id | card title>")
            return
        key = " ".join(args)
        card_id = self._card_id_for_key(key)
        if not card_id:
            self.error(f"No bridge key matches {key!r} — nothing to release")
            return
        if identity.release_card(card_id):
            print(f"Released claim on card {card_id}")
        else:
            print(
                f"No claim held by this session on card {card_id} "
                "(nothing released)"
            )
