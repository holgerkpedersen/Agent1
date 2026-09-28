"""Habits command — inspect and maintain the workspace habit ledger.

Usage:
    habits                 List mined/pinned habits with evidence counts
    habits pin "<text>"    Append a user-pinned habit (shown first)
    habits forget <n>      Remove the n-th habit from the list
    habits mine            Re-mine chat history + traces, save the ledger
    habits off | on         Toggle the habits_enabled workspace preference

The ledger is the workspace-local ``.habits.json`` (see
:mod:`agent_core.habits`); the prompt block is rebuilt from it every turn.
``habits off`` sets the ``habits_enabled`` preference to ``False`` —
:func:`agent_core.habits.load_habits` returns ``[]`` then, which suppresses
the prompt block without deleting any mined habit.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any

from agent_core.habits import (
    load_habits,
    mine_habits,
    save_habits,
)

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

logger = logging.getLogger(__name__)


def _trace_records(workspace: pathlib.Path) -> list[dict[str, Any]]:
    """Parse every ``reports/traces/*.jsonl`` record (never raises)."""
    records: list[dict[str, Any]] = []
    trace_dir = workspace / "reports" / "traces"
    try:
        for path in sorted(trace_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(record, dict):
                            records.append(record)
            except OSError:
                logger.debug("Unreadable trace file %s", path, exc_info=True)
    except OSError:
        logger.debug("Unreadable traces directory", exc_info=True)
    return records


class HabitsCommand(Command):
    """List, pin, forget, mine or toggle workspace habits."""

    @property
    def name(self) -> str:
        return "habits"

    @property
    def help_text(self) -> str:
        return ('habits [list|pin "<text>"|forget <n>|mine|off|on] - Workspace '
                "user-habit ledger")

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        workspace = pathlib.Path(agent._effective_ws_dir())
        sub = (args[0].lower() if args else "list")

        if sub == "pin":
            text = " ".join(args[1:]).strip()
            if not text:
                print('  Usage: habits pin "<text>"')
                return True
            habits = load_habits(workspace)
            habits.insert(0, {"text": text, "count": 1, "category": "pinned"})
            save_habits(workspace, habits)
            print(f"  Pinned: {text}")
            return True

        if sub == "forget":
            habits = load_habits(workspace)
            try:
                index = int(args[1])
                target = habits[index]
            except (IndexError, ValueError, TypeError):
                print("  Usage: forget <n> (see 'habits' for the numbers)")
                return True
            habits.pop(index)
            save_habits(workspace, habits)
            print(f"  Forgot: {target.get('text', target)}")
            return True

        if sub == "mine":
            history = list(getattr(agent, "_chat_history", None) or [])
            traces = _trace_records(workspace)
            mined = mine_habits(history, traces)
            # Keep pinned habits; replace the mined ones.
            pinned = [h for h in load_habits(workspace)
                      if isinstance(h, dict) and h.get("category") == "pinned"]
            merged = pinned + [h for h in mined
                               if h.get("category") != "pinned"]
            save_habits(workspace, merged)
            print(f"  Mined {len(mined)} habit(s) from "
                  f"{len(history)} message(s) + {len(traces)} trace record(s); "
                  f"ledger now has {len(merged)}.")
            return True

        if sub in ("off", "on"):
            try:
                from agent_core.llm.workspace_prefs import set_pref

                set_pref(workspace, "habits_enabled", sub == "on")
                state = "enabled" if sub == "on" else "disabled"
                print(f"  Habits {state}. "
                      + ("Prompt block restored next turn."
                         if sub == "on" else "Prompt block suppressed."))
            except Exception:  # noqa: BLE001 - pref write must not kill the REPL
                logger.exception("Could not toggle habits_enabled:\n")
                print("  Could not update the habits_enabled preference.")
            return True

        # default: list
        habits = load_habits(workspace)
        if not habits:
            print("  No habits yet. 'habits mine' learns from history/traces.")
            return True
        print(f"\n  Habits ({len(habits)}; number shown for 'forget'):\n")
        for index, habit in enumerate(habits):
            if isinstance(habit, dict):
                text = str(habit.get("text") or "")
                count = habit.get("count", 1)
                category = habit.get("category", "pinned")
                print(f"  [{index}] {text}  (seen {count}×, {category})")
            else:
                print(f"  [{index}] {habit}")
        print()
        return True
