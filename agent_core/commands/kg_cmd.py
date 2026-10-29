"""Knowledge-graph command — add and inspect KG entries from the REPL.

The agent's knowledge graph (``agent._knowledge_graph``) is persisted with
cross-session memory (see ``agent_core.memory.persistence``) and counted by
``memory_stats``, so the ``clear`` command reports it as
``knowledge graph: <n> sections``.  Until now nothing ever *wrote* into it,
so the counter was permanently 0.

Usage:
    kg add fact <key> <value...>        Add a fact entry (arbitrary JSON value)
    kg add file <path> [note...]      Record a file the agent worked on
    kg add tag <tag> <ref...>         Tag an entity (file, decision, symbol)
    kg add symbol <name> <definition> Record a code symbol
    kg show                            Print every entry grouped by section
    kg show <section> [key]          Print one section, or one key of it
    kg remove <section> <key>        Delete one entry

Sections are the four keys ``clear``/persistence already understand:
``facts``, ``files``, ``tags``, ``symbols``.  Singular aliases (fact, file,
tag, symbol) are accepted too.  Values are JSON when they parse as JSON
(``{"a":1}``, ``[1,2]``, ``true``, ``5``), otherwise plain text; a file
entry with no note stores an empty string.

The command is additive: it never rewrites sections it was not given, and
it saves memory after every mutation so the entry survives the session.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

logger = logging.getLogger(__name__)

#: The four canonical sections of the knowledge graph.
SECTIONS = ("facts", "files", "tags", "symbols")

#: Singular aliases accepted on the command line (fact -> facts, ...).
_ALIASES = {"fact": "facts", "file": "files", "tag": "tags", "symbol": "symbols"}


def _canonical(section: str) -> str | None:
    """Return the canonical section name for *section*, or None if unknown."""
    if section in SECTIONS:
        return section
    return _ALIASES.get(section)


def _render(value: Any) -> str:
    """Render a raw value on one line (JSON for non-strings)."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _parse_value(text: str) -> Any:
    """Parse *text* as JSON if it parses, else keep it as plain text."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def add_entry(
    agent: "Agent", section: str, key: str, value: Any
) -> dict[str, Any] | None:
    """Add/replace one entry in *section* and stamp it with a timestamp.

    Returns the stored entry, or ``None`` when *section* is unknown.
    Creates missing sections; never touches other sections.
    """
    section = _canonical(section)
    if section is None:
        return None
    graph = getattr(agent, "_knowledge_graph", None)
    if not isinstance(graph, dict):
        graph = {}
        agent._knowledge_graph = graph
    entry = graph.setdefault(section, {})
    if not isinstance(entry, dict):  # corrupt section from a bad memory file
        entry = {}
        graph[section] = entry
    entry[key] = {"value": value, "ts": datetime.now().isoformat(timespec="seconds")}
    return entry[key]


def remove_entry(agent: "Agent", section: str, key: str) -> bool:
    """Delete one entry; True when it existed. Prunes empty sections."""
    graph = getattr(agent, "_knowledge_graph", None)
    if not isinstance(graph, dict):
        return False
    entry = graph.get(section)
    if isinstance(entry, dict) and key in entry:
        del entry[key]
        if not entry:
            del graph[section]
        return True
    return False


class KGCommand(Command):
    """Add and inspect knowledge-graph entries from the REPL."""

    @property
    def name(self) -> str:
        return "kg"

    @property
    def help_text(self) -> str:
        return "kg add|show|remove ... — edit the knowledge graph shown by clear"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        """Dispatch the subcommand.  Always returns True (continue REPL)."""
        if not args:
            self._usage()
            return True
        verb, rest = args[0], args[1:]
        if verb == "add":
            return await self._add(agent, rest)
        if verb == "show":
            return self._show(agent, rest)
        if verb == "remove":
            return self._remove(agent, rest)
        self.error(f"unknown kg subcommand: {verb}")
        self._usage()
        return True

    # ── subcommands ────────────────────────────────────────────────────

    async def _add(self, agent: "Agent", rest: list[str]) -> bool:
        if len(rest) < 2:
            self._usage()
            return True
        section, key = rest[0], rest[1]
        value = " ".join(rest[2:])
        entry = add_entry(agent, section, key, _parse_value(value))
        if entry is None:
            self.error(f"unknown section {section!r} (known: {', '.join(SECTIONS)})")
            return True
        agent._save_memory()
        self.success(f"  [kg] added {section}/{key}")
        return True

    def _show(self, agent: "Agent", rest: list[str]) -> bool:
        graph = getattr(agent, "_knowledge_graph", None)
        graph = graph if isinstance(graph, dict) else {}
        if not rest:
            print("Knowledge graph:")
            for section in SECTIONS:
                entry = graph.get(section)
                items = entry if isinstance(entry, dict) else {}
                print(f"  {section} ({len(items)}):")
                for key, value in items.items():
                    print(f"    {key}: {self._render(value)}")
            return True
        section = _canonical(rest[0])
        if section is None:
            self.error(f"unknown section {rest[0]!r} (known: {', '.join(SECTIONS)})")
            return True
        entry = graph.get(section)
        items = entry if isinstance(entry, dict) else {}
        if len(rest) > 1:
            key = rest[1]
            if key in items:
                print(f"  {section}/{key}: {self._render(items[key])}")
            else:
                self.error(f"no entry {section}/{key}")
            return True
        print(f"  {section} ({len(items)}):")
        for key, value in items.items():
            print(f"    {key}: {self._render(value)}")
        return True

    def _remove(self, agent: "Agent", rest: list[str]) -> bool:
        if len(rest) < 2:
            self._usage()
            return True
        section = _canonical(rest[0])
        if section is None:
            self.error(f"unknown section {rest[0]!r} (known: {', '.join(SECTIONS)})")
            return True
        if remove_entry(agent, section, rest[1]):
            agent._save_memory()
            self.success(f"  [kg] removed {section}/{rest[1]}")
        else:
            self.error(f"no entry {section}/{rest[1]}")
        return True

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _render(value: Any) -> str:
        """One-line rendering of a stored entry (dict with value/ts, raw)."""
        if isinstance(value, dict) and "value" in value:
            ts = value.get("ts", "")
            rendered = _render(value["value"])
            return f"{rendered} [{ts}]" if ts else rendered
        return _render(value)

    @staticmethod
    def _usage() -> None:
        print("  Usage: kg add <fact|file|tag|symbol> <key> [value...]")
        print("         kg show [section [key]]  |  kg remove <section> <key>")
