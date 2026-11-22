"""Machine-readable catalogue of the registered REPL commands.

The terminal REPL discovers its commands from
:class:`~agent_core.commands.registry.CommandRegistry` — ``_register_commands``
in ``agent_core/cli/interactive.py`` is the single registration point, and the
banner, ``help`` and dispatch all read from it.  The dashboard chat needs the
*same* list (an opencode-style ``/`` slash menu in ``static/chat.html``); the
only thing it may not do is keep a second, hand-maintained copy, which would
drift the moment a command is added.

This module is the adapter: it turns a registry into plain dicts for JSON.

    >>> from agent_core.cli.interactive import _build_registry
    >>> names = [c["name"] for c in command_catalog(_build_registry())]
    >>> "read" in names and "kanban" in names
    True

Two derived facts are attached to every entry, because the browser must know
them to render honestly:

``needs_input``
    The command prompts on the terminal (``read_choice`` / ``read_input`` /
    ``read_stdin``).  Nothing is reading stdin behind an HTTP request, so the
    dashboard reports it as terminal-only instead of hanging the request until
    the socket times out (see ``chat_api`` for the guard).

``summary`` / ``usage``
    ``help_text`` is one line shaped ``"<usage> - <summary>"``; the menu wants
    the summary, the reference card wants both.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Pattern

from .registry import CommandRegistry

#: Interactive prompts a command may use (see ``commands/base.py``).  Matched
#: against the command CLASS source: the helper is called by the class, so the
#: name appearing in its own module is what identifies a blocking command.
_INPUT_CALL: Pattern[str] = re.compile(r"\bread_(?:choice|input|stdin)\s*\(")

#: Summary separator inside ``help_text``.  Commands use either an ASCII
#: hyphen or an em dash (``kg add|show|remove ... — edit the knowledge graph``).
_SUMMARY_SEP: Pattern[str] = re.compile(r"\s+[-\u2014\u2013]\s+")

#: Per-CLASS cache of the stdin probe.  ``inspect.getsource`` is not free and
#: the answer can never change for a loaded class.
_NEEDS_INPUT_CACHE: Dict[type, bool] = {}


def needs_terminal_input(command: Any) -> bool:
    """True when *command* prompts on stdin and so cannot run in a browser.

    Best effort: if the source cannot be read (frozen/compiled install) the
    command is assumed safe.  A wrong ``False`` only means the request waits
    for a timeout; a wrong ``True`` would wrongly hide a usable command.
    """
    declared = getattr(command, "declared_needs_terminal_input", None)
    if declared is not None:
        # An explicit answer wins: the source probe cannot tell a mandatory
        # prompt from an optional picker that is skipped without a TTY.
        return bool(declared)
    cls = type(command)
    cached = _NEEDS_INPUT_CACHE.get(cls)
    if cached is not None:
        return cached
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):  # pragma: no cover - frozen installs
        return False
    answer = bool(_INPUT_CALL.search(source))
    _NEEDS_INPUT_CACHE[cls] = answer
    return answer


def split_synopsis(help_text: str) -> tuple[str, str]:
    """Split ``help_text`` into ``(usage, summary)`` on the first separator.

    ``"read <path>        - Read a file"`` becomes ``("read <path>", "Read a
    file")``.  Text without a separator is its own usage and its own summary,
    so a terse command still renders something meaningful.
    """
    first_line = (help_text or "").strip()
    lines = first_line.splitlines()
    first_line = lines[0].strip() if lines else ""
    if not first_line:
        return "", ""
    match = _SUMMARY_SEP.search(first_line)
    if match is None:
        return first_line, first_line
    usage = first_line[: match.start()].strip()
    summary = first_line[match.end() :].strip()
    return usage, summary or usage


def command_catalog(registry: CommandRegistry) -> List[Dict[str, Any]]:
    """Return every registered command as a JSON-ready dict, sorted by name."""
    entries: List[Dict[str, Any]] = []
    for name in sorted(registry.names()):
        command = registry.get(name)
        if command is None:  # pragma: no cover - names() comes from the registry
            continue
        help_text = str(getattr(command, "help_text", "") or "").strip()
        usage, summary = split_synopsis(help_text)
        entries.append(
            {
                "name": name,
                "usage": usage,
                "summary": summary,
                "detail": help_text,
                "needs_input": needs_terminal_input(command),
            }
        )
    return entries


def catalog_for_agent(agent: Any) -> Dict[str, Any]:
    """Build the catalogue from a live agent's registry, never raising.

    The dashboard resolves its registry through the REPL builder, but a broken
    optional dependency (an import inside ``_register_commands``) must degrade
    to an empty menu, not a 500 on the chat page.
    """
    from agent_core.cli.interactive import _build_registry

    catalog = command_catalog(_build_registry())
    return {
        "commands": catalog,
        "count": len(catalog),
        "model": str(getattr(agent, "model_name", "") or "unloaded"),
    }


__all__ = [
    "catalog_for_agent",
    "command_catalog",
    "needs_terminal_input",
    "split_synopsis",
]
