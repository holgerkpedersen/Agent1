"""Resolve a leading-slash command WITHOUT the model.

WHY THIS EXISTS
---------------
In the dashboard chat a slash command used to be plain text: ``/model`` went to
the LLM as a prompt, so the model *invented* an answer to a question it cannot
see.  The list of models is a fact the process already holds — the REPL's own
registry and providers — so it must not be inferred.  The same is true of
``/help``, ``/kg show`` and anything else that only prints.

This module is the offline path:

``parse_slash``
    ``"/model list"`` -> ``("model", ["list"])``; not a slash command -> None.

``run_slash``
    Handles the message through :class:`~agent_core.commands.registry
    .CommandRegistry` — the SAME registry the terminal dispatches on — and
    returns the answer as a dict.  Nothing is sent to the LLM.

SAFETY
------
An allowlist (:data:`OFFLINE_COMMANDS`) decides what may run this way, so the
browser gains no reach it did not already have: these are the commands whose
answer is a printed fact.  A command that prompts on stdin
(``needs_terminal_input``) is refused with an explanation instead of blocking
an HTTP request on a terminal nobody is watching.  An unlisted command is left
to the normal turn pipeline — ``None`` is returned and nothing changes.

``/model`` and ``/models`` are special: instead of capturing their terminal
output, they return the SHARED catalogue from
:mod:`agent_core.commands.model_catalog` as structured data, which is what the
page renders as a picker.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import shlex
import threading
from typing import Any, Dict, List, Optional, Tuple

from .catalog import needs_terminal_input
from .model_catalog import model_catalog, select_model

logger = logging.getLogger(__name__)

#: Commands whose answer is a fact we already hold, so the model is not asked.
#: Ordered only for documentation; membership is what matters.
OFFLINE_COMMANDS: frozenset[str] = frozenset(
    {
        "model",
        "models",
        "help",
        "kg",
        "memory",
        "session",
    }
)

#: ``/model`` subcommands that only report the catalogue (no switch).
_READ_ONLY_MODEL_SUBS: frozenset[str] = frozenset({"", "list", "ls"})

#: Subcommands that CHANGE agent state.  A browser message must not reach
#: these through the offline path: the dashboard is a convenience surface, not
#: a second way to edit the knowledge graph or switch sessions behind the
#: terminal's back.  Everything here still works from the REPL.
_MUTATING_SUBS: Dict[str, frozenset[str]] = {
    "kg": frozenset({"add", "remove", "rm", "delete", "del", "set"}),
    "session": frozenset(
        {
            "new",
            "use",
            "switch",
            "rename",
            "delete",
            "del",
            "rm",
            "remove",
            "create",
        }
    ),
}


def _refusal(name: str, args: List[str]) -> Optional[str]:
    """Reason to refuse ``/{name} {args}``, or ``None`` when it is read-only."""
    verb = (args[0].strip().lower() if args else "").lstrip("-")
    blocked = _MUTATING_SUBS.get(name, frozenset())
    if verb in blocked:
        return (
            f"/{name} {verb} changes agent state, so it is not run from the "
            f"browser. Use the REPL, or the dashboard's own controls."
        )
    return None


def parse_slash(text: str) -> Optional[Tuple[str, List[str]]]:
    """Split ``"/model list"`` into ``("model", ["list"])``.

    Returns ``None`` for anything that is not a one-line slash command, so the
    caller keeps the normal turn path.  A blank name (a bare ``"/"``) is also
    ``None``: that is the autocomplete affordance, not a command.
    """
    stripped = (text or "").strip()
    if not stripped.startswith("/") or "\n" in stripped:
        return None
    body = stripped[1:].strip()
    if not body:
        return None
    name, _, raw_args = body.partition(" ")
    name = name.strip().lower()
    if not name:
        return None
    raw_args = raw_args.strip()
    if not raw_args:
        return name, []
    try:
        args = shlex.split(raw_args)
    except ValueError:  # an unbalanced quote must not raise at the browser
        args = raw_args.split()
    return name, args


def _scale_catalog_text(catalog: Dict[str, Any]) -> str:
    """Compact text form of a catalogue (fallback for clients without the panel)."""
    lines = [
        f"Current model: {catalog.get('current') or 'none'}"
        f"  (provider: {catalog.get('active_provider') or 'unknown'})"
    ]
    for group in catalog.get("groups") or []:
        lines.append(f"[{group.get('title') or group.get('provider')}]")
        for entry in group.get("entries") or []:
            flags = []
            if entry.get("current"):
                flags.append("current")
            if entry.get("loaded"):
                flags.append("loaded")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {entry.get('value')}{suffix}")
        if group.get("hint"):
            lines.append(f"  {group['hint']}")
    if len(lines) == 1:
        lines.append("No models found: no provider is reachable.")
    return "\n".join(lines)


def _run_coro(coro: Any) -> Any:
    """Run *coro* from a synchronous caller (a live loop gets a worker thread).

    Mirrors ``chat_api._run_coro``: an HTTP handler may or may not already own
    an event loop, and a command must behave the same either way.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    outcome: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - re-raised in caller
            outcome["error"] = exc

    thread = threading.Thread(target=_worker, name="slash-command", daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _models_payload(
    agent: Any, name: str, args: List[str], fresh: bool
) -> Optional[Dict[str, Any]]:
    """Answer ``/model``, ``/model list`` and ``/models``; ``None`` otherwise."""
    if name == "models":
        # ``models <name>`` is a switch, exactly like ``model <name>``.
        if args:
            return None
        catalog = model_catalog(agent, fresh=fresh)
        return {
            "ok": True,
            "reply": _scale_catalog_text(catalog),
            "kind": "models",
            "catalog": catalog,
            "command": name,
            "tools": True,
        }
    sub = args[0].strip().lower() if args else ""
    if sub not in _READ_ONLY_MODEL_SUBS:
        # Every other subcommand (provider/profile/jev/load/...) is handled by
        # the registry below, so the REPL stays the only implementation.
        return None
    catalog = model_catalog(agent, fresh=fresh)
    return {
        "ok": True,
        "reply": _scale_catalog_text(catalog),
        "kind": "models",
        "catalog": catalog,
        "command": "model",
        "tools": True,
    }


def run_slash(
    agent: Any, text: str, *, fresh: bool = False
) -> Optional[Dict[str, Any]]:
    """Answer *text* offline, or return ``None`` to let the turn pipeline run.

    Never raises: a broken command becomes ``{"ok": False, "error": ...}``, the
    same contract the chat endpoints already speak.
    """
    parsed = parse_slash(text)
    if parsed is None:
        return None
    name, args = parsed
    if name not in OFFLINE_COMMANDS:
        return None

    special = _models_payload(agent, name, args, fresh)
    if special is not None:
        return special

    refusal = _refusal(name, args)
    if refusal:
        return {
            "ok": False,
            "error": refusal,
            "command": name,
            "tools": True,
        }

    try:
        from agent_core.cli.interactive import _build_registry

        registry = _build_registry()
        command = registry.get(name)
    except Exception as exc:  # a missing optional dep must not 500
        logger.debug("slash: registry unavailable", exc_info=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tools": True}
    if command is None:
        return None
    if needs_terminal_input(command):
        # It prompts on stdin; nothing is reading a terminal behind HTTP.
        return {
            "ok": False,
            "error": f"/{name} asks for input on the terminal, so it cannot run "
            f"from the browser. Use the REPL, or a form that takes arguments.",
            "command": name,
            "tools": True,
        }

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            _run_coro(command.execute(list(args), agent))
    except Exception as exc:
        logger.debug("slash: /%s failed", name, exc_info=True)
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "command": name,
            "tools": True,
        }
    output = buffer.getvalue().strip()
    return {
        "ok": True,
        "reply": output or f"/{name} produced no output.",
        "kind": "text",
        "command": name,
        "tools": True,
    }


def switch_model(
    agent: Any, model: str, provider: Optional[str] = None
) -> Dict[str, Any]:
    """Switch *agent* to *model* (the dashboard's model picker action)."""
    result = select_model(agent, model, provider)
    if result.get("ok"):
        result["reply"] = (
            f"Switched to {result.get('model')} (provider {result.get('provider')})"
            if result.get("changed")
            else f"Already using {result.get('model')}"
        )
        result["kind"] = "text"
        result["tools"] = True
    return result


__all__ = [
    "OFFLINE_COMMANDS",
    "parse_slash",
    "run_slash",
    "switch_model",
]
