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

``/clear`` is special for the opposite reason.  ``ClearCommand.execute`` asks
for confirmation with the BUILTIN ``input()``, and :func:`needs_terminal_input`
only matches ``commands.base.read_*`` — so the stdin guard cannot see it and
the request would park on a terminal nobody is watching.  The browser is also a
confirmation surface of its own (see the prompt in ``static/chat.html``), so
``clear`` is driven in its ``--force`` form here: same REPL command, same
``agent.clear_history()``, no prompt.
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
#: Ordered only for documentation; membership is what matters.  Every entry is
#: non-interactive (``needs_terminal_input`` is False) and read-only in its
#: default form — anything that executes shell commands, mutates external
#: systems or spawns agents stays out of this set and falls through to the
#: normal turn pipeline unchanged.  Subcommand-level mutations are refused by
#: :data:`_MUTATING_SUBS` below.
OFFLINE_COMMANDS: frozenset[str] = frozenset(
    {
        "model",
        "models",
        "help",
        "kg",
        "memory",
        "session",
        "clear",
        # Read-only reports / lookups.  Each is verified below against its
        # command source: bare or flagged forms only print; the forms that
        # write files, clear stats or switch modes are refused via
        # ``_MUTATING_SUBS`` (or simply absent from the set).
        "display",   # show current verbosity (setting persists to .env -> blocked)
        "perf",      # summary / --detail report
        # (--reset clears stats, --html writes a file -> blocked)
        "read",      # prints one file's content
        "search",    # searches files for a string
        "habits",    # list only (pin/forget/mine/off/on mutate ledger -> blocked)
        "subagent",  # roles|list|summary|route read-only
        # (create|run|reset spawn children -> blocked)
        "mode",      # bare shows current mode (plan|build switch state -> blocked)
        # cleanup: bare prints unreferenced-file report;
        # --delete removes files and prompts via auto_choice -> blocked.
        "cleanup",
        # issue: list/show/autonomy are read-only reports (bare prints help);
        # add/resolve/note/update/promote write .issues.json -> blocked.
        "issue",
        # git: read-only subcommands only (see _GIT_READ_ONLY); a mutating
        # subcommand is refused rather than sent to the model.
        "git",
    }
)

#: Git subcommands safe to run from the browser.  Mutating verbs (commit,
#: push, checkout, reset, merge, rebase, branch -d, ...) are deliberately
#: absent, so ``/git status`` runs the CLI while ``/git push`` is refused.
_GIT_READ_ONLY: frozenset[str] = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "branch",
        "remote",
        "rev-parse",
        "describe",
        "ls-files",
        "shortlog",
        "blame",
    }
)

#: Commands whose terminal form BLOCKS on ``input()`` that
#: :func:`needs_terminal_input` cannot detect, mapped to the arguments that
#: make them non-interactive.  ``clear`` asks "Clear all memory? (y/n)"; the
#: page has already asked, so the browser path runs its ``--force`` form and
#: never reaches the prompt.  A bare ``/clear`` is exactly ``--force``;
#: ``/clear stats`` stays a read-only report.
_NON_INTERACTIVE_ARGS: Dict[str, List[str]] = {"clear": ["--force"]}

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
    # display: bare shows the current verbosity; setting it persists to .env.
    "display": frozenset({"verbose", "normal", "clean", "quiet", "intent"}),
    # perf: --reset wipes stats and --html writes performance_dashboard.html;
    # bare /perf and /perf --detail are read-only reports.
    "perf": frozenset({"reset", "html"}),
    # habits: list only; pin/forget edit the ledger, mine rewrites it from
    # history and off/on toggles a preference — all state changes.
    "habits": frozenset({"pin", "forget", "mine", "off", "on"}),
    # subagent: roles|list|summary only; create/run spawn children, reset wipes them.
    "subagent": frozenset({"create", "run", "reset"}),
    # mode: bare shows the current mode; plan/build switch the session state.
    "mode": frozenset({"plan", "build"}),
    # cleanup: --delete removes files and prompts via auto_choice — never
    # from the browser.
    "cleanup": frozenset({"delete"}),
    # issue: add/resolve/note/update/promote write .issues.json;
    # list/show/autonomy are reads.
    "issue": frozenset({"add", "resolve", "note", "update", "promote"}),
}


def _refusal(name: str, args: List[str]) -> Optional[str]:
    """Reason to refuse ``/{name} {args}``, or ``None`` when it is read-only.

    The FIRST argument is the subcommand — how execute dispatches; any flag
    anywhere in the arguments is also checked, because a mutating flag does not
    have to sit in position one for execute to act on it (e.g. ``/perf --detail
    --reset`` still reaches the reset branch). Free text after a subcommand is
    never inspected: routing an issue whose text mentions "reset" must stay
    readable.
    """
    blocked = _MUTATING_SUBS.get(name, frozenset())
    if not args or not blocked:
        return None

    def _verb_of(token: str) -> str:
        return token.strip().lower().lstrip("-")

    # The subcommand verb — how execute dispatches (first argument only).
    first = _verb_of(args[0])
    if first in blocked:
        return _refusal_message(name, first)
    # Flags anywhere in the arguments are always control flags, never free text.
    for token in args[1:]:
        if token.lstrip().startswith("--") and _verb_of(token) in blocked:
            return _refusal_message(name, _verb_of(token))
    return None


def _refusal_message(name: str, verb: str) -> str:
    """The browser-surface refusal message for a mutating subcommand/flag."""
    return (
        f"/{name} {verb} changes agent state, so it is not run from the "
        f"browser. Use the REPL, or the dashboard's own controls."
    )


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
    if name != "model":
        # Only ``/model`` has catalogue subcommands.  Without this guard any
        # allowlisted command called with no args (or with "list") fell
        # through to the catalogue branch below, so ``/help`` and ``/memory``
        # answered with the model list instead of running at all.
        return None
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

    if name == "git":
        sub = args[0].strip().lower() if args else ""
        if sub not in _GIT_READ_ONLY:
            return {
                "ok": False,
                "error": (
                    f"/git {sub or '<subcommand>'} is not a read-only git "
                    f"command, so it is not run from the browser. "
                    f"Allowed: {', '.join(sorted(_GIT_READ_ONLY))}."
                ),
                "command": name,
                "tools": True,
            }

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
    # ``clear`` confirms with a builtin input() that the stdin probe above
    # cannot see, so it runs in the non-interactive form that skips the
    # prompt.  Reporting subcommands ("clear stats") are left untouched.
    call_args = list(args)
    non_interactive = _NON_INTERACTIVE_ARGS.get(name)
    if non_interactive and not any(a.lstrip("-") in ("stats",) for a in call_args):
        call_args.extend(non_interactive)
    try:
        with contextlib.redirect_stdout(buffer):
            _run_coro(command.execute(call_args, agent))
    except Exception as exc:
        logger.debug("slash: /%s failed", name, exc_info=True)
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "command": name,
            "tools": True,
        }
    output = buffer.getvalue().strip()
    # A clear that actually destroyed something is reported as its own kind:
    # the transcript behind the page is now empty, which the page must reflect.
    # "Nothing to clear" and "clear stats" stay plain text - nothing changed.
    kind = "text"
    if name == "clear" and "Agent memory cleared." in output:
        kind = "clear"
    return {
        "ok": True,
        "reply": output or f"/{name} produced no output.",
        "kind": kind,
        "command": name,
        "tools": True,
        **({"cleared": True} if kind == "clear" else {}),
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
