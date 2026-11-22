"""Model catalogue shared by the REPL and the dashboard chat.

``model list`` (and the interactive ``model`` picker) prints for a terminal.
The dashboard chat needs the SAME facts in a browser: which models exist per
provider, which one this session is pinned to, and which provider is active.
Printing to stdout cannot cross an HTTP boundary, so this module is the shared
answer:

``model_catalog``
    Collects the catalogue from the REPL's OWN fetchers —
    ``ModelCommand._fetch_models`` / ``_opencode_catalog`` / ``_llama_catalog``
    / ``_openrouter_catalog`` / ``_lemonade_catalog`` — and returns plain
    dicts.  There is no second model list here to drift out of step with the
    terminal, exactly like ``commands/catalog.py`` is the single command list.

``select_model``
    Applies a choice the same way ``model <name>`` does: rebuild the provider
    through ``ModelCommand._rebuild_provider`` (which wraps
    ``build_provider``) and persist it with ``persist_model_choice``.  One
    implementation of "switch model", now reachable from the browser.

WHY THIS EXISTS
---------------
A slash command typed in the dashboard was just text: ``/model`` went to the
LLM as a prompt, so the model *guessed* a model list.  Worse, the ``/`` menu
filled in the command's synopsis, so ``/model`` expanded to
``/model [pick|list|load|...] ...`` and the optional-argument brackets ended up
in the message.  A command whose answer is known must not be answered by
inference: the page fetches this catalogue instead and renders it.

NETWORK COST
------------
Collecting a catalogue talks to every reachable provider, which is what the
terminal does too.  A dashboard may ask repeatedly (the panel can be reopened),
so results are cached per agent for :data:`CACHE_TTL_SECONDS`.  A failed fetch
is an empty group, never an exception: a provider being down must not break the
panel.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from .base import Command
from .model_cmd import ModelCommand, _format_size, _live_provider_name

logger = logging.getLogger(__name__)

#: How long a collected catalogue is reused for one agent.  Short on purpose:
#: ``model load`` in LM Studio must show up quickly, and the panel is opened by
#: hand, not polled.
CACHE_TTL_SECONDS = 20.0

#: Provider groups, in the order the terminal prints them.
GROUP_ORDER: tuple[str, ...] = (
    "lmstudio",
    "opencode",
    "opencode-zen",
    "llama",
    "openrouter",
    "lemonade",
)

#: Human labels for the group headers.
GROUP_TITLES: Dict[str, str] = {
    "lmstudio": "LM Studio",
    "opencode": "opencode",
    "opencode-zen": "opencode-zen (free)",
    "llama": "llama.cpp",
    "openrouter": "OpenRouter",
    "lemonade": "Lemonade (NPU)",
}

#: What to do when the ACTIVE provider serves nothing.  Other empty providers
#: are dropped from the payload: an unconfigured gateway is noise, not choice.
GROUP_HINTS: Dict[str, str] = {
    "lmstudio": "LM Studio not reachable — Developer tab > Start Server",
    "opencode": "set OPENCODE_API_KEY to reach the hosted catalog",
    "opencode-zen": "free tier unreachable — check the network",
    "llama": "llama-server not reachable — start it",
    "openrouter": "set OPENROUTER_API_KEY",
    "lemonade": "start it: 'lemonade run <model>'",
}

#: Serialises collection: two dashboard requests must not double-fetch.
_FETCH_LOCK = threading.Lock()

#: ``(agent_id, current_model, provider) -> (timestamp, payload)``.
_CACHE: Dict[tuple, tuple[float, Dict[str, Any]]] = {}


def _current_model(agent: Any) -> str:
    """The model this session is pinned to, or ``""``."""
    name = getattr(agent, "model_name", None)
    if not name:
        name = getattr(getattr(agent, "llm", None), "model_name", None)
    return str(name) if name else ""


def _active_provider(agent: Any) -> str:
    """Provider name of the live client, falling back to the routing rules."""
    live = _live_provider_name(agent)
    if live:
        return str(live)
    try:
        from agent_core.config import load_agent_settings
        from agent_core.constants import load_model_json
        from agent_core.llm.provider import provider_for

        return provider_for(
            _current_model(agent),
            load_agent_settings().llm_provider,
            str(load_model_json().get("provider") or ""),
        )
    except Exception:
        return "lmstudio"


def _entry(
    value: str,
    *,
    provider: str,
    current: str,
    label: str = "",
    extra: str = "",
    loaded: bool = False,
) -> Dict[str, Any]:
    """One selectable model row (``value`` is what ``model <value>`` accepts)."""
    return {
        "value": value,
        "label": label or value,
        "provider": provider,
        "current": value == current,
        "loaded": bool(loaded),
        "detail": extra,
    }


def _group(
    provider: str,
    entries: List[Dict[str, Any]],
    *,
    active: str,
) -> Optional[Dict[str, Any]]:
    """Wrap *entries* as a group, or ``None`` when it would only be noise.

    An empty group is kept ONLY when it is the active provider: that is where
    the hint explains why nothing is reachable.  Five empty gateways would bury
    the one source that works.
    """
    if entries:
        return {
            "provider": provider,
            "title": GROUP_TITLES.get(provider, provider),
            "entries": entries,
            "active": provider == active,
        }
    if provider != active:
        return None
    return {
        "provider": provider,
        "title": GROUP_TITLES.get(provider, provider),
        "entries": [],
        "active": True,
        "hint": GROUP_HINTS.get(provider, "not reachable"),
    }


def collect_catalog(
    agent: Any, command: Optional[ModelCommand] = None
) -> Dict[str, Any]:
    """Collect the model catalogue from the REPL's own fetchers.

    Never raises: a provider that answers with an exception becomes an empty
    group (or is dropped), because a dead gateway must not break the panel.
    """
    cmd = command or ModelCommand()
    current = _current_model(agent)
    active = _active_provider(agent)
    groups: List[Optional[Dict[str, Any]]] = []

    # LM Studio first: it is the default provider and the only one that also
    # reports which model currently sits in VRAM.
    lmstudio: List[Dict[str, Any]] = []
    try:
        models, _loaded_ids = cmd._fetch_models()
    except Exception:  # pragma: no cover - guarded by the fetcher itself
        models = []
    for model in models or []:
        try:
            key = str(model["key"])
        except (KeyError, TypeError):  # pragma: no cover - defensive
            continue
        params = str(model.get("params_string") or "")
        size_bytes = model.get("size_bytes") or 0
        size = ""
        try:
            size = _format_size(int(size_bytes)) if size_bytes else ""
        except (TypeError, ValueError):  # pragma: no cover - defensive
            size = ""
        detail = " · ".join(part for part in (params, size) if part)
        lmstudio.append(
            _entry(
                key,
                provider="lmstudio",
                current=current,
                label=str(model.get("display_name") or key),
                extra=detail,
                loaded=bool(model.get("loaded")),
            )
        )
    groups.append(_group("lmstudio", lmstudio, active=active))

    # opencode returns BOTH catalogs in one call (keyed go + keyless zen free
    # tier), so it is fetched once and split rather than fetched per group:
    # each fetch is a live HTTP round trip.
    try:
        opencode_go, opencode_zen, _api_mode = cmd._opencode_catalog(agent)
    except Exception:
        logger.debug("model catalog: opencode fetch failed", exc_info=True)
        opencode_go, opencode_zen = [], []

    # Every other provider is a flat list of namespaced ids.
    fetchers = (
        ("opencode", lambda: (opencode_go,)),
        ("llama", lambda: (cmd._llama_catalog(agent),)),
        ("openrouter", lambda: (cmd._openrouter_catalog(agent),)),
        ("lemonade", lambda: (cmd._lemonade_catalog(agent),)),
    )
    for provider, fetch in fetchers:
        try:
            fetched = fetch()[0]
        except Exception:
            logger.debug("model catalog: %s fetch failed", provider, exc_info=True)
            fetched = []
        entries = [
            _entry(str(item), provider=provider, current=current)
            for item in (fetched or [])
        ]
        groups.append(_group(provider, entries, active=active))

    # The keyless opencode-zen FREE tier, from the same single call above.
    zen = [
        _entry(str(item), provider="opencode-zen", current=current)
        for item in (opencode_zen or [])
    ]
    groups.append(_group("opencode-zen", zen, active=active))

    ordered = sorted(
        (group for group in groups if group),
        key=lambda group: GROUP_ORDER.index(group["provider"]),
    )
    return {
        "current": current,
        "active_provider": active,
        "groups": ordered,
        "count": sum(len(group["entries"]) for group in ordered),
    }


def model_catalog(agent: Any, *, fresh: bool = False) -> Dict[str, Any]:
    """Return the catalogue for *agent*, cached for :data:`CACHE_TTL_SECONDS`."""
    current = _current_model(agent)
    active = _active_provider(agent)
    key = (id(agent), current, active)
    now = time.monotonic()
    if not fresh:
        cached = _CACHE.get(key)
        if cached and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
    if agent is None:
        return {"current": "", "active_provider": active, "groups": [], "count": 0}
    with _FETCH_LOCK:
        cached = _CACHE.get(key)
        if not fresh and cached and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
        payload = collect_catalog(agent)
        _CACHE[key] = (now, payload)
        # Drop stale keys for this agent so a long-lived process cannot grow
        # one entry per switch.
        for stale in [k for k in _CACHE if k[0] == key[0] and k != key]:
            _CACHE.pop(stale, None)
        return payload


def clear_cache() -> None:
    """Forget every cached catalogue (called after a switch, and by tests)."""
    with _FETCH_LOCK:
        _CACHE.clear()


def select_model(
    agent: Any, model_key: str, provider: Optional[str] = None
) -> Dict[str, Any]:
    """Switch *agent* to *model_key*, exactly as ``model <name>`` does.

    *provider* forces the provider (the dashboard passes it per row, because a
    name alone is ambiguous across gateways).  When it is ``None`` the routing
    rules decide — the same ``provider_for`` the REPL uses.

    Returns ``{"ok": bool, "model": str, "changed": bool}``; never raises.
    """
    key = (model_key or "").strip()
    if not key:
        return {"ok": False, "error": "a model name is required"}
    llm = getattr(agent, "llm", None)
    if llm is None:
        return {"ok": False, "error": "this agent has no LLM client"}

    from agent_core.llm.provider import provider_for

    target = (provider or "").strip() or provider_for(key)
    previous = _current_model(agent)
    if key == previous and (not provider or provider == _active_provider(agent)):
        # Same model, same provider: nothing to rebuild, nothing to persist.
        return {"ok": True, "model": key, "provider": target, "changed": False}

    try:
        command = ModelCommand()
        command._rebuild_provider(agent, key, provider or None)
    except Exception as exc:
        logger.debug("model switch failed", exc_info=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        llm.model_name = key
    except Exception:  # pragma: no cover - a client without a settable name
        logger.debug("could not set llm.model_name", exc_info=True)

    persisted = True
    try:
        from agent_core.constants import persist_model_choice

        persist_model_choice(key, provider=target)
    except Exception:  # the switch itself already happened
        logger.debug("could not persist the model choice", exc_info=True)
        persisted = False

    clear_cache()
    return {
        "ok": True,
        "model": key,
        "provider": target,
        "previous": previous,
        "changed": True,
        "persisted": persisted,
    }


class ModelsCommand(Command):
    """``/models`` — show the model catalogue and pick from it.

    The REPL's ``model`` command prints its catalogue; this one is the
    non-printing form of the same thing, so the dashboard can render it.  It
    shares ``model_catalog`` with the dashboard endpoint, which is what keeps
    one list instead of two.
    """

    @property
    def name(self) -> str:
        return "models"

    @property
    def help_text(self) -> str:
        return (
            "models - List the available models per provider and pick one "
            "(same catalogue as 'model list'; the dashboard renders it)"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:  # noqa: F821
        from agent_core.commands.model_cmd import ModelCommand as _ModelCommand

        catalogue = model_catalog(agent, fresh=True)
        if not catalogue["groups"]:
            print("  No models found: no provider is reachable.")
            return True
        print(
            f"\n  Current model: {catalogue['current'] or 'none'}"
            f"  (provider: {catalogue['active_provider']})"
        )
        for group in catalogue["groups"]:
            marker = "*" if group["active"] else " "
            print(f"\n {marker} [{group['title']}] {len(group['entries'])} model(s)")
            for entry in group["entries"]:
                flags = []
                if entry["current"]:
                    flags.append("current")
                if entry["loaded"]:
                    flags.append("loaded")
                suffix = f"  [{', '.join(flags)}]" if flags else ""
                print(f"     {entry['value']}{suffix}")
            if group.get("hint"):
                print(f"     {group['hint']}")
        if args:
            # ``models <name>`` switches, so the alias has a terminal meaning.
            print()
            await _ModelCommand().execute(list(args), agent)
        return True


__all__ = [
    "CACHE_TTL_SECONDS",
    "GROUP_ORDER",
    "ModelsCommand",
    "clear_cache",
    "collect_catalog",
    "model_catalog",
    "select_model",
]
