"""One accessor for the monkeypatched globals a moved unit may read.

When a function moves from ``agent.py`` into ``agent_core.<module>``, a bare
read of a name that tests patch on ``agent`` (``agent.CHAT_HISTORY_JSON_PATH``,
``agent.LLMClient``, ...) resolves against the NEW module's globals and the
patch is silently ignored.  ``live()`` closes the gap without a per-name
accessor module: it resolves the name against ``agent``'s live globals at call
time.

``import agent`` is deferred into the function on purpose.  ``agent`` is the
composition root and imports everything else, so no ``agent_core`` module may
import it at module level (circular import).  The first call happens long
after import, so the deferred import is free.
"""
from __future__ import annotations

from typing import Any


def live(name: str, default: Any = None) -> Any:
    """Return ``agent.<name>`` as it is right now, or ``default`` if unset.

    Use it in a moved function for any global that tests patch on ``agent``:

        CHAT_HISTORY_JSON_PATH -> live("CHAT_HISTORY_JSON_PATH")

    A missing name yields ``default`` (``None`` when not given) rather than
    raising, so a module that has not been wired into ``agent``'s namespace
    cannot crash a call path.
    """
    try:
        import agent
    except Exception:  # pragma: no cover - import of the root must not crash a call
        return default
    return getattr(agent, name, default)
