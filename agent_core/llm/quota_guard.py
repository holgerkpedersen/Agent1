"""Memory of provider tiers whose free quota is exhausted.

Why this exists (see ``docs/plans/2026-11-27-zen-free-tier-quota-plan.md``): the
opencode-zen free tier answers HTTP 429 ``FreeUsageLimitError`` for EVERY free
model at the same moment, because the quota is ACCOUNT-wide, not per model.
Rotating between free models therefore cannot succeed — it only multiplies
latency (3 retries = 2+4+8 s = 14 s wasted per turn, then the same 429).

Remembering the verdict turns the second and every later turn into a 0 s,
zero-HTTP answer with an actionable message.

Design constraints, all deliberate:

* **Keyed by TIER, not by model** — ``"opencode-zen"``.  That matches the
  measured reality: one dead tier, not one dead model.
* **Injected clock** — expiry is testable without ``sleep``.
* **Thread-safe** — providers run requests in worker threads
  (``asyncio.to_thread``), so the cache is touched concurrently.
* **Opt-in persistence** — the process-wide guard restores its verdicts from
  ``quota_state.json`` (next to the session's chat history), so restarting the
  agent does not spend one more request re-probing a tier that is known dead.
  A bare ``QuotaGuard()`` stays pure in-memory, which keeps unit tests free of
  filesystem side effects.

Persistence stores WALL-CLOCK deadlines, never ``time.monotonic()`` values: a
monotonic reading is meaningless in another process.  The in-memory map stays
monotonic (immune to clock jumps) and is converted on the way in and out.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

#: How long a tier stays blocked after an observed quota error, when the
#: gateway sends no ``Retry-After``.  Short on purpose: a user who subscribes
#: mid-session should not stay blocked for the rest of the day.
DEFAULT_COOLDOWN_SECONDS = 900.0

#: Env override for the default cooldown (seconds).
COOLDOWN_ENV_VAR = "AGENT_QUOTA_COOLDOWN_SECONDS"

#: Schema version of the persisted state file.
STATE_VERSION = 1


def _env_cooldown() -> float:
    """Cooldown from the environment, falling back to the default."""
    raw = os.environ.get(COOLDOWN_ENV_VAR)
    if not raw:
        return DEFAULT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_SECONDS


def _session_state_path() -> str | None:
    """``quota_state.json`` for the active session, or ``None`` if unknown.

    Resolved lazily per call (never at import) so the session layer can be
    pointed elsewhere — ``conftest.py`` redirects it for the test suite — and so
    an import cycle at module load is impossible.
    """
    try:
        from agent_core.session.context import current_paths

        return str(current_paths().quota_state_path)
    except Exception:  # noqa: BLE001 - state is an optimisation, never required
        logger.debug("No session path for quota state", exc_info=True)
        return None


class QuotaGuard:
    """Remembers which provider tiers are out of free quota, with an expiry."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        cooldown: float | None = None,
        state_path: Callable[[], str | None] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        #: Monotonic time source; injected by tests.
        self._clock: Callable[[], float] = clock or time.monotonic
        #: Wall-clock time source, used only to persist across processes.
        self._wall_clock: Callable[[], float] = wall_clock or time.time
        #: Fixed cooldown when given; otherwise read from the env on each mark.
        self._cooldown = cooldown
        #: Resolver for the state file; ``None`` means memory-only.
        self._state_path = state_path
        #: tier -> monotonic deadline after which the tier may be tried again.
        self._until: dict[str, float] = {}
        #: Reentrant: public methods load state while holding the lock.
        self._lock = threading.RLock()
        self._loaded = False

    # ------------------------------------------------------------------ state

    def _path(self) -> str | None:
        if self._state_path is None:
            return None
        try:
            return self._state_path()
        except Exception:  # noqa: BLE001
            logger.debug("Could not resolve quota state path", exc_info=True)
            return None

    def _ensure_loaded(self) -> None:
        """Restore persisted verdicts once, translating wall clock -> monotonic."""
        if self._loaded or self._state_path is None:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            path = self._path()
            if not path or not os.path.exists(path):
                return
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:  # noqa: BLE001 - corrupt state must not break turns
                logger.debug("Could not read quota state (no-op)", exc_info=True)
                return
            tiers = data.get("tiers") if isinstance(data, dict) else None
            if not isinstance(tiers, dict):
                return
            now_mono = self._clock()
            now_wall = self._wall_clock()
            for tier, epoch in tiers.items():
                try:
                    left = float(epoch) - now_wall
                except (TypeError, ValueError):
                    continue
                # Ignore already-expired entries: the next turn may probe again.
                if left > 0:
                    self._until[str(tier)] = now_mono + left

    def _persist(self) -> None:
        """Write the live verdicts out; never raises."""
        if self._state_path is None:
            return
        try:
            now_mono = self._clock()
            now_wall = self._wall_clock()
            tiers = {
                tier: round(now_wall + (until - now_mono), 3)
                for tier, until in self._until.items()
                if until > now_mono
            }
            path = self._path()
            if not path:
                return
            payload = {
                "version": STATE_VERSION,
                "tiers": tiers,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now_wall)),
            }
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - never break a turn over bookkeeping
            logger.debug("Could not save quota state (no-op)", exc_info=True)

    # ------------------------------------------------------------------- API

    def _cooldown_seconds(self) -> float:
        return self._cooldown if self._cooldown is not None else _env_cooldown()

    def is_exhausted(self, tier: str) -> bool:
        """True when *tier* is blocked and its cooldown has not elapsed."""
        self._ensure_loaded()
        with self._lock:
            until = self._until.get(tier)
            if until is None:
                return False
            if self._clock() >= until:
                # Expired: forget it so the next turn probes the tier again.
                self._until.pop(tier, None)
                self._persist()
                return False
            return True

    def mark_exhausted(self, tier: str, retry_after: float | None = None) -> None:
        """Block *tier* until the cooldown ends.

        *retry_after* (seconds, from the gateway's ``Retry-After`` header) wins
        over the default cooldown when it is a positive number.
        """
        self._ensure_loaded()
        cooldown = self._cooldown_seconds()
        if retry_after is not None:
            try:
                ra = float(retry_after)
            except (TypeError, ValueError):
                ra = 0.0
            if ra > 0:
                cooldown = ra
        with self._lock:
            self._until[tier] = self._clock() + cooldown
            self._persist()

    def remaining(self, tier: str) -> float:
        """Seconds left on *tier*'s block (0.0 when it is not blocked)."""
        self._ensure_loaded()
        with self._lock:
            until = self._until.get(tier)
            if until is None:
                return 0.0
            left = until - self._clock()
            if left <= 0:
                self._until.pop(tier, None)
                self._persist()
                return 0.0
            return left

    def reset(self, tier: str | None = None) -> None:
        """Clear *tier*, or every tier when *tier* is ``None``."""
        self._ensure_loaded()
        with self._lock:
            if tier is None:
                self._until.clear()
            else:
                self._until.pop(tier, None)
            self._persist()


#: The process-wide guard used by the providers.
_GUARD = QuotaGuard(state_path=_session_state_path)


def quota_guard() -> QuotaGuard:
    """Return the process-wide :class:`QuotaGuard`."""
    return _GUARD


def reset_quota_guard() -> None:
    """Clear the process-wide guard (used when the user re-selects a model)."""
    _GUARD.reset()


__all__: list[str] = [
    "COOLDOWN_ENV_VAR",
    "DEFAULT_COOLDOWN_SECONDS",
    "STATE_VERSION",
    "QuotaGuard",
    "quota_guard",
    "reset_quota_guard",
]
