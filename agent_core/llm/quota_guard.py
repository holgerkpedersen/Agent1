"""Process-local memory of provider tiers whose free quota is exhausted.

Why this exists (see ``docs/plans/2026-11-27-zen-free-tier-quota-plan.md``): the
opencode-zen free tier answers HTTP 429 ``FreeUsageLimitError`` for EVERY free
model at the same moment, because the quota is ACCOUNT-wide, not per model.
Rotating between free models therefore cannot succeed — it only multiplies
latency (3 retries = 2+4+8 s = 14 s wasted per turn, then the same 429).

Remembering the verdict turns the second and every later turn in a session into
a 0 s, zero-HTTP answer with an actionable message.

Design constraints, all deliberate:

* **Process-local** — no new files, no cross-session state.  Quota is per-run and
  per-account, and a restart (or a subscription) is the natural reset.
* **Keyed by TIER, not by model** — ``"opencode-zen"``.  That matches the
  measured reality: one dead tier, not one dead model.
* **Injected clock** — expiry is testable without ``sleep``.
* **Thread-safe** — providers run requests in worker threads
  (``asyncio.to_thread``), so the cache is touched concurrently.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

#: How long a tier stays blocked after an observed quota error, when the
#: gateway sends no ``Retry-After``.  Short on purpose: a user who subscribes
#: mid-session should not stay blocked for the rest of the day.
DEFAULT_COOLDOWN_SECONDS = 900.0

#: Env override for the default cooldown (seconds).
COOLDOWN_ENV_VAR = "AGENT_QUOTA_COOLDOWN_SECONDS"


def _env_cooldown() -> float:
    """Cooldown from the environment, falling back to the default."""
    raw = os.environ.get(COOLDOWN_ENV_VAR)
    if not raw:
        return DEFAULT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_SECONDS


class QuotaGuard:
    """Remembers which provider tiers are out of free quota, with an expiry."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        cooldown: float | None = None,
    ) -> None:
        #: Monotonic time source; injected by tests.
        self._clock: Callable[[], float] = clock or time.monotonic
        #: Fixed cooldown when given; otherwise read from the env on each mark.
        self._cooldown = cooldown
        #: tier -> monotonic deadline after which the tier may be tried again.
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _cooldown_seconds(self) -> float:
        return self._cooldown if self._cooldown is not None else _env_cooldown()

    def is_exhausted(self, tier: str) -> bool:
        """True when *tier* is blocked and its cooldown has not elapsed."""
        with self._lock:
            until = self._until.get(tier)
            if until is None:
                return False
            if self._clock() >= until:
                # Expired: forget it so the next turn probes the tier again.
                self._until.pop(tier, None)
                return False
            return True

    def mark_exhausted(self, tier: str, retry_after: float | None = None) -> None:
        """Block *tier* until the cooldown ends.

        *retry_after* (seconds, from the gateway's ``Retry-After`` header) wins
        over the default cooldown when it is a positive number.
        """
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

    def remaining(self, tier: str) -> float:
        """Seconds left on *tier*'s block (0.0 when it is not blocked)."""
        with self._lock:
            until = self._until.get(tier)
            if until is None:
                return 0.0
            left = until - self._clock()
            if left <= 0:
                self._until.pop(tier, None)
                return 0.0
            return left

    def reset(self, tier: str | None = None) -> None:
        """Clear *tier*, or every tier when *tier* is ``None``."""
        with self._lock:
            if tier is None:
                self._until.clear()
            else:
                self._until.pop(tier, None)


#: The process-wide guard used by the providers.
_GUARD = QuotaGuard()


def quota_guard() -> QuotaGuard:
    """Return the process-wide :class:`QuotaGuard`."""
    return _GUARD


def reset_quota_guard() -> None:
    """Clear the process-wide guard (used when the user re-selects a model)."""
    _GUARD.reset()


__all__: list[str] = [
    "COOLDOWN_ENV_VAR",
    "DEFAULT_COOLDOWN_SECONDS",
    "QuotaGuard",
    "quota_guard",
    "reset_quota_guard",
]
