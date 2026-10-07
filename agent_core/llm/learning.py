"""Meta-policy learning loop — the production wiring for plan item #4.

Closes ``docs/AGENTIC_IMPROVEMENT_PLAN.md`` item "close the meta-policy
loop": ``MetricsTracker`` and ``MetaPolicyEvolver`` existed but were never
called from production code.  This module owns the two singletons and the
three operations the chat pipeline needs:

* :func:`record_turn_outcome` — record one finished chat turn (success or
  failure) under the inferred :class:`TaskType` and the active
  :class:`ProfileType`, then evolve + persist the evolver weights every
  ``EVOLVE_EVERY`` recorded turns.
* :func:`recommend_profile` — return the profile name that
  ``TASK_PROFILE_MAP`` maps this input to, but only when it clearly
  outscores the currently running profile's evolved weight.
* :func:`save_weights` — persist ``EVOLVER`` weights (shutdown hook).

Persistence: ``meta_policy.json`` lives in the same state directory as
``CHAT_HISTORY_JSON_PATH`` (resolved at call time, so tests can monkeypatch
the constant), written atomically (tmp + ``os.replace``) as
``{"weights": {<profile>: <weight>}, "updated": <iso>}``.

Contract for everything exported here: **never raises**.  A broken state
file or a half-initialised agent must not kill a chat turn — the same
try/except-no-op rule as ``Agent._record_llm_experience`` and the A3
turn-log hook (harness-layer observability only, decision #014: nothing in
this module changes sampling behaviour unless the caller opts in via the
``profile_auto`` workspace pref, which defaults off).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import agent_core.constants as _constants
from agent_core.llm.config import TASK_PROFILE_MAP
from agent_core.llm.llm_types import ProfileType
from agent_core.llm.meta_policy import MetaPolicyEvolver
from agent_core.llm.metrics_tracker import MetricsTracker
from agent_core.llm.provider import get_last_metrics
from agent_core.llm.task_type import infer_task_type

logger = logging.getLogger(__name__)

__all__ = [
    "METRICS",
    "EVOLVER",
    "DEFAULT_PROFILE_TYPE",
    "EVOLVE_EVERY",
    "load_weights",
    "save_weights",
    "profile_type_for_name",
    "profile_name_for_type",
    "record_turn_outcome",
    "recommend_profile",
]

#: Per-((task, profile)) success/failure/latency/token counters.
METRICS = MetricsTracker()

#: Evolved preference weights over :class:`ProfileType`.
EVOLVER = MetaPolicyEvolver()

#: Evolver weights are updated after this many recorded turns, then saved.
EVOLVE_EVERY = 10

#: Built-in profile names (agent_core/llm/model_profiles.py:29) — the only
#: names that map to a ProfileType; custom user profiles map to ``None``.
_NAME_TO_TYPE: dict[str, ProfileType] = {
    "fast-codegen": ProfileType.FAST_CODEGEN,
    "deep-analysis": ProfileType.DEEP_ANALYSIS,
    "precise": ProfileType.PRECISE,
}

#: "No profile name" is recorded under DEEP_ANALYSIS: providers default to
#: temperature 0.7 (agent_core/llm/lmstudio.py:482), which is exactly the
#: built-in deep-analysis profile's temperature (model_profiles.py:36).
#: The same type doubles as the baseline weight in :func:`recommend_profile`,
#: so un-pinned turns evolve the baseline they are compared against.
DEFAULT_PROFILE_TYPE: ProfileType = ProfileType.DEEP_ANALYSIS

#: Loaded-at-least-once flag for the persisted weights (lazy first use).
_weights_loaded = False

#: Recorded turns since the last ``EVOLVER.update_weights`` call.
_turns_since_evolve = 0


def _state_path() -> str:
    """``meta_policy.json`` next to ``chat_history.json`` (call-time resolve)."""
    return os.path.join(
        os.path.dirname(_constants.CHAT_HISTORY_JSON_PATH),
        "meta_policy.json",
    )


def load_weights(*, force: bool = False) -> None:
    """Lazily restore ``EVOLVER`` weights from ``meta_policy.json``.

    Idempotent (``force=True`` re-reads, e.g. for tests); never raises —
    a missing or corrupt file leaves the weights at their defaults.
    """
    global _weights_loaded
    if _weights_loaded and not force:
        return
    _weights_loaded = True
    try:
        path = _state_path()
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        weights = data.get("weights") if isinstance(data, dict) else None
        if not isinstance(weights, dict):
            return
        for key, value in weights.items():
            try:
                weight = float(value)
                profile = ProfileType(str(key))
            except (TypeError, ValueError):
                continue
            # Clamp to the evolver's own bounds (meta_policy.py:24).
            EVOLVER._weights[profile] = max(0.1, min(weight, 2.0))
    except Exception:  # noqa: BLE001 - observability must not break turns
        logger.debug("Could not load meta-policy weights (no-op)", exc_info=True)


def save_weights() -> None:
    """Persist ``EVOLVER`` weights atomically; never raises."""
    try:
        load_weights()  # never overwrite a good file with unloaded defaults
        payload = {
            "weights": {
                profile.value: round(weight, 6)
                for profile, weight in EVOLVER.get_preference_scores().items()
            },
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        path = _state_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - observability must not break turns
        logger.debug("Could not save meta-policy weights (no-op)", exc_info=True)


def profile_type_for_name(name: str | None) -> ProfileType | None:
    """Map an active profile name to its :class:`ProfileType`, else ``None``.

    Built-in names (``fast-codegen`` / ``deep-analysis`` / ``precise``) match
    directly; the underscore/hyphen-insensitive fallback covers custom
    profiles whose name happens to equal a ProfileType value.  Custom user
    profiles with arbitrary names return ``None``.
    """
    if not name:
        return None
    normalized = str(name).strip().lower()
    mapped = _NAME_TO_TYPE.get(normalized)
    if mapped is not None:
        return mapped
    try:
        return ProfileType(normalized.replace("-", "_"))
    except ValueError:
        return None


def profile_name_for_type(profile_type: ProfileType) -> str:
    """Built-in profile name for a ProfileType (``fast_codegen`` → ``fast-codegen``)."""
    return profile_type.value.replace("_", "-")


#: A trace-derived turn quality at or above this value is recorded as a
#: success; below it, a failure (plan item #4 residual).  score_run() gives
#: completed runs >= 0.7 (latency-penalized) and incomplete/errored runs
#: 0.0, so 0.5 separates the two without clipping penalized successes.
QUALITY_SUCCESS_THRESHOLD = 0.5


def record_turn_outcome(
    user_input: str,
    *,
    profile_name: str | None,
    success: bool,
    latency_seconds: float | None = None,
    provider: Any = None,
    mutated_files: list[str] | None = None,
    quality: float | None = None,
) -> None:
    """Record one finished chat turn into ``METRICS``; never raises.

    The task type is inferred from *user_input* (``task_type.py``); the
    profile type comes from *profile_name* (falls back to
    ``DEFAULT_PROFILE_TYPE`` — un-pinned turns are recorded under the
    profile whose temperature matches the provider default).

    *quality* is the trace-derived turn score
    (``harnessfix.evolution_metrics.score_run`` over the turn's own events).
    When present it decides the recorded success/failure
    (``quality >= QUALITY_SUCCESS_THRESHOLD``) instead of the caller's
    *success* proxy, so profile weights evolve from real trace outcomes —
    e.g. a loop that ended ``no_progress`` without a provider error is
    recorded as the failure it was.  ``None`` keeps the *success* flag.

    On success, ``MetricsTracker.record_turn`` is used when the provider
    exposes ``last_response_metrics`` (token/cost accounting) — it already
    counts the success itself (metrics_tracker.py:113), so ``record_success``
    is only the fallback when no metrics are available; calling both would
    double-count successes.  Failures use ``record_failure`` (no latency
    slot there).

    Every ``EVOLVE_EVERY`` recorded turns the evolver runs and the weights
    are persisted.
    """
    global _turns_since_evolve
    try:
        load_weights()
        if quality is not None:
            success = float(quality) >= QUALITY_SUCCESS_THRESHOLD
        task_type = infer_task_type(
            user_input, mutated_files=list(mutated_files) if mutated_files else None,
        )
        profile_type = profile_type_for_name(profile_name) or DEFAULT_PROFILE_TYPE
        if success:
            latency = float(latency_seconds) if latency_seconds else 0.0
            metrics = get_last_metrics(provider) if provider is not None else None
            if metrics is not None:
                METRICS.record_turn(task_type, profile_type, latency, metrics=metrics)
            else:
                METRICS.record_success(task_type, profile_type, latency)
        else:
            METRICS.record_failure(task_type, profile_type)
        _turns_since_evolve += 1
        if _turns_since_evolve >= EVOLVE_EVERY:
            _turns_since_evolve = 0
            EVOLVER.update_weights(METRICS)
            save_weights()
    except Exception:  # noqa: BLE001 - observability must not break turns
        logger.debug("Meta-policy recording failed (no-op)", exc_info=True)


def recommend_profile(
    user_input: str,
    current_profile_name: str | None = None,
) -> str | None:
    """Profile name to suggest for *user_input*, or ``None``; never raises.

    Rule (plan B2): ``TASK_PROFILE_MAP[infer_task_type(user_input)]`` must
    differ from the currently running profile **and** its evolved weight must
    be strictly higher than the current profile's weight.  The "current"
    profile when no name is set is ``DEFAULT_PROFILE_TYPE`` — the same type
    un-pinned turns are recorded under, so the baseline evolves from real
    outcomes: the suggestion appears once the running profile underperforms
    the mapped alternative.  With fresh, equal weights nothing is suggested.
    """
    try:
        load_weights()
        suggested = TASK_PROFILE_MAP[infer_task_type(user_input)]
        current = profile_type_for_name(current_profile_name) or DEFAULT_PROFILE_TYPE
        if suggested == current:
            return None
        scores = EVOLVER.get_preference_scores()
        if scores.get(suggested, 1.0) <= scores.get(current, 1.0):
            return None
        return profile_name_for_type(suggested)
    except Exception:  # noqa: BLE001 - observability must not break turns
        logger.debug("Profile recommendation failed (no-op)", exc_info=True)
        return None
