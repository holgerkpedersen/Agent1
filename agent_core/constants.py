"""Shared constants for agent configuration."""

from __future__ import annotations
import json as _json
import logging
import os
from typing import Any, cast

logger = logging.getLogger(__name__)

_MODEL_JSON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Path to the external model catalog JSON (loaded at import time).
_MODEL_CATALOG_PATH = os.path.join(_MODEL_JSON_DIR, "model_catalog.json")


def _load_model_catalog() -> dict[str, Any]:
    """Load the model catalog from ``model_catalog.json``.

    The catalog is the single source of truth for model names, provider
    routing, tier prefixes, and default models.  A missing, malformed, or
    incomplete catalog is a hard error at import time: the agent must never
    silently fall back to a hardcoded model name.
    """
    try:
        with open(_MODEL_CATALOG_PATH, "r", encoding="utf-8") as f:
            data: Any = _json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"model catalog not found at {_MODEL_CATALOG_PATH} — all model "
            "names, provider routing, and defaults must come from this file"
        ) from None
    except _json.JSONDecodeError as exc:
        raise RuntimeError(
            f"model catalog {_MODEL_CATALOG_PATH} is malformed: {exc}"
        ) from None
    if not isinstance(data, dict):
        raise RuntimeError(
            f"model catalog {_MODEL_CATALOG_PATH} root must be a JSON object"
        )

    routing = data.get("_routing")
    if not isinstance(routing, dict) or not routing:
        raise RuntimeError(
            "model catalog must define a non-empty '_routing' table "
            "(model-name prefix -> provider)"
        )
    defaults = data.get("_defaults")
    if not isinstance(defaults, dict):
        raise RuntimeError("model catalog must define a '_defaults' object")
    for key in ("model", "opencode_model", "openrouter_model"):
        if not str(defaults.get(key) or "").strip():
            raise RuntimeError(f"model catalog must define '_defaults.{key}'")
    llm_chain = defaults.get("llm_chain")
    if not isinstance(llm_chain, list) or not llm_chain:
        raise RuntimeError(
            "model catalog must define a non-empty '_defaults.llm_chain' list "
            "(ordered provider failover chain, entries may be 'provider:model')"
        )
    for key in (
        "opencode_server_url",
        "opencode_api_base",
        "opencode_zen_api_base",
        "llama_base_url",
        "openrouter_api_base",
        "lemonade_base_url",
    ):
        if not str(defaults.get(key) or "").strip():
            raise RuntimeError(f"model catalog must define '_defaults.{key}'")
    zen_prefixes = data.get("_zen_free_tier_prefixes")
    if not isinstance(zen_prefixes, list) or not zen_prefixes:
        raise RuntimeError(
            "model catalog must define a non-empty '_zen_free_tier_prefixes' list"
        )
    thinking_gates = data.get("_thinking_gates", {})
    if not isinstance(thinking_gates, dict):
        raise RuntimeError(
            "model catalog '_thinking_gates' must be an object mapping "
            "model-name substrings to {/think, /no_think} directive pairs"
        )
    for token, pair in thinking_gates.items():
        if (
            not isinstance(pair, dict)
            or not str(pair.get("/think") or "").strip()
            or not str(pair.get("/no_think") or "").strip()
        ):
            raise RuntimeError(
                f"model catalog '_thinking_gates.{token}' must define both "
                "'/think' and '/no_think' directives"
            )
    return data


_CATALOG: dict[str, Any] = _load_model_catalog()

KNOWN_MODELS: dict[str, dict[str, Any]] = {
    k: v for k, v in _CATALOG.items() if not k.startswith("_")
}

#: Provider routing table from ``model_catalog.json`` ``_routing``.
#: Maps a model-name prefix (lowercase) to the provider that handles it.
ROUTER: dict[str, str] = {str(k): str(v) for k, v in _CATALOG["_routing"].items()}

#: Default model names from ``model_catalog.json`` ``_defaults``.
_DEFAULTS: dict[str, str] = {str(k): str(v) for k, v in _CATALOG["_defaults"].items()}

#: Model-name prefixes that select the keyless opencode-zen FREE tier
#: (from the catalog — no concrete names in code).
_ZEN_TIER_PREFIXES: tuple[str, ...] = tuple(
    str(p) for p in _CATALOG["_zen_free_tier_prefixes"]
)

#: Reasoning-gate directives per model family, from ``model_catalog.json``
#: ``_thinking_gates``: maps a lowercase model-name substring (e.g. the
#: Nemotron-3 family token) to its {"/think", "/no_think"} system-prompt
#: directive pair.  Providers consult this table instead of hardcoding
#: concrete model names.
THINKING_GATES: dict[str, dict[str, str]] = {
    str(token): {str(k): str(v) for k, v in pair.items()}
    for token, pair in _CATALOG.get("_thinking_gates", {}).items()
}

#: Default local model (LM Studio family) when nothing else is configured.
#: Env override first, then the catalog — no hardcoded model name.
DEFAULT_MODEL = os.environ.get("AGENT_MODEL") or _DEFAULTS["model"]

#: Default opencode-go model when the opencode provider is active and no
#: explicit model is configured.  Reads AGENT_OPENCODE_MODEL from .env so
#: this constant and AgentSettings.opencode_model share one source of truth.
DEFAULT_OPENCODE_MODEL = (
    os.environ.get("AGENT_OPENCODE_MODEL") or _DEFAULTS["opencode_model"]
)

#: Default openrouter model (env override first, then the catalog).
DEFAULT_OPENROUTER_MODEL = (
    os.environ.get("AGENT_OPENROUTER_MODEL") or _DEFAULTS["openrouter_model"]
)

#: Default model for the Jev decision engine (env override first, then the
#: catalog ``_defaults.jev_model``).  Jev uses a DEDICATED small model (the
#: non-thinking ``qwen2.5-coder-1.5b-instruct``) for typed yesno/choice/score
#: micro-decisions; it is never the agent's main model.  Absent from an older
#: catalog -> "" (the Jev engine then reports a clear configuration error
#: instead of guessing).
DEFAULT_JEV_MODEL = os.environ.get("AGENT_JEV_MODEL") or _DEFAULTS.get("jev_model", "")

#: Default ordered provider failover chain from ``model_catalog.json``
#: ``_defaults.llm_chain`` (entries may be "provider:model" overrides).
#: No concrete provider/model names in code.
DEFAULT_LLM_CHAIN: tuple[str, ...] = tuple(
    str(e) for e in _CATALOG["_defaults"]["llm_chain"]
)

#: Default endpoint URLs from ``model_catalog.json`` ``_defaults`` — the
#: provider endpoints are catalog data, not code (env overrides happen in
#: agent_core.config where the user's environment is read).
DEFAULT_OPENCODE_SERVER_URL = _DEFAULTS["opencode_server_url"]
DEFAULT_OPENCODE_API_BASE = _DEFAULTS["opencode_api_base"]
DEFAULT_OPENCODE_ZEN_API_BASE = _DEFAULTS["opencode_zen_api_base"]
DEFAULT_LLAMA_BASE_URL = _DEFAULTS["llama_base_url"]
DEFAULT_OPENROUTER_API_BASE = _DEFAULTS["openrouter_api_base"]
#: Local AMD Lemonade (NPU/iGPU) OpenAI-compatible endpoint.
DEFAULT_LEMONADE_API_BASE = _DEFAULTS["lemonade_base_url"]

MODEL_JSON_PATH = os.path.join(_MODEL_JSON_DIR, "model.json")
CHAT_HISTORY_JSON_PATH = os.path.join(_MODEL_JSON_DIR, "chat_history.json")
#: Cross-session memory (files read, semantic index, knowledge graph, working
#: memory) — reloaded on the next session so work done is not forgotten.
AGENT_MEMORY_JSON_PATH = os.path.join(_MODEL_JSON_DIR, "agent_memory.json")
#: Atomic-write sidecars: persistence writes go to "<path>.tmp" first and are
#: then os.replace()d into place, so a crash mid-write can never leave a
#: half-written JSON behind (a corrupt file used to mean the whole
#: conversation was silently dropped on next load).
CHAT_HISTORY_TMP_PATH = CHAT_HISTORY_JSON_PATH + ".tmp"
AGENT_MEMORY_TMP_PATH = AGENT_MEMORY_JSON_PATH + ".tmp"

#: Per-turn outcome log (JSONL, bounded to the newest 500 lines): one record
#: per finished chat turn {ts, user_input_tail, mutated_files, llm_error,
#: duration} — harness-layer observability only (decision #014) feeding habit
#: mining and the self-improvement loop.
TURN_LOG_PATH = os.path.join(_MODEL_JSON_DIR, "turn_log.jsonl")

#: Message key marking loop-INJECTED user notes (the auto-continue note).
#: Tagged messages are stripped from history when a turn ends and MUST be
#: stripped again at the LLM payload boundary (sanitize_message_roles) so
#: the internal marker never reaches a provider as an unknown field.
LOOP_NOTE_TAG_KEY = "_loop_note"

#: Root directory for benchmark reports, harnessfix output, etc.  Created on
#: demand by save_report(); read-only in normal operation.
REPORTS_DIR = os.path.join(_MODEL_JSON_DIR, "reports")


# ---------------------------------------------------------------------------
#  Single-source-of-truth model resolution
# ---------------------------------------------------------------------------

#: Process-lifetime cache for the LM Studio "what is loaded right now" poll.
#: ``None`` = not polled yet; a list = the polled model dicts.
_LIVE_POLL_CACHE: list[dict[str, Any]] | None = None


def _live_loaded_models(poll: Any) -> list[dict[str, Any]]:
    """Run the LM Studio model poll at most ONCE per process.

    The poll is a blocking HTTP request made from every provider
    construction; without this cache a single ``Agent()`` cost ~4.5s of
    socket wait (four calls) and the full test suite spent minutes in it.
    Tests that need a fresh poll call :func:`reset_live_poll_cache`.
    """
    global _LIVE_POLL_CACHE
    if _LIVE_POLL_CACHE is None:
        try:
            _LIVE_POLL_CACHE = list(poll())
        except Exception:
            _LIVE_POLL_CACHE = []
    return _LIVE_POLL_CACHE


def reset_live_poll_cache() -> None:
    """Forget the cached live LM Studio poll (tests / explicit refresh)."""
    global _LIVE_POLL_CACHE
    _LIVE_POLL_CACHE = None


def _lmstudio_live_models() -> list[dict[str, Any]]:
    """Poll LM Studio for its model list (deliberately NOT the process cache).

    The persisted-model gate uses the LIVE list: a model installed after
    process start must still be recognised, and a stale cache from a
    previous session must never veto the persisted choice.  One poll per
    gate evaluation, which only runs when the static catalog misses.
    Raises when LM Studio is unreachable.
    """
    from agent_core.llm.lmstudio import get_models_status

    return list(get_models_status())


def _lmstudio_installation_known() -> bool:
    """True when this process has evidence of LM Studio's installed models.

    Evidence = a successful live poll, or the process-cached list from an
    earlier successful poll.  False means LM Studio is unreachable AND no
    list was ever polled (first run with LM Studio down): the caller then
    falls back to the static catalog gate, because the bare persisted name
    (the ONLY evidence of a model in that state) is exactly what the
    ``model list`` auto-sync bug used to poison, and the live-poll fallback
    / catalog default are the safe outcomes.
    """
    try:
        return bool(_lmstudio_live_models())
    except Exception:
        return bool(_LIVE_POLL_CACHE)


def _lmstudio_reports_model(name: str) -> bool:
    """True when LM Studio has *name* installed (live, uncached poll).

    Used by :func:`resolve_model` to honour a persisted LM Studio selection
    that the static ``model_catalog.json`` does not know: the live model
    list, not the catalog, is the authority on what the provider can serve.
    When LM Studio is unreachable, the process-cached poll IS the
    last-known installation list and is consulted instead — a persisted
    model that was in that list must not be vetoed by a transient outage.
    """
    low = (name or "").strip().lower()
    if not low:
        return False
    try:
        models = _lmstudio_live_models()
    except Exception as exc:
        logger.warning("Live model poll failed for persisted-model check: %s", exc)
        models = list(_LIVE_POLL_CACHE) if _LIVE_POLL_CACHE else []
    for m in models:
        key = str(m.get("key") or "").strip().lower()
        disp = str(m.get("display_name") or "").strip().lower()
        if key == low or disp == low:
            return True
    return False


def resolve_model(explicit: str | None = None) -> str:
    """Return the best model name to use, checking sources in priority order.

    1. Explicit argument (caller override) — its prefix selects the provider
    2. Configured opencode model when the active provider is opencode
    3. Persisted model.json (set by ``model`` command, provider-aware)
    4. ``AGENT_MODEL`` environment variable
    5. What is actually loaded in LM Studio right now (via API) — lmstudio only
    6. Catalog default (``model_catalog.json`` ``_defaults.model``)

    The persisted choice outranks the live LM Studio poll (multi-shell
    safety): a second ``agent.py`` shell that loads a different model must
    not silently re-target this session at startup — the session keeps its
    own chosen model and LMStudioProvider reloads it on demand when a
    request finds it missing from VRAM.  The live poll remains as a
    first-run fallback when nothing is persisted yet.  For a persisted
    lmstudio model the "keep it" test is: the model is in the static
    catalog, OR LM Studio reports it installed (live poll) — so a
    user-installed model that the catalog never listed is not forgotten.
    """
    if explicit:
        return explicit

    from agent_core.llm.provider import provider_for
    from agent_core.config import load_agent_settings

    # Provider selection: persisted/env setting, or a model prefix if present.
    persisted = load_model_json()
    persisted_model = str(persisted.get("model") or "")
    persisted_provider = str(persisted.get("provider") or "")
    try:
        settings = load_agent_settings()
        provider_setting = settings.llm_provider
        opencode_model = settings.opencode_model
        # getattr: partial settings stubs (older tests) may lack jev_model;
        # an AttributeError here would wrongly reset provider_setting too.
        jev_setting = str(getattr(settings, "jev_model", "") or "")
    except Exception as exc:
        logger.warning("Failed to load agent settings, using defaults: %s", exc)
        provider_setting = "lmstudio"
        opencode_model = DEFAULT_OPENCODE_MODEL
        jev_setting = ""

    provider = provider_for(persisted_model, provider_setting, persisted_provider)
    if provider == "opencode":
        return (
            persisted_model
            if persisted_model.startswith("opencode")
            else opencode_model
        )

    # Persisted choice for a NON-lmstudio provider (llama/, lemonade/,
    # openrouter/, opencode-zen/ ...) is authoritative.  ``KNOWN_MODELS`` is
    # an LM-Studio-era shortlist and must not silently discard it: on
    # 2026-10-24 a persisted ``llama/swift-1.5-iq3_s`` failed this gate, the
    # live poll then adopted the loaded JEV decision model (a 1.5B non-agent
    # model) as the chat model, and the session lost tool calling entirely.
    if persisted_model and provider != "lmstudio":
        return persisted_model

    # A persisted LM Studio choice wins over what another shell may have
    # loaded since — IF this installation can serve it: the live poll (or
    # this process's last-known model list when LM Studio is unreachable)
    # reports it installed, OR the static catalog lists it.  The gate is NOT
    # ``KNOWN_MODELS`` alone: a model the operator has installed in LM Studio
    # but never listed in ``model_catalog.json`` (e.g.
    # ``qwen2.5-coder-1.5b-instruct``) used to fail the catalog-only gate,
    # the live-poll fallback below then adopted whatever sat in VRAM, and
    # the session forgot the last selection at every restart.  The gate
    # exists to stop ``model list`` auto-sync leftovers and names that can
    # no longer be served from persisting forever, so LM Studio's own model
    # list is the authority when it answers; the static catalog decides only
    # when LM Studio is unreachable and no list was ever polled.
    lmstudio_gate_failed = False
    if persisted_model and provider == "lmstudio":
        _evidence = _lmstudio_installation_known()
        keep = (
            _evidence and _lmstudio_reports_model(persisted_model)
            or persisted_model in KNOWN_MODELS
        )
        if keep:
            return persisted_model
        lmstudio_gate_failed = True

    # First-run fallback: nothing usable persisted — adopt whatever LM
    # Studio currently has loaded (if anything).
    #
    # The poll is a live HTTP round-trip (~2s against a busy LM Studio) and
    # this function runs on EVERY provider construction — ``Agent.__init__``
    # alone calls it four times, and the test suite builds ~370 Agents.  An
    # uncached poll therefore added minutes of pure socket wait to a single
    # pytest run.  Cache the answer for the process: a session that starts
    # with nothing persisted must not keep re-asking which model is in VRAM,
    # and the poll is only a *first-run* fallback anyway.  ``model`` (the
    # REPL switch) and LMStudioProvider's own reload path do not go through
    # here, so an explicit switch still takes effect immediately.
    #
    # The dedicated Jev decision model is NEVER eligible: it is a tiny
    # non-agent model (terse yesno/choice answers) that cannot drive the
    # tool loop.  Because the Jev engine loads it into LM Studio on demand,
    # the live poll would otherwise keep adopting it after every restart.
    jev_key = (jev_setting or DEFAULT_JEV_MODEL).split("/")[-1].strip().lower()
    try:
        from agent_core.llm.lmstudio import get_models_status

        models = _live_loaded_models(get_models_status)
        loaded = [
            str(m["key"])
            for m in models
            if m["loaded"]
            and str(m["key"]).split("/")[-1].strip().lower() != jev_key
        ]
        if loaded:
            return str(loaded[0])
    except Exception as exc:
        logger.warning("Could not determine the active LM Studio model: %s", exc)

    if persisted_model and not lmstudio_gate_failed:
        # Last-resort adoption of the persisted name: only when the live
        # poll found nothing loadable AND the lmstudio gate above never
        # ran (nothing persisted as lmstudio, e.g. a provider-less model.json
        # with no routable prefix).  A name the lmstudio gate explicitly
        # REJECTED (LM Studio unreachable, no evidence, catalog miss) must
        # NOT be adopted — it may be a ``model list`` auto-sync leftover the
        # provider cannot serve; fall through to the catalog default.
        return persisted_model

    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------


def load_model_json() -> dict[str, Any]:
    """Load persisted model state from model.json."""
    import json as _json

    try:
        with open(MODEL_JSON_PATH, "r") as f:
            return cast(dict[str, Any], _json.load(f))
    except (FileNotFoundError, _json.JSONDecodeError):
        logger.debug("model.json missing or unreadable — using empty state")
        return {}


def save_model_json(data: dict[str, Any]) -> None:
    """Persist model state to model.json."""
    import json as _json

    try:
        with open(MODEL_JSON_PATH, "w") as f:
            _json.dump(data, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to save model.json: %s", exc)


def persist_model_choice(model_name: str, provider: str | None = None) -> None:
    """Write *model_name* (and its provider) to model.json and .env.

    The provider is inferred from the model prefix when not given.
    """
    from agent_core.llm.provider import provider_for

    effective_provider = provider or provider_for(model_name)
    data = load_model_json()
    data["model"] = model_name
    data["provider"] = effective_provider
    save_model_json(data)

    env_path = ".env"
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r") as ef:
            lines = ef.readlines()
    with open(env_path, "w") as ef:
        for line in lines:
            if line.startswith("AGENT_MODEL="):
                ef.write(f"AGENT_MODEL={model_name}\n")
                found = True
            else:
                ef.write(line)
        if not found:
            ef.write(f"\nAGENT_MODEL={model_name}\n")


def persist_jev_model(model_name: str) -> None:
    """Write the dedicated Jev model to model.json and .env.

    The Jev model is INDEPENDENT of the selected chat model: ``jev``,
    ``jev_decide`` and ``speculate --judge jev`` always resolve it from
    ``AGENT_JEV_MODEL`` / model.json ``jev_model`` / the catalog, never from
    the active agent model.  Persisting here means every future session picks
    it up automatically.
    """
    data = load_model_json()
    data["jev_model"] = model_name
    save_model_json(data)

    env_path = ".env"
    lines: list[str] = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r") as ef:
            lines = ef.readlines()
    with open(env_path, "w") as ef:
        for line in lines:
            if line.startswith("AGENT_JEV_MODEL="):
                ef.write(f"AGENT_JEV_MODEL={model_name}\n")
                found = True
            else:
                ef.write(line)
        if not found:
            ef.write(f"\nAGENT_JEV_MODEL={model_name}\n")


__all__: list[str] = [
    "AGENT_MEMORY_JSON_PATH",
    "AGENT_MEMORY_TMP_PATH",
    "CHAT_HISTORY_JSON_PATH",
    "CHAT_HISTORY_TMP_PATH",
    "DEFAULT_LLM_CHAIN",
    "DEFAULT_MODEL",
    "DEFAULT_OPENCODE_API_BASE",
    "DEFAULT_OPENCODE_MODEL",
    "DEFAULT_OPENCODE_SERVER_URL",
    "DEFAULT_OPENCODE_ZEN_API_BASE",
    "DEFAULT_LEMONADE_API_BASE",
    "DEFAULT_LLAMA_BASE_URL",
    "DEFAULT_OPENROUTER_API_BASE",
    "DEFAULT_OPENROUTER_MODEL",
    "DEFAULT_JEV_MODEL",
    "KNOWN_MODELS",
    "LOOP_NOTE_TAG_KEY",
    "MODEL_JSON_PATH",
    "ROUTER",
    "THINKING_GATES",
    "TURN_LOG_PATH",
    "load_model_json",
    "persist_jev_model",
    "persist_model_choice",
    "resolve_model",
    "save_model_json",
]
