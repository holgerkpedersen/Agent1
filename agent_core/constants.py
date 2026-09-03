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
        "opencode_server_url", "opencode_api_base", "opencode_zen_api_base",
        "llama_base_url", "openrouter_api_base",
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
        if not isinstance(pair, dict) or not str(pair.get("/think") or "").strip() \
                or not str(pair.get("/no_think") or "").strip():
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

#: Message key marking loop-INJECTED user notes (the auto-continue note).
#: Tagged messages are stripped from history when a turn ends and MUST be
#: stripped again at the LLM payload boundary (sanitize_message_roles) so
#: the internal marker never reaches a provider as an unknown field.
LOOP_NOTE_TAG_KEY = "_loop_note"


# ---------------------------------------------------------------------------
#  Single-source-of-truth model resolution
# ---------------------------------------------------------------------------

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
    first-run fallback when nothing is persisted yet.
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
    except Exception as exc:
        logger.warning("Failed to load agent settings, using defaults: %s", exc)
        provider_setting = "lmstudio"
        opencode_model = DEFAULT_OPENCODE_MODEL

    provider = provider_for(persisted_model, provider_setting, persisted_provider)
    if provider == "opencode":
        return persisted_model if persisted_model.startswith("opencode") else opencode_model

    # Persisted choice wins over what another shell may have loaded since.
    if persisted_model in KNOWN_MODELS:
        return persisted_model

    # First-run fallback: nothing usable persisted — adopt whatever LM
    # Studio currently has loaded (if anything).
    try:
        from agent_core.llm.lmstudio import get_models_status
        models = get_models_status()
        loaded = [m["key"] for m in models if m["loaded"]]
        if loaded:
            return str(loaded[0])
    except Exception as exc:
        logger.warning("Could not determine the active LM Studio model: %s", exc)

    if persisted_model:
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


