"""Shared constants for agent configuration."""

from __future__ import annotations
import logging
import os
from typing import Any, cast

logger = logging.getLogger(__name__)

KNOWN_MODELS = {
    "qwen3.6-27b-mtp": {
        "desc": "Qwen 3.6 27B — chat, codegen, large context",
        "max_tokens": 100000,
        "size_gb": 15.8,
        "params": "27B",
        "thinking": True,
        "disable_thinking_kwargs": {"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}},
    },
    "qwen3.5-9b-mtp": {
        "desc": "Qwen 3.5 9B MTP — coding, thinking",
        "max_tokens": 100000,
        "size_gb": 5.3,
        "params": "9B",
        "thinking": True,
        "disable_thinking_kwargs": {"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}},
    },
    "qwen/qwen3.8-27b": {
        "desc": "Qwen 3.8 27B — chat, codegen, reasoning (2026-08-18: no explicit disable_kwargs; uses the safe minimal fallback — aggressive switches cause a full-budget reasoning burn)",
        "max_tokens": 100000,
        "size_gb": 16.0,
        "params": "27B",
        "thinking": True,
    },
    "google/gemma-4-12b": {
        "desc": "Gemma 4 12B — chat, reasoning, fast token gen",
        "max_tokens": 100000,
        "size_gb": 7.0,
        "params": "12B",
        "thinking": True,
    },
    "google/gemma-4-31b": {
        "desc": "Gemma 4 31B — chat, reasoning, fast token gen",
        "max_tokens": 100000,
        "size_gb": 18.1,
        "params": "31B",
        "thinking": True,
    },
    "kwaipilot_kat-coder-v2.5-dev": {
        "desc": "Kwaipilot Kat-Coder 2.5 dev — coding, thinking",
        "max_tokens": 100000,
        "size_gb": 8.1,
        "params": "8B",
        "thinking": True,
    },
    "qwen3-coder-30b-a3b-instruct": {
        "desc": "Qwen3 Coder 30B A3B MoE — coding, thinking",
        "max_tokens": 100000,
        "size_gb": 18.1,
        "params": "30B-A3B",
        "thinking": True,
        "disable_thinking_kwargs": {"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}},
    },
    "laguna-s-2.1": {
        "desc": "Laguna S 2.1 MoE A8B — agentic coding, thinking",
        "max_tokens": 100000,
        "size_gb": 4.2,
        "params": "8B-MoE",
        "thinking": True,
        "disable_thinking_kwargs": {
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
            "thinking": {"type": "disabled"},
            "enableThinking": False,
            "preserve_thinking": False,
            "reasoning": "off",
        },
    },
}

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "laguna-s-2.1")

#: Default opencode-go model when the opencode provider is active and no
#: explicit model is configured.
DEFAULT_OPENCODE_MODEL = "opencode-go/deepseek-v4-flash"

_MODEL_JSON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    6. Hardcoded fallback (``laguna-s-2.1``)

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


