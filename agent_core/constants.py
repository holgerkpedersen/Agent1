"""Shared constants for agent configuration."""
import os

KNOWN_MODELS = {
    "qwen3.6-27b-mtp": {
        "desc": "Qwen 3.6 27B — chat, codegen, large context",
        "max_tokens": 100000,
        "size_gb": 15.8,
        "params": "27B",
        "thinking": True,
        "disable_thinking_kwargs": {"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}},
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

_MODEL_JSON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_JSON_PATH = os.path.join(_MODEL_JSON_DIR, "model.json")


# ---------------------------------------------------------------------------
#  Single-source-of-truth model resolution
# ---------------------------------------------------------------------------

def resolve_model(explicit: str | None = None) -> str:
    """Return the best model name to use, checking sources in priority order.

    1. Explicit argument (caller override)
    2. What is actually loaded in LM Studio right now (via API)
    3. Persisted model.json (set by ``model`` command)
    4. ``AGENT_MODEL`` environment variable
    5. Hardcoded fallback (``laguna-s-2.1``)
    """
    if explicit:
        return explicit

    # Query LM Studio — what model is actually in VRAM?
    try:
        from agent_core.llm.lmstudio import get_models_status
        models = get_models_status()
        loaded = [m["key"] for m in models if m["loaded"]]
        if loaded:
            return loaded[0]
    except Exception:
        pass

    # Persisted choice from model.json
    persisted = load_model_json()
    if persisted.get("model") in KNOWN_MODELS:
        return persisted["model"]

    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------

def load_model_json() -> dict:
    """Load persisted model state from model.json."""
    import json as _json
    try:
        with open(MODEL_JSON_PATH, "r") as f:
            return _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        return {}


def save_model_json(data: dict) -> None:
    """Persist model state to model.json."""
    import json as _json
    try:
        with open(MODEL_JSON_PATH, "w") as f:
            _json.dump(data, f, indent=2)
    except OSError:
        pass


def persist_model_choice(model_name: str) -> None:
    """Write *model_name* to model.json and .env so it survives restarts."""
    data = load_model_json()
    data["model"] = model_name
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
