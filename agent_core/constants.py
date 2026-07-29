"""Shared constants for agent configuration."""
import os

KNOWN_MODELS = {
    "qwen3.6-27b-mtp": {
        "desc": "Qwen 3.6 27B — chat, codegen, large context",
        "max_tokens": 100000,
        "size_gb": 15.8,
        "params": "27B",
        "thinking": True,
    },
    "google/gemma-4-31b": {
        "desc": "Gemma 4 31B — chat, reasoning, fast token gen",
        "max_tokens": 100000,
        "size_gb": 18.1,
        "params": "31B",
        "thinking": True,
    },
    "laguna-s-2.1": {
        "desc": "Laguna S 2.1 MoE A8B — agentic coding, thinking",
        "max_tokens": 100000,
        "size_gb": 4.2,
        "params": "8B-MoE",
        "thinking": True,
    },
}

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "laguna-s-2.1")

MODEL_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.json")

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
