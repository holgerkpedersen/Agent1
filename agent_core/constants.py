"""Shared constants for agent configuration."""
import os

KNOWN_MODELS = {
    "qwen3.6-27b-mtp": {"desc": "Qwen 3.6 27B - chat, codegen, large context", "max_tokens": 100000},
    "google/gemma-4-31b": {"desc": "Gemma 4 31B - chat, reasoning, fast token gen", "max_tokens": 100000},
    "laguna-s-2.1": {"desc": "Laguna S 2.1 MoE A8B - agentic coding, thinking", "max_tokens": 100000},
}

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "laguna-s-2.1")
