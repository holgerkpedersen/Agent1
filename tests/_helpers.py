"""Shared test helpers — single source of truth for the default LLM model.

The opencode-go regression tests pin a concrete model name so partial-name
matching, catalog lookups, and provider routing stay meaningful.  When the
env var ``AGENT_OPENCODE_MODEL`` is set (CI / .env), the helper returns that
value instead — giving one knob to retarget all model-aware tests at once.
"""
from __future__ import annotations

import os

# Concrete fixture model pinned by opencode-go regression tests.  The live
# catalog default is "opencode-go/deepseek-v4-flash"; these tests
# intentionally pin "hy3" so partial-name matching stays meaningful.
_DEFAULT_OPENCODE_MODEL = "opencode-go/mimo-v2.5"

# Concrete fixture model for the opencode-zen FREE tier (keyless).  This is a
# *different* model from _default_llm() (which is the go-mode keyed model), so
# it is pinned separately rather than derived from _default_llm().
_DEFAULT_ZEN_FREE_MODEL = "opencode-zen/mimo-v2.5-free"


def _default_llm() -> str:
    """Full opencode-go model name for tests.

    Honours ``AGENT_OPENCODE_MODEL`` when explicitly set (CI / .env), else
    falls back to the concrete fixture model.  Never returns *None*.
    """
    return os.environ.get("AGENT_OPENCODE_MODEL") or _DEFAULT_OPENCODE_MODEL


def _default_llm_short() -> str:
    """Model tail without the ``opencode-go/`` prefix (e.g. ``hy3``)."""
    return _default_llm().removeprefix("opencode-go/")


def _default_zen_free_model() -> str:
    """Full opencode-zen FREE tier model name for tests (keyless).

    Honours ``AGENT_ZEN_FREE_MODEL`` when explicitly set (CI / .env), else
    falls back to the concrete fixture model.  Never returns *None*.
    """
    return os.environ.get("AGENT_ZEN_FREE_MODEL") or _DEFAULT_ZEN_FREE_MODEL
