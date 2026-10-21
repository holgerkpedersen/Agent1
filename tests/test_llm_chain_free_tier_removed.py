"""Regression guard: the default LLM chain ships no free-tier models.

The opencode-zen free tier has been fully removed (code, config and docs):
every default provider chain — the catalog ``_defaults.llm_chain`` (source of
``DEFAULT_LLM_CHAIN``) and the documented ``AGENT_LLM_PROVIDERS`` sample in
``.env.example`` — must list only keyed ``opencode-go`` slots plus local /
openrouter fallbacks.  This test pins that configuration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_core.constants import DEFAULT_LLM_CHAIN
from agent_core.llm.provider import _split_entry

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Free-tier shapes that must never appear in a default chain model id.
_FREE_PREFIXES = ("opencode-zen/", "zen/")


def _assert_entry_is_paid(entry: str) -> None:
    """A chain entry is either bare (provider default) or a paid model id."""
    provider, model = _split_entry(entry)
    assert provider in ("opencode", "openrouter", "lmstudio", "llama"), entry
    if model is None:
        return
    assert not model.startswith(_FREE_PREFIXES), f"free-tier model in chain: {entry}"
    assert not model.endswith("-free"), f"free model in chain: {entry}"
    if provider == "opencode":
        # The remaining opencode slots are the keyed go tier only.
        assert model.startswith("opencode-go/"), f"non-go opencode slot: {entry}"


def test_default_llm_chain_has_no_free_tier_entries() -> None:
    assert DEFAULT_LLM_CHAIN, "default chain must not be empty"
    for entry in DEFAULT_LLM_CHAIN:
        _assert_entry_is_paid(entry)


def test_catalog_defaults_llm_chain_matches_constant() -> None:
    """model_catalog.json is the source of truth — check it directly too."""
    catalog = json.loads((_REPO_ROOT / "model_catalog.json").read_text("utf-8"))
    chain = catalog["_defaults"]["llm_chain"]
    assert chain, "catalog default chain must not be empty"
    assert tuple(chain) == DEFAULT_LLM_CHAIN
    for entry in chain:
        _assert_entry_is_paid(entry)


def test_env_example_sample_chain_has_no_free_tier_entries() -> None:
    """The documented AGENT_LLM_PROVIDERS sample must also be free of zen."""
    text = (_REPO_ROOT / ".env.example").read_text("utf-8")
    samples = [
        m.group(1).strip()
        for line in text.splitlines()
        if (m := re.match(r"#?\s*AGENT_LLM_PROVIDERS=(.+)", line))
    ]
    assert samples, "no AGENT_LLM_PROVIDERS sample found in .env.example"
    for sample in samples:
        assert "opencode-zen/" not in sample, sample
        assert not sample.endswith("-free"), sample
        for entry in sample.split(","):
            _assert_entry_is_paid(entry.strip())


def test_repo_env_chain_has_no_free_tier_entries() -> None:
    """The live repo ``.env`` chain (when present) must be free-tier free."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text("utf-8").splitlines():
        if line.startswith("AGENT_LLM_PROVIDERS="):
            sample = line.split("=", 1)[1].strip()
            for entry in sample.split(","):
                _assert_entry_is_paid(entry.strip())
