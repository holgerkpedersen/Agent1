"""Regression: the model catalog JSON is the single source of truth.

No concrete LLM name may be hardcoded as a fallback in Python code —
a missing, malformed, or incomplete model_catalog.json must fail loudly
at import time instead of silently falling back to a specific model.
"""
import importlib
import json

import pytest

import agent_core.constants as constants


@pytest.fixture(autouse=True)
def _restore_constants():
    """Reload the real constants module after each test.

    Tests below monkeypatch the catalog path and/or reload the module;
    always restore the pristine import state afterwards.
    """
    yield
    importlib.reload(constants)


def _write_catalog(path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _valid_catalog() -> dict:
    return {
        "_defaults": {
            "model": "model-a",
            "opencode_model": "opencode-go/model-b",
            "openrouter_model": "openrouter/some/model:free",
            "llm_chain": ["opencode:model-b", "lmstudio", "llama"],
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_api_base": "https://example.invalid/zen/go/v1",
            "opencode_zen_api_base": "https://example.invalid/zen/v1",
            "llama_base_url": "http://127.0.0.1:8080/v1",
            "openrouter_api_base": "https://example.invalid/api/v1",
        },
        "_routing": {"model-a": "lmstudio", "opencode": "opencode"},
        "_zen_free_tier_prefixes": ["opencode-zen/", "zen/"],
        "model-a": {"desc": "a", "max_tokens": 1000},
    }


class TestCatalogLoader:
    def test_missing_catalog_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            constants, "_MODEL_CATALOG_PATH", str(tmp_path / "nope.json")
        )
        with pytest.raises(RuntimeError, match="model catalog not found"):
            constants._load_model_catalog()

    def test_malformed_catalog_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        p.write_text("{ not valid json", encoding="utf-8")
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="malformed"):
            constants._load_model_catalog()

    def test_non_dict_root_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="root must be a JSON object"):
            constants._load_model_catalog()

    def test_missing_routing_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        del data["_routing"]
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="_routing"):
            constants._load_model_catalog()

    def test_empty_routing_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        data["_routing"] = {}
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="_routing"):
            constants._load_model_catalog()

    def test_missing_defaults_object_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        del data["_defaults"]
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="_defaults"):
            constants._load_model_catalog()

    @pytest.mark.parametrize(
        "key", ["model", "opencode_model", "openrouter_model"]
    )
    def test_missing_default_key_raises(self, tmp_path, monkeypatch, key):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        del data["_defaults"][key]
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match=f"_defaults\\.{key}"):
            constants._load_model_catalog()

    @pytest.mark.parametrize("chain", [None, []])
    def test_missing_llm_chain_raises(
        self, tmp_path, monkeypatch, chain
    ):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        if chain is None:
            del data["_defaults"]["llm_chain"]
        else:
            data["_defaults"]["llm_chain"] = chain
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="_defaults\\.llm_chain"):
            constants._load_model_catalog()

    @pytest.mark.parametrize(
        "key",
        [
            "opencode_server_url",
            "opencode_api_base",
            "opencode_zen_api_base",
            "llama_base_url",
            "openrouter_api_base",
        ],
    )
    def test_missing_endpoint_url_raises(self, tmp_path, monkeypatch, key):
        """Provider endpoint URLs are catalog data — a missing one must fail
        loudly instead of letting a hardcoded URL in Python win."""
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        del data["_defaults"][key]
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match=f"_defaults\\.{key}"):
            constants._load_model_catalog()

    def test_missing_zen_prefixes_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        del data["_zen_free_tier_prefixes"]
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="_zen_free_tier_prefixes"):
            constants._load_model_catalog()

    def test_non_object_thinking_gates_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        data["_thinking_gates"] = ["nemotron"]
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="_thinking_gates"):
            constants._load_model_catalog()

    def test_incomplete_thinking_gate_pair_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        data["_thinking_gates"] = {"somefamily": {"/think": "/think"}}
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        with pytest.raises(RuntimeError, match="_thinking_gates\\.somefamily"):
            constants._load_model_catalog()

    def test_thinking_gates_come_from_catalog(self, tmp_path, monkeypatch):
        """THINKING_GATES must mirror the catalog exactly — providers must
        not hardcode which model families gate reasoning."""
        p = tmp_path / "model_catalog.json"
        data = _valid_catalog()
        data["_thinking_gates"] = {
            "family-x": {"/think": "/think", "/no_think": "/no_think"},
            "family-y": {"/think": "ON", "/no_think": "OFF"},
        }
        _write_catalog(p, data)
        monkeypatch.setattr(constants, "_MODEL_CATALOG_PATH", str(p))
        loaded = constants._load_model_catalog()
        assert loaded["_thinking_gates"] == data["_thinking_gates"]


class TestDefaultsFromCatalog:
    """The public default constants must come from the catalog (or env),
    never from a name hardcoded in Python."""

    def test_defaults_match_catalog(self, monkeypatch):
        # Strip any ambient env overrides so the catalog values win.
        for var in ("AGENT_MODEL", "AGENT_OPENCODE_MODEL", "AGENT_OPENROUTER_MODEL"):
            monkeypatch.delenv(var, raising=False)
        importlib.reload(constants)
        assert constants.DEFAULT_MODEL == constants._DEFAULTS["model"]
        assert constants.DEFAULT_OPENCODE_MODEL == constants._DEFAULTS["opencode_model"]
        assert constants.DEFAULT_OPENROUTER_MODEL == constants._DEFAULTS["openrouter_model"]

    def test_env_overrides_still_win(self, monkeypatch):
        monkeypatch.setenv("AGENT_MODEL", "env-model-a")
        monkeypatch.setenv("AGENT_OPENCODE_MODEL", "env-model-b")
        monkeypatch.setenv("AGENT_OPENROUTER_MODEL", "env-model-c")
        importlib.reload(constants)
        assert constants.DEFAULT_MODEL == "env-model-a"
        assert constants.DEFAULT_OPENCODE_MODEL == "env-model-b"
        assert constants.DEFAULT_OPENROUTER_MODEL == "env-model-c"

    def test_zen_prefixes_come_from_catalog(self, monkeypatch):
        for var in ("AGENT_MODEL", "AGENT_OPENCODE_MODEL", "AGENT_OPENROUTER_MODEL"):
            monkeypatch.delenv(var, raising=False)
        importlib.reload(constants)
        # Whatever the catalog says, the code must mirror it exactly.
        with open(constants._MODEL_CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        assert constants._ZEN_TIER_PREFIXES == tuple(catalog["_zen_free_tier_prefixes"])
