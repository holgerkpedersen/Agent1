"""Tests for the engine/accelerator abstraction (NPU + iGPU parallelism)."""
from __future__ import annotations

from types import SimpleNamespace

from agent_core.llm import engines


class TestDeviceMapping:
    def test_provider_devices(self):
        assert engines.device_for_provider("lemonade") == "npu"
        assert engines.device_for_provider("lmstudio") == "igpu"
        assert engines.device_for_provider("llama") == "igpu"
        assert engines.device_for_provider("opencode") == "cloud"
        assert engines.device_for_provider("openrouter") == "cloud"
        assert engines.device_for_provider("wat") == "cloud"

    def test_device_for_model(self):
        assert engines.device_for_model("lemonade/qwen3.5-4b-FLM") == "npu"
        assert engines.device_for_model("laguna-s-2.1") == "igpu"


class TestEngines:
    def test_resolve_engine(self):
        assert engines.resolve_engine("npu").provider == "lemonade"
        assert engines.resolve_engine("NPU").device == "npu"
        assert engines.resolve_engine("igpu").provider == "lmstudio"
        assert engines.resolve_engine("nope") is None

    def test_engine_model_defaults_without_probe(self):
        settings = SimpleNamespace(lemonade_model="qwen3.5-4b-FLM")
        assert engines.engine_model(engines.ENGINES["npu"], settings, probe=False) == (
            "lemonade/qwen3.5-4b-FLM"
        )

    def test_resolve_engines_with_overrides(self):
        settings = SimpleNamespace(
            lemonade_model="qwen3.5-4b-FLM",
            opencode_model="opencode-go/mimo-v2.5",
        )
        models, overrides, errors = engines.resolve_engines(
            ["npu", "cloud"], settings, probe=False
        )
        assert models == ["lemonade/qwen3.5-4b-FLM", "opencode-go/mimo-v2.5"]
        assert overrides["lemonade/qwen3.5-4b-FLM"] == "lemonade"
        assert overrides["opencode-go/mimo-v2.5"] == "opencode"
        assert errors == []

    def test_resolve_engines_reports_unknown_and_missing(self):
        models, overrides, errors = engines.resolve_engines(
            ["gpu"], None, probe=False
        )
        assert models == [] and overrides == {}
        assert errors and "unknown engine" in errors[0]

        models2, _, errors2 = engines.resolve_engines(["npu"], None, probe=False)
        assert models2 == []
        assert errors2 and "no model available" in errors2[0]
