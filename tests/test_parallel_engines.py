"""Tests for device-aware parallel dispatch (NPU + iGPU in one run)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_core.llm import context_budget, engines, parallel


class _Fake:
    def __init__(self, text: str, ctx: int | None = None):
        self.model_name = "fake"
        self.text = text
        self.context_limit = ctx
        self.last_response_metrics = None
        self.seen: list = []

    def apply_profile(self, *a) -> None:
        pass

    async def chat(self, messages, **kw):
        self.seen.append(list(messages))
        return self.text


def _patch_providers(monkeypatch, provs: dict):
    monkeypatch.setattr(
        parallel, "build_provider",
        lambda settings, m, provider_override=None: provs[m],
    )


def test_device_and_provider_override(monkeypatch):
    p_npu, p_gpu = _Fake("npu-ok"), _Fake("gpu-ok")
    _patch_providers(monkeypatch, {"lemonade/x": p_npu, "zai/glm": p_gpu})
    run = asyncio.run(parallel.run_parallel(
        [{"role": "user", "content": "hi"}],
        ["lemonade/x", "zai/glm"],
        settings=object(),
        provider_overrides={"lemonade/x": "lemonade"},
    ))
    assert [r.device for r in run.results] == ["npu", "igpu"]
    assert run.results[0].provider == "lemonade"
    assert run.results[1].provider == "lmstudio"


def test_small_context_trims_only_that_provider(monkeypatch):
    old = {"role": "assistant", "content": "y" * 4000}
    new = {"role": "user", "content": "z" * 4000}
    p_npu = _Fake("ok", ctx=2000)  # small NPU window -> oldest turn trimmed
    p_gpu = _Fake("ok", ctx=None)  # unknown iGPU window -> untouched
    _patch_providers(monkeypatch, {"lemonade/x": p_npu, "zai/glm": p_gpu})
    asyncio.run(parallel.run_parallel(
        [old, new], ["lemonade/x", "zai/glm"], settings=object(),
        provider_overrides={"lemonade/x": "lemonade"},
    ))
    assert any(m.get("content") == context_budget.TRIM_NOTE for m in p_npu.seen[0])
    assert [m["content"] for m in p_gpu.seen[0]] == [old["content"], new["content"]]


def test_multillm_npu_alias_expands():
    from agent_core.commands.multillm_cmd import _expand_model_aliases

    assert _expand_model_aliases(
        ["npu/qwen3.5-4b-FLM", "npu:Qwen3-4B-Hybrid", "laguna-s-2.1"]
    ) == ["lemonade/qwen3.5-4b-FLM", "lemonade/Qwen3-4B-Hybrid", "laguna-s-2.1"]


def test_tool_loop_trims_per_iteration(monkeypatch):
    """The tool loop re-trims each iteration so a small NPU window is safe."""
    prov = _Fake("done", ctx=2000)
    hist = [
        {"role": "assistant", "content": "y" * 4000},
        {"role": "user", "content": "z" * 4000},
    ]
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]

    async def fake_execute(name, args):
        return "tool-result"

    from agent_core.llm import parallel as _p
    monkeypatch.setattr(_p, "build_provider", lambda *a, **k: prov)
    # ToolLoopRunner.run is the real one; the fake provider returns plain text
    # after its (trimmed) first call, so the loop makes exactly one LLM call.
    run = asyncio.run(_p.run_parallel(
        hist, ["lemonade/x", "zai/glm"], settings=object(),
        provider_overrides={"lemonade/x": "lemonade"},
        tools=tools, execute_tool_fn=fake_execute,
    ))
    # Both providers get the fake; assert trimming happened on the NPU call.
    assert any(
        m.get("content") == context_budget.TRIM_NOTE
        for m in prov.seen[0]
    )


@pytest.mark.anyio
async def test_multillm_engines_flag_expands_to_models(monkeypatch):
    from agent_core.commands.multillm_cmd import MultiLlmCommand

    captured: dict = {}

    class _Run:
        template_id = "parallel"
        results: list = []

        def consensus(self, *a) -> str:
            return ""

    async def fake_run(messages, models, settings, **kw):
        captured["models"] = list(models)
        captured["overrides"] = kw.get("provider_overrides")
        return _Run()

    monkeypatch.setattr(parallel, "run_parallel", fake_run)
    monkeypatch.setattr(
        engines, "resolve_engines",
        lambda names, settings, probe=True: (
            ["lemonade/qwen3.5-4b-FLM", "opencode-go/mimo-v2.5"],
            {"lemonade/qwen3.5-4b-FLM": "lemonade"},
            [],
        ),
    )
    agent = SimpleNamespace(mode="build")
    ok = await MultiLlmCommand().execute(
        ["what is 2+2", "--engines", "npu,cloud"], agent
    )
    assert ok
    assert captured["models"] == ["lemonade/qwen3.5-4b-FLM", "opencode-go/mimo-v2.5"]
    assert captured["overrides"] == {"lemonade/qwen3.5-4b-FLM": "lemonade"}


def test_summarize_includes_device(monkeypatch):
    p_npu, p_gpu = _Fake("a"), _Fake("b")
    _patch_providers(monkeypatch, {"lemonade/x": p_npu, "zai/glm": p_gpu})
    run = asyncio.run(parallel.run_parallel(
        [{"role": "user", "content": "hi"}], ["lemonade/x", "zai/glm"],
        settings=object(), provider_overrides={"lemonade/x": "lemonade"},
    ))
    text = parallel.summarize(run)
    assert "npu/lemonade" in text
    assert "igpu/lmstudio" in text
