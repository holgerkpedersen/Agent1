"""Regression tests for the llama-server lifecycle manager (model reconcile).

These tests mock the HTTP/process layer so they are deterministic and never
touch a real llama-server.  They encode the fix for "the agent silently uses
whatever the pre-started server serves instead of the requested model":

* a router that can resolve the id -> POST /models/load (+ unload old)
* a router that cannot resolve it (no --models-dir) -> relaunch with
  --models-dir then load
* no server at all -> launch a router and load the model
"""
from types import SimpleNamespace

import pytest

from agent_core.llm import llama_server as mod


def _router_props() -> dict:
    return {"role": "router", "max_instances": 4, "models_autoload": True}


def _single_props() -> dict:
    return {"role": "model", "max_instances": 1}


def _models_payload(ids):
    return {"data": [{"id": i, "status": {"value": "loaded"}} for i in ids]}


class _FakeHTTP:
    """Replaces mod._http_json with a programmable in-memory server."""

    def __init__(self, props, models, load_ok=True, load_status=200):
        self.props = props
        self.models = list(models)          # currently served ids
        self.load_ok = load_ok
        self.load_status = load_status
        self.calls = []                     # (method, path, payload)
        self.launched = False

    def __call__(self, method, url, payload=None, timeout=10.0):
        self.calls.append((method, url, payload))
        if url.endswith("/props"):
            return 200, self.props
        if url.endswith("/models") and method == "GET":
            return 200, _models_payload(self.models)
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/models/load"):
            if self.load_ok:
                if payload and payload.get("model") not in self.models:
                    self.models.append(payload["model"])
                return self.load_status, {"success": True}
            return 404, {"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}}
        if url.endswith("/models/unload"):
            if payload and payload.get("model") in self.models:
                self.models.remove(payload["model"])
            return 200, {"success": True}
        if url.endswith("/shutdown"):
            return 200, {}
        return 404, {"error": {"message": "nf"}}


@pytest.fixture
def patch_http(monkeypatch):
    def _install(fake):
        monkeypatch.setattr(mod, "_http_json", fake)
        return fake
    return _install


class TestEnsureModelServed:
    def test_already_served_is_noop(self, patch_http):
        fake = _FakeHTTP(_router_props(), ["Bonsai-27B-Q1_0"])
        patch_http(fake)
        ok, msg = mod.ensure_model_served("http://x/v1", "llama/Bonsai-27B-Q1_0")
        assert ok is True
        assert "already serves" in msg
        # No load/unload posted.
        assert not any(c[1].endswith("/models/load") for c in fake.calls)

    def test_router_loads_and_unloads_old(self, patch_http):
        fake = _FakeHTTP(_router_props(), ["gemma-4b"])
        patch_http(fake)
        ok, msg = mod.ensure_model_served("http://x/v1", "llama/Bonsai-27B-Q1_0")
        assert ok is True
        assert "Bonsai-27B-Q1_0" in fake.models
        assert "gemma-4b" not in fake.models
        assert any(c[1].endswith("/models/load") for c in fake.calls)
        assert any(c[1].endswith("/models/unload") for c in fake.calls)

    def test_router_relaunch_when_cannot_resolve(self, patch_http, monkeypatch):
        # Router returns 404 on load (no --models-dir) -> must shutdown + relaunch.
        fake = _FakeHTTP(_router_props(), ["gemma-4b"], load_ok=False)
        patch_http(fake)
        # Prevent the OS-global kill/wait from touching real processes or hanging.
        monkeypatch.setattr(mod, "_taskkill_by_image", lambda: True)
        monkeypatch.setattr(mod, "running_server_pids", lambda api_url: [])
        monkeypatch.setattr(mod, "is_server_up", lambda api_url: False)
        launched = {}
        def _fake_launch(api_url, bare, extra_args=None):
            launched["api_url"] = api_url
            launched["bare"] = bare
            # Simulate a freshly launched server that CAN serve the model,
            # but DON'T recurse into the real _launch_server.
            fake.props = _router_props()
            fake.models = [bare]
            fake.load_ok = True
            return True, f"launched and serving '{bare}'"
        monkeypatch.setattr(mod, "_launch_server", _fake_launch)
        monkeypatch.setattr(mod, "server_binary_path", lambda api_url: "llama-server.exe")
        monkeypatch.setattr(mod, "_wait_until_up", lambda api_url, timeout=0: True)
        ok, msg = mod.ensure_model_served("http://x/v1", "llama/Bonsai-27B-Q1_0")
        assert ok is True
        # Relaunch path: _launch_server was invoked (not just a dynamic load).
        assert launched.get("bare") == "Bonsai-27B-Q1_0"
        assert "shutdown" in msg or launched, "server should have been relaunched"

    def test_single_model_server_relaunches(self, patch_http, monkeypatch):
        fake = _FakeHTTP(_single_props(), ["gemma-4b"])
        patch_http(fake)
        monkeypatch.setattr(mod, "_taskkill_by_image", lambda: True)
        monkeypatch.setattr(mod, "running_server_pids", lambda api_url: [])
        monkeypatch.setattr(mod, "is_server_up", lambda api_url: False)
        monkeypatch.setattr(mod, "_wait_until_up", lambda api_url, timeout=0: True)
        launched = {}
        def _fake_launch(api_url, bare, extra_args=None):
            launched["bare"] = bare
            fake.props = _single_props()
            fake.models = [bare]
            return True, "launched"
        monkeypatch.setattr(mod, "_launch_server", _fake_launch)
        monkeypatch.setattr(mod, "server_binary_path", lambda api_url: "llama-server.exe")
        ok, msg = mod.ensure_model_served("http://x/v1", "llama/Bonsai-27B-Q1_0")
        assert ok is True
        assert launched.get("bare") == "Bonsai-27B-Q1_0"
        assert "shutdown" in msg or launched, "server should have been relaunched"


class TestToolCallParserRestart:
    """ensure_model_served restarts a server that lacks --tool-call-parser."""

    def test_restart_when_parser_missing(self, patch_http, monkeypatch):
        """Server serves Qwen3 but was launched without --tool-call-parser."""
        # Server is up, serves the right model.
        fake = _FakeHTTP(_router_props(), ["Qwen3.8-Flash-Next"])
        patch_http(fake)

        # Patch _server_has_tool_call_parser to return False (missing parser).
        monkeypatch.setattr(mod, "_server_has_tool_call_parser",
                            lambda api_url: False)
        # Prevent real shutdown/launch.
        monkeypatch.setattr(mod, "shutdown_server",
                            lambda api_url: (True, "stopped"))
        launched = {}
        def _fake_launch(api_url, bare, extra_args=None):
            launched["bare"] = bare
            return True, f"relaunched with parser for '{bare}'"
        monkeypatch.setattr(mod, "_launch_server", _fake_launch)

        ok, msg = mod.ensure_model_served(
            "http://x/v1", "llama/Qwen3.8-Flash-Next")
        assert ok is True
        assert launched.get("bare") == "Qwen3.8-Flash-Next"
        assert "relaunched" in msg.lower() or "parser" in msg.lower()

    def test_no_restart_when_parser_present(self, patch_http, monkeypatch):
        """Server serves Qwen3 and already has --tool-call-parser."""
        fake = _FakeHTTP(_router_props(), ["Qwen3.8-Flash-Next"])
        patch_http(fake)
        monkeypatch.setattr(mod, "_server_has_tool_call_parser",
                            lambda api_url: True)
        ok, msg = mod.ensure_model_served(
            "http://x/v1", "llama/Qwen3.8-Flash-Next")
        assert ok is True
        assert "already serves" in msg

    def test_no_restart_when_model_unknown(self, patch_http, monkeypatch):
        """Unknown model family — no parser needed, no restart."""
        fake = _FakeHTTP(_router_props(), ["some-random-model"])
        patch_http(fake)
        monkeypatch.setattr(mod, "_server_has_tool_call_parser",
                            lambda api_url: False)
        ok, msg = mod.ensure_model_served(
            "http://x/v1", "llama/some-random-model")
        assert ok is True
        assert "already serves" in msg


class TestResolveLocalGguf:
    def test_maps_routing_label_to_local_file(self, monkeypatch):
        # Point the llama models dir (NOT the LM Studio dir) at a temp tree.
        import os, tempfile
        from agent_core.llm import llama_provider
        d = tempfile.mkdtemp()
        rel = os.path.join(d, "lmstudio-community", "Bonsai-27B-GGUF")
        os.makedirs(rel)
        gguf = os.path.join(rel, "Bonsai-27B-Q1_0.gguf")
        with open(gguf, "w") as f:
            f.write("x")
        monkeypatch.setattr(llama_provider, "_llama_models_dir", lambda: d)
        monkeypatch.setattr(llama_provider, "_lmstudio_models_dir", lambda: None)
        got = mod._resolve_local_gguf("lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0")
        assert got is not None and got.replace(os.sep, "/").endswith(
            "lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0.gguf")

    def test_returns_none_when_missing(self, monkeypatch):
        from agent_core.llm import llama_provider
        monkeypatch.setattr(llama_provider, "_llama_models_dir", lambda: None)
        assert mod._resolve_local_gguf("nope/Bonsai") is None

    def test_never_reads_lmstudio_dir(self, monkeypatch):
        """A GGUF in the LM Studio models dir must NOT resolve for llama."""
        import os, tempfile
        from agent_core.llm import llama_provider
        d = tempfile.mkdtemp()
        rel = os.path.join(d, "lmstudio-community", "Bonsai-27B-GGUF")
        os.makedirs(rel)
        with open(os.path.join(rel, "Bonsai-27B-Q1_0.gguf"), "w") as f:
            f.write("x")
        monkeypatch.setattr(llama_provider, "_lmstudio_models_dir", lambda: d)
        monkeypatch.setattr(llama_provider, "_llama_models_dir", lambda: None)
        assert mod._resolve_local_gguf(
            "lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0") is None


class TestLaunchServer:
    def test_single_model_when_local_gguf_resolves(self, monkeypatch):
        """A local GGUF -> single-model launch with --model + --alias (robust)."""
        launched = {}
        monkeypatch.setattr(mod, "server_binary_path", lambda api_url: "llama-server.exe")
        monkeypatch.setattr(mod, "_wait_until_up", lambda api_url, timeout=0: True)
        monkeypatch.setattr(mod, "_resolve_local_gguf",
                            lambda bare: r"C:\models\Bonsai-27B-Q1_0.gguf")
        def _fake_popen(cmd, **kw):
            launched["cmd"] = list(cmd)
            return None
        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
        ok, msg = mod._launch_server("http://127.0.0.1:8080/v1", "Bonsai-27B-Q1_0")
        assert ok is True
        assert "--model" in launched["cmd"]
        assert r"C:\models\Bonsai-27B-Q1_0.gguf" in launched["cmd"]
        assert "--alias" in launched["cmd"]
        assert "Bonsai-27B-Q1_0" in launched["cmd"]
        assert "--models-dir" not in launched["cmd"]

    def test_router_when_no_local_gguf(self, monkeypatch):
        """No local GGUF -> router launch with --models-dir + dynamic load."""
        launched = {}
        monkeypatch.setattr(mod, "server_binary_path", lambda api_url: "llama-server.exe")
        monkeypatch.setattr(mod, "_wait_until_up", lambda api_url, timeout=0: True)
        monkeypatch.setattr(mod, "_resolve_local_gguf", lambda bare: None)
        monkeypatch.setattr(mod, "_models_dir_for_launch", lambda: r"C:\models")
        monkeypatch.setattr(mod, "_dynamic_load", lambda api_url, bare, served: (True, "ok"))
        def _fake_popen(cmd, **kw):
            launched["cmd"] = list(cmd)
            return None
        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
        ok, msg = mod._launch_server("http://127.0.0.1:8080/v1", "ggml-org/gemma-4b")
        assert ok is True
        assert "--models-dir" in launched["cmd"]
        assert "--model" not in launched["cmd"]


class TestAgentReconcileHook:
    def test_reconcile_invokes_server_manager_for_llama(self, monkeypatch):
        """LLMClient._reconcile_llama_model calls ensure_model_served for a
        llama model and pins the served id on the provider."""
        from agent_core.llm.llama_provider import LlamaProvider

        called = {}
        monkeypatch.setattr(
            mod, "ensure_model_served",
            lambda api_url, model_name: called.update(api_url=api_url, model=model_name) or (True, "ok"),
        )
        # Server is serving a DIFFERENT model so the reconcile hook calls
        # ensure_model_served to switch it.  After the switch, the re-check
        # returns the correct model.
        call_count = {"n": 0}
        def _list_served(api_url):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ["OldModel"]
            return ["Bonsai-27B-Q1_0"]
        monkeypatch.setattr(mod, "list_served_models", _list_served)

        # Build a real LLMClient but stub out the heavy init pieces so we only
        # exercise the reconcile hook against the in-memory server manager.
        from agent import LLMClient
        monkeypatch.setattr("agent_core.constants.resolve_model", lambda e=None: "llama/Bonsai-27B-Q1_0")
        monkeypatch.setattr("agent_core.config.load_agent_settings",
                            lambda: SimpleNamespace(llm_provider="llama"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})
        provider = LlamaProvider(model_name="llama/Bonsai-27B-Q1_0",
                                 api_url="http://127.0.0.1:8080/v1")
        monkeypatch.setattr("agent_core.llm.provider.build_provider",
                            lambda settings, name: provider)

        client = LLMClient("llama/Bonsai-27B-Q1_0")
        assert called.get("model") == "llama/Bonsai-27B-Q1_0"
        assert called.get("api_url") == "http://127.0.0.1:8080/v1"
        # The served id is pinned on the provider so chat uses the right model.
        assert provider._cached_server_model_id == "Bonsai-27B-Q1_0"

    def test_reconcile_is_noop_for_non_llama(self, monkeypatch):
        from agent import LLMClient
        monkeypatch.setattr("agent_core.constants.resolve_model", lambda e=None: "laguna-s-2.1")
        monkeypatch.setattr("agent_core.config.load_agent_settings",
                            lambda: SimpleNamespace(llm_provider="lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})
        from agent_core.llm.lmstudio import LMStudioProvider
        monkeypatch.setattr("agent_core.llm.provider.build_provider",
                            lambda settings, name: LMStudioProvider(model_name=name))
        called = {}
        monkeypatch.setattr(mod, "ensure_model_served",
                            lambda api_url, model_name: called.update(x=1) or (True, "ok"))
        client = LLMClient("laguna-s-2.1")
        assert "x" not in called  # llama manager must NOT be invoked


class TestToolCallParserDetection:
    """_tool_call_parser_for_model returns the right --tool-call-parser value."""

    def test_qwen3_flash_next(self):
        assert mod._tool_call_parser_for_model(
            "Qwen/Qwen3.8-Flash-Next") == "qwen3_coder"

    def test_qwen3_variant_paths(self):
        assert mod._tool_call_parser_for_model(
            "llama//home/holger/.huggingface/Qwen3.8-Flash-Next-UD-IQ3_XXS.gguf"
        ) == "qwen3_coder"

    def test_qwen4_experimental(self):
        assert mod._tool_call_parser_for_model(
            "Qwen/Qwen4-Experimental-Next") == "qwen3_coder"

    def test_qwen2(self):
        assert mod._tool_call_parser_for_model(
            "Qwen/Qwen2.5-7B-Instruct") == "qwen2"

    def test_llama3(self):
        assert mod._tool_call_parser_for_model(
            "meta-llama/Meta-Llama-3-8B-Instruct") == "llama3"

    def test_llama4(self):
        assert mod._tool_call_parser_for_model(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct") == "llama3"

    def test_mistral(self):
        assert mod._tool_call_parser_for_model(
            "mistralai/Mistral-7B-Instruct-v0.3") == "mistral"

    def test_mixtral(self):
        assert mod._tool_call_parser_for_model(
            "mistralai/Mixtral-8x7B-Instruct-v0.1") == "mistral"

    def test_command_r(self):
        assert mod._tool_call_parser_for_model(
            "CohereForAI/c4ai-command-r-v01") == "command-r"

    def test_unknown_model_returns_none(self):
        assert mod._tool_call_parser_for_model("gpt-4o") is None
        assert mod._tool_call_parser_for_model("some-random-model") is None


class TestLaunchServerAutoParser:
    """_launch_server injects --tool-call-parser when model family is known."""

    def test_qwen3_gets_parser(self, monkeypatch):
        launched = {}
        monkeypatch.setattr(mod, "server_binary_path", lambda api_url: "llama-server.exe")
        monkeypatch.setattr(mod, "_wait_until_up", lambda api_url, timeout=0: True)
        monkeypatch.setattr(mod, "_resolve_local_gguf", lambda bare: None)
        monkeypatch.setattr(mod, "_models_dir_for_launch", lambda: r"C:\models")
        monkeypatch.setattr(mod, "_dynamic_load", lambda api_url, bare, served: (True, "ok"))
        def _fake_popen(cmd, **kw):
            launched["cmd"] = list(cmd)
            return None
        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
        ok, msg = mod._launch_server(
            "http://127.0.0.1:8080/v1",
            "Qwen/Qwen3.8-Flash-Next")
        assert ok is True
        assert "--tool-call-parser" in launched["cmd"]
        idx = launched["cmd"].index("--tool-call-parser")
        assert launched["cmd"][idx + 1] == "qwen3_coder"

    def test_user_extra_args_override_not_duplicated(self, monkeypatch):
        """If the user already supplies --tool-call-parser in llama_extra_args,
        we must NOT add a second one."""
        launched = {}
        monkeypatch.setattr(mod, "server_binary_path", lambda api_url: "llama-server.exe")
        monkeypatch.setattr(mod, "_wait_until_up", lambda api_url, timeout=0: True)
        monkeypatch.setattr(mod, "_resolve_local_gguf", lambda bare: None)
        monkeypatch.setattr(mod, "_models_dir_for_launch", lambda: r"C:\models")
        monkeypatch.setattr(mod, "_dynamic_load", lambda api_url, bare, served: (True, "ok"))
        monkeypatch.setattr(mod, "_llama_extra_args",
                            lambda: ["--tool-call-parser", "llama3"])
        def _fake_popen(cmd, **kw):
            launched["cmd"] = list(cmd)
            return None
        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
        ok, msg = mod._launch_server(
            "http://127.0.0.1:8080/v1",
            "Qwen/Qwen3.8-Flash-Next")
        assert ok is True
        assert launched["cmd"].count("--tool-call-parser") == 1

    def test_unknown_model_no_parser(self, monkeypatch):
        launched = {}
        monkeypatch.setattr(mod, "server_binary_path", lambda api_url: "llama-server.exe")
        monkeypatch.setattr(mod, "_wait_until_up", lambda api_url, timeout=0: True)
        monkeypatch.setattr(mod, "_resolve_local_gguf", lambda bare: None)
        monkeypatch.setattr(mod, "_models_dir_for_launch", lambda: r"C:\models")
        monkeypatch.setattr(mod, "_dynamic_load", lambda api_url, bare, served: (True, "ok"))
        def _fake_popen(cmd, **kw):
            launched["cmd"] = list(cmd)
            return None
        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
        ok, msg = mod._launch_server(
            "http://127.0.0.1:8080/v1",
            "some-unknown-model")
        assert ok is True
        assert "--tool-call-parser" not in launched["cmd"]
