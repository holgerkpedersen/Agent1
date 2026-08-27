"""Live integration tests for the llama-server lifecycle manager.

These tests start a *real* ``llama-server`` binary (router mode) on an isolated
port (8099) and exercise the genuine ``model switch -> relaunch -> verify``
path end-to-end -- the gap the mocked unit tests in ``test_llama_server.py``
leave behind.

Safety guarantees:
* They run only when a ``llama-server`` binary is discoverable on PATH / known
  install locations; otherwise the whole module is skipped (so CI on machines
  without llama.cpp stays green).
* They use a dedicated high port (8099), never 8080, so the user's own server
  is untouched.
* Teardown kills **only the PIDs this test spawned** (computed as the delta of
  llama-server PIDs before/after launch) and never calls the host-wide
  ``taskkill /IM`` fallback -- so even a co-located user server is safe.
* ``shutdown_server``'s ``_taskkill_by_image`` fallback is monkeypatched to a
  no-op inside these tests as defense-in-depth.

Run with: ``pytest -m integration`` (or the whole suite -- they skip elsewhere).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import psutil
import pytest

from agent_core.llm import llama_server as mod

# Isolated port so we never touch a server the user may run on 8080.
LIVE_API = "http://127.0.0.1:8099/v1"

_BIN_CANDIDATES = [
    r"C:\tools\llama-b10655-bin-win-vulkan-x64\llama-server.exe",
    r"C:\tools\llama\llama-server.exe",
    os.path.expanduser(r"~\tools\llama-server.exe"),
]


def _find_binary() -> str | None:
    on_path = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if on_path:
        return on_path
    for c in _BIN_CANDIDATES:
        if os.path.isfile(c):
            return c
    return None


_LLAMA_BIN = _find_binary()


def _current_llama_pids() -> set[int]:
    names = ("llama-server", "llama-server.exe", "llama-swap", "llama-swap.exe")
    return {
        p.info["pid"]
        for p in psutil.process_iter(["pid", "name", "cmdline"])
        if (p.info.get("cmdline") or [])
        and os.path.basename(str(p.info["cmdline"][0])).lower() in names
    }


def _live_args(api_url: str) -> list[str] | None:
    """Return the launch argv of the running instance (from GET /v1/models)."""
    status, body = mod._http_json("GET", f"{api_url}/models", timeout=8.0)
    if status != 200 or not isinstance(body, dict):
        return None
    for m in body.get("data") or []:
        if isinstance(m, dict):
            args = (m.get("status") or {}).get("args")
            if isinstance(args, list) and args:
                return [str(a) for a in args]
    return None


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_router():
    """Start a real router (no model) on port 8099; tear it down after tests.

    Yields ``(api_url, our_pids)``.  If the binary is missing or the launch
    fails, the module is skipped before any test runs.
    """
    if not _LLAMA_BIN:
        pytest.skip("llama-server binary not found on PATH or in known locations")

    # Defense-in-depth: never let teardown nuke other hosts' servers.
    import agent_core.llm.llama_server as ls
    original_taskkill = ls._taskkill_by_image
    ls._taskkill_by_image = lambda: False  # type: ignore[assignment]

    before = _current_llama_pids()
    ok, msg = mod._launch_server(
        LIVE_API, None, extra_args=["--ctx-size", "65536"]
    )
    if not ok:
        ls._taskkill_by_image = original_taskkill
        pytest.skip(f"could not launch live llama-server (router): {msg}")

    our_pids = sorted(_current_llama_pids() - before)
    assert our_pids, "launched router but could not identify its PID"

    try:
        yield LIVE_API, our_pids
    finally:
        # Graceful first; targeted kill only our own PIDs as a backstop.
        try:
            mod.shutdown_server(LIVE_API)
        except Exception:
            pass
        mod.kill_only_pids(our_pids)
        ls._taskkill_by_image = original_taskkill


def test_router_is_up_and_reports_role(live_router):
    api_url, _ = live_router
    assert mod.is_server_up(api_url) is True
    # We launched router mode (no --model, only --models-dir), so role == router.
    assert mod.get_role(api_url) == "router"


def test_extra_args_forwarded_to_live_process(live_router):
    """The --ctx-size we passed to _launch_server must appear in the real
    server's argv (proves tuning survives a launch against reality)."""
    api_url, _ = live_router
    args = _live_args(api_url)
    assert args is not None, "could not read live server argv via GET /v1/models"
    assert "--ctx-size" in args, f"--ctx-size not in live argv: {args}"
    assert "65536" in args, f"ctx-size value not in live argv: {args}"


def test_reconcile_loads_real_model_when_available(live_router):
    """If the host has a local GGUF, ensure_model_served should load it on the
    running router; otherwise this assertion is skipped (model-agnostic)."""
    from agent_core.llm.llama_provider import discover_local_gguf_models

    available = discover_local_gguf_models()
    if not available:
        pytest.skip("no local GGUF discoverable; cannot exercise real load path")

    api_url, _ = live_router
    model_id = available[0]  # e.g. "llama/unsloth/Qwen3-..."
    bare = model_id[len("llama/"):] if model_id.startswith("llama/") else model_id

    ok, msg = mod.ensure_model_served(api_url, model_id)
    assert ok is True, f"ensure_model_served failed: {msg}"
    served = mod.list_served_models(api_url)
    assert bare in served, f"'{bare}' not served after reconcile; served={served}"


def test_graceful_shutdown_frees_port(live_router):
    api_url, our_pids = live_router
    ok, msg = mod.shutdown_server(api_url)
    assert ok is True, f"shutdown reported failure: {msg}"
    # Give the OS a moment to release the socket.
    time.sleep(1.0)
    assert mod.is_server_up(api_url) is False, "port still answering after shutdown"
    # Backstop: ensure our PIDs are gone (no lingering process).
    still = [p for p in our_pids if psutil.pid_exists(p)]
    assert not still, f"our llama-server PIDs still alive after teardown: {still}"
