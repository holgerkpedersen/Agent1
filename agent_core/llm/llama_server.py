"""llama-server lifecycle manager (model reconciliation for the agent).

The agent's ``model_name`` (e.g. ``llama/lmstudio-community/Bonsai-27B-GGUF/
Bonsai-27B-Q1_0``) is a *routing label*.  The running ``llama-server`` is what
physically serves a model.  This module reconciles the two so the agent can
"handle the model directly" instead of silently using whatever the server
happens to be serving:

* If the running server is a **router** (``GET /props`` -> ``role:"router"``)
  it can dynamically load/unload models via ``POST /models/load`` and
  ``POST /models/unload``.  We use that path when the router can resolve the
  requested model id (local GGUF or HF repo).
* If the router is running but cannot resolve the model (e.g. it was started
  *without* ``--models-dir`` and the id is a local file), we **relaunch** the
  router with ``--models-dir`` pointed at the LM Studio models directory so it
  can resolve local GGUFs, then load the requested model.
* If no server is running, we launch one (router mode) at the configured URL.

The server binary path is recovered from the running instance's argv (exposed
by ``GET /v1/models`` -> ``status.args``), falling back to ``llama-server`` on
PATH / common install locations.  This avoids hard-coding a path.

All public functions are best-effort and never raise: they return a
``(ok, message)`` tuple so callers degrade gracefully when a server cannot be
managed (e.g. on a host without llama.cpp installed).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _llama_extra_args() -> list[str]:
    """Parse the user's persisted ``llama_extra_args`` setting into argv tokens.

    These are extra CLI flags (e.g. "--gpu-layers 999 --ctx-size 262144") that
    should be appended to any auto-launched / relaunched llama-server so the
    user's manual tuning survives a model switch.  Returns [] when unset.
    """
    raw = ""
    try:
        from agent_core.config import load_agent_settings
        raw = (getattr(load_agent_settings(), "llama_extra_args", "") or "").strip()
    except Exception:
        return []
    if not raw:
        return []
    # Split on whitespace; keep quoted segments intact where possible.
    try:
        import shlex
        return shlex.split(raw)
    except Exception:
        return raw.split()


# How long to wait for a freshly (re)started server to begin answering.
_LAUNCH_TIMEOUT_S = 60.0
_POLL_INTERVAL_S = 0.5


# ---------------------------------------------------------------------------
#  HTTP helpers (no auth requirement for a local server)
# ---------------------------------------------------------------------------

def _http_json(method: str, url: str, payload: dict[str, Any] | None = None,
               timeout: float = 10.0) -> tuple[int | None, Any]:
    """GET/POST JSON.  Returns ``(status_or_None, parsed_or_raw_text)``."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except Exception as exc:  # noqa: BLE001 - urllib.error.URLError etc.
        return None, repr(exc)


def _server_base_url(api_url: str) -> str:
    if api_url.endswith("/v1"):
        return api_url[:-3]
    return api_url


# ---------------------------------------------------------------------------
#  Server introspection
# ---------------------------------------------------------------------------

def is_server_up(api_url: str) -> bool:
    base = _server_base_url(api_url)
    status, _ = _http_json("GET", f"{base}/health", timeout=3.0)
    if status == 200:
        return True
    # /health may be absent on older builds; fall back to /v1/models.
    status, _ = _http_json("GET", f"{api_url}/models", timeout=3.0)
    return status == 200


def get_role(api_url: str) -> str | None:
    """Return the server role ('router' | 'model' | None) via ``GET /props``."""
    base = _server_base_url(api_url)
    status, body = _http_json("GET", f"{base}/props", timeout=5.0)
    if status == 200 and isinstance(body, dict):
        role = body.get("role")
        return str(role) if role else None
    return None


def list_served_models(api_url: str) -> list[str]:
    """Return bare model ids currently registered on the server."""
    status, body = _http_json("GET", f"{api_url}/models", timeout=8.0)
    if status != 200 or not isinstance(body, dict):
        return []
    items = body.get("data") or []
    return [m["id"] for m in items if isinstance(m, dict) and m.get("id")]


def server_binary_path(api_url: str) -> str | None:
    """Best-effort recovery of the llama-server binary path from running argv.

    ``GET /v1/models`` exposes each instance's launch argv under
    ``status.args``; the first element is the executable.  Falls back to
    ``llama-server`` on PATH and a few common Windows install locations.
    """
    status, body = _http_json("GET", f"{api_url}/models", timeout=8.0)
    if status == 200 and isinstance(body, dict):
        for m in (body.get("data") or []):
            if isinstance(m, dict):
                args = (m.get("status") or {}).get("args")
                if isinstance(args, list) and args and isinstance(args[0], str):
                    if os.path.isfile(args[0]):
                        return args[0]
    # Fallbacks.
    import shutil
    on_path = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if on_path:
        return on_path
    candidates = [
        r"C:\tools\llama-b10655-bin-win-vulkan-x64\llama-server.exe",
        r"C:\tools\llama\llama-server.exe",
        os.path.expanduser(r"~\tools\llama-server.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ---------------------------------------------------------------------------
#  Model reconciliation
# ---------------------------------------------------------------------------

def _bare_id(model_name: str) -> str:
    return model_name[len("llama/"):] if model_name.startswith("llama/") else model_name


def _resolve_local_gguf(bare_id: str) -> str | None:
    """Map a routing label to a local GGUF file path, if one exists.

    Resolves against the llama.cpp models dir ONLY (``_llama_models_dir``) —
    never the LM Studio models dir, so the two engines keep separate model
    folders.  Returns the absolute path of the first matching ``.gguf``
    (ignoring mmproj files).
    """
    from .llama_provider import _llama_models_dir

    models_dir = _llama_models_dir()
    if not models_dir:
        return None
    # bare_id is like "lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0"
    rel = bare_id.replace("/", os.sep)
    # Try exact path first (with and without .gguf).
    for cand in (os.path.join(models_dir, rel),
                 os.path.join(models_dir, rel + ".gguf")):
        if os.path.isfile(cand):
            return cand
    # Fall back to a walk (handles sharded / nested layouts).
    want = bare_id.lower()
    for root, _dirs, files in os.walk(models_dir):
        for f in files:
            if not f.lower().endswith(".gguf") or f.lower().startswith("mmproj"):
                continue
            stem = f[:-5]
            key = os.path.relpath(os.path.join(root, stem), models_dir).replace(os.sep, "/").lower()
            if key == want or key == want.replace("/", "/") + "-00001-of" in key:
                return os.path.join(root, f)
    return None


def ensure_model_served(
    api_url: str, model_name: str, *, extra_args: list[str] | None = None
) -> tuple[bool, str]:
    """Make the running llama-server serve *model_name* (best effort).

    Steps:
      1. If the server already serves the bare id, we're done.
      2. If it's a router that can resolve the id, ``POST /models/load`` it
         (and unload the previously served model to free VRAM).
      3. Otherwise relaunch the router with ``--models-dir`` so local GGUFs
         resolve, then load the requested model.
      4. If no server is running, launch one (router mode) and load the model.

    Returns ``(ok, message)``.  ``ok`` is True when the requested model is
    confirmed served (or was already served).  Never raises.
    """
    bare = _bare_id(model_name)
    if is_server_up(api_url):
        served = list_served_models(api_url)
        if bare in served:
            return True, f"llama-server already serves '{bare}'"
        role = get_role(api_url)
        if role == "router":
            ok, msg = _dynamic_load(api_url, bare, served)
            if ok:
                return True, msg
            # Router couldn't resolve it (e.g. no --models-dir). Relaunch.
            print(f"  [llama] router could not load '{bare}' ({msg}); "
                  "relaunching router with --models-dir")
            shut = shutdown_server(api_url)
            if not shut[0]:
                return False, f"could not stop existing server to relaunch: {shut[1]}"
        else:
            # Single-model server serving the wrong model: must relaunch.
            print(f"  [llama] server serves '{served[0] if served else 'none'}'; "
                  f"relaunching to serve '{bare}'")
            shutdown_server(api_url)

    return _launch_server(api_url, bare, extra_args=extra_args)


def _dynamic_load(api_url: str, bare: str, currently_served: list[str]) -> tuple[bool, str]:
    """Load *bare* via POST /models/load on a router; unload others first.

    Tries, in order:
      1. the bare id as registered (HF repo id / --alias),
      2. the absolute local GGUF path (works once the router has --models-dir
         or the file is in the HF cache).
    """
    base = _server_base_url(api_url)
    candidates = [bare]
    local_path = _resolve_local_gguf(bare)
    if local_path:
        candidates.append(local_path)

    last_err = f"HTTP {None}"
    for model_ref in candidates:
        status, body = _http_json("POST", f"{base}/models/load",
                                  {"model": model_ref}, timeout=30.0)
        if status == 200 and isinstance(body, dict) and body.get("success"):
            # Unload previously-served models to free VRAM (keep the one we want).
            for old in currently_served:
                if old != bare:
                    _http_json("POST", f"{base}/models/unload",
                               {"model": old}, timeout=20.0)
            return True, f"loaded '{bare}' on running router"
        last_err = (body.get("error", {}).get("message")
                    if isinstance(body, dict) else str(body)) or f"HTTP {status}"
    return False, last_err
    err = body.get("error", {}).get("message") if isinstance(body, dict) else str(body)
    return False, err or f"HTTP {status}"


def running_server_pids(api_url: str) -> list[int]:
    """Recover PIDs of the running llama-server processes via ``GET /v1/models``.

    Each instance's launch argv is exposed under ``status.args``; we match the
    executable name to find the owning PID.  Best-effort: returns [] on any
    failure (callers fall back to ``POST /shutdown``).
    """
    status, body = _http_json("GET", f"{api_url}/models", timeout=8.0)
    if status != 200 or not isinstance(body, dict):
        return []
    pids: list[int] = []
    try:
        import psutil
    except Exception:
        return pids
    names = ("llama-server", "llama-server.exe", "llama-swap", "llama-swap.exe")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cli = proc.info.get("cmdline") or []
            if cli and os.path.basename(str(cli[0])).lower() in names:
                pids.append(proc.info["pid"])
        except Exception:
            continue
    return pids


def _kill_pids(pids: list[int]) -> None:
    import signal
    for pid in pids:
        try:
            if os.name == "nt":
                # Use taskkill to terminate the whole process tree on Windows.
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F", "/T"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def kill_only_pids(pids: list[int]) -> None:
    """Public, targeted kill of *only* the given llama-server PIDs.

    Unlike ``_taskkill_by_image`` (which kills every llama-server on the host),
    this terminates just the supplied PIDs (and their process trees on Windows).
    Used by the live integration tests so teardown never touches a server the
    user started themselves.
    """
    _kill_pids([int(p) for p in pids if p])


def _taskkill_by_image() -> bool:
    """Kill every llama-server / llama-swap process tree (Windows).

    Returns True if a kill was attempted (best effort).  Used to free a port
    that a lingering server still holds so a relaunch can bind it.
    """
    if os.name != "nt":
        return False
    killed = False
    for img in ("llama-server.exe", "llama-swap.exe"):
        try:
            r = subprocess.run(["taskkill", "/IM", img, "/F", "/T"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=15)
            if r.returncode == 0:
                killed = True
        except Exception:
            pass
    return killed


def shutdown_server(api_url: str) -> tuple[bool, str]:
    """Stop the running llama-server (best effort).

    1. ``POST /shutdown`` (graceful, the documented unload endpoint).
    2. If the port is still occupied, kill the server process tree
       (``taskkill /IM`` on Windows, or SIGTERM by PID via psutil) so a
       relaunch can bind the port.

    Returns ``(ok, message)``.  ``ok`` is True when the port is no longer
    serving (or was already free).
    """
    base = _server_base_url(api_url)
    status, _ = _http_json("POST", f"{base}/shutdown", timeout=10.0)
    time.sleep(1.0)
    if not is_server_up(api_url):
        return True, "llama-server stopped"

    # Port still held: kill the process tree so a relaunch can bind it.
    _taskkill_by_image()
    pids = running_server_pids(api_url)
    if pids:
        _kill_pids(pids)

    # Wait for the port to actually free up.
    for _ in range(int(_LAUNCH_TIMEOUT_S / _POLL_INTERVAL_S)):
        if not is_server_up(api_url):
            return True, "llama-server stopped (killed)"
        time.sleep(_POLL_INTERVAL_S)

    if status in (200, 404) or status is None:
        return True, "llama-server stopped"
    return False, "shutdown issued but port still occupied"


def _wait_until_up(api_url: str, timeout: float = _LAUNCH_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_server_up(api_url):
            return True
        time.sleep(_POLL_INTERVAL_S)
    return False


def _launch_server(api_url: str, bare: str | None,
                   *, extra_args: list[str] | None = None) -> tuple[bool, str]:
    """Launch a llama-server so it serves *bare* (best effort).

    Strategy (most robust first):
      * If *bare* resolves to a **local GGUF file**, launch a single-model
        server with ``--model <abs path> --alias <bare>``.  This is the most
        reliable mode: the server serves exactly that model under the alias we
        send in the chat ``model`` field, with no router/indexing ambiguity.
      * Otherwise launch a **router** with ``--models-dir`` (for HF-cache /
        presets models) and dynamically load *bare*.

    The server is started detached (new process group) so it keeps running
    after the agent exits.  Returns ``(ok, message)``.
    """
    # Honor the user's persisted tuning (e.g. --gpu-layers / --ctx-size) on any
    # auto-launch / relaunch so a manual launch's flags survive a model switch.
    if not extra_args:
        extra_args = _llama_extra_args()
    bin_path = server_binary_path(api_url)
    if not bin_path:
        return False, ("llama-server binary not found on PATH or in common "
                       "locations; cannot auto-launch. Start it manually with "
                       "`--model <path-to-gguf>`.")
    base = _server_base_url(api_url)
    host = "127.0.0.1"
    port = 8080
    try:
        from urllib.parse import urlparse
        p = urlparse(base)
        host = p.hostname or host
        port = p.port or port
    except Exception:
        pass

    local_gguf = _resolve_local_gguf(bare) if bare else None
    if local_gguf:
        # Single-model server: serve the local GGUF under the requested id.
        cmd = [bin_path, "--host", host, "--port", str(port),
               "--model", local_gguf, "--alias", bare or "local-model"]
    else:
        # Router mode: resolves HF-cache / --models-dir / preset models.
        cmd = [bin_path, "--host", host, "--port", str(port),
               "--models-dir", _models_dir_for_launch()]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  [llama] launching server: {' '.join(cmd)}")
    try:
        if os.name == "nt":
            subprocess.Popen(
                cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                cmd, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to launch llama-server: {exc}"

    if not _wait_until_up(api_url):
        return False, f"llama-server did not come up at {api_url} within {_LAUNCH_TIMEOUT_S}s"

    if bare and not local_gguf:
        # Router mode: dynamically load the model id.
        ok, msg = _dynamic_load(api_url, bare, [])
        if ok:
            return True, f"launched router and loaded '{bare}'"
        return False, f"router up but could not load '{bare}': {msg}"
    # Single-model (or no model requested): it's already serving / up.
    return True, f"llama-server launched serving '{bare}'" if bare else "llama-server launched"


def _models_dir_for_launch() -> str:
    """Pick the --models-dir for a launched router.

    Uses the llama.cpp models dir ONLY (``_llama_models_dir``) so a launched
    router never reaches into the LM Studio models folder.  Falls back to the
    llama.cpp cache dir if the project-local folder is absent.
    """
    from .llama_provider import _llama_models_dir
    d = _llama_models_dir()
    if d:
        return d
    # Fallback: a llama.cpp cache dir if present.
    cache = os.path.expanduser("~/.cache/llama.cpp")
    if os.path.isdir(cache):
        return cache
    # Last resort: the project-local llama folder (may not exist yet).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "models", "llama")


@dataclass
class ReconcileResult:
    ok: bool
    message: str
    server_model_id: str | None = None
