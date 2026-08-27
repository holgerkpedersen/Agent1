"""llama.cpp (llama-server) provider implementation.

OpenAI-compatible chat provider for a local ``llama-server``.  Unlike
:class:`LMStudioProvider` it performs NO model management (no load/unload, no
auto-reload on a "not loaded" 400): the external llama-server owns loading,
and the model is expected to be served before the agent talks to it.

The standard OpenAI ``/v1/chat/completions`` and ``/v1/models`` endpoints are
used directly — no protocol translation needed for any GGUF-backed arch that
the *running* llama-server supports.

NOTE: experimental architectures such as "qwen4exp" (used by some Qwen3.8 /
Qwen3.8-Flash-Next GGUFs) are NOT present in any tagged/released Windows
build of llama.cpp — not in the official releases (which ship no Windows
binaries), not in LM Studio's bundled backends, and not in community Vulkan
builds. To serve such a model you must build llama.cpp from the latest
``main`` branch (``-DGGML_VULKAN=ON`` on Windows/AMD) yourself. This provider
makes no attempt to paper over that gap.
"""

from __future__ import annotations
import asyncio
import json
import time as _time
from typing import Any, cast
import logging
import os
import urllib.request
import urllib.error

from .provider import ResponseMetrics
from .retry import RetryPolicy, TRANSIENT_HTTP_STATUSES, TransientHTTPError
from agent_core.constants import KNOWN_MODELS, resolve_model

logger = logging.getLogger(__name__)


def _lmstudio_models_dir() -> str | None:
    """Return the LM Studio local models directory, or None if not found.

    Checks, in priority order:
      1. ``LMSTUDIO_MODELS_DIR`` env var (explicit override)
      2. ``%USERPROFILE%\\.lmstudio\\models`` (default Windows LM Studio path)
      3. ``~/.lmstudio/models`` (default path on macOS/Linux)
    """
    explicit = os.environ.get("LMSTUDIO_MODELS_DIR")
    if explicit:
        return explicit
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".lmstudio", "models"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def discover_local_gguf_models() -> list[str]:
    """Scan the LM Studio models directory for ``.gguf`` files.

    Returns a sorted list of model ids namespaced as ``llama/<relative_path>``
    where ``<relative_path>`` is the file path relative to the models dir with
    the ``.gguf`` extension stripped.  For example a file at
    ``unsloth/Qwen3-Coder-30B-A3B-Instruct-Q4_K_S.gguf`` becomes
    ``llama/unsloth/Qwen3-Coder-30B-A3B-Instruct-Q4_K_S``.

    Only the *filename stem* (without extension) is used as the id component,
    so sharded models (``-00001-of-00003.gguf``) collapse to a single id.

    Returns ``[]`` if the directory does not exist or is not readable.
    """
    models_dir = _lmstudio_models_dir()
    if not models_dir:
        return []
    discovered: set[str] = set()
    for root, _dirs, files in os.walk(models_dir):
        for fname in files:
            if not fname.lower().endswith(".gguf"):
                continue
            # Skip mmproj (vision projection) files — they are not chat models.
            if fname.lower().startswith("mmproj"):
                continue
            rel = os.path.relpath(os.path.join(root, fname), models_dir)
            # Normalise to forward slashes for cross-platform consistent ids.
            rel = rel.replace(os.sep, "/")
            # Strip the .gguf extension (case-insensitive) for the id.
            stem = rel[:-5] if rel.lower().endswith(".gguf") else rel
            # Collapse sharded models (-00001-of-00003) to a single id.
            stem = _shard_stem(stem)
            discovered.add(f"llama/{stem}")
    return sorted(discovered)


def _shard_stem(stem: str) -> str:
    """Collapse a sharded filename stem to its base id.

    ``Qwen3.8-...-00001-of-00003`` -> ``Qwen3.8-...``
    Non-sharded stems are returned unchanged.
    """
    import re
    return re.sub(r"-\d{5}-of-\d{5}$", "", stem)


def _chat_timeout() -> int:
    """Per-request socket-inactivity timeout (seconds). Override with
    ``LLAMA_CHAT_TIMEOUT``; floor 30s, default 600s."""
    raw = os.environ.get("LLAMA_CHAT_TIMEOUT", "600")
    try:
        return max(30, int(raw))
    except ValueError:
        return 600


class LlamaProvider:
    """Concrete LLM provider for a local llama.cpp ``llama-server``."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        name = resolve_model(model_name)
        # Keep the routing prefix so provider detection / persistence agree
        # (the bare id is only added back at the HTTP boundary, see
        # ``_server_model_id``).
        self.model_name = name if name.startswith("llama/") else f"llama/{name}"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_url = (api_url or os.environ.get("AGENT_LLAMA_URL")
                        or "http://127.0.0.1:8080/v1").rstrip("/")
        self.retry_policy = retry_policy or RetryPolicy(max_retries=3, base_delay=2.0)
        self.temperature: float = 0.7
        self.max_tokens: int = 50000
        self._profile_name: str | None = None
        self.last_response_metrics: ResponseMetrics | None = None
        self._last_label: str | None = None
        #: Cached model id the running llama-server actually serves (from
        #: GET /v1/models).  llama-server loads exactly one model at startup,
        #: so this is the only id valid for the request ``model`` field.
        #: Cleared (refreshed) whenever the server reports "model not found".
        self._cached_server_model_id: str | None = None

    def apply_profile(self, name: str, temperature: float, max_tokens: int) -> None:
        """Activate a named profile (temperature/max_tokens)."""
        self._profile_name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _server_model_id(self) -> str:
        """Model id the llama-server expects (no routing prefix).

        Internally ``model_name`` keeps the ``llama/`` routing prefix (so
        provider routing, persistence and ``model list`` agree).  The actual
        HTTP request must use the bare id the server registered under
        ``GET /v1/models`` (e.g. ``ggml-org/gemma-4-e4b-it-GGUF:Q4_0``).

        llama-server loads exactly ONE model at startup (via ``--model``) and
        registers it under a specific id — it does NOT know the user-typed
        routing label (e.g. ``oss-20b``).  So we prefer the *cached server
        model id* discovered from ``GET /v1/models``; only when that is
        unknown do we fall back to the bare routing label.  This is what makes
        ``model oss-20b -p llama`` actually work against a server started with
        ``--model llama/gpt-oss-20b-MXFP4``.
        """
        if self._cached_server_model_id:
            return self._cached_server_model_id
        name = self.model_name
        return name[len("llama/"):] if name.startswith("llama/") else name

    def refresh_server_model_id(self) -> str | None:
        """Ask the running llama-server which model it actually serves.

        Returns the bare server model id (cached on the instance) or ``None``
        if the server is unreachable / has no model loaded.  This is the id
        that must be sent in the request ``model`` field — never the routing
        label.
        """
        try:
            req = urllib.request.Request(
                f"{self.api_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("data") if isinstance(data, dict) else []
            ids = [m.get("id") for m in (items or []) if isinstance(m, dict) and m.get("id")]
            if ids:
                self._cached_server_model_id = ids[0]
                return self._cached_server_model_id
        except Exception:
            return None
        return None

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        override_max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> dict[str, Any]:
        """Build the OpenAI /v1/chat/completions payload.

        Reuses :func:`agent_core.llm.lmstudio.sanitize_message_roles` so role
        normalization (mid-conversation system -> user, orphan tool drops,
        loop-tag stripping) is identical to LM Studio and opencode providers.
        """
        from .lmstudio import sanitize_message_roles
        model_info = KNOWN_MODELS.get(self.model_name, {})
        max_tok = override_max_tokens or self.max_tokens
        payload: dict[str, Any] = {
            "model": self._server_model_id(),
            "messages": sanitize_message_roles(messages),
            "temperature": self.temperature,
            "max_tokens": max_tok,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        # Disable reasoning budget when requested or when the model is known to
        # not support thinking.  Qwen's Jinja rejects a trailing system msg and
        # also honours chat_template_kwargs.enable_thinking=False, so we set it.
        if disable_thinking or model_info.get("thinking") is False:
            payload["reasoning"] = "off"
            extra = model_info.get("disable_thinking_kwargs")
            if isinstance(extra, dict):
                payload.update(cast(dict[str, Any], extra))
            else:
                ctk = payload.setdefault("chat_template_kwargs", {})
                assert isinstance(ctk, dict)
                ctk.setdefault("enable_thinking", False)
        return payload

    def _open_chat(self, req: urllib.request.Request, timeout: int) -> Any:
        """Open *req*; surface HTTP errors as typed exceptions (no auto-load)."""
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace").strip()
            detail = body or str(exc.reason)
            if exc.code in TRANSIENT_HTTP_STATUSES:
                raise TransientHTTPError(exc.code, detail) from exc
            raise RuntimeError(f"HTTP Error {exc.code}: {detail}") from exc

    def _make_request(self, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        """Non-streaming chat completion (runs in a thread for async).

        Self-heals the "model not found" 400: llama-server loads exactly one
        model (the ``--model`` it started with) and registers it under that
        exact id.  The agent's ``model_name`` is a routing label that may not
        match, so if the server rejects the id we sent, we discover the real
        served id via ``GET /v1/models`` and retry once with it.  This makes
        ``model oss-20b -p llama`` work against a server started with a
        different ``--model`` without the user having to restart it.
        """
        t = timeout if timeout is not None else _chat_timeout()
        sent_id = payload.get("model")
        try:
            return self._post_chat(payload, t)
        except RuntimeError as exc:
            msg = str(exc)
            if ("model" in msg.lower() and "not found" in msg.lower()
                    and sent_id == self._server_model_id()):
                refreshed = self.refresh_server_model_id()
                if refreshed and refreshed != sent_id:
                    payload = dict(payload)
                    payload["model"] = refreshed
                    print(f"  [llama] server serves '{refreshed}'; "
                          f"retrying with that id")
                    return self._post_chat(payload, t)
            raise

    def _post_chat(self, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        """POST one chat completion request and parse the JSON response."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with self._open_chat(req, timeout) as response:
                return cast(dict[str, Any], json.loads(response.read().decode()))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"llama-server returned non-JSON response: {exc}") from exc

    def _check_thinking_error(
        self, content: str, reasoning: str, finish_reason: str | None = None
    ) -> str | None:
        """Detect the 'reasoning ate all budget' failure mode and return an
        error string (or None).  Mirrors LMStudioProvider's guard so callers
        see a consistent [Error: ...] message."""
        if not content and reasoning and len(reasoning) > 500:
            return ("[Error: model consumed "
                    f"{len(reasoning)} reasoning bytes with no output. "
                    "Retry with thinking disabled or a larger output budget.]")
        if (finish_reason == "length" and reasoning and len(reasoning) > 300
                and (not content or len(content) < len(reasoning))):
            return ("[Error: model hit the output limit "
                    f"({len(reasoning)} reasoning bytes) and the response was "
                    "truncated before the output could complete. Retry with "
                    "thinking disabled or a larger output budget.]")
        return None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Non-streaming chat completion.  Returns text content (or an
        ``[Error: ...]`` string on failure — never raises to callers)."""
        payload = self._build_payload(
            messages, tools, override_max_tokens=max_tokens, disable_thinking=disable_thinking,
        )
        label = f"[model: {payload['model']}]"
        if self._profile_name:
            label = (f"[model: {payload['model']} | profile={self._profile_name} "
                     f"t={self.temperature} tok={self.max_tokens}]")
        if label != self._last_label:
            print(f"  {label}", end="", flush=True)
            self._last_label = label

        async def _do_request() -> Any:
            start_time = _time.monotonic()
            result = await asyncio.to_thread(self._make_request, payload)
            elapsed_ms = (_time.monotonic() - start_time) * 1000.0
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message", {})
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                usage = result.get("usage") if isinstance(result, dict) else None
                self.last_response_metrics = ResponseMetrics(
                    prompt_tokens=int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0,
                    completion_tokens=int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0,
                    latency_ms=elapsed_ms,
                )
                if tools and message.get("tool_calls"):
                    return json.dumps(message)
                thinking_err = self._check_thinking_error(
                    content, reasoning, finish_reason=choice.get("finish_reason"))
                if thinking_err:
                    return thinking_err
                return content or reasoning
            return ""

        def _on_retry(attempt: int, error_msg: str, wait_time: float) -> None:
            print(f"  [retry {attempt}/{self.retry_policy.max_retries}] {error_msg}, waiting {wait_time}s...")

        try:
            return await self.retry_policy.execute_with_retry(_do_request, on_retry=_on_retry)
        except Exception as e:  # noqa: BLE001 - never raise to callers
            return f"[Error: {e}]"

    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """Streaming chat completion.  Prints tokens inline; returns the full
        assembled content (or ``[llama-server stream error: ...]``)."""
        payload = self._build_payload(messages, stream=True)

        async def _do_stream() -> Any:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.api_url}/chat/completions",
                data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            full_content = ""
            reasoning_content = ""
            finish_reason: str | None = None
            with self._open_chat(req, timeout=_chat_timeout()) as response:
                for line_bytes in response:
                    line = line_bytes.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        token = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        if reasoning:
                            reasoning_content += reasoning
                        if token:
                            print(token, end="", flush=True)
                            full_content += token
                    except json.JSONDecodeError as e:
                        logger.warning("llama-server stream JSON decode error: %s", e)
            print()
            thinking_err = self._check_thinking_error(
                full_content, reasoning_content, finish_reason=finish_reason)
            if thinking_err:
                return thinking_err
            return full_content

        def _on_retry(attempt: int, error_msg: str, wait_time: float) -> None:
            print(f"\n  [retry {attempt}/{self.retry_policy.max_retries}] {error_msg}, waiting {wait_time}s...")

        try:
            return await self.retry_policy.execute_with_retry(_do_stream, on_retry=_on_retry)
        except Exception as e:  # noqa: BLE001 - never raise to callers
            return f"[llama-server stream error: {e}]"

    async def analyze_code(self, code: str) -> str:
        """Analyze Python code using LLM (one-shot review)."""
        prompt = (
            "Analyze this Python code and identify:\n"
            "1. Bugs or issues\n2. Code quality concerns\n"
            "3. Potential improvements\n"
            "4. Circular imports - which modules import each other, creating cycles\n"
            "5. Missing or broken cross-module references\n\nCode:\n" + code
        )
        messages = [
            {"role": "system", "content": "You are an expert code reviewer."},
            {"role": "user", "content": prompt},
        ]
        return await self.chat(messages)

    def list_models(self) -> list[str]:
        """Return llama-server model ids namespaced as ``llama/<id>``.

        Merges two sources:

        1. **Live server** — ``GET {api_url}/models`` returns the models the
           running llama-server has registered (i.e. what it can *serve*).
        2. **Local GGUF files** — :func:`discover_local_gguf_models` scans the
           LM Studio models directory for ``.gguf`` files so that
           ``model list`` shows every model available to load, even if the
           server hasn't registered it yet.

        Server models are listed first (they are the ones chat can actually
        use right now); local-only models follow alphabetically.  Returns
        ``[]`` only if the server is unreachable *and* no local models are
        found.
        """
        server_models: list[str] = []
        try:
            req = urllib.request.Request(
                f"{self.api_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("data") if isinstance(data, dict) else []
            server_models = sorted(
                f"llama/{m.get('id')}"
                for m in (items or [])
                if isinstance(m, dict) and m.get("id")
            )
        except Exception:
            pass

        local_models = discover_local_gguf_models()

        # Server models first, then local-only (dedup by id).
        seen: set[str] = set()
        merged: list[str] = []
        for mid in server_models + local_models:
            if mid not in seen:
                seen.add(mid)
                merged.append(mid)
        return merged

    def _server_base_url(self) -> str:
        """Root URL of the llama-server process (no /v1 suffix).

        Management endpoints like ``POST /shutdown`` live at the server root,
        not under ``/v1``.
        """
        return self.api_url.rsplit("/v1", 1)[0] if self.api_url.endswith("/v1") else self.api_url

    def shutdown(self) -> tuple[bool, str]:
        """Shut down the llama-server process, unloading whatever model it has.

        llama-server exposes ``POST /shutdown`` at the root (not under
        ``/v1``).  This unloads the currently loaded model.  Returns
        ``(success, message)``.  If the server is already stopped, reports
        that as success (the desired end-state is achieved).
        """
        base = self._server_base_url()
        try:
            req = urllib.request.Request(f"{base}/shutdown", method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            return True, f"llama-server shut down (model unloaded)"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False, f"server does not support /shutdown (HTTP {exc.code})"
            return False, f"HTTP Error {exc.code}: {exc.reason}"
        except urllib.error.URLError:
            return True, "llama-server already stopped (nothing to unload)"
        except Exception as exc:  # noqa: BLE001
            return False, f"shutdown failed: {exc}"
