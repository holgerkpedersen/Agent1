"""Lemonade LLM provider — OpenAI-compatible server for the AMD XDNA NPU.

`Lemonade <https://lemonade-server.ai>`_ is AMD's open-source local AI server.
It runs models on the Ryzen AI NPU (XDNA/XDNA2, via FastFlowLM or the Ryzen AI
LLM/OGA backends) as well as the iGPU, and exposes them over the standard
OpenAI ``/chat/completions`` API — so the agent talks to it exactly like LM
Studio, only the base URL differs.

Model ids are namespaced ``lemonade/<id>`` (e.g.
``lemonade/qwen3.5-4b-FLM``, ``lemonade/Qwen3-4B-Hybrid``).  The routing prefix
is kept on :attr:`model_name` so provider resolution / persistence / ``model
list`` agree; it is stripped at the HTTP boundary (see :func:`_http_model_id`).

There is NO model management (no LM Studio load/unload): Lemonade owns its own
model lifecycle and auto-loads a model on first request.  Errors are returned
as ``[Error: ...]`` strings; transport-level failures carry a
``(connection error)`` marker so :func:`~agent_core.llm.provider.is_connection_failure`
fails over to the next provider correctly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from .provider import ResponseMetrics
from .pricing import estimate_cost
from agent_core.constants import DEFAULT_LEMONADE_API_BASE
from agent_core.timeout import (
    DEFAULT_CHAT_TIMEOUT,
    MODEL_LOAD_TIMEOUT,
    MODEL_REFRESH_TIMEOUT,
)

logger = logging.getLogger(__name__)

#: HTTP statuses worth retrying (429 rate limit + the classic transient
#: gateway/server failures).  The NPU runtime can blip while (re)loading a
#: model, so a single failure must not abort the turn.
_TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

DEFAULT_API_BASE = DEFAULT_LEMONADE_API_BASE


def _http_model_id(model_name: str) -> str:
    """Strip the ``lemonade/`` routing prefix for the request payload.

    Lemonade model ids are bare (``qwen3.5-4b-FLM``); the extra prefix is an
    agent-side namespace so provider_for() / persistence / ``model list`` can
    tell an NPU model apart from an LM Studio one.
    """
    if model_name.startswith("lemonade/"):
        return model_name[len("lemonade/"):]
    return model_name


def _read_http_error_detail(exc: "urllib.error.HTTPError") -> str:
    """Return the server's error detail string from an ``HTTPError``."""
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - the fp may already be consumed
        body = ""
    return body or str(exc.reason)


def _format_http_error(code: int, detail: str) -> str:
    """Turn a Lemonade HTTP error into an actionable ``[Error: ...]`` string."""
    lowered = (detail or "").lower()
    if "max length" in lowered or "context length" in lowered or "too long" in lowered:
        return (
            "[Error: lemonade NPU model context is too small for this prompt. "
            "Reload it with a larger context, e.g. "
            "`lemonade load <model> --ctx-size 65536 --save-options`. "
            f"({detail})]"
        )
    if code == 404 and "model" in lowered:
        return (
            "[Error: lemonade model not found — it may not be downloaded. "
            f"List available models with `model list` / `lemonade list`. ({detail})]"
        )
    if "not loaded" in lowered or ("not found" in lowered and "model" in lowered):
        return (
            "[Error: lemonade model is not loaded and the server did not "
            f"auto-load it. Check the Lemonade server logs. ({detail})]"
        )
    return f"[Error: lemonade API request failed: HTTP Error {code}: {detail}]"


def _postprocess_content(content: str, reasoning: str) -> str:
    """Return usable text, falling back to a pure-reasoning block.

    Some NPU models (Qwen3.5 family) return ``content=None`` plus a
    ``reasoning_content`` block; a pure-reasoning answer is still useful, so it
    is recovered instead of collapsing the turn to ``(no output)``.
    """
    if content and content.strip():
        return content
    if reasoning and reasoning.strip() and len(reasoning.strip()) > 1:
        return reasoning
    return ""


class LemonadeProvider:
    """Concrete LLM provider for the local AMD Lemonade (NPU/iGPU) server."""

    def __init__(
        self,
        model_name: str | None = None,
        api_url: str | None = None,
        max_retries: int | None = None,
        retry_base_delay: float = 2.0,
    ) -> None:
        name = (model_name or "").strip()
        # Keep the routing prefix so provider detection / persistence agree
        # (the bare id is only used at the HTTP boundary).
        if name and not name.startswith("lemonade/"):
            name = f"lemonade/{name}"
        self.model_name = name or "lemonade/"
        self.api_url = (
            api_url or os.environ.get("LEMONADE_API_URL") or DEFAULT_API_BASE
        ).rstrip("/")
        self.temperature: float = 0.7
        self.max_tokens: int = 50000
        self._profile_name: str | None = None
        self.last_response_metrics: ResponseMetrics | None = None
        self._last_label: str | None = None
        #: Long reads are normal for big prompts; 600s matches the other local
        #: providers. Override via LEMONADE_TIMEOUT.
        self._api_timeout = float(
            os.environ.get("LEMONADE_TIMEOUT", str(DEFAULT_CHAT_TIMEOUT))
        )
        self._max_retries = (
            max(0, int(os.environ.get("LEMONADE_MAX_RETRIES", "3")))
            if max_retries is None
            else max(0, int(max_retries))
        )
        self._retry_base_delay = max(0.0, float(retry_base_delay))
        #: Real context window (tokens) of the loaded model; filled lazily from
        #: the server's ``/models`` response (``context_length``).
        #: ``LEMONADE_CTX_SIZE`` seeds it when the server is unreachable.
        self.context_limit: int | None = None
        seed = os.environ.get("LEMONADE_CTX_SIZE")
        if seed:
            try:
                seeded = int(seed)
                if seeded > 0:
                    self.context_limit = seeded
            except ValueError:
                pass

    def apply_profile(self, name: str, temperature: float, max_tokens: int) -> None:
        """Activate *name* with its sampling parameters in one step."""
        self._profile_name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Labeling (once per session, like the other providers)
    # ------------------------------------------------------------------

    def _label(self) -> None:
        label = f"[model: {_http_model_id(self.model_name)} | provider=lemonade, NPU]"
        if self._profile_name:
            label += f" profile={self._profile_name}"
        if label != self._last_label:
            print(f"  {label}", end="", flush=True)
            self._last_label = label

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self, json_body: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _request(
        self, method: str, url: str, body: Any = None, timeout: float | None = None
    ) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, headers=self._headers(body is not None), method=method
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else None

    async def _with_retry(
        self, factory: Callable[[], Any], *, label: str
    ) -> Any:
        """Run *factory* with exponential backoff on transient failures.

        The factory performs a BLOCKING urllib call — it is dispatched to a
        worker thread so it never stalls the event loop.
        """
        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.to_thread(factory)
            except urllib.error.HTTPError as exc:
                if exc.code not in _TRANSIENT_HTTP_STATUSES:
                    raise
                last_error = exc
            except (TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < self._max_retries:
                wait = self._retry_base_delay * (2 ** attempt)
                print(
                    f"  [retry {attempt + 1}/{self._max_retries}] {label}: "
                    f"{last_error}, waiting {wait:.0f}s...",
                    flush=True,
                )
                await asyncio.sleep(wait)
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Chat (OpenAI-compatible, native tool calling)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Send a chat request to Lemonade (NPU or iGPU backend).

        ``disable_thinking`` is accepted for protocol parity but ignored —
        Lemonade owns its own model/backend lifecycle and the NPU runtimes do
        not expose a portable reasoning knob.
        """
        self._label()
        from .lmstudio import sanitize_message_roles

        payload: dict[str, Any] = {
            "model": _http_model_id(self.model_name),
            "messages": sanitize_message_roles(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        try:
            result = await self._with_retry(
                lambda: self._request(
                    "POST", f"{self.api_url}/chat/completions", payload,
                    timeout=self._api_timeout,
                ),
                label="lemonade chat/completions",
            )
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_detail(exc)
            # Some NPU backends reject tool schemas; retry without them so a
            # plain chat still succeeds instead of failing the whole turn.
            if exc.code == 400 and tools and "tool" in detail.lower():
                payload.pop("tools", None)
                try:
                    result = await self._with_retry(
                        lambda: self._request(
                            "POST", f"{self.api_url}/chat/completions", payload,
                            timeout=self._api_timeout,
                        ),
                        label="lemonade chat/completions (no tools)",
                    )
                except urllib.error.HTTPError as exc2:
                    return _format_http_error(
                        exc2.code, _read_http_error_detail(exc2)
                    )
                except (urllib.error.URLError, TimeoutError, OSError) as exc2:
                    return (
                        "[Error: lemonade API request failed (connection "
                        f"error): {exc2}]"
                    )
                except Exception as exc2:  # noqa: BLE001 - defensive per-turn
                    return f"[Error: lemonade API request failed: {exc2}]"
            else:
                return _format_http_error(exc.code, detail)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Transport-level failure — the "(connection error)" marker makes
            # is_connection_failure() fail over to the next provider.
            return f"[Error: lemonade API request failed (connection error): {exc}]"
        except Exception as exc:  # noqa: BLE001 - defensive per-turn
            return f"[Error: lemonade API request failed: {exc}]"

        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            return "[Error: lemonade API returned no choices]"
        first = choices[0] if isinstance(choices, list) else choices
        message = first.get("message") or {}
        content_raw = message.get("content")
        reasoning_raw = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )

        tool_calls = message.get("tool_calls")
        usage = result.get("usage") if isinstance(result, dict) else None
        prompt_tokens = (
            int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
        )
        completion_tokens = (
            int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
        )
        # Local inference is free — cost is 0.0 by definition (pricing has no
        # entry for the "lemonade" provider).
        self.last_response_metrics = ResponseMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0.0,
            cost=estimate_cost(
                prompt_tokens, completion_tokens, self.model_name, "lemonade"
            ),
        )
        if tool_calls:
            return json.dumps({"content": content_raw, "tool_calls": tool_calls})
        content = _postprocess_content(
            str(content_raw or ""), str(reasoning_raw or "")
        )
        return content if content else "(no output)"

    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """No streaming support — return the full text response."""
        return await self.chat(messages)

    async def analyze_code(self, code: str) -> str:
        """Analyze code using the Lemonade model."""
        return await self.chat([
            {"role": "system",
             "content": "Analyze the following code for bugs and improvements."},
            {"role": "user", "content": code},
        ])

    # ------------------------------------------------------------------
    # model command / health support
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Live Lemonade catalog (GET /models), namespaced ``lemonade/<id>``.

        Returns ``[]`` on any failure so ``model list`` never crashes.
        """
        try:
            data = self._request(
                "GET", f"{self.api_url}/models", timeout=MODEL_REFRESH_TIMEOUT
            )
        except Exception:  # noqa: BLE001 - listing must never crash
            return []
        items = data.get("data") if isinstance(data, dict) else []
        out = [
            f"lemonade/{m['id']}"
            for m in (items or [])
            if isinstance(m, dict) and m.get("id")
        ]
        return sorted(out)

    def health_check(self) -> tuple[bool, str]:
        """Probe the Lemonade server; return ``(ok, human-readable note)``.

        Never raises — a dead/unreachable server is reported, not thrown, so
        `model list` and the REPL stay usable.
        """
        try:
            data = self._request(
                "GET", f"{self.api_url}/models", timeout=MODEL_REFRESH_TIMEOUT
            )
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return False, f"Lemonade unreachable at {self.api_url}: {exc}"
        items = data.get("data") if isinstance(data, dict) else None
        count = len(items or [])
        if count == 0:
            return False, (
                f"Lemonade reachable at {self.api_url} but returned no models "
                "(run `lemonade run <model>` / check `lemonade list`)"
            )
        return True, f"Lemonade reachable at {self.api_url} ({count} model(s))"

    def context_length(self) -> int | None:
        """Fetch and cache the loaded model's real context window (tokens).

        Reads ``context_length`` (falling back to ``max_context_window``) from
        the server's ``/models`` entry for this model.  Returns the cached
        ``context_limit`` when the model/server is not found — never raises.
        """
        try:
            data = self._request(
                "GET", f"{self.api_url}/models", timeout=MODEL_REFRESH_TIMEOUT
            )
        except Exception:  # noqa: BLE001 - discovery must never break a turn
            return self.context_limit
        bare = _http_model_id(self.model_name)
        items = data.get("data") if isinstance(data, dict) else None
        for entry in items or []:
            if not isinstance(entry, dict) or entry.get("id") != bare:
                continue
            value = entry.get("context_length") or entry.get("max_context_window")
            try:
                parsed = int(value) if value is not None else 0
            except (TypeError, ValueError):
                parsed = 0
            if parsed > 0:
                self.context_limit = parsed
            break
        return self.context_limit

    def ensure_model_loaded(self) -> tuple[bool, str]:
        """Ensure this provider's model is resident (residency manager hook).

        Lemonade's ``/load`` is idempotent — a no-op when the model is already
        loaded — so calling it is the safe way to guarantee the NPU model is
        warm before a parallel/Jev run without paying load latency mid-call.
        Returns ``(ok, note)`` and never raises.
        """
        bare = _http_model_id(self.model_name)
        if not bare:
            return False, "no lemonade model configured"
        try:
            result = self._request(
                "POST", f"{self.api_url}/load", {"model_name": bare},
                timeout=MODEL_LOAD_TIMEOUT,
            )
        except urllib.error.HTTPError as exc:
            return False, f"load failed: {_read_http_error_detail(exc)}"
        except Exception as exc:  # noqa: BLE001 - availability must not raise
            return False, f"Lemonade unreachable at {self.api_url}: {exc}"
        status = str((result or {}).get("status") or "")
        if status and status != "success":
            return False, f"load status={status}: {result}"
        self.context_length()
        return True, f"{bare} loaded on the NPU"

    def unload_model(self) -> tuple[bool, str]:
        """Unload this provider's model from the Lemonade server."""
        bare = _http_model_id(self.model_name)
        if not bare:
            return False, "no lemonade model configured"
        try:
            self._request(
                "POST", f"{self.api_url}/unload", {"model_name": bare},
                timeout=MODEL_LOAD_TIMEOUT,
            )
        except urllib.error.HTTPError as exc:
            return False, f"unload failed: {_read_http_error_detail(exc)}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Lemonade unreachable at {self.api_url}: {exc}"
        return True, f"{bare} unloaded"

    #: Alias used by generic health tooling.
    is_available = health_check


__all__: list[str] = [
    "DEFAULT_API_BASE",
    "LemonadeProvider",
]
