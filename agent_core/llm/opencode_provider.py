"""Opencode LLM provider (opencode-go family) for the agent.

Two connection modes, both implementing the SAME chat contract as
:class:`LMStudioProvider` (plain text, or an OpenAI-style JSON string with
``tool_calls``; errors as ``[Error ...]`` strings):

1. **Direct API mode** (default when a key is available): OpenAI-compatible
   ``https://opencode.ai/v1`` chat/completions with NATIVE tool calling —
   the agent's tool schemas are passed through unchanged.  The API key
   resolves from ``OPENCODE_API_KEY`` first, then from opencode's
   credentials store (``~/.local/share/opencode/auth.json``, ``opencode-go``
   entry).  The key is never logged or persisted in agent state.
2. **Server mode**: a local ``opencode serve`` session whose message API
   executes its OWN built-in tools server-side (tools are an enable/disable
   map, not custom schemas), so the agent's tool loop is TEXT-MEDIATED:
   pending tool parts are mapped to this agent's tools via a fixed table and
   results fed back as text parts on the next turn.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

#: HTTP statuses that are safe to retry — the hosted gateway sits behind
#: Cloudflare and intermittently returns 5xx on healthy requests (observed
#: live: a single HTTP 500 that succeeded on the immediate retry).
_TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

#: OpenAI-compatible hosted endpoint for opencode-go (verified live: the
#: embedded base URL in the opencode CLI binary, /models returns the
#: opencode-go catalog with UNPREFIXED ids like "deepseek-v4-flash").
DEFAULT_API_BASE = "https://opencode.ai/zen/go/v1"

#: Model ids on the hosted API are unprefixed; this agent's persisted names
#: keep the "opencode-go/..." prefix for provider resolution.
def _hosted_model_id(model_name: str) -> str:
    for prefix in ("opencode-go/", "opencode/"):
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name

#: Default port for ``opencode serve``.
DEFAULT_SERVER_URL = "http://127.0.0.1:4096"

#: Map opencode's built-in tool names to this agent's NLP tools (server mode).
#: Only tools with overlapping semantics are mapped; anything else is reported
#: as an unmapped-tool error so the loop can continue.
_TOOL_MAP: dict[str, str] = {
    "bash": "run",
    "read": "read",
    "write": "write",
    "edit": "edit",
    "list": "list_files",
    "grep": "search",
    "webfetch": "web_search",
}


def _apply_thinking_knob(
    messages: list[dict[str, Any]], enabled: bool
) -> list[dict[str, Any]]:
    """Prepend the Nemotron-3 reasoning gate (``/think`` / ``/no_think``) to the
    leading system message.

    NVIDIA documents the ``/think`` / ``/no_think`` system-prompt directive as
    the control for Nemotron-3-Super reasoning (RAG blueprint docs).  Honoring
    ``disable_thinking`` this way stops these models from burning their whole
    output budget in ``reasoning_content`` and returning empty ``content``.
    """
    directive = "/think" if enabled else "/no_think"
    out: list[dict[str, Any]] = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            content = str(m.get("content") or "")
            m["content"] = f"{directive}\n\n{content}".rstrip()
            break
    return out


def _reasoning_exhausted(
    content: str, reasoning: str, finish_reason: str | None = None
) -> str | None:
    """Detect when a reasoning model used its budget thinking and left no usable
    output (Nemotron-3 and similar).  Returns an actionable ``[Error: ...]``
    string instead of silently collapsing to ``(no output)`` — callers retry
    with thinking disabled rather than burning more reasoning tokens."""
    if not content.strip() and reasoning and len(reasoning) > 300:
        return (
            f"[Error: model consumed {len(reasoning)} reasoning bytes with "
            "no output — reasoning models burn their budget thinking. Retry "
            "with disable_thinking=True (injects /no_think for Nemotron-3).]"
        )
    if (
        finish_reason == "length"
        and reasoning
        and len(reasoning) > 300
        and len(content) < len(reasoning)
    ):
        return (
            f"[Error: model hit the output limit ({len(reasoning)} reasoning "
            "bytes) and the response was truncated before the output completed. "
            "Retry with disable_thinking=True or a larger output budget.]"
        )
    return None

#: opencode built-in tools we enable on sessions (the model may call these).
_ENABLED_TOOLS: dict[str, bool] = {
    "bash": True,
    "read": True,
    "write": True,
    "edit": True,
    "list": True,
    "grep": True,
    "webfetch": True,
}


def _api_key_from_store() -> str:
    """Read the opencode-go API key from opencode's credentials store."""
    candidates = [
        Path.home() / ".local" / "share" / "opencode" / "auth.json",
        Path(os.environ.get("APPDATA", "")) / "opencode" / "auth.json",
    ]
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            key = data.get("opencode-go", {}).get("key")
            if key:
                return str(key)
        except Exception:
            continue
    return ""


class OpencodeProvider:
    """opencode-go LLM provider — direct hosted API or local server mode."""

    def __init__(
        self,
        model_name: str,
        server_url: str = DEFAULT_SERVER_URL,
        password: str = "",
        api_url: str = DEFAULT_API_BASE,
        api_key: str = "",
        profile_name: str | None = None,
        read_store: bool = True,
        max_retries: int | None = None,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.model_name = model_name
        self.server_url = server_url.rstrip("/")
        self.password = password
        self.api_url = api_url.rstrip("/")
        stored = _api_key_from_store() if read_store else ""
        self.api_key = api_key or os.environ.get("OPENCODE_API_KEY", "") or stored
        #: Direct API mode when a key is available; otherwise local server.
        self.api_mode = bool(self.api_key)
        #: Compatibility attributes — callers poke provider state directly.
        self.temperature: float = 0.7
        self.max_tokens: int = 50000
        self._profile_name: str | None = profile_name
        self._session_id: str | None = None
        self._last_label: str | None = None
        self._server_timeout = float(os.environ.get("OPENCODE_SERVER_TIMEOUT", "600"))
        #: Long reads are normal for big prompts (workflow plan steps); 600s
        #: matches the server-mode default. Override via OPENCODE_TIMEOUT.
        self._api_timeout = float(os.environ.get("OPENCODE_TIMEOUT", "600"))
        #: Transient-failure retry (default 3 attempts, exponential backoff).
        #: A single intermittent gateway 5xx must not abort a workflow run.
        self._max_retries = (
            max(0, int(os.environ.get("OPENCODE_MAX_RETRIES", "3")))
            if max_retries is None
            else max(0, int(max_retries))
        )
        self._retry_base_delay = max(0.0, float(retry_base_delay))

    # ------------------------------------------------------------------
    # Labeling (once per session, like LM Studio)
    # ------------------------------------------------------------------

    def _label(self) -> str:
        mode = "api" if self.api_mode else "server"
        label = f"[model: {self.model_name} | provider=opencode ({mode})]"
        if label != self._last_label:
            print(f"  {label}", end="", flush=True)
            self._last_label = label
        return label

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self, json_body: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {
            #: The hosted gateway sits behind Cloudflare, which rejects the
            #: default "Python-urllib" user agent (error 1010).
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_mode and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.password:
            token = base64.b64encode(f"opencode:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
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

        Retries HTTP 429/5xx responses and network timeouts — the hosted
        gateway intermittently returns HTTP 500 on healthy requests
        (observed live). Permanent client errors (4xx other than 429)
        propagate immediately. Raises the last error after retries are
        exhausted so callers can format it as an ``[Error ...]`` string.
        """
        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return factory()
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
    # Direct API mode (OpenAI-compatible, NATIVE tool calling)
    # ------------------------------------------------------------------

    async def _chat_api(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        disable_thinking: bool = False,
    ) -> str:
        from .lmstudio import sanitize_message_roles

        # Nemotron-3 models gate reasoning through /think vs /no_think in the
        # system prompt (NVIDIA docs). Honor disable_thinking for them — no
        # other model responds to the directive, so the payload stays as-is.
        if disable_thinking and "nemotron" in self.model_name.lower():
            messages = _apply_thinking_knob(messages, enabled=False)
        payload: dict[str, Any] = {
            "model": _hosted_model_id(self.model_name),
            "messages": sanitize_message_roles(messages),
        }
        if tools:
            payload["tools"] = tools
        payload["temperature"] = self.temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            result = await self._with_retry(
                lambda: self._request(
                    "POST", f"{self.api_url}/chat/completions", payload,
                    timeout=self._api_timeout,
                ),
                label="opencode chat/completions",
            )
        except urllib.error.HTTPError as exc:
            # Surface the gateway's explanation (e.g. "Messages with role
            # 'tool' must be a response to a preceding message with
            # 'tool_calls'") — the bare "400: Bad Request" is useless.
            body = exc.read().decode("utf-8", "replace").strip()
            detail = body or str(exc.reason)
            return f"[Error: opencode API request failed: HTTP Error {exc.code}: {detail}]"
        except Exception as exc:
            return f"[Error: opencode API request failed: {exc}]"
        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            return "[Error: opencode API returned no choices]"
        first = choices[0] if isinstance(choices, list) else choices
        message = first.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        finish_reason = first.get("finish_reason")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return json.dumps({"content": content, "tool_calls": tool_calls})
        thinking_err = _reasoning_exhausted(content, reasoning, finish_reason)
        if thinking_err:
            return thinking_err
        return content if content else "(no output)"

    # ------------------------------------------------------------------
    # Server mode (opencode serve session)
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> str:
        if self._session_id is not None:
            return self._session_id
        try:
            session = self._request(
                "POST", f"{self.server_url}/session", {"title": "agent1"},
                timeout=self._server_timeout,
            )
        except Exception as exc:
            return f"[Error: opencode server unreachable at {self.server_url}: {exc}]"
        sid = session.get("id") if isinstance(session, dict) else None
        self._session_id = str(sid) if sid else None
        if not self._session_id:
            return "[Error: opencode did not return a session id]"
        return self._session_id

    @staticmethod
    def _messages_to_parts(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Split *messages* into (system_text, request_parts).

        tool_call_id of the previous assistant turn are irrelevant here:
        opencode's message API takes text parts only, so tool results are
        folded into text ("Tool <name> result: ...").
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        parts: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            if role == "user":
                parts.append({"type": "text", "text": str(m.get("content") or "")})
            elif role == "assistant":
                if m.get("content"):
                    parts.append({"type": "text", "text": str(m.get("content"))})
                for tc in m.get("tool_calls", []) or []:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    parts.append({"type": "text", "text": f"[tool call] {name}({json.dumps(args)})"})
            elif role == "tool":
                parts.append({"type": "text", "text": f"[tool result] {m.get('content', '')}"})
        return "\n".join(system_parts), parts

    def _parts_to_response(self, parts: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Map response parts to (content, tool_calls).

        tool-call parts (state pending, input present) become OpenAI-style
        tool_calls with the opencode tool name mapped to this agent's tools.
        """
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts or []:
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                if text:
                    content_parts.append(text)
            elif ptype == "tool":
                state = part.get("state") or {}
                if state.get("status") == "pending":
                    tool_name = part.get("tool", "")
                    mapped = _TOOL_MAP.get(tool_name, "")
                    call = {
                        "id": part.get("callID") or f"oc_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": mapped or f"unmapped:{tool_name}",
                            "arguments": json.dumps(state.get("input") or {}),
                        },
                    }
                    tool_calls.append(call)
        return "\n".join(content_parts), tool_calls

    async def _chat_server(
        self, messages: list[dict[str, Any]], max_tokens: int | None,
        disable_thinking: bool = False,
    ) -> str:
        session = await self._ensure_session()
        if session.startswith("[Error"):
            return session
        system, parts = self._messages_to_parts(messages)
        if disable_thinking and "nemotron" in self.model_name.lower():
            system = "/no_think\n\n" + system
        body: dict[str, Any] = {
            "model": self.model_name,
            "system": system,
            "parts": parts,
            "tools": dict(_ENABLED_TOOLS),
        }
        if max_tokens is not None:
            body["maxTokens"] = max_tokens
        try:
            result = await self._with_retry(
                lambda: self._request(
                    "POST", f"{self.server_url}/session/{session}/message", body,
                    timeout=self._server_timeout,
                ),
                label="opencode message",
            )
        except Exception as exc:
            return f"[Error: opencode request failed: {exc}]"
        if not isinstance(result, dict):
            return "[Error: unexpected opencode response]"
        raw_parts = result.get("parts")
        parts = raw_parts if isinstance(raw_parts, list) else []
        content, tool_calls = self._parts_to_response(parts)
        if tool_calls:
            return json.dumps({"content": content, "tool_calls": tool_calls})
        return content if content else "(no output)"

    # ------------------------------------------------------------------
    # LLMProvider contract
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Send chat request — direct API (native tools) or server mode.

        *disable_thinking* is honored for Nemotron models via the documented
        ``/no_think`` system-prompt gate (and surfaced as a clear error for
        other reasoning backends that exhaust their budget thinking).  LM
        Studio API knobs never reach this provider (decision #011).
        """
        self._label()
        if self.api_mode:
            return await self._chat_api(messages, tools, max_tokens, disable_thinking)
        return await self._chat_server(messages, max_tokens, disable_thinking)

    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """No streaming support — return the full text response."""
        return await self.chat(messages)

    async def analyze_code(self, code: str) -> str:
        """Analyze code using the opencode model."""
        return await self.chat([
            {"role": "system", "content": "Analyze the following code for bugs and improvements."},
            {"role": "user", "content": code},
        ])

    # ------------------------------------------------------------------
    # model command support
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Models for this provider.

        API mode: GET /models (the hosted opencode-go catalog, ids like
        ``opencode-go/...``).  Server mode: /config/providers (model ids of
        every provider, grouped as ``provider/model``).
        """
        try:
            if self.api_mode:
                data = self._request("GET", f"{self.api_url}/models", timeout=15)
                items = data.get("data") if isinstance(data, dict) else []
                return sorted(
                    f"opencode-go/{m.get('id')}" for m in (items or []) if isinstance(m, dict)
                )
            config = self._request("GET", f"{self.server_url}/config/providers", timeout=15)
        except Exception:
            return []
        providers = config.get("providers") if isinstance(config, dict) else None
        out: list[str] = []
        for prov in providers or []:
            pid = prov.get("id") if isinstance(prov, dict) else ""
            models = prov.get("models") if isinstance(prov, dict) else {}
            if isinstance(models, dict):
                for mid in models.keys():
                    out.append(f"{pid}/{mid}" if pid else str(mid))
        return sorted(out)
