"""Opencode LLM provider (opencode-go family) for the agent.

Talks to a running ``opencode serve`` HTTP server (port 4096 by default)
and implements the SAME chat contract as :class:`LMStudioProvider`:

- ``chat(...)`` returns plain text, or an OpenAI-style JSON string with
  ``tool_calls`` when the model requests tools
- errors are returned as ``[Error ...]`` strings
- ``temperature``/``max_tokens``/``_profile_name`` attributes exist so
  callers that poke provider state (e.g. implement's profile switch) work

opencode's message API executes its OWN built-in tools server-side and
takes ``tools`` as an enable/disable map, not custom schemas.  The agent's
tool loop is therefore TEXT-MEDIATED: pending tool parts from the response
are mapped to this agent's tools via a fixed table, executed by the caller,
and fed back as text parts on the next turn.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Any

#: Default port for ``opencode serve``.
DEFAULT_SERVER_URL = "http://127.0.0.1:4096"

#: Map opencode's built-in tool names to this agent's NLP tools.  Only tools
#: with overlapping semantics are mapped; anything else is reported as an
#: unmapped-tool error so the loop can continue.
_TOOL_MAP: dict[str, str] = {
    "bash": "run",
    "read": "read",
    "write": "write",
    "edit": "edit",
    "list": "list_files",
    "grep": "search",
    "webfetch": "web_search",
}

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


class OpencodeProvider:
    """opencode server-backed LLM provider (opencode-go models)."""

    def __init__(
        self,
        model_name: str,
        server_url: str = DEFAULT_SERVER_URL,
        password: str = "",
        profile_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.server_url = server_url.rstrip("/")
        self.password = password
        #: Compatibility attributes — callers poke provider state directly.
        self.temperature: float = 0.7
        self.max_tokens: int = 50000
        self._profile_name: str | None = profile_name
        self._session_id: str | None = None
        self._last_label: str | None = None
        self._timeout = float(os.environ.get("OPENCODE_SERVER_TIMEOUT", "600"))

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.password:
            token = base64.b64encode(f"opencode:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        return headers

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.server_url + path,
            data=data,
            headers=self._headers(),
            method=method,
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else None

    def _label(self) -> str:
        label = f"[model: {self.model_name} | provider=opencode]"
        if label != self._last_label:
            print(f"  {label}", end="", flush=True)
            self._last_label = label
        return label

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> str:
        if self._session_id is not None:
            return self._session_id
        try:
            session = self._request("POST", "/session", {"title": "agent1"})
        except Exception as exc:
            return f"[Error: opencode server unreachable at {self.server_url}: {exc}]"
        sid = session.get("id") if isinstance(session, dict) else None
        self._session_id = str(sid) if sid else None
        if not self._session_id:
            return "[Error: opencode did not return a session id]"
        return self._session_id

    # ------------------------------------------------------------------
    # Message <-> parts mapping
    # ------------------------------------------------------------------

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
        """Send chat request to the opencode server.

        *disable_thinking* and *tools* are intentionally ignored: opencode
        manages reasoning and its own tool set server-side (decision #011 —
        LM Studio knobs never reach other providers).
        """
        self._label()
        session = await self._ensure_session()
        if session.startswith("[Error"):
            return session
        system, parts = self._messages_to_parts(messages)
        body: dict[str, Any] = {
            "model": self.model_name,
            "system": system,
            "parts": parts,
            "tools": dict(_ENABLED_TOOLS),
        }
        if max_tokens is not None:
            body["maxTokens"] = max_tokens
        try:
            result = self._request("POST", f"/session/{session}/message", body)
        except Exception as exc:
            return f"[Error: opencode request failed: {exc}]"
        if not isinstance(result, dict):
            return "[Error: unexpected opencode response]"
        raw_parts = result.get("parts")
        parts = raw_parts if isinstance(raw_parts, list) else []
        content, tool_calls = self._parts_to_response(parts)
        if tool_calls:
            return json.dumps({
                "content": content,
                "tool_calls": tool_calls,
            })
        return content if content else "(no output)"

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
        """Models available from the opencode provider (via /config/providers).

        Returns the model ids of every provider listed by the server (the
        model command renders them grouped by provider).
        """
        try:
            config = self._request("GET", "/config/providers")
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
