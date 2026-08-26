"""LM Studio provider implementation."""

from __future__ import annotations
import asyncio
import json
import time as _time
from typing import Any, cast
import logging
import os
import urllib.request
import urllib.error

import httpx

from .provider import ResponseMetrics
from .retry import RetryPolicy, TRANSIENT_HTTP_STATUSES, TransientHTTPError
from agent_core.constants import (
    KNOWN_MODELS,
    resolve_model,
    LOOP_NOTE_TAG_KEY,
)
from agent_core.config import lmstudio_base_url

logger = logging.getLogger(__name__)


def sanitize_message_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert mid-conversation ``system`` messages to ``user`` role.

    Strict chat templates (qwen's Jinja: "System message must be at the
    beginning") reject any system message that is not in the leading block.
    Steering notes injected by the tool loop — and any system message that
    leaked into a persisted chat history — must therefore travel as user
    messages.  The leading system block is preserved untouched.

    Also drops ORPHAN ``tool`` messages (tool_call_id without a matching
    assistant tool_calls message): strict gateways (opencode Console Go)
    reject those with HTTP 400.

    Also strips loop-internal tags (``LOOP_NOTE_TAG_KEY``): they mark
    agent-injected notes for the history manager and must never reach a
    provider as an unknown message field.
    """
    seen_non_system = False
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system" and seen_non_system:
            content = str(message.get("content") or "")
            converted = {
                **message,
                "role": "user",
                "content": f"[System note] {content}" if content else "",
            }
            converted.pop(LOOP_NOTE_TAG_KEY, None)
            out.append(converted)
        else:
            if message.get("role") != "system":
                seen_non_system = True
            plain = dict(message)
            plain.pop(LOOP_NOTE_TAG_KEY, None)
            out.append(plain)
    valid_ids: set[str] = set()
    for message in out:
        if message.get("role") == "assistant":
            for tc in message.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    valid_ids.add(str(tc["id"]))
    return [
        m for m in out
        if not (m.get("role") == "tool" and m.get("tool_call_id") not in valid_ids)
    ]


def _model_load_hint(detail: str) -> bool:
    """True when LM Studio's error body says the model is not in VRAM.

    LM Studio answers such requests with HTTP 400 ("model is not loaded,
    load it first"); retrying does not help — the model must be loaded first.
    """
    d = detail.lower()
    return any(
        needle in d
        for needle in ("not loaded", "is not loaded", "load it first", "load the model")
    )


def _management_url() -> str:
    """Return the LM Studio model-management base URL (REST API, not OpenAI-compat)."""
    base = lmstudio_base_url()
    # Strip trailing /v1 and construct /api/v1
    if base.endswith("/v1"):
        base = base[:-3]
    elif base.endswith("/v1/"):
        base = base[:-4]
    return f"{base}/api/v1"


def chat_timeout() -> int:
    """Socket cap (seconds) for one chat request to LM Studio.

    Was hard-wired at 3600: a stalled engine (KV-cache thrash at ~33k prompt
    tokens, observed live 2026-08-25 as a silent 12-minute freeze) then held
    the turn hostage for up to an hour with zero feedback.  The cap is a
    SOCKET-INACTIVITY timeout, not a total-duration limit — a healthy slow
    generation keeps streaming/receiving and never trips it, while a dead
    engine surfaces as a timeout error that the existing retry policy treats
    as transient.  Override with ``LMSTUDIO_CHAT_TIMEOUT``.
    """
    raw = os.environ.get("LMSTUDIO_CHAT_TIMEOUT", "600")
    try:
        return max(30, int(raw))
    except ValueError:
        return 600


def _http_get_json(url: str, timeout: int = 10) -> dict[str, Any] | None:
    """Synchronous HTTP GET that returns parsed JSON, or None on failure."""
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("LM Studio API error: %s at %s", e.response.status_code, url)
        return None
    except Exception as e:
        logger.warning("LM Studio API error at %s: %s", url, e)
        return None


def _http_post_json(url: str, body: dict[str, Any], timeout: int = 30) -> dict[str, Any] | None:
    """Synchronous HTTP POST that returns parsed JSON, or None on connection failure.

    HTTP error responses (4xx, 5xx) are returned as-is so callers can inspect
    the error body (e.g. LM Studio's ``{"error": {"message": "..."}}``).
    Only connection failures return None.
    """
    try:
        resp = httpx.post(url, json=body, timeout=timeout)
        return cast(dict[str, Any], resp.json())
    except (httpx.ConnectError, httpx.ConnectTimeout):
        logger.warning("LM Studio POST error: could not connect to %s", url)
        return None
    except Exception as e:
        logger.warning("LM Studio POST error at %s: %s", url, e)
        return None


def get_models_status() -> list[dict[str, Any]]:
    """Return every LLM model from LM Studio with loaded status, size, and params.

    Each dict has keys: key, display_name, size_bytes, params_string,
    loaded (bool), instance_id (str or None), context_length.
    """
    base = _management_url()
    data = _http_get_json(f"{base}/models")
    if not data or "models" not in data:
        return []

    # List comprehension avoids per-iteration append overhead; the nested
    # ``for inst`` binds loaded_instances once to avoid repeated lookups.
    return [
        {
            "key": m["key"],
            "display_name": m.get("display_name", m["key"]),
            "size_bytes": m.get("size_bytes", 0),
            "params_string": m.get("params_string", "?"),
            "loaded": len(inst) > 0,
            "instance_id": inst[0]["id"] if inst else None,
            "context_length": (inst[0].get("config", {}).get("context_length", 0)
                               if inst else m.get("max_context_length", 0)),
            "architecture": m.get("architecture"),
        }
        for m in data["models"]
        if m.get("type") == "llm"
        for inst in [m.get("loaded_instances", [])]
    ]


def get_vram_info() -> dict[str, Any]:
    """Return VRAM usage summary.

    Returns: {"total_bytes": int, "loaded_count": int, "models": [dict, ...]}
    """
    models = get_models_status()
    loaded = [m for m in models if m["loaded"]]
    return {
        "total_bytes": sum(m["size_bytes"] for m in loaded),
        "loaded_count": len(loaded),
        "models": models,
    }


def load_model(model_key: str, eval_batch_size: int = 4096) -> tuple[bool, str]:
    """Load a model via LM Studio REST API, with ``lms load`` CLI fallback.

    Returns (success, message).
    """
    base = _management_url()
    resp = _http_post_json(f"{base}/models/load", {
        "model": model_key,
        "eval_batch_size": eval_batch_size,
    }, timeout=300)  # Load can take a while (5 min)
    if resp and resp.get("status") == "loaded":
        return True, f"loaded ({resp.get('load_time_seconds', '?')}s) — {resp.get('instance_id', model_key)}"
    if resp:
        err = resp.get("error", "unknown")
        if isinstance(err, dict):
            err = err.get("message", str(err))
        return False, str(err)

    # REST API failed — try lms CLI as fallback
    import subprocess
    import shutil as _shutil
    lms = _shutil.which("lms") or _shutil.which("lms.exe")
    if lms:
        try:
            r = subprocess.run(
                [str(lms), "load", model_key, "--yes"],
                capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
            )
            if r.returncode == 0:
                return True, f"loaded via lms — {model_key}"
            return False, r.stderr.strip() or r.stdout.strip() or "unknown lms error"
        except subprocess.TimeoutExpired:
            return False, "lms load timed out"
        except Exception as e:
            return False, str(e)
    return False, "could not reach LM Studio (REST API timed out, lms CLI not found)"


def unload_model(instance_id: str | None = None) -> tuple[bool, str]:
    """Unload a model instance, or all if *instance_id* is None.

    Returns (success, message).
    """
    base = _management_url()
    models = get_models_status()
    loaded = [m for m in models if m["loaded"]]
    if not loaded:
        return True, "nothing to unload"

    target = instance_id or loaded[0]["instance_id"]

    resp = _http_post_json(f"{base}/models/unload", {"instance_id": target})
    if resp and resp.get("instance_id"):
        return True, f"unloaded {resp['instance_id']}"
    return False, (resp.get("error", "unknown") if resp else "could not reach LM Studio")


def resolve_model_name(query: str) -> str | None:
    """Fuzzy-match *query* against real LM Studio model keys and display names.

    Returns the matched model key, or None.
    """
    import difflib
    models = get_models_status()
    if not models:
        return None

    keys = [str(m["key"]) for m in models]
    qlo: str = query.lower()

    # Exact key match
    if query in keys:
        return query

    # Search display names and params as well, but return the key
    by_display = {m["display_name"].lower(): m["key"] for m in models if m["display_name"]}
    by_params: dict[str, str] = {m["params_string"].lower(): str(m["key"]) for m in models if m["params_string"]}

    # Exact display name match
    if qlo in by_display:
        return by_display.get(qlo)

    # Exact params match (e.g. "9b")
    if qlo in by_params:
        return by_params.get(qlo)

    # Substring match on keys
    sub_keys = [k for k in keys if qlo in k.lower()]
    if len(sub_keys) == 1 and (key := sub_keys[0]):
        return key

    # Substring match on display names
    sub_display = [v for k, v in by_display.items() if qlo in k]
    if len(sub_display) == 1:
        return sub_display[0] if sub_display else None

    # Substring match on params
    sub_params = [v for k, v in by_params.items() if qlo in k]
    if len(sub_params) == 1:
        return sub_params[0] if sub_params else None

    # difflib fuzzy on keys
    matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
    if matches:
        return matches[0] if matches else None

    # difflib on display names
    matches = difflib.get_close_matches(query, list(by_display.keys()), n=1, cutoff=0.3)
    if matches:
        return by_display.get(matches[0])

    return None


class LMStudioProvider:
    """Concrete LLM provider for LM Studio.
    
    Implements LLMProvider protocol for communicating with LM Studio's
    OpenAI-compatible API endpoint.
    """
    
    def __init__(
        self, 
        model_name: str | None = None, 
        api_key: str | None = None,
        retry_policy: RetryPolicy | None = None
    ):
        self.model_name = resolve_model(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.lmstudio_url = lmstudio_base_url()
        self.retry_policy = retry_policy or RetryPolicy(max_retries=3, base_delay=2.0)
        self.temperature: float = 0.7
        self.max_tokens: int = 50000
        self._profile_name: str | None = None
        #: Per-turn token/latency/cost of the last chat call (plan ARCH 17).
        self.last_response_metrics: ResponseMetrics | None = None
        #: Last printed status label — printed once per session and only
        #: re-printed when it changes (model/profile/temperature/tokens).
        self._last_label: str | None = None

    def apply_profile(
        self, name: str, temperature: float, max_tokens: int,
    ) -> None:
        """Activate *name* with its sampling parameters in one step.

        The single sanctioned way for callers to switch profiles on a provider
        (see ``LLMProvider.apply_profile``) — no direct attribute pokes.
        """
        self._profile_name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _build_payload(
        self, 
        messages: list[dict[str, Any]], 
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        override_max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> dict[str, Any]:
        """Build request payload for LM Studio API.

        When thinking is disabled, sends ``reasoning: "off"`` (safe on every
        probed model) plus any per-model ``disable_thinking_kwargs`` declared
        in KNOWN_MODELS (e.g. ``chat_template_kwargs: {"enable_thinking":
        False}`` for Qwen/Laguna jinja templates).  Models without an explicit
        entry get only the minimal safe fallback — see the probes on
        qwen/qwen3.8-27b (2026-08-18) where the aggressive switches
        (``thinking.disabled`` / ``enableThinking`` / ``preserve_thinking``)
        made the model burn its whole output budget on reasoning_content with
        zero content (finish_reason=length).
        """
        model_info = KNOWN_MODELS.get(self.model_name, {})
        max_tok = override_max_tokens or self.max_tokens
        payload = {
            "model": self.model_name,
            "messages": sanitize_message_roles(messages),
            "temperature": self.temperature,
            "max_tokens": max_tok,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        if disable_thinking or model_info.get("thinking") is False:
            # reasoning:"off" is the universal, safe knob — every probed model
            # answers normally with it and none burn their budget on it.
            payload["reasoning"] = "off"
            extra = model_info.get("disable_thinking_kwargs")
            if isinstance(extra, dict):
                # Explicit per-model knob set (laguna-s-2.1, qwen3.* entries)
                # — declared only when that model needs the extra switches.
                payload.update(cast(dict[str, Any], extra))
            else:
                # Minimal fallback for models without an explicit entry.  The
                # aggressive switches below are deliberately NOT sent: on
                # qwen/qwen3.8-27b they caused the model to spend all output
                # tokens on reasoning_content and emit nothing.
                ctk = payload.setdefault("chat_template_kwargs", {})
                assert isinstance(ctk, dict)
                ctk.setdefault("enable_thinking", False)
        return payload
    
    def _open_chat(self, req: urllib.request.Request, timeout: int) -> Any:
        """Open *req* against LM Studio, auto-loading the pinned model on a
        "not loaded" 400 (multi-shell recovery).

        When another ``agent.py`` shell (or the LM Studio GUI) evicted this
        session's model from VRAM, LM Studio answers HTTP 400 ("model is not
        loaded").  Instead of failing, load OUR model back and retry once —
        this is what keeps a running session pinned to its own choice.
        Shared by :meth:`_make_request` and :meth:`chat_stream`.
        """
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', 'replace').strip()
            detail = body or str(exc.reason)
            if exc.code == 400 and _model_load_hint(detail):
                ok, msg = load_model(self.model_name)
                if ok:
                    return urllib.request.urlopen(req, timeout=timeout)
                detail = f"{detail} (auto-load failed: {msg})"
            # Transient statuses become a typed error the RetryPolicy
            # understands (rate limit / gateway blip) — every other status
            # stays a plain RuntimeError so permanent failures fail fast.
            if exc.code in TRANSIENT_HTTP_STATUSES:
                raise TransientHTTPError(exc.code, detail) from exc
            raise RuntimeError(f"HTTP Error {exc.code}: {detail}") from exc

    def _make_request(self, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        """Make synchronous HTTP request to LM Studio.

        HTTP errors surface their response body (LM Studio explains the
        reason there — e.g. "model is not loaded"); a 400 whose body says the
        model is not in VRAM triggers an automatic load + single retry.
        """
        t = timeout if timeout is not None else chat_timeout()
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.lmstudio_url}/chat/completions",
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            },
            method='POST'
        )
        try:
            with self._open_chat(req, t) as response:
                return cast(dict[str, Any], json.loads(response.read().decode()))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LM Studio returned non-JSON response: {exc}") from exc
    
    def _check_thinking_error(self, content: str, reasoning: str, finish_reason: str | None = None) -> str | None:
        """Check if model used all tokens thinking with no output.

        Returns a hard error instead of silently doubling ``max_tokens``:
        for reasoning models, a larger budget only buys more reasoning, so the
        retry would burn the whole doubled budget again (observed live: 31,922
        reasoning tokens, zero output).  Callers that need to proceed should
        retry with thinking disabled instead.
        """
        if not content and reasoning and len(reasoning) > 500:
            return (
                f"[Error: model consumed {len(reasoning)} reasoning bytes "
                "with no output. Retry with thinking disabled or a larger "
                "output budget.]"
            )
        if (
            finish_reason == "length"
            and reasoning
            and len(reasoning) > 300
            and (not content or len(content) < len(reasoning))
        ):
            return (
                f"[Error: model hit the output limit ({len(reasoning)} reasoning bytes"
                ") and the response was truncated before the output could "
                "complete. Retry with thinking disabled or a larger output budget."
                "]"
            )
        return None
    
    async def chat(
        self, 
        messages: list[dict[str, Any]], 
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Send chat request to LLM via LM Studio with retry."""
        payload = self._build_payload(
            messages, tools, override_max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        label = f"[model: {payload['model']}]"
        if self._profile_name:
            label = f"[model: {payload['model']} | profile={self._profile_name} t={self.temperature} tok={self.max_tokens}]"
        # Print the status label once per session; re-print only when the
        # model/profile/temperature/tokens change mid-session.
        if label != self._last_label:
            print(f"  {label}", end="", flush=True)
            self._last_label = label
        
        async def _do_request() -> Any:
            start_time = _time.monotonic()
            # _make_request is a BLOCKING urllib call — run it in a worker
            # thread so it never stalls the event loop.  Without this,
            # asyncio.gather in run_parallel() serializes every model: the
            # first coroutine's sync HTTP round-trip blocks the loop and the
            # second model's chat() cannot even start until it finishes.
            result = await asyncio.to_thread(self._make_request, payload)
            elapsed_ms = (_time.monotonic() - start_time) * 1000.0

            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                message = choice.get('message', {})
                content = message.get('content') or ""
                reasoning = message.get('reasoning_content') or ""

                # Per-turn token/latency/cost accounting (plan ARCH item 17).
                usage = result.get("usage") if isinstance(result, dict) else None
                self.last_response_metrics = ResponseMetrics(
                    prompt_tokens=int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0,
                    completion_tokens=int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0,
                    latency_ms=elapsed_ms,
                )

                # If tools present and model returned tool_calls, return full message
                if tools and message.get('tool_calls'):
                    return json.dumps(message)

                # Check for thinking error
                thinking_err = self._check_thinking_error(
                    content, reasoning, finish_reason=choice.get('finish_reason')
                )
                if thinking_err:
                    return thinking_err

                return content or reasoning

            return ""
        
        def _on_retry(attempt: int, error_msg: str, wait_time: float) -> None:
            print(f"  [retry {attempt}/{self.retry_policy.max_retries}] {error_msg}, waiting {wait_time}s...")
        
        try:
            return await self.retry_policy.execute_with_retry(
                _do_request, 
                on_retry=_on_retry
            )
        except Exception as e:
            return f"[Error: {e}]"
    
    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """Chat with real-time token streaming to console."""
        payload = self._build_payload(messages, stream=True)
        
        async def _do_stream() -> Any:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.lmstudio_url}/chat/completions",
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                },
                method='POST'
            )
            
            full_content = ""
            reasoning_content = ""
            finish_reason = None
            
            # _open_chat (not raw urlopen): a 400 "model is not loaded" here
            # means another shell evicted our pinned model — reload + retry
            # once, same recovery as the non-streaming path.
            with self._open_chat(req, timeout=chat_timeout()) as response:
                for line_bytes in response:
                    line = line_bytes.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        choice = chunk.get('choices', [{}])[0]
                        delta = choice.get('delta', {})
                        token = delta.get('content', '')
                        reasoning = delta.get('reasoning_content', '')
                        if choice.get('finish_reason'):
                            finish_reason = choice['finish_reason']
                        if reasoning:
                            reasoning_content += reasoning
                        if token:
                            print(token, end='', flush=True)
                            full_content += token
                    except json.JSONDecodeError as e:
                        logger.warning("LM Studio stream JSON decode error: %s", e)
            
            print()
            
            # Check for thinking error
            thinking_err = self._check_thinking_error(
                full_content, reasoning_content, finish_reason=finish_reason
            )
            if thinking_err:
                return thinking_err
            
            return full_content
        
        def _on_retry(attempt: int, error_msg: str, wait_time: float) -> None:
            print(f"\n  [retry {attempt}/{self.retry_policy.max_retries}] {error_msg}, waiting {wait_time}s...")
        
        try:
            return await self.retry_policy.execute_with_retry(
                _do_stream,
                on_retry=_on_retry
            )
        except Exception as e:
            return f"[LM Studio stream error: {e}]"
    
    async def analyze_code(self, code: str) -> str:
        """Analyze code using LLM."""
        prompt = f"""Analyze this Python code and identify:
1. Bugs or issues
2. Code quality concerns
3. Potential improvements
4. Circular imports - which modules import each other, creating cycles
5. Missing or broken cross-module references

Code:
{code}"""
        
        messages = [
            {"role": "system", "content": "You are an expert code reviewer. Analyze the provided code and give detailed feedback."},
            {"role": "user", "content": prompt}
        ]
        
        return await self.chat(messages)
