"""LM Studio provider implementation."""
import json
import logging
import os
import urllib.request
import urllib.error

import httpx

from .retry import RetryPolicy
from agent_core.constants import KNOWN_MODELS, resolve_model

logger = logging.getLogger(__name__)


def _management_url() -> str:
    """Return the LM Studio model-management base URL (REST API, not OpenAI-compat)."""
    base = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
    # Strip trailing /v1 and construct /api/v1
    if base.endswith("/v1"):
        base = base[:-3]
    elif base.endswith("/v1/"):
        base = base[:-4]
    return f"{base}/api/v1"


def _http_get_json(url: str, timeout: int = 10) -> dict | None:
    """Synchronous HTTP GET that returns parsed JSON, or None on failure."""
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("LM Studio API error: %s at %s", e.response.status_code, url)
        return None
    except Exception as e:
        logger.warning("LM Studio API error at %s: %s", url, e)
        return None


def _http_post_json(url: str, body: dict, timeout: int = 30) -> dict | None:
    """Synchronous HTTP POST that returns parsed JSON, or None on connection failure.

    HTTP error responses (4xx, 5xx) are returned as-is so callers can inspect
    the error body (e.g. LM Studio's ``{"error": {"message": "..."}}``).
    Only connection failures return None.
    """
    try:
        resp = httpx.post(url, json=body, timeout=timeout)
        return resp.json()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        logger.warning("LM Studio POST error: could not connect to %s", url)
        return None
    except Exception as e:
        logger.warning("LM Studio POST error at %s: %s", url, e)
        return None


def get_models_status() -> list[dict]:
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


def get_vram_info() -> dict:
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

    keys = [m["key"] for m in models]
    qlo = query.lower()

    # Exact key match
    if query in keys:
        return query

    # Search display names and params as well, but return the key
    by_display = {m["display_name"].lower(): m["key"] for m in models if m["display_name"]}
    by_params = {m["params_string"].lower(): m["key"] for m in models if m["params_string"]}

    # Exact display name match
    if qlo in by_display:
        return by_display[qlo]

    # Exact params match (e.g. "9b")
    if qlo in by_params:
        return by_params[qlo]

    # Substring match on keys
    sub_keys = [k for k in keys if qlo in k.lower()]
    if len(sub_keys) == 1:
        return sub_keys[0]

    # Substring match on display names
    sub_display = [v for k, v in by_display.items() if qlo in k]
    if len(sub_display) == 1:
        return sub_display[0]

    # Substring match on params
    sub_params = [v for k, v in by_params.items() if qlo in k]
    if len(sub_params) == 1:
        return sub_params[0]

    # difflib fuzzy on keys
    matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
    if matches:
        return matches[0]

    # difflib on display names
    matches = difflib.get_close_matches(query, list(by_display.keys()), n=1, cutoff=0.3)
    if matches:
        return by_display[matches[0]]

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
        self.lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
        self.retry_policy = retry_policy or RetryPolicy(max_retries=3, base_delay=2.0)
        self.temperature: float = 0.7
        self.max_tokens: int = 50000
        self._profile_name: str | None = None
    
    def _build_payload(
        self, 
        messages: list[dict], 
        tools: list[dict] | None = None,
        stream: bool = False,
        override_max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> dict:
        """Build request payload for LM Studio API.

        When thinking is disabled, sends both the OpenAI-style ``thinking``
        field (honored by some templates) and any per-model extra kwargs from
        ``KNOWN_MODELS`` (e.g. ``chat_template_kwargs: {"enable_thinking": False}``
        for Qwen3-family templates that gate reasoning on the jinja
        ``enable_thinking`` variable).  Sending both keeps the disable working
        across models regardless of which mechanism the template honors.
        """
        model_info = KNOWN_MODELS.get(self.model_name, {})
        max_tok = override_max_tokens or self.max_tokens
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tok,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        if disable_thinking or model_info.get("thinking") is False:
            payload["thinking"] = {"type": "disabled"}
            extra = model_info.get("disable_thinking_kwargs")
            if extra:
                payload.update(extra)
        return payload
    
    def _make_request(self, payload: dict, timeout: int = 3600) -> dict:
        """Make synchronous HTTP request to LM Studio."""
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
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    
    def _check_thinking_error(self, content: str, reasoning: str) -> str | None:
        """Check if model used all tokens thinking with no output.

        Returns a hard error instead of silently doubling ``max_tokens``:
        for reasoning models, a larger budget only buys more reasoning, so the
        retry would burn the whole doubled budget again (observed live: 31,922
        reasoning tokens, zero output).  Callers that need to proceed should
        retry with thinking disabled instead.
        """
        if not content and reasoning and len(reasoning) > 500:
            return (
                f"[Error: model consumed {len(reasoning)} tokens reasoning "
                "with no output. Retry with thinking disabled or a larger "
                "output budget.]"
            )
        return None
    
    async def chat(
        self, 
        messages: list[dict], 
        tools: list[dict] | None = None,
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
        print(f"  {label}", end="", flush=True)
        
        async def _do_request():
            result = self._make_request(payload)
            
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']
                content = message.get('content') or ""
                reasoning = message.get('reasoning_content') or ""
                
                # If tools present and model returned tool_calls, return full message
                if tools and message.get('tool_calls'):
                    return json.dumps(message)
                
                # Check for thinking error
                thinking_err = self._check_thinking_error(content, reasoning)
                if thinking_err:
                    return thinking_err
                
                return content or reasoning
            
            return ""
        
        def _on_retry(attempt, error_msg, wait_time):
            print(f"  [retry {attempt}/{self.retry_policy.max_retries}] {error_msg}, waiting {wait_time}s...")
        
        try:
            return await self.retry_policy.execute_with_retry(
                _do_request, 
                on_retry=_on_retry
            )
        except Exception as e:
            return f"[Error after {self.retry_policy.max_retries} retries: {e}]"
    
    async def chat_stream(self, messages: list[dict]) -> str:
        """Chat with real-time token streaming to console."""
        payload = self._build_payload(messages, stream=True)
        
        async def _do_stream():
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
            
            with urllib.request.urlopen(req, timeout=3600) as response:
                for line_bytes in response:
                    line = line_bytes.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        token = delta.get('content', '')
                        reasoning = delta.get('reasoning_content', '')
                        if reasoning:
                            reasoning_content += reasoning
                        if token:
                            print(token, end='', flush=True)
                            full_content += token
                    except json.JSONDecodeError as e:
                        logger.warning("LM Studio stream JSON decode error: %s", e)
            
            print()
            
            # Check for thinking error
            thinking_err = self._check_thinking_error(full_content, reasoning_content)
            if thinking_err:
                return thinking_err
            
            return full_content
        
        def _on_retry(attempt, error_msg, wait_time):
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