"""LM Studio provider implementation."""
import asyncio
import json
import os
import urllib.request
import urllib.error
import socket

import httpx

from .retry import RetryPolicy
from agent_core.constants import KNOWN_MODELS, DEFAULT_MODEL


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
        print(f"  LM Studio API error: {e.response.status_code} at {url}")
        return None
    except Exception as e:
        print(f"  LM Studio API error: {e}")
        return None


def _http_post_json(url: str, body: dict, timeout: int = 30) -> dict | None:
    """Synchronous HTTP POST that returns parsed JSON, or None on failure."""
    try:
        resp = httpx.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return None
    except httpx.HTTPStatusError as e:
        print(f"  LM Studio API error: {e.response.status_code} at {url}")
        return None
    except Exception as e:
        print(f"  LM Studio API error: {e}")
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

    result = []
    for m in data["models"]:
        if m.get("type") != "llm":
            continue
        instances = m.get("loaded_instances", [])
        result.append({
            "key": m["key"],
            "display_name": m.get("display_name", m["key"]),
            "size_bytes": m.get("size_bytes", 0),
            "params_string": m.get("params_string", "?"),
            "loaded": len(instances) > 0,
            "instance_id": instances[0]["id"] if instances else None,
            "context_length": instances[0].get("config", {}).get("context_length", 0) if instances else m.get("max_context_length", 0),
            "architecture": m.get("architecture"),
        })
    return result


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


def load_model(model_key: str, parallel: int = 4) -> tuple[bool, str]:
    """Load a model via LM Studio REST API.

    Returns (success, message).
    """
    base = _management_url()
    resp = _http_post_json(f"{base}/models/load", {
        "model": model_key,
        "eval_batch_size": parallel,
    })
    if resp and resp.get("status") == "loaded":
        return True, f"loaded ({resp.get('load_time_seconds', '?')}s) — {resp.get('instance_id', model_key)}"
    return False, resp.get("error", "unknown") if resp else "could not reach LM Studio"


def unload_model(instance_id: str | None = None) -> tuple[bool, str]:
    """Unload a model instance, or all if *instance_id* is None.

    Returns (success, message).
    """
    base = _management_url()
    models = get_models_status()
    loaded = [m for m in models if m["loaded"]]
    if not loaded:
        return True, "nothing to unload"

    target = instance_id
    if not target:
        target = loaded[0]["instance_id"]

    resp = _http_post_json(f"{base}/models/unload", {"instance_id": target})
    if resp and resp.get("instance_id"):
        return True, f"unloaded {resp['instance_id']}"
    return False, resp.get("error", "unknown") if resp else "could not reach LM Studio"


def resolve_model_name(query: str) -> str | None:
    """Fuzzy-match *query* against real LM Studio model keys.

    Returns the matched model key, or None.
    """
    import difflib
    models = get_models_status()
    if not models:
        return None

    keys = [m["key"] for m in models]
    # Exact match
    if query in keys:
        return query
    # Substring match
    sub = [k for k in keys if query.lower() in k.lower()]
    if len(sub) == 1:
        return sub[0]
    # difflib fuzzy match
    matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
    if matches:
        return matches[0]
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
        self.model_name = model_name or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
        self.retry_policy = retry_policy or RetryPolicy(max_retries=3, base_delay=2.0)
    
    def _build_payload(
        self, 
        messages: list[dict], 
        tools: list[dict] | None = None,
        stream: bool = False,
        override_max_tokens: int | None = None,
    ) -> dict:
        """Build request payload for LM Studio API."""
        model_info = KNOWN_MODELS.get(self.model_name, {})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": override_max_tokens or model_info.get("max_tokens", 50000),
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        if model_info.get("thinking") is False:
            payload["thinking"] = {"type": "disabled"}
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
        """Check if model used all tokens thinking with no output."""
        if not content and reasoning and len(reasoning) > 500:
            model = self.model_name
            alternates = [m for m in KNOWN_MODELS if m != model]
            alt_hint = f" Try: model {alternates[0]}" if alternates else ""
            return f"[Error: {model} used all tokens thinking, zero code output. Use 'model reload' or switch model.{alt_hint}]"
        return None
    
    async def chat(
        self, 
        messages: list[dict], 
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat request to LLM via LM Studio with retry."""
        payload = self._build_payload(messages, tools, override_max_tokens=max_tokens)
        
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
                    except json.JSONDecodeError:
                        pass
            
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
