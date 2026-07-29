"""LM Studio provider implementation."""
import asyncio
import json
import os
import urllib.request
import urllib.error
import socket

from .retry import RetryPolicy
from agent_core.constants import KNOWN_MODELS, DEFAULT_MODEL


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
