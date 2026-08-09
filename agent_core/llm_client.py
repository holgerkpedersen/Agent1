from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmResponse:
    content: str
    is_error: bool = False
    error_code: str | None = None


class LLMClient:
    def __init__(self, api_url: str, timeout_sec: float = 30.0) -> None:
        self.api_url = api_url
        self.timeout_sec = timeout_sec

    async def chat(self, prompt: str) -> LlmResponse:
        """Send message to LLM and return structured response."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(
                    self.api_url + "/chat/completions",
                    json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]},
                )
            if resp.status_code != 200:
                logger.warning("LLM returned HTTP %s", resp.status_code)
                return LlmResponse(
                    content=f"Service error occurred (HTTP {resp.status_code})",
                    is_error=True,
                    error_code="SERVICE_ERROR",
                )
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                logger.warning("LLM response contained no choices")
                return LlmResponse(
                    content=f"No response from LLM",
                    is_error=True,
                    error_code="NO_RESPONSE",
                )
            message_content = (
                choices[0].get("message", {}).get("content", "") or ""
            ).strip()
            if not message_content:
                logger.warning("LLM returned empty content")
                return LlmResponse(
                    content=f"Empty response from LLM",
                    is_error=True,
                    error_code="EMPTY_RESPONSE",
                )
            return LlmResponse(content=message_content)
        except httpx.RequestError as e:
            logger.exception("LLM request failed")
            return LlmResponse(
                content=f"Service error occurred ({e})",
                is_error=True,
                error_code="SERVICE_ERROR",
            )
        except Exception as e:
            logger.exception("Unexpected LLM failure")
            return LlmResponse(
                content=f"Service error occurred ({e})",
                is_error=True,
                error_code="UNEXPECTED_ERROR",
            )