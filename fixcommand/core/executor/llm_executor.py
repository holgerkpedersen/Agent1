import asyncio
from typing import List, Dict, Any, Optional
import openai

from ..parser.structured import parse_tool_calls, execute_tool_call
from ..prompt.generator import generate_prompt_with_tools
from ..tools.definitions import TOOLS, ToolCallResult


class LLMExecutorError(Exception):
    """Raised when LLM execution or tool dispatch fails."""


class LLMExecutor:
    """Executor that queries an LLM and dispatches returned tool calls."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run_fix_command(self, issue_description: str) -> List[str]:
        """Generate a prompt, query the LLM, and execute returned tool calls concurrently."""
        if not issue_description.strip():
            raise LLMExecutorError("issue_description must not be empty")

        messages = generate_prompt_with_tools(issue_description)

        try:
            response = await asyncio.to_thread(self._create_chat_completion, messages)
        except openai.OpenAIError as exc:
            raise LLMExecutorError(f"OpenAI API call failed: {exc}") from exc

        if not response.choices:
            raise LLMExecutorError("No choices returned in LLM response")

        response_dict = self._extract_response_dict(response.choices[0].message)
        parsed_calls: List[ToolCallResult] = parse_tool_calls(response_dict)

        if not parsed_calls:
            return []

        results = await asyncio.gather(
            *(self._execute_tool_safe(call) for call in parsed_calls),
            return_exceptions=True,
        )
        return [str(r) if not isinstance(r, Exception) else f"Error: {r}" for r in results]

    def _create_chat_completion(self, messages: List[Dict[str, Any]]):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _extract_response_dict(message: Any) -> Dict[str, Any]:
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return dict(message)

    @staticmethod
    async def _execute_tool_safe(call: ToolCallResult) -> str:
        try:
            return await asyncio.to_thread(execute_tool_call, call)
        except Exception as exc:
            return f"Error: {exc}"
