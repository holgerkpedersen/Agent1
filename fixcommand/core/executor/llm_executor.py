import asyncio
from typing import List, Dict, Any

import openai

from ..parser.structured import parse_tool_calls, execute_tool_call
from ..prompt.generator import generate_prompt_with_tools
from ..tools.definitions import TOOLS


class LLMExecutor:
    def __init__(self, api_key: str) -> None:
        self.client = openai.OpenAI(api_key=api_key)

    async def run_fix_command(self, issue_description: str) -> List[str]:
        messages = generate_prompt_with_tools(issue_description)

        response = await asyncio.to_thread(
            lambda: self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        )

        message = response.choices[0].message

        if hasattr(message, "model_dump"):
            response_dict: Dict[str, Any] = message.model_dump()
        else:
            response_dict = dict(message)

        parsed_calls = parse_tool_calls(response_dict)

        results: List[str] = []
        for call in parsed_calls:
            result = await asyncio.to_thread(execute_tool_call, call)
            results.append(result)

        return results
