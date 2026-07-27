"""Tool calling loop orchestrator for LLM conversations."""
import json
from typing import Callable, Awaitable


class ToolLoopRunner:
    """Orchestrates tool calling loop with LLM.
    
    Extracted from LLMClient.chat_with_tool_loop to separate tool orchestration
    from LLM communication. This enables testing tool logic without a running LLM.
    """
    
    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations
    
    async def run(
        self,
        messages: list[dict],
        llm_chat_fn: Callable[[list[dict], list[dict]], Awaitable[tuple[str, list[dict]]]],
        execute_tool_fn: Callable[[str, dict], Awaitable[str]],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:
        """Run conversation with automatic tool calling loop.
        
        Args:
            messages: Initial conversation messages
            llm_chat_fn: Async function that sends messages to LLM and returns
                        (response_text, updated_messages_with_reasoning)
            execute_tool_fn: Async function that executes a tool and returns result
            tools: Optional tool schemas (uses default if None)
            
        Returns:
            Tuple of (final_text, updated_messages)
        """
        if not tools:
            from agent import AGENT_TOOL_SCHEMAS
            tools = AGENT_TOOL_SCHEMAS
        
        all_text_parts = []
        current_messages = [dict(m) for m in messages]
        
        for iteration in range(self.max_iterations):
            # Call LLM
            response_text, updated_messages = await llm_chat_fn(current_messages, tools)
            current_messages = updated_messages
            
            if response_text:
                all_text_parts.append(response_text)
            
            # Check for tool calls in the last assistant message
            last_msg = current_messages[-1] if current_messages else {}
            tool_calls = last_msg.get("tool_calls", [])
            
            # No tool calls - we're done
            if not tool_calls:
                break
            
            # Execute each tool call
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                
                result_str = await execute_tool_fn(tool_name, args)
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str
                })
        
        return "\n".join(all_text_parts), current_messages
