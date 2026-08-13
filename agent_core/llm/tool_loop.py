"""Tool calling loop orchestrator for LLM conversations."""
import json
from typing import Callable, Awaitable, Any


class ToolLoopRunner:
    """Orchestrates tool calling loop with LLM.
    
    Extracted from LLMClient.chat_with_tool_loop to separate tool orchestration
    from LLM communication. This enables testing tool logic without a running LLM.
    """
    
    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations
    
    async def run(
        self,
        messages: list[dict[str, Any]],
        llm_chat_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[tuple[str, list[dict[str, Any]]]]],
        execute_tool_fn: Callable[[str, dict[str, Any]], Awaitable[str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
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
            tools = []
        
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
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                
                print(f"  [tool] {tool_name}({_fmt_args(args)})")
                try:
                    result_str = await execute_tool_fn(tool_name, args)
                except Exception as exc:
                    result_str = f"Tool error: {exc}"
                print(f"  [result] {result_str[:200]}")
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str
                })
        
        # Only the LAST non-empty text matters: earlier texts are the model's
        # intermediate narration ("I will read the file...") and would clutter
        # the final answer the user sees.
        final_text = ""
        for part in reversed(all_text_parts):
            if part and part.strip():
                final_text = part
                break
        return final_text, current_messages


def _fmt_args(args: dict[str, Any]) -> str:
    """Short one-line rendering of tool arguments for the console."""
    pieces = []
    for key, value in list(args.items())[:4]:
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        pieces.append(f"{key}={text}")
    return ", ".join(pieces)
