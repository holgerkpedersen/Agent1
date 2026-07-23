"""
src/agent_v2_tools.py
Step 2: Tool-Use Agent
Introduces function calling, tool registration, argument parsing, and safe execution.
Mental Model: Input -> LLM decides on tool -> Parse JSON args -> Execute safely -> Inject result -> Output
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List

# Shared contracts & utilities (Phase 2 dependency)
try:
    from .types import ToolDefinition, parse_json_safely, format_tool_result
except ImportError:
    # Fallback stubs for standalone execution or incomplete environment setup
    def parse_json_safely(text: str) -> Any: ...
    def format_tool_result(res: Any) -> str: ...


class ToolAgent:
    """
    An agent capable of registering and executing external tools/functions.
    
    Pedagogical Focus:
    - How to expose Python functions as "tools" for an LLM
    - Parsing structured JSON outputs from unstructured LLM text
    - Safe execution sandboxing & error handling
    - Injecting tool results back into the context window
    """

    def __init__(self, llm: Any, system_prompt: str = "You are a helpful assistant."):
        self.llm = llm
        self.system_prompt = system_prompt
        # Internal registry: name -> {definition, function}
        self._tools_registry: Dict[str, Dict] = {}

    def register_tool(self, name: str, func: Callable[..., Any], description: str) -> None:
        """
        Register a Python function as an available tool.
        Automatically extracts argument names from the function signature for schema generation.
        """
        sig = inspect.signature(func)
        args_schema = list(sig.parameters.keys())
        
        # Construct definition compatible with ToolDefinition TypedDict/Dataclass contracts
        tool_def: Any = {
            "name": name,
            "description": description,
            "args": args_schema
        }
            
        self._tools_registry[name] = {
            "definition": tool_def,
            "function": func
        }

    def _parse_tool_calls(self, llm_output: str) -> List[Dict]:
        """
        Extract tool calls from raw LLM text.
        Expects JSON format: [{"name": "...", "arguments": {...}}, ...]
        Returns empty list if parsing fails or structure is invalid.
        """
        parsed = parse_json_safely(llm_output)  # type: ignore[possibly-undefined, assignment]
        
        if isinstance(parsed, dict):
            # Some LLMs return a single call object instead of a list
            parsed = [parsed]
        elif not isinstance(parsed, list):
            return []
            
        valid_calls = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item and "arguments" in item:
                valid_calls.append(item)
                
        return valid_calls

    def execute_tools(self, calls: List[Dict]) -> List[str]:
        """
        Safely execute registered tools with provided arguments.
        Catches execution errors and wraps results using shared formatting utilities.
        Returns a list of formatted result strings ready for LLM consumption.
        """
        results = []
        for call in calls:
            tool_name = call.get("name")
            args = call.get("arguments", {})
            
            if tool_name not in self._tools_registry:
                raw_err = f"Error: Tool '{tool_name}' is not registered."
                results.append(format_tool_result(raw_err))  # type: ignore[possibly-undefined, arg-type]
                continue
                
            func = self._tools_registry[tool_name]["function"]
            
            try:
                # Execute function with unpacked keyword arguments
                raw_result = func(**args)
                results.append(format_tool_result(raw_result))  # type: ignore[possibly-undefined, arg-type]
            except Exception as e:
                error_msg = f"Execution failed for '{tool_name}': {type(e).__name__}: {e}"
                results.append(format_tool_result(error_msg))  # type: ignore[possibly-undefined, arg-type]
                
        return results

    def run(self, user_input: str) -> str:
        """
        Main execution loop for tool-augmented responses.
        
        Flow:
        1. Build initial context (system + user)
        2. Query LLM
        3. If tool calls detected -> parse & execute
        4. Inject results back into context & query again for final answer
        """
        # 1. Initial context setup
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # 2. First LLM pass: decision & tool selection
        llm_response = self.llm.chat(messages)
        
        # 3. Check for structured tool requests
        tool_calls = self._parse_tool_calls(llm_response)
        
        if not tool_calls:
            # Direct answer path: no tools needed
            return llm_response
            
        # 4. Execution phase: run requested tools safely
        execution_results = self.execute_tools(tool_calls)
        
        # 5. Synthesis phase: feed results back to LLM for final response generation
        messages.append({"role": "assistant", "content": llm_response})
        combined_results = "\n---\n".join(execution_results)
        messages.append({"role": "tool", "content": combined_results})
        
        # Second pass: generate human-readable answer based on tool outputs
        final_response = self.llm.chat(messages)
        return final_response

    def get_available_tools(self) -> List[Dict]:
        """Return a list of registered tool definitions (useful for prompt injection/debugging)."""
        return [data["definition"] for data in self._tools_registry.values()]