import sys
import os
from typing import Any, Callable, Dict, List, Optional

# Ensure correct module resolution when running standalone or via test runners
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.types import (
    AgentBase,
    LLMProvider,
    Message,
    ToolDefinition,
    parse_json_safely,
    format_tool_result,
)


class ToolsAgent(AgentBase):
    """
    V2 Agent: Implements tool registration and execution routing.
    Demonstrates the Input -> Process (Tool Call) -> Output mental model.
    """
    
    def __init__(self, llm: LLMProvider, system_prompt: str = "You are a helpful assistant with access to tools.", max_tool_calls: int = 3):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tool_calls = max_tool_calls
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._tool_defs: List[ToolDefinition] = []

    def register_tool(self, name: str, func: Callable[..., Any], description: str) -> None:
        """Register a function as an available tool for the agent."""
        self._tools[name] = func
        # Store definition for prompt injection. Assumes ToolDefinition accepts 'name' and 'description'.
        self._tool_defs.append(ToolDefinition(name=name, description=description))

    def _format_tools_for_prompt(self) -> str:
        """Serialize registered tools into a format the LLM can parse."""
        if not self._tool_defs:
            return ""
        
        tool_descriptions = []
        for t in self._tool_defs:
            tool_descriptions.append(f"- {t.name}: {t.description}")
            
        prompt_section = (
            "\n\nYou have access to the following tools. "
            "When you need to use a tool, respond with JSON containing 'tool_calls'.\n"
            f"{chr(10).join(tool_descriptions)}\n"
            "If no tool is needed, just answer normally."
        )
        return prompt_section

    def _execute_tool_call(self, call: Dict[str, Any]) -> str:
        """Safely execute a single tool invocation in an isolated scope."""
        tool_name = call.get("name", "")
        arguments = call.get("arguments", {})
        
        if tool_name not in self._tools:
            return f"Error: Unknown tool '{tool_name}' requested."
            
        try:
            func = self._tools[tool_name]
            # Execute with isolated kwargs to prevent unexpected side effects or scope leakage
            result = func(**arguments) if isinstance(arguments, dict) else func(arguments)
            return format_tool_result(result)
        except Exception as e:
            return f"Execution failed for '{tool_name}': {type(e).__name__}: {e}"

    def run(self, user_input: str) -> str:
        """Process user input, route to tools if necessary, and return final response."""
        system_context = f"{self.system_prompt}{self._format_tools_for_prompt()}"
        
        conversation_history = [
            Message(role="system", content=system_context),
            Message(role="user", content=user_input)
        ]
        
        # 1. Initial generation to get tool calls or direct answer
        raw_response = self.llm.chat(conversation_history)
        
        # 2. Parse LLM output safely using shared utility
        parsed_data = parse_json_safely(raw_response)
        
        # Fallback if parsing fails or returns non-dict/list structure
        if not isinstance(parsed_data, (dict, list)):
            return raw_response
            
        tool_calls: List[Dict[str, Any]] = []
        if isinstance(parsed_data, dict):
            tool_calls = parsed_data.get("tool_calls", []) or []
        elif isinstance(parsed_data, list):
            tool_calls = parsed_data
            
        # If no valid tool calls found, return raw text response directly
        if not tool_calls:
            return raw_response
            
        # 3. Execute tools sequentially up to the configured limit
        observations: List[str] = []
        for call in tool_calls[:self.max_tool_calls]:
            obs_text = self._execute_tool_call(call)
            observations.append(f"[{call.get('name', 'unknown')}] {obs_text}")
            
        if not observations:
            return raw_response
            
        # 4. Inject results back into context for final synthesis
        observation_prompt = (
            f"\n\nTool execution results:\n"
            + "\n".join(observations) + 
            "\nBased on these results, provide a clear and concise answer to the user's original request."
        )
        
        conversation_history.append(Message(role="assistant", content=raw_response))
        conversation_history.append(Message(role="user", content=observation_prompt))
        
        return self.llm.chat(conversation_history)