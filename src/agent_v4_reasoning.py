"""
Agent V4: Reasoning & Planning (ReAct Pattern)

Implements a multi-step reasoning loop where the agent thinks, 
calls tools iteratively, observes results, and formulates a final answer.
Teaches planning -> tool use -> reflection -> final answer.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Callable

# Import shared types & utilities from the project root
try:
    from .types import LLMProvider, ToolDefinition, ReasoningStep, parse_json_safely, format_tool_result
except ImportError:
    # Fallback stubs for isolated execution/testing if types.py isn't available yet
    class LLMProvider: pass
    class ToolDefinition: pass
    class ReasoningStep: 
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def parse_json_safely(text): return text
    def format_tool_result(res): return str(res)

class ReasoningAgent:
    """
    An agent that uses a ReAct (Reason + Act) loop to solve complex tasks.
    
    It iteratively generates thoughts, decides on actions (tool calls), 
    executes them, observes the output, and repeats until it finds a final answer 
    or reaches the maximum step limit.
    """
    
    REACT_SYSTEM_PROMPT = """You are an AI assistant that solves problems step-by-step.
Use the following format:

Thought: Reason about what to do next.
Action: The tool to use (one of [{tool_names}]).
Action Input: The input for the tool.
Observation: The result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: [Your final response]

Always end with 'Final Answer:' when you are ready to respond."""

    def __init__(self, llm: LLMProvider, tools: Optional[List[Dict[str, Any]]] = None, max_steps: int = 5):
        """
        Initialize the ReasoningAgent.
        
        Args:
            llm: An object satisfying the LLMProvider protocol (must have .chat() method).
            tools: List of tool definitions containing 'name', 'description', and 'func'.
            max_steps: Maximum number of reasoning iterations before forcing a stop.
        """
        self.llm = llm
        self.tools = tools or []
        self.max_steps = max_steps
        
    def _get_tool_names(self) -> str:
        """Extract available tool names for the prompt."""
        if not self.tools:
            return "None"
        return ", ".join(t.get("name", t.get("function", {}).get("name", "")) for t in self.tools)

    def _format_react_prompt(self, user_input: str, observations: List[str]) -> str:
        """Build the complete ReAct prompt with history and available tools."""
        tool_names = self._get_tool_names()
        system_prompt = self.REACT_SYSTEM_PROMPT.format(tool_names=tool_names)
        
        obs_history = "\n".join(observations) if observations else ""
        history_part = f"\n{obs_history}" if obs_history else ""
        
        return f"""{system_prompt}

User Query: {user_input}
{history_part}
Thought:"""

    def _parse_llm_output(self, text: str) -> Dict[str, Any]:
        """Extract Thought, Action, Action Input, or Final Answer from LLM output."""
        # Regex patterns for ReAct format
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)\s*Action Input:\s*(.*)", text, re.DOTALL)
        final_answer_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        
        result: Dict[str, Any] = {}
        
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        else:
            result["thought"] = "Proceeding based on previous context."
            
        if action_match:
            result["action"] = action_match.group(1).strip().lower()
            result["action_input"] = action_match.group(2).strip()
        elif final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip()
            result["is_final"] = True
        else:
            # Fallback if LLM doesn't follow format strictly
            result["thought"] = text.strip()
            result["final_answer"] = text.strip()
            result["is_final"] = True
            
        return result

    def _execute_tool(self, tool_name: str, tool_input: Any) -> str:
        """Execute a registered tool and return the observation string."""
        for tool in self.tools:
            t_name = tool.get("name", tool.get("function", {}).get("name", ""))
            if t_name.lower() == tool_name.lower():
                func = tool.get("func") or tool.get("execute")
                if callable(func):
                    try:
                        # Safely parse JSON input if provided as string, else pass directly
                        inp = parse_json_safely(tool_input) if isinstance(tool_input, str) and tool_input.strip().startswith("{") else tool_input
                        res = func(**inp) if isinstance(inp, dict) else func(inp)
                        return format_tool_result(res)
                    except Exception as e:
                        return f"Error executing tool '{tool_name}': {str(e)}"
        return f"Unknown tool '{tool_name}'. Available tools: {self._get_tool_names()}"

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Run the ReAct reasoning loop.
        
        Returns a dictionary containing:
            - reasoning_trace: List of step dictionaries (compatible with ReasoningStep)
            - final_answer: The resolved answer string
            - steps_taken: Number of iterations performed
        """
        steps = []
        observations = []
        
        for step_num in range(1, self.max_steps + 1):
            # 1. Prompt LLM for next thought/action
            prompt = self._format_react_prompt(user_input, observations)
            llm_response = self.llm.chat([{"role": "user", "content": prompt}])
            
            # 2. Parse structured output
            parsed = self._parse_llm_output(llm_response)
            
            current_step: Dict[str, Any] = {
                "step_number": step_num,
                "thought": parsed.get("thought", ""),
                "action": parsed.get("action"),
                "action_input": parsed.get("action_input"),
                "observation": None,
                "is_final": False
            }
            
            # 3. Check for early exit (Final Answer)
            if parsed.get("is_final"):
                current_step["is_final"] = True
                current_step["final_answer"] = parsed.get("final_answer", "")
                steps.append(current_step)
                return {
                    "reasoning_trace": steps,
                    "final_answer": current_step["final_answer"],
                    "steps_taken": step_num
                }
            
            # 4. Execute Action if present
            if parsed.get("action") and parsed.get("action_input"):
                observation = self._execute_tool(parsed["action"], parsed["action_input"])
                current_step["observation"] = observation
                observations.append(f"Action: {parsed['action']}\nAction Input: {parsed['action_input']}\nObservation: {observation}")
            else:
                # Fallback if LLM skips action but isn't final
                current_step["is_final"] = True
                current_step["final_answer"] = parsed.get("thought", "No further actions needed.")
                
            steps.append(current_step)
            
        # 5. Max steps reached without explicit final answer
        fallback_answer = "Reached maximum reasoning steps. Could not determine a definitive final answer."
        if not any(s.get("is_final") for s in steps):
            steps[-1]["is_final"] = True
            steps[-1]["final_answer"] = fallback_answer
            
        return {
            "reasoning_trace": steps,
            "final_answer": fallback_answer,
            "steps_taken": self.max_steps
        }

# Module-level convenience for CLI/Tutorial integration
if __name__ == "__main__":
    print("✅ agent_v4_reasoning.py loaded successfully.")
    print("👉 Instantiate ReasoningAgent(llm=..., tools=[...], max_steps=5)")