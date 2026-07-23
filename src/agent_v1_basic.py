"""
📘 Step 1: Minimal Agent Loop (BasicAgent)

Teaches the fundamental anatomy of an AI agent:
  Input → Process → Output

This version focuses purely on message formatting and a single LLM call.
No tools, no memory, no multi-step reasoning—just the core I/O loop.

Mental Model:
  1️⃣ FORMAT   : Combine system instructions + user input into a message list
  2️⃣ PROCESS  : Send messages to the LLM provider
  3️⃣ OUTPUT   : Return the raw string response
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    """Minimal contract for any LLM wrapper (MockLLM, OpenAI, Anthropic, etc.)"""
    
    def chat(self, messages: list[dict]) -> str: ...


class BasicAgent:
    """
    A minimal agent that satisfies the AgentBase mental model.
    
    Usage:
        agent = BasicAgent(llm=my_llm, system_prompt="You are a helpful assistant.")
        response = agent.run("What is 2+2?")
    """

    def __init__(self, llm: Any, system_prompt: str) -> None:
        """
        Initialize the basic agent.
        
        Args:
            llm: An object satisfying the LLMProvider protocol (must implement .chat(messages)).
            system_prompt: Instructions that define the agent's behavior/persona.
        """
        if not hasattr(llm, "chat"):
            raise TypeError("llm must have a .chat(messages) method")
            
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, user_input: str) -> str:
        """
        Execute a single turn of the agent loop.
        
        Args:
            user_input: The text prompt from the user.
            
        Returns:
            str: The generated response from the LLM.
        """
        # 1️⃣ FORMAT: Build context window
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 2️⃣ PROCESS: Call LLM provider
        response = self.llm.chat(messages)

        # 3️⃣ OUTPUT: Return raw string
        return response


# ---------------------------------------------------------------------------
# 🧪 Quick Self-Test / Dry-Run (Optional for beginners)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class _DummyLLM:
        def chat(self, messages: list[dict]) -> str:
            # Echo back to prove message structure works
            return f"[System] {messages[0]['content']}\n[User] {messages[1]['content']}"

    dummy = _DummyLLM()
    agent = BasicAgent(llm=dummy, system_prompt="You are a concise math tutor.")
    
    print("🚀 Running BasicAgent dry-test...")
    out = agent.run("What is 5 * 3?")
    print(f"✅ Output:\n{out}")