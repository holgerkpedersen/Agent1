"""
V3 Agent: Context Window & Conversation History Management

Pedagogical Focus:
- Statefulness across turns (unlike V1's stateless request/response)
- Sliding window technique to simulate token limits
- System prompt persistence vs. conversation history pruning
- Turn alternation tracking (user ↔ assistant)

Mental Model: 📖 Bookmarks & Margins
Imagine reading a long book. You can't hold every page in your head at once,
so you keep the last few pages open (context window) while remembering the 
chapter title (system prompt). Old pages are gently closed to make room for new ones.
"""

from __future__ import annotations

import sys
from typing import List, Dict, Optional, Protocol, runtime_checkable

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies: Shared types & utilities from Phase 2
# ──────────────────────────────────────────────────────────────────────────────
try:
    from src.types import Message, LLMProvider, AgentBase, build_context_window
except ImportError:
    # Fallback definitions for standalone execution or early-phase testing
    class Message:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content
            
    @runtime_checkable
    class LLMProvider(Protocol):
        def chat(self, messages: List[Dict[str, str]]) -> str: ...
        
    @runtime_checkable
    class AgentBase(Protocol):
        def run(self, user_input: str) -> str: ...
        
    def build_context_window(messages: List[Message], max_turns: int) -> List[Dict[str, str]]:
        """Stateless utility to prune history into a sliding window."""
        # Keep only the most recent `max_turns` exchanges
        window = messages[-(max_turns * 2):] if len(messages) > (max_turns * 2) else messages
        return [{"role": m.role, "content": m.content} for m in window]


# ──────────────────────────────────────────────────────────────────────────────
# Core Implementation: Memory-Aware Agent
# ──────────────────────────────────────────────────────────────────────────────

class MemoryAgent(AgentBase):
    """
    An AI agent that maintains conversation history and respects context limits.
    
    Attributes:
        llm: Provider satisfying the LLMProvider protocol (e.g., MockLLM, OpenAI)
        system_prompt: Persistent instructions sent with every request
        max_turns: Maximum number of user/assistant exchanges to retain in memory
        history: Internal buffer storing all past messages as Message objects
    """
    
    def __init__(
        self, 
        llm: LLMProvider, 
        system_prompt: str = "You are a helpful assistant. Keep responses concise.",
        max_turns: int = 5
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.history: List[Message] = []
        
    def add_to_history(self, role: str, content: str) -> None:
        """Append a new message to the conversation buffer."""
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role '{role}'. Must be 'user', 'assistant', or 'system'.")
        
        self.history.append(Message(role=role, content=content))
        
    def get_context_window(self) -> List[Dict[str, str]]:
        """
        Prepare the payload for the LLM API.
        
        Applies the sliding window via build_context_window() to enforce 
        max_turns limits, then prepends the immutable system prompt.
        
        Returns:
            List of dictionaries matching standard OpenAI/ChatGPT message format.
        """
        # 1. Prune conversation history to respect turn limits
        pruned_history = build_context_window(self.history, self.max_turns)
        
        # 2. Reconstruct context with system prompt at index 0
        context: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        context.extend(pruned_history)
        
        return context
        
    def run(self, user_input: str) -> str:
        """
        Execute a single conversational turn.
        
        Flow:
          1. Record user message in history
          2. Build pruned context window
          3. Forward to LLM provider
          4. Record assistant response in history
          5. Return raw text response
        """
        # Step 1: Persist user turn
        self.add_to_history("user", user_input)
        
        # Step 2 & 3: Context assembly + inference
        context = self.get_context_window()
        assistant_response = self.llm.chat(context)
        
        # Step 4: Persist assistant turn (enables multi-turn continuity)
        self.add_to_history("assistant", assistant_response)
        
        return assistant_response


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Demo / Self-Test Block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Import mock provider for zero-API-key demonstration
    try:
        from src.mock_llm import MockLLM
    except ImportError:
        print("⚠️  src/mock_llm.py not found. Install dependencies or run via pytest.")
        sys.exit(0)

    print("🧠 V3 Memory Agent Demo")
    print("=" * 40)
    
    # Initialize with strict memory limits to observe pruning behavior
    demo_llm = MockLLM(config={"temperature": 0.1})
    agent = MemoryAgent(
        llm=demo_llm, 
        system_prompt="You are a history tutor. Answer briefly.",
        max_turns=2  # Only keep last 2 exchanges in context
    )
    
    test_inputs = [
        "Who built the pyramids?",
        "What materials did they use?",
        "How long did construction take?",
        "Where are they located?"
    ]
    
    for i, query in enumerate(test_inputs, 1):
        print(f"\n👤 Turn {i}: {query}")
        response = agent.run(query)
        print(f"🤖 Response: {response}")
        
        # Debug view: show what's actually sent to the LLM
        window = agent.get_context_window()
        print(f"📜 Context Window Size: {len(window)} messages")
        for msg in window:
            role_tag = "SYS" if msg["role"] == "system" else ("USR" if msg["role"] == "user" else "AST")
            print(f"   [{role_tag}] {msg['content'][:50]}...")
            
    print("\n✅ Demo complete. Notice how older turns drop out as max_turns is enforced.")