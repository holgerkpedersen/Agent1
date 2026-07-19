# Architecture of an Agentic Loop with Tool Calling

We need an agentic loop to be able to solve long-running tasks, so we need to adopt a memory system, which can support long contexts.

## Overview

The system is built around a **ReAct-style agentic loop** (Reasoning + Acting) that iteratively:
1. Observes the current state (user input, tool results, memory)
2. Reasons about what to do next
3. Executes a tool call or produces a final response
4. Stores the interaction in memory

```
┌─────────────────────────────────────────────────┐
│                  Agentic Loop                     │
│                                                   │
│  ┌──────┐    ┌──────────┐    ┌────────────┐      │
│  │Observe│───>│  Reason  │───>│   Act      │      │
│  └──────┘    └──────────┘    └────────────┘      │
│       ^                           │               │
│       │                           v               │
│       └───────────────────┐       │               │
│                           │       │               │
│                    ┌──────┴───────┴──┐            │
│                    │  Memory Store   │            │
│                    └─────────────────┘            │
└─────────────────────────────────────────────────┘
```

## 1. Memory System

The memory system is **layered** to balance context window limits with long-term retention.

### Layers

| Layer | Scope | Storage | Eviction Policy |
|-------|-------|---------|-----------------|
| **Working Memory** | Current turn | In-memory dict | Cleared after turn |
| **Episodic Buffer** | Last N turns | Sliding window (queue) | FIFO when full |
| **Summary Store** | Compressed history | LLM-generated summaries | LRU |
| **Persistent Store** | Cross-session | File / Vector DB | Manual / TTL |

### Episodic Buffer

Stores the last N (e.g., 20) turns as raw message pairs (user + assistant). Managed as a fixed-size deque:

```python
class EpisodicBuffer:
    def __init__(self, max_turns: int = 20):
        self.turns: deque[dict] = deque(maxlen=max_turns)

    def add(self, turn: dict) -> None:
        self.turns.append(turn)

    def get_context(self) -> list[dict]:
        return list(self.turns)
```

### Summary Store

When the episodic buffer is full and older turns need to be evicted, they are **summarized** by the LLM and stored in the summary store. Summaries are injected into the system prompt on subsequent turns:

```
[System]
You are an agent. Below is a summary of earlier context:
{summary}

Current conversation:
{turns}
```

### Persistent Store

For cross-session memory, summaries and key facts are persisted to a file or vector database keyed by session/conversation ID.

## 2. Tool Calling Mechanism

Tools are registered with a **name**, **description**, and **JSON Schema** for parameters. The LLM emits structured tool calls which are dispatched to registered handlers.

### Registration

```python
@tool_registry.register(
    name="search_web",
    description="Search the web for information",
    params={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    }
)
async def search_web(query: str, num_results: int = 5) -> str:
    ...
```

### Dispatch Loop
```
LLM output ──> Parse tool call ──> Validate params ──> Execute handler ──> Return result ──> Feed back to LLM
```

### Error Handling
- **Validation errors**: Return a structured error message to the LLM so it can retry with corrected params.
- **Execution errors**: Catch exceptions, return error description to LLM.
- **Max retries**: Limit consecutive tool call retries (e.g., 3) to prevent infinite loops.

## 3. Agentic Loop (Main Orchestrator)

```python
class AgenticLoop:
    def __init__(self, llm, tool_registry, memory, max_iterations=20):
        self.llm = llm
        self.tools = tool_registry
        self.memory = memory
        self.max_iterations = max_iterations

    async def run(self, user_input: str) -> str:
        turn_count = 0
        self.memory.add_user_turn(user_input)

        while turn_count < self.max_iterations:
            context = self.memory.build_context()
            response = await self.llm.generate(context, tools=self.tools.schemas())

            if response.tool_call:
                result = await self.tools.dispatch(response.tool_call)
                self.memory.add_tool_result(response.tool_call, result)
            else:
                final = response.text
                self.memory.add_assistant_turn(final)
                return final

            turn_count += 1

        return "Max iterations reached."
```

## 4. Context Budget Management

To avoid exceeding context limits:

1. **Always include**: System prompt, latest user input, tool definitions
2. **Priority inclusion**: Recent turns from episodic buffer (most recent first)
3. **Fallback**: If buffer exceeds budget, replace oldest turns with the summary
4. **Token counting**: Use a tokenizer-aware counter before assembling context

```
┌──────────────────────────────────────────────┐
│               Context Window                  │
├──────────────────────────────────────────────┤
│ System Prompt (fixed)                        │
│ Summary (if needed)                          │
│ Recent Turns (sliding window)                │
│ Current User Input                           │
│ Tool Results (from current turn)             │
│ Tool Definitions (schema registry)           │
└──────────────────────────────────────────────┘
```

## 5. State Persistence

The full agent state (memory buffer, summary store, turn count) can be serialized and restored for resumability:

```json
{
  "session_id": "abc-123",
  "episodic_buffer": [...],
  "summary": "...",
  "turn_count": 7,
  "created_at": "...",
  "updated_at": "..."
}
```

## Summary

| Component | Responsibility |
|-----------|---------------|
| **Episodic Buffer** | Keep recent turn history for coherent multi-step reasoning |
| **Summary Store** | Compress older context to stay within token limits |
| **Tool Registry** | Define, validate, and dispatch tool calls |
| **Agentic Loop** | Orchestrate observe-reason-act cycles |
| **State Persistence** | Enable pause/resume across sessions |

This architecture scales from simple single-turn tool use to complex multi-hour agentic sessions by intelligently managing memory and context.
