"""
Context optimizer — decides the best way to represent a file to an LLM.

This component implements a "Smart Selection" strategy to reduce token usage 
by providing either a full file, a skeleton (signatures), or a snippet, 
based on the user's intent and file size.
"""

from typing import List, Optional
from pathlib import Path
from agent_core.file_context_retriever import FileContextRetriever


class ContextOptimizer:
    """
    Decides the most efficient way to represent a file in the LLM context.

    Strategies:
    - FULL: The entire file content (for small files or explicit requests).
    - SKELETON: Only class/function signatures and docstrings (for large files or structural queries).
    - SNIPPET: A subset of lines (for error-specific or targeted queries).
    """

    # Thresholds for deciding between FULL and SKELETON
    # If a file is larger than this, we default to SKELETON unless requested.
    _MAX_FULL_SIZE_BYTES = 10_000  # ~10KB
    _MAX_FULL_LINES = 300

    def __init__(self, retriever: FileContextRetriever) -> None:
        self._retriever = retriever

    def optimize(self, message: str, filenames: List[str]) -> List[str]:
        """
        Analyze the message and filenames to return the best list of context strings.
        """
        optimized_contexts: List[str] = []
        message_lower = message.lower()

        for fn in filenames:
            # 1. Check for explicit "skeleton" intent
            if any(word in message_lower for word in ["skeleton", "structure", "signatures", "what is the api"]):
                skeleton = self._retriever.retrieve_skeleton(fn)
                if skeleton:
                    optimized_contexts.append(f"--- SKELETON OF {fn} ---\n{skeleton}")
                    continue

            # 2. Check for snippet intent (very basic heuristic)
            # If message contains "line" or "error at", we might want to do something,
            # but without line numbers in the message, we can't easily use retrieve_snippet.
            # For now, we'll skip snippet optimization unless the user provides lines.

            # 3. Handle size-based optimization (Default behavior)
            content = self._retriever.retrieve(fn)
            if content is None:
                continue

            lines = content.splitlines()
            file_size = len(content.encode('utf-8'))

            # If file is large, prioritize the skeleton to save tokens
            if file_size > self._MAX_FULL_SIZE_BYTES or len(lines) > self._MAX_FULL_LINES:
                # If the user didn't explicitly ask for the whole file (via a keyword like "full")
                if "full" not in message_lower and "entire" not in message_lower:
                    skeleton = self._retriever.retrieve_skeleton(fn)
                    if skeleton:
                        optimized_contexts.append(f"--- SKELETON OF {fn} (File is large) ---\n{skeleton}")
                        continue

            # 4. Fallback to full content
            optimized_contexts.append(content)

        return optimized_contexts
