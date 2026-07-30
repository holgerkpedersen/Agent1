"""Parser package for structured tool call parsing.

Re-exports the canonical ``parse_tool_calls`` from ``.structured``.
"""
from .structured import parse_tool_calls

__all__ = ["parse_tool_calls"]
