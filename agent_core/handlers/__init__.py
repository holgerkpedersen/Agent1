"""Command handler base interfaces and the analyze handler.

``base_handler`` defines the :class:`CommandHandler` protocol and the
:class:`BaseCommandHandler` ABC; ``analyze_handler`` provides the AST-based
analyze command; ``nlp_tools`` holds the NLP tool handlers extracted from
Agent (imported lazily by agent.py — not re-exported here to avoid a
circular import).
"""

from .base_handler import BaseCommandHandler, CommandHandler
from .analyze_handler import AnalyzeCommand

__all__ = ["BaseCommandHandler", "CommandHandler", "AnalyzeCommand"]
