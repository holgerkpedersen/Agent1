"""Command package for agent interactive mode — the REPL command API.

Every REPL command is a :class:`Command` subclass registered in
:class:`CommandRegistry` and dispatched from ``agent.py``'s interactive loop.
This module is the API reference for the command surface:

- :mod:`agent_core.commands.fix` (``fix``) — repair compile errors, mypy
  errors, and runtime tracebacks in the workspace.  ``fix --desc <text>``
  applies an LLM-described change, ``fix --mypy`` types-fixes grouped errors,
  and the default mode fixes a pasted traceback.  Exit codes: 0 on success,
  1 when no fix was applicable (reported to the caller as ``execute()``'s
  return value).
- :mod:`agent_core.commands.workflow` (``workflow``) — run the full
  spec -> analyze -> plan -> entities -> taskplan pipeline in ``.docs/``.
  ``workflow . --desc <text>`` (or ``--auto``) writes the phase docs and
  prints the tailored next ``implement`` command.
- :mod:`agent_core.commands.optimize` (``optimize``) — batched static
  analysis over ``.py`` files with deterministic mechanical fixes, an
  LLM patch loop per finding, and an optional ``--apply`` write phase.
- ``chat_nlp`` (``agent.py``) — the natural-language tool loop
  (:class:`~agent_core.llm.tool_loop.ToolLoopRunner` + ``NLPParser``);
  see :mod:`agent_core.nlp_parser` for the intent contract.
"""
from .base import Command
from .registry import CommandRegistry
from .read_cmd import ReadCommand
from .write_cmd import WriteCommand
from .search_cmd import SearchCommand
from .clear_cmd import ClearCommand
from .model_cmd import ModelCommand
from .analyze_cmd import AnalyzeCommand
from .plan_cmd import PlanCommand
from .entities_cmd import EntitiesCommand
from .taskplan_cmd import TaskplanCommand

# Plan workflow components
from . import plan_schema
from . import plan_lifecycle
from . import plan_dry_run
from . import plan_decision_gate

__all__ = [
    "Command",
    "CommandRegistry",
    "ReadCommand",
    "WriteCommand",
    "SearchCommand",
    "ClearCommand",
    "ModelCommand",
    "AnalyzeCommand",
    "PlanCommand",
    "EntitiesCommand",
    "TaskplanCommand",
    "plan_schema",
    "plan_lifecycle",
    "plan_dry_run",
    "plan_decision_gate",
]
