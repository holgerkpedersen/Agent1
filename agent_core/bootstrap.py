"""Import bootstrap for the agent entry point (refactor step t02).

Why this module exists
---------------------
``agent.py`` used to carry ~71 ``import`` statements plus two module-level
side effects:

1. ``OPENBLAS_NUM_THREADS`` defaulted to ``"1"`` — this **must** happen before
   ``numpy``/``scipy`` are imported anywhere in the process, so it runs FIRST
   in this module, before any third-party import.
2. ``sys.stdout``/``sys.stderr`` reconfigured to UTF-8 — the Windows console
   default codec (cp1252) cannot encode the box-drawing glyphs the UI prints.

Both side effects and the whole import block now live here, and ``agent.py``
does ``from agent_core.bootstrap import *`` as its first statement. ``*``
respects ``__all__`` below, which is the exact public surface the import block
produced at baseline (see ``tests/test_agent_module_surface.py``).

Import order is load-bearing: keep the ``os.environ`` line above every
import that can pull numpy in at module-import time.
"""

import os

# Restrict OpenBLAS to a single thread before any numpy import.  This must
# stay the first thing that touches the environment: none of the imports
# below pull numpy in at module-import time, and moving this after them is
# only safe for that reason.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import asyncio
import concurrent.futures
import contextlib
import io
import json
import logging
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, cast

from agent_core import to_windows_path
from agent_core.commands.analyze_cmd import AnalyzeCommand
from agent_core.commands.base import (
    Command,
    FlowStopped,
    chat_stoppable,
    clear_stop,
    save_file_py,
)
from agent_core.commands.clear_cmd import ClearCommand
from agent_core.commands.cleanup_cmd import CleanupCommand
from agent_core.commands.decide_cmd import DecideCommand
from agent_core.commands.demo_data_cmd import DemoDataCommand
from agent_core.commands.display_cmd import DisplayCommand
from agent_core.commands.entities_cmd import EntitiesCommand
from agent_core.commands.fix_cmd import FixCommand
from agent_core.commands.git_cmd import GitCommand
from agent_core.commands.implement_cmd import ImplementCommand
from agent_core.commands.model_cmd import ModelCommand
from agent_core.commands.multillm_cmd import MultiLlmCommand
from agent_core.commands.optimize_cmd import OptimizeCommand
from agent_core.commands.paste_cmd import PasteCommand
from agent_core.commands.paste_image_cmd import PasteImageCommand
from agent_core.commands.perf_cmd import PerfCommand, PerfTracker
from agent_core.commands.plan_cmd import PlanCommand
from agent_core.commands.read_cmd import ReadCommand
from agent_core.commands.reconstruct_cmd import ReconstructCommand
from agent_core.commands.registry import CommandRegistry
from agent_core.commands.review_cmd import ReviewCommand
from agent_core.commands.run_cmd import RunCommand
from agent_core.commands.search_cmd import SearchCommand
from agent_core.commands.self_heal_cmd import SelfHealCommand
from agent_core.commands.taskplan_cmd import TaskplanCommand
from agent_core.commands.workflow_cmd import WorkflowCommand
from agent_core.commands.write_cmd import WriteCommand
from agent_core.config import AgentDisplayMode, load_agent_settings
from agent_core.constants import (  # noqa: F401
    AGENT_MEMORY_JSON_PATH,
    AGENT_MEMORY_TMP_PATH,
    CHAT_HISTORY_JSON_PATH,
    CHAT_HISTORY_TMP_PATH,
    LOOP_NOTE_TAG_KEY,
    TURN_LOG_PATH,
    resolve_model,
)
from agent_core.colors import blue, cyan, gray, green, magenta, red, yellow
from agent_core.context_management import CORRELATION_ID_CTX, CorrelationIdContext
from agent_core.decisions import decisions_as_system_prompt
from agent_core.file_history import FILE_HISTORY_MARKER, file_notes_block
from agent_core.file_searcher import FileSearcher
from agent_core.file_system import FileSystem
from agent_core.habits import HABITS_MARKER, habits_block, load_habits
from agent_core.hue.bridge import HueBridge, HueBridgeError
from agent_core.llm.learning import record_turn_outcome, recommend_profile
from agent_core.llm.learning import save_weights as save_meta_policy_weights
from agent_core.llm.provider import (
    get_last_metrics,
    is_connection_failure,
    provider_supports_streaming,
)
from agent_core.llm.tool_loop import ToolLoopRunner
from agent_core.memory import (
    MAX_GOAL_CHARS,
    ORIGINAL_GOAL_MARKER,
    SEMANTIC_MEMORY_MARKER,
    load_semantic_memory,
    original_goal_block,
    original_goal_from_block,
    semantic_memory_block,
    semantic_search_memories,
)
from agent_core.modes import (
    MODE_BUILD,
    check_tool_allowed,
    filter_tool_schemas,
    is_plan_mode,
    plan_mode_system_suffix,
    plan_mode_turn_note,
)
from agent_core.path_utils import resolve_path, safe_path
from agent_core.pytest_gate import (
    DEFAULT_FULL_SUITE_TIMEOUT,
    FullRunGate,
    record_full_run_seconds,
)
from agent_core.security.allowlist import (
    DESTRUCTIVE_SHELL_PATTERNS,
    find_destructive_shell_pattern,
)
from agent_core.skills import (
    DEFAULT_PAGE_LINES,
    SKILL_INDEX_MARKER,
    SkillError,
    discover_skills,
    format_skill_page,
    load_skill_index,
    match_skills_for_input,
    read_skill,
    skill_hint_block,
)
from agent_core.subagent_roles import get_role, role_names
from agent_core.symbol_intel import collect_definitions, collect_references
from agent_core.text_encoding import (
    NonUtf8TextError,
    UnrepresentableTextError,
    encode_in_encoding,
    read_text_preserving,
)
from agent_core.tool_dispatcher import ToolDispatcher
from agent_core.tool_schemas import NLP_TOOL_NAMES, NLP_TOOL_SCHEMAS

try:
    from harnessfix.tracing import TraceWriter, trace_enabled
except Exception:  # pragma: no cover - tracing degrades gracefully if unavailable
    TraceWriter = None  # type: ignore[assignment, misc]

    def trace_enabled() -> bool:
        return False


if TYPE_CHECKING:
    pass

# Windows console default codec is cp1252, which cannot encode many Unicode
# glyphs the UI prints (box-drawing chars, arrows). Reconfigure stdout/stderr
# to UTF-8 at startup so no print() crashes on non-cp1252 characters.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (ValueError, OSError, AttributeError):
            pass

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_MEMORY_JSON_PATH",
    "AGENT_MEMORY_TMP_PATH",
    "AgentDisplayMode",
    "AnalyzeCommand",
    "Any",
    "Awaitable",
    "CHAT_HISTORY_JSON_PATH",
    "CHAT_HISTORY_TMP_PATH",
    "CORRELATION_ID_CTX",
    "Callable",
    "CleanupCommand",
    "ClearCommand",
    "Command",
    "CommandRegistry",
    "CorrelationIdContext",
    "DEFAULT_FULL_SUITE_TIMEOUT",
    "DEFAULT_PAGE_LINES",
    "DESTRUCTIVE_SHELL_PATTERNS",
    "DecideCommand",
    "DemoDataCommand",
    "DisplayCommand",
    "EntitiesCommand",
    "FILE_HISTORY_MARKER",
    "FileSearcher",
    "FileSystem",
    "FixCommand",
    "FlowStopped",
    "FullRunGate",
    "GitCommand",
    "HABITS_MARKER",
    "HueBridge",
    "HueBridgeError",
    "ImplementCommand",
    "Iterator",
    "LOOP_NOTE_TAG_KEY",
    "MAX_GOAL_CHARS",
    "MODE_BUILD",
    "ModelCommand",
    "MultiLlmCommand",
    "NLP_TOOL_NAMES",
    "NLP_TOOL_SCHEMAS",
    "NonUtf8TextError",
    "ORIGINAL_GOAL_MARKER",
    "OptimizeCommand",
    "Optional",
    "PasteCommand",
    "PasteImageCommand",
    "PerfCommand",
    "PerfTracker",
    "PlanCommand",
    "ReadCommand",
    "ReconstructCommand",
    "ReviewCommand",
    "RunCommand",
    "SEMANTIC_MEMORY_MARKER",
    "SKILL_INDEX_MARKER",
    "SearchCommand",
    "SelfHealCommand",
    "SkillError",
    "TURN_LOG_PATH",
    "TYPE_CHECKING",
    "TaskplanCommand",
    "ToolDispatcher",
    "ToolLoopRunner",
    "TraceWriter",
    "UnrepresentableTextError",
    "WorkflowCommand",
    "WriteCommand",
    "asyncio",
    "blue",
    "cast",
    "chat_stoppable",
    "check_tool_allowed",
    "clear_stop",
    "collect_definitions",
    "collect_references",
    "concurrent",
    "contextlib",
    "cyan",
    "datetime",
    "decisions_as_system_prompt",
    "defaultdict",
    "discover_skills",
    "encode_in_encoding",
    "file_notes_block",
    "filter_tool_schemas",
    "find_destructive_shell_pattern",
    "format_skill_page",
    "get_last_metrics",
    "get_role",
    "gray",
    "green",
    "habits_block",
    "io",
    "is_connection_failure",
    "is_plan_mode",
    "json",
    "load_agent_settings",
    "load_habits",
    "load_semantic_memory",
    "load_skill_index",
    "logger",
    "logging",
    "magenta",
    "match_skills_for_input",
    "original_goal_block",
    "original_goal_from_block",
    "os",
    "plan_mode_system_suffix",
    "plan_mode_turn_note",
    "provider_supports_streaming",
    "re",
    "read_skill",
    "read_text_preserving",
    "recommend_profile",
    "record_full_run_seconds",
    "record_turn_outcome",
    "red",
    "resolve_model",
    "resolve_path",
    "role_names",
    "safe_path",
    "save_file_py",
    "save_meta_policy_weights",
    "semantic_memory_block",
    "semantic_search_memories",
    "shlex",
    "signal",
    "skill_hint_block",
    "subprocess",
    "sys",
    "time",
    "to_windows_path",
    "trace_enabled",
    "traceback",
    "yellow",
]
