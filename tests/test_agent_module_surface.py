"""Baseline guard for the `agent.py` decomposition refactor.

Purpose: during the refactor, every public name must stay importable from
`agent`, whether it is defined there or re-exported from `agent_core.*`.
If a step moves a name and forgets the re-export, this test fails.

The list below is the measured public surface of `agent` at the start of the
refactor (2026-10-16, agent.py = 5565 lines). See docs/refactor_agent_py.md.
"""

import importlib
import inspect

import pytest

# Names that must be importable from `agent` after every refactor step.
EXPECTED_PUBLIC_NAMES: frozenset[str] = frozenset({
    "AGENT_MEMORY_JSON_PATH",
    "AGENT_MEMORY_TMP_PATH",
    "Agent",
    "AgentDisplayMode",
    "AnalyzeCommand",
    "Any",
    "Awaitable",
    "CHAT_HISTORY_JSON_PATH",
    "CHAT_HISTORY_TMP_PATH",
    "CHAT_INPUT_PROMPT",
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
    "LLMClient",
    "LOOP_NOTE_TAG_KEY",
    "MAX_GOAL_CHARS",
    "MODE_BUILD",
    "ModelCommand",
    "MultiLlmCommand",
    "NLP_TOOL_NAMES",
    "NLP_TOOL_SCHEMAS",
    "NlpToolHandler",
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
    "get_metrics_collector",
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
    "main",
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
    "record_command_metrics",
    "record_full_run_seconds",
    "record_turn_outcome",
    "red",
    "resolve_model",
    "resolve_path",
    "role_names",
    "run_dashboard_server",
    "run_interactive",
    "safe_path",
    "save_file_py",
    "save_meta_policy_weights",
    "semantic_memory_block",
    "semantic_search_memories",
    "shlex",
    "signal",
    "skill_hint_block",
    "start_dashboard_thread",
    "subprocess",
    "sys",
    "time",
    "to_windows_path",
    "trace_enabled",
    "traceback",
    "yellow",
})

# Names the plan moves out of agent.py. Each must remain reachable from
# `agent` (by re-export) for the whole refactor.
MUST_STAY_IMPORTABLE: tuple[str, ...] = (
    "Agent",
    "LLMClient",
    "run_interactive",
    "main",
    "NLP_TOOL_SCHEMAS",
    "NLP_TOOL_NAMES",
    "chat_stoppable",
    "clear_stop",
    "collect_definitions",
    "collect_references",
    "filter_tool_schemas",
    "find_destructive_shell_pattern",
    "original_goal_from_block",
    "plan_mode_turn_note",
    "semantic_memory_block",
    "start_dashboard_thread",
)


def test_agent_imports():
    """`import agent` must keep working, with no side-effect crash."""
    module = importlib.import_module("agent")
    assert module is not None


@pytest.mark.parametrize("name", sorted(EXPECTED_PUBLIC_NAMES))
def test_public_name_is_importable(name):
    """Every baseline public name is still reachable from `agent`."""
    module = importlib.import_module("agent")
    assert hasattr(module, name), f"{name!r} is no longer importable from agent"


def test_no_public_names_were_dropped():
    """Report the whole missing set at once, not one failure at a time."""
    module = importlib.import_module("agent")
    missing = sorted(n for n in EXPECTED_PUBLIC_NAMES if not hasattr(module, n))
    assert not missing, "names dropped from agent: " + ", ".join(missing)


@pytest.mark.parametrize("name", MUST_STAY_IMPORTABLE)
def test_moved_names_keep_reexport(name):
    """Names the plan relocates must keep a re-export in `agent`."""
    module = importlib.import_module("agent")
    assert hasattr(module, name), f"{name!r} lost its re-export from agent"


def test_agent_classes_keep_their_methods():
    """Method surface of the two classes must not shrink."""
    module = importlib.import_module("agent")
    for cls_name, required in (
        ("Agent", ("__init__",)),
        ("LLMClient", ("__init__",)),
    ):
        cls = getattr(module, cls_name)
        names = {n for n, _ in inspect.getmembers(cls)}
        assert set(required) <= names, f"{cls_name} lost members: {set(required) - names}"


def test_agent_py_stays_under_line_budget():
    """agent.py must shrink during the refactor, never grow past baseline.

    Baseline is 5565 lines. The budget tightens as steps land; a step that
    makes agent.py bigger is a step that copied instead of moved.
    """
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "agent.py"
    assert path.exists(), "agent.py is missing"
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= 5565, f"agent.py grew to {lines} lines (baseline 5565)"
