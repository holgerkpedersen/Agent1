"""Contract guard for the module surface of ``agent.py``.

Baseline: agent.py, 5558 lines (refactor step t01). The lists below are the
frozen surface of the module: names agent.py defines in its own body, plus
names it exposes through its import block.

Why this test exists
--------------------
The agent.py refactor moves code into ``agent_core`` submodules and re-exports
it from agent.py. A re-export that is forgotten, a name that is renamed, or a
helper that quietly goes private breaks, at runtime and silently:

* tests that do ``from agent import _thing`` (monkeypatching, white-box tests),
* ``python -m agent`` and the CLI entry points,
* the tool-handler dispatch table and command registry built at import time.

The tests import the live module and compare ``dir()`` against the frozen
lists, so any surface change is a loud failure with the exact diff -- not a
late ImportError.

Two lists, two rules
--------------------
* ``EXPECTED_DEFINITIONS`` -- names agent.py's own body must keep defining.
  When a step moves code out, remove the name from this list in the SAME commit
  and add it to ``EXPECTED_IMPORTED`` (the re-export). That keeps every move a
  deliberate, reviewable edit.
* ``EXPECTED_IMPORTED`` -- names reachable through agent.py's import block,
  including the whole ``from agent_core.bootstrap import *`` surface.

If a change to the surface is INTENTIONAL, update the list in the same commit
and say why in the commit message.
"""

import ast
import importlib
import io

# Names agent.py's own body defines (functions, classes, module constants).
EXPECTED_DEFINITIONS = frozenset(
    (
    "Agent",
    "CHAT_INPUT_PROMPT",
    "LLMClient",
    "NlpToolHandler",
    "_ACTION_VERB_RE",
    "_COMPLETE_MARKERS",
    "_CONTEXT_AST_THRESHOLD_KB",
    "_CONTINUE_NOTE",
    "_CONTINUE_NOTE_TAG",
    "_CONTINUE_NOTE_TAG_KEY",
    "_DANGEROUS_SHELL_PATTERNS",
    "_DELEGATE_TIMEOUT_S",
    "_ENV_FILE_PATH",
    "_FULL_SUITE_MARGIN",
    "_GOAL_RELATED_THRESHOLD",
    "_INCOMPLETE_MARKERS",
    "_LLM_ERROR_BACKOFF_S",
    "_LLM_ERROR_MAX_RETRIES",
    "_MAX_ACTIVE_SUBAGENTS",
    "_MAX_CHAINED_RUNS",
    "_MAX_CONSECUTIVE_READS",
    "_MAX_READ_FILES",
    "_MAX_READ_LINES",
    "_MAX_RUN_TIMEOUT_S",
    "_PYTEST_BARE_TOKENS",
    "_PYTEST_FULL_SUITE_TIMEOUT_KEY",
    "_PYTEST_LAST_FULL_RUN_KEY",
    "_PYTEST_SUBSET_FLAGS",
    "_PYTEST_VALUE_FLAGS",
    "_QUESTION_LEAD_RE",
    "_READ_LOOP_NOTE",
    "_SHELL_NAME",
    "_SHUTDOWN_FLAG",
    "_SYSTEM_PROMPT",
    "_VERIFY_NOTE",
    "_append_turn_log",
    "_begin_trace_turn",
    "_blocked_shell_command",
    "_build_chat_prompt",
    "_build_registry",
    "_current_git_branch",
    "_detect_shell",
    "_escape_json_string_controls",
    "_final_answer_fallback",
    "_git_branch",
    "_input_executor",
    "_install_signal_handlers",
    "_interruptible_input",
    "_is_full_pytest_command",
    "_is_pytest_command",
    "_is_subset_pytest_command",
    "_kill_process_tree",
    "_looks_incomplete",
    "_pytest_full_suite_timeout",
    "_read_env_value",
    "_read_json_quarantining",
    "_register_commands",
    "_repair_tool_arguments",
    "_resolve_display_mode",
    "_resolve_stream_fn",
    "_run_subprocess_captured",
    "_shape_run_stderr",
    "_shell_command_hint",
    "_shell_name_token",
    "_shutdown_save_memory",
    "_shutdown_trace_close",
    "_signal_break",
    "_signal_break_with_cleanup",
    "_start_kanban_inbound",
    "_strip_dynamic_system_blocks",
    "_suppress_and_log",
    "_truncate_output",
    "_turn_quality_score",
    "_unix_command_hint",
    "_unverified_action_claim",
    "_warn_uncommitted",
    "_windows_command_hint",
    "logger",
    "main",
    "run_interactive",
    )
)

# Names agent.py exposes through imports (``import x``, ``from y import z``,
# and everything re-exported by agent_core.bootstrap.__all__).
EXPECTED_IMPORTED = frozenset(
    (
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
    "Handler",
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
    "_GOAL_RELATED_THRESHOLD",
    "_GOAL_TOPIC_STOPWORDS",
    "_HISTORY_CHAR_BUDGET",
    "_HISTORY_TRIM_GOAL_CAP",
    "_HISTORY_TRIM_NOTE",
    "_MAX_CHAT_MESSAGES",
    "_build_dashboard",
    "_compaction_note_goal",
    "_content_tokens",
    "_dashboard_port",
    "_default_alert_rules",
    "_drop_orphan_tool_messages",
    "_emit_command_metrics",
    "_emit_tool_metrics",
    "_first_user_message_text",
    "_is_compaction_note",
    "_message_size",
    "_project_chat_history",
    "_recover_original_goal",
    "_repair_unanswered_tool_calls",
    "_shared_metrics_collector",
    "_strip_image_blocks",
    "_strip_tool_args",
    "_strip_turn_wrappers",
    "_trim_chat_history",
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
    )
)


def _agent_source() -> str:
    return io.open("agent.py", encoding="utf-8").read()


def _ast_lists(source: str) -> tuple:
    """(names defined in the body, names brought in by the import block).

    Star imports are expanded from the module's declared ``__all__``.
    """
    tree = ast.parse(source)
    defined, imported = set(), set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                mod = importlib.import_module(node.module or "")
                names = getattr(mod, "__all__", None)
                if names is None:
                    names = [n for n in dir(mod) if not n.startswith("_")]
                imported.update(names)
                continue
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    defined = {n for n in defined if not (n.startswith("__") and n.endswith("__"))}
    return defined, imported


def _live_public_names() -> set:
    """Public, non-None attributes of the imported agent module.

    Names bound to ``None`` are excluded: the bootstrap module deliberately
    binds ``TraceWriter = None`` when the optional tracing package is missing,
    and ``dir()`` shows those names regardless of how they are bound.
    """
    agent = importlib.import_module("agent")
    return {
        n
        for n in dir(agent)
        if not n.startswith("__") and getattr(agent, n, None) is not None
    }


def test_agent_imports_cleanly():
    assert importlib.import_module("agent") is not None


def test_expected_definitions_are_live():
    """Every frozen definition must be reachable as an attribute of agent.

    Names bound to ``None`` at module level (lazy placeholders assigned later,
    plus bootstrap's optional-dependency placeholder) never appear in the live
    public set -- ``_live_public_names`` filters None out -- so they are
    exempted here.
    """
    exempt = {"_input_executor", "_shutdown_save_memory", "_shutdown_trace_close"}
    missing = sorted(EXPECTED_DEFINITIONS - _live_public_names() - exempt)
    assert missing == [], "agent.py lost names: %s" % missing


def test_agent_body_defines_exactly_the_frozen_definitions():
    """No top-level name may appear or vanish in agent.py's body unannounced.

    Moving code to agent_core: delete the name from EXPECTED_DEFINITIONS and
    add it to EXPECTED_IMPORTED (the re-export). Both lists are checked.
    """
    defined, imported = _ast_lists(_agent_source())
    added = sorted(defined - EXPECTED_DEFINITIONS)
    removed = sorted(EXPECTED_DEFINITIONS - defined)
    assert added == [], "new top-level names in agent.py: %s" % added
    assert removed == [], (
        "names no longer defined in agent.py (move them to EXPECTED_IMPORTED "
        "only if they are re-exported): %s" % removed
    )


def test_import_block_exposes_exactly_the_frozen_imports():
    """The import surface is frozen too, so a lost re-export fails loudly."""
    defined, imported = _ast_lists(_agent_source())
    added = sorted(imported - EXPECTED_IMPORTED)
    removed = sorted(EXPECTED_IMPORTED - imported)
    assert added == [], "new names in agent.py's import block: %s" % added
    assert removed == [], "names dropped from agent.py's import block: %s" % removed


def test_live_namespace_has_no_unexpected_names():
    """Nothing extra may leak into agent's namespace (typo, accidental global)."""
    defined, imported = _ast_lists(_agent_source())
    expected = defined | imported
    extra = sorted(_live_public_names() - expected)
    assert extra == [], "unexpected names in agent namespace: %s" % extra


def test_no_dunder_leaks_in_frozen_lists():
    bad = sorted(
        n for n in EXPECTED_DEFINITIONS | EXPECTED_IMPORTED if n.startswith("__")
    )
    assert bad == []
