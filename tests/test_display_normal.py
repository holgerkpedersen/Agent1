"""Tests for the ``normal`` (opencode-style) display mode.

NORMAL is the chat transcript mode: it shows the model's short thinking note,
the round's actions in plain words and ONE evidence line per result — never a
tool name, an argument dump or raw tool output.  These tests pin that contract
(the screen) and the untouched model payload (the tool result the LLM gets).
"""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from agent_core.config import AgentDisplayMode
from agent_core.llm.tool_loop import (
    DisplayMode,
    ToolLoopRunner,
    _condense_thought,
    _normal_action_phrase,
    _normal_evidence,
    _result_reports_failure,
)
from agent_core.llm.tool_reply import parse_tool_reply
from agent_core.run_utils import (
    normal_mode_system_suffix,
    strip_dynamic_system_blocks,
)
from agent_core.turn_loop import _normal_turn_footer


class _ScriptedLLM:
    """Tiny fake LLM: text answers, single calls, or a list of parallel calls."""

    def __init__(self, script):
        self.script = list(script)

    async def chat(self, messages, tools=None, **kwargs):
        step = self.script.pop(0)
        if isinstance(step, list):
            calls = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                for i, (name, args) in enumerate(step)
            ]
            return json.dumps(
                {"role": "assistant", "content": "", "tool_calls": calls}
            )
        if isinstance(step, tuple):
            name, args = step[0], step[1]
            narration = step[2] if len(step) > 2 else ""
            return json.dumps(
                {
                    "role": "assistant",
                    "content": narration,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                }
            )
        return step


def _run(script, execute_tool, *, display_mode="normal"):
    fake = _ScriptedLLM(script)

    async def llm_chat_fn(messages, tools):
        raw = await fake.chat(messages, tools)
        return parse_tool_reply(raw, messages, repair_arguments=lambda *a, **k: None)

    runner = ToolLoopRunner(max_iterations=5, display_mode=display_mode)
    return asyncio.run(
        runner.run(
            messages=[{"role": "user", "content": "do the thing"}],
            llm_chat_fn=llm_chat_fn,
            execute_tool_fn=execute_tool,
            tools=[],
        )
    )


# ---------------------------------------------------------------------------
# Mode plumbing
# ---------------------------------------------------------------------------


class TestNormalModePlumbing:
    def test_agent_display_mode_parses_normal(self):
        assert AgentDisplayMode("normal") is AgentDisplayMode.NORMAL

    def test_runner_maps_agent_normal_mode(self):
        runner = ToolLoopRunner(max_iterations=5, display_mode=AgentDisplayMode.NORMAL)
        assert runner.display_mode is DisplayMode.NORMAL

    def test_display_command_accepts_normal(self, tmp_path, monkeypatch, capsys):
        from agent_core.commands.display_cmd import DisplayCommand

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_DISPLAY_MODE", raising=False)
        assert asyncio.run(DisplayCommand().execute(["normal"], agent=None)) is True
        assert "Display mode set to 'normal'" in capsys.readouterr().out
        env_text = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "AGENT_DISPLAY_MODE=normal" in env_text

    def test_normal_system_suffix_and_strip_marker(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_DISPLAY_MODE", raising=False)
        assert normal_mode_system_suffix() == ""

        monkeypatch.setenv("AGENT_DISPLAY_MODE", "normal")
        suffix = normal_mode_system_suffix()
        assert "NORMAL DISPLAY MODE" in suffix
        # A rebuild never accumulates a stale copy of the block.
        rebuilt = strip_dynamic_system_blocks("BASE" + suffix + suffix)
        assert rebuilt == "BASE"

    def test_other_modes_do_not_get_the_normal_suffix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for mode in ("verbose", "clean", "intent", "quiet"):
            monkeypatch.setenv("AGENT_DISPLAY_MODE", mode)
            assert normal_mode_system_suffix() == ""


# ---------------------------------------------------------------------------
# Pure display helpers
# ---------------------------------------------------------------------------


class TestNormalHelpers:
    def test_action_phrases_are_plain_language(self):
        assert _normal_action_phrase("read", {"path": "a.py"}) == "reading a.py"
        assert (
            _normal_action_phrase("read", {"path": "a.py", "offset": 40})
            == "reading a.py from line 40"
        )
        assert (
            _normal_action_phrase("search", {"query": "needle"})
            == 'searching for "needle"'
        )
        assert _normal_action_phrase("write", {"path": "b.py"}) == "writing b.py"
        assert (
            _normal_action_phrase("tests", {"path": "tests"})
            == "running the tests in tests"
        )
        assert (
            _normal_action_phrase("references", {"symbol": "foo"})
            == 'finding references to "foo"'
        )
        assert (
            _normal_action_phrase("run", {"command": "python -m pytest -q"})
            == "running python -m pytest -q"
        )
        assert (
            _normal_action_phrase("git", {"subcommand": "diff"})
            == "checking git diff"
        )

    def test_unknown_tool_is_humanized_never_raw_payload(self):
        phrase = _normal_action_phrase("some_custom_tool", {"secret": "x"})
        assert phrase == "some custom tool"
        assert "secret" not in phrase

    def test_condense_thought_strips_think_blocks_and_caps(self):
        text = "<think>secret reasoning</think>I will check the config now."
        assert _condense_thought(text) == "I will check the config now."
        long = "This is a sentence. " * 80
        condensed = _condense_thought(long)
        assert len(condensed) <= 430  # cap plus the ellipsis marker
        assert condensed.endswith("...")

    def test_evidence_read_reports_line_count(self):
        ok, text = _normal_evidence("read", {"path": "a.py"}, "l1\nl2\nl3")
        assert ok is True
        assert text == "a.py — 3 lines"

    def test_evidence_read_reports_more_available_when_paged(self):
        ok, text = _normal_evidence(
            "read", {"path": "a.py"}, "l1\n[truncated — use read with offset=2]"
        )
        assert ok is True
        assert "more available" in text

    def test_evidence_search_counts_matches_and_files(self):
        result = (
            "  agent_core/a.py:1: needle\n"
            "  agent_core/a.py:9: needle\n"
            "  agent_core/b.py:3: needle\n"
        )
        ok, text = _normal_evidence("search", {"query": "needle"}, result)
        assert ok is True
        assert text.startswith('"needle" — 3 match(es) in 2 file(s); first ')
        assert "agent_core/a.py:1" in text

    def test_evidence_search_handles_absolute_windows_paths(self):
        """Drive-letter colons must not be mistaken for the file separator."""
        result = (
            "  D:\\Dev\\Agent1\\agent_core\\a.py:1: needle\n"
            "  D:\\Dev\\Agent1\\agent_core\\b.py:2: needle\n"
        )
        ok, text = _normal_evidence("search", {"query": "needle"}, result)
        assert ok is True
        assert "2 match(es) in 2 file(s)" in text

    def test_evidence_search_notes_the_capped_match_count(self):
        result = "  a.py:1: x\n... and 7 more matches"
        ok, text = _normal_evidence("search", {"query": "x"}, result)
        assert ok is True
        assert "1 match(es) in 1 file(s), +7 more" in text

    def test_evidence_no_match_is_still_conclusive(self):
        ok, text = _normal_evidence(
            "search", {"query": "zzz"}, "No files found matching that query."
        )
        assert ok is True
        assert text == '"zzz" — no matches'

    def test_evidence_write_reports_syntax_verification(self):
        result = (
            "Written a.py (12 bytes)\n"
            "[verify] py_compile \u2713 a.py"
        )
        ok, text = _normal_evidence("write", {"path": "a.py"}, result)
        assert ok is True
        assert text == "a.py — written, syntax ok"

    def test_evidence_write_syntax_failure_is_negative(self):
        result = (
            "Written a.py (12 bytes)\n"
            "[verify] py_compile \u2717 a.py: SyntaxError"
        )
        ok, text = _normal_evidence("write", {"path": "a.py"}, result)
        assert ok is False
        assert "syntax ERROR" in text

    def test_evidence_skipped_write_is_reported_as_unchanged(self):
        ok, text = _normal_evidence(
            "write", {"path": "a.py"}, "Skipped a.py (no changes)"
        )
        assert ok is True
        assert text == "a.py — unchanged (nothing to write)"

    def test_evidence_missing_path_is_negative_evidence(self):
        ok, text = _normal_evidence(
            "read", {"path": "gone.py"}, "File not found: gone.py"
        )
        assert ok is False
        assert "File not found" in text

    def test_evidence_offset_past_eof_is_negative_evidence(self):
        ok, text = _normal_evidence(
            "read",
            {"path": "a.py", "offset": 999},
            "Offset 999 is beyond the end of a.py (3 lines).",
        )
        assert ok is False
        assert "beyond the end" in text

    def test_evidence_test_run_reports_summary_and_exit_code(self):
        result = "=== 4 passed, 1 skipped in 0.30s ===\n[EXIT CODE: 0]"
        ok, text = _normal_evidence("run", {}, result)
        assert ok is True
        assert text == "4 passed, 1 skipped in 0.30s (exit 0)"

    def test_evidence_failed_command_is_marked_failed(self):
        result = "boom\n[EXIT CODE: 2]"
        ok, text = _normal_evidence("run", {}, result)
        assert ok is False
        assert text == "boom"

    def test_footer_reports_duration_steps_and_files(self):
        loop = SimpleNamespace(tool_calls_made=9)
        footer = _normal_turn_footer(12.34, loop, ["a.py", "b.py"])
        assert footer == "  ✓ done · 12.3s · 9 steps · 2 files changed"

    def test_footer_handles_a_tool_less_turn(self):
        loop = SimpleNamespace(tool_calls_made=0)
        assert _normal_turn_footer(1.0, loop, []) == "  ✓ done · 1.0s"
        assert _normal_turn_footer(None, loop, []) == ""

    def test_footer_does_not_claim_done_when_the_guard_stopped_the_run(self):
        loop = SimpleNamespace(tool_calls_made=4, termination_reason="stuck")
        footer = _normal_turn_footer(2.0, loop, [])
        assert footer == "  ! ended early · 2.0s · 4 steps"


# ---------------------------------------------------------------------------
# End-to-end loop display
# ---------------------------------------------------------------------------


class TestNormalLoopDisplay:
    def test_round_shows_thought_actions_and_evidence_not_tools(self, capsys):
        raw_result = "line one\nline two"

        async def execute_tool(name, args):
            return raw_result

        final_text, messages = _run(
            [
                (
                    "read",
                    {"path": "agent_core/config.py"},
                    "I need to see how display modes are defined.",
                ),
                "Here is the conclusion.",
            ],
            execute_tool,
        )

        out = capsys.readouterr().out
        # Thought (the model's own words) and a plain action row.
        assert "I need to see how display modes are defined." in out
        assert "· reading agent_core/config.py" in out
        # Evidence of what the read established.
        assert "✓ agent_core/config.py — 2 lines" in out
        # No technical dump anywhere.
        assert "[tool]" not in out
        assert "[result]" not in out
        assert "read(path=" not in out
        # The model still receives the full payload untouched.
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert tool_msg["content"] == raw_result
        assert final_text == "Here is the conclusion."

    def test_actions_are_capped_per_round(self, capsys):
        calls = [("read", {"path": f"f{i}.py"}) for i in range(6)]

        async def execute_tool(name, args):
            return "x"

        _run([calls, "Done."], execute_tool)
        out = capsys.readouterr().out
        assert "· reading f0.py" in out
        assert "· reading f3.py" in out
        assert "· reading f4.py" not in out
        assert "+2 more action(s)" in out
        # Evidence is capped too (6 rows + a count line).
        assert "✓ f5.py — 1 lines" in out

    def test_parallel_round_prints_evidence_in_announced_order(self, capsys):
        calls = [
            ("read", {"path": "a.py"}),
            ("read", {"path": "b.py"}),
            ("read", {"path": "c.py"}),
        ]
        delays = {"a.py": 0.03, "b.py": 0.02, "c.py": 0.0}

        async def execute_tool(name, args):
            await asyncio.sleep(delays[args["path"]])
            return args["path"] + " content"

        _run([calls, "Done."], execute_tool)
        out = capsys.readouterr().out
        assert out.count("✓") == 3
        # c.py finishes FIRST, but evidence follows the announced a, b, c order
        # so the rows line up with the action rows above them.
        assert out.index("✓ a.py") < out.index("✓ b.py") < out.index("✓ c.py")

    def test_error_result_is_shown_as_negative_evidence(self, capsys):
        async def execute_tool(name, args):
            return "File not found: nowhere.py"

        _run([("read", {"path": "nowhere.py"}), "It does not exist."], execute_tool)
        out = capsys.readouterr().out
        assert "✗ File not found: nowhere.py" in out
        assert "[result]" not in out

    def test_quiet_still_hides_everything_but_the_answer(self, capsys):
        """Guard against the new branch leaking rows into QUIET."""

        async def execute_tool(name, args):
            return "line one"

        _run(
            [("read", {"path": "a.py"}), "Answer."],
            execute_tool,
            display_mode="quiet",
        )
        out = capsys.readouterr().out
        assert "· reading" not in out
        assert "✓ a.py" not in out


class TestFailureColoring:
    """Verbose mode must redden a result only when the TOOL failed.

    History (Kanban card ``Color red - false positive``): the guard tested
    ``"error" in shown``, so reading a source file that merely mentioned
    "error" painted the entire tool result red.
    """

    def test_failure_detection_requires_a_prefix_or_nonzero_exit(self):
        assert _result_reports_failure("Read error: nope") is True
        assert _result_reports_failure("Tool error (ValueError): boom") is True
        assert _result_reports_failure("boom\n[EXIT CODE: 2]") is True
        assert _result_reports_failure('raise RuntimeError("Error: text")') is False
        assert _result_reports_failure("done\n[EXIT CODE: 0]") is False
        assert _result_reports_failure("a note about errors") is False

    def test_source_text_mentioning_error_is_not_reddened(self, capsys, monkeypatch):
        import agent_core.llm.tool_loop as tool_loop

        monkeypatch.setattr(
            tool_loop, "red", lambda text, **kwargs: f"<RED>{text}</RED>"
        )

        async def execute_tool(name, args):
            return 'def handle():\n    raise RuntimeError("Error: source text")\n'

        _run(
            [("read", {"path": "a.py"}), "Done."],
            execute_tool,
            display_mode="verbose",
        )
        out = capsys.readouterr().out
        assert "Error: source text" in out
        assert "<RED>" not in out

    def test_real_tool_error_still_reddens_the_result(self, capsys, monkeypatch):
        import agent_core.llm.tool_loop as tool_loop

        monkeypatch.setattr(
            tool_loop, "red", lambda text, **kwargs: f"<RED>{text}</RED>"
        )

        async def execute_tool(name, args):
            return "Read error: permission denied"

        _run(
            [("read", {"path": "a.py"}), "Done."],
            execute_tool,
            display_mode="verbose",
        )
        out = capsys.readouterr().out
        assert "<RED>Read error: permission denied</RED>" in out


# ---------------------------------------------------------------------------
# Turn footer (finish_turn)
# ---------------------------------------------------------------------------


class _FakeAgent:
    workspace = "."
    session_id = ""
    _chat_history: list = []
    _original_goal = None
    _turn_start_index = 0
    _turn_started_at = datetime.now()
    _last_user_input = "add a display mode"
    _turn_run_id = None
    _nlp_workspace = None

    def _mutating_files_this_turn(self):
        return ["agent_core/config.py"]

    def _record_llm_experience(self, **kwargs):
        return None

    def is_plan_mode(self):
        return False

    def _snapshot_session_meta(self, **kwargs):
        return None

    def _save_chat_history(self):
        return None

    def _save_memory(self):
        return None

    def _print_self_review_note(self, files):
        print(f"  [self-review] changed {len(files)} file(s)")


def test_finish_turn_prints_normal_footer_not_self_review(monkeypatch, capsys):
    import agent
    from agent_core import turn_loop

    monkeypatch.setattr(turn_loop, "_append_turn_log", lambda record: None)
    monkeypatch.setattr(turn_loop, "_turn_quality_score", lambda *a, **k: None)
    monkeypatch.setattr(agent, "record_turn_outcome", lambda *a, **k: None)

    loop = SimpleNamespace(tool_calls_made=3, final_answer_streamed=False)
    turn_loop._finish_turn(
        _FakeAgent(),
        "All done: `normal` is wired.",
        None,
        loop,
        AgentDisplayMode.NORMAL,
    )

    out = capsys.readouterr().out
    assert "All done: `normal` is wired." in out
    assert "✓ done" in out
    assert "3 steps" in out
    assert "1 file changed" in out
    assert "agent_core/config.py" in out
    assert "[self-review]" not in out


def test_finish_turn_keeps_self_review_for_verbose(monkeypatch, capsys):
    import agent
    from agent_core import turn_loop

    monkeypatch.setattr(turn_loop, "_append_turn_log", lambda record: None)
    monkeypatch.setattr(turn_loop, "_turn_quality_score", lambda *a, **k: None)
    monkeypatch.setattr(agent, "record_turn_outcome", lambda *a, **k: None)

    loop = SimpleNamespace(tool_calls_made=3, final_answer_streamed=False)
    turn_loop._finish_turn(
        _FakeAgent(),
        "All done.",
        None,
        loop,
        AgentDisplayMode.VERBOSE,
    )

    out = capsys.readouterr().out
    assert "[self-review]" in out
    assert "✓ done" not in out


class _FakeChatLLM:
    """Scripted LLM client good enough for a real ``Agent.chat_nlp`` turn."""

    model_name = "fake/model"
    _profile_name = "default"
    _profile_pinned = True

    class _Provider:
        pass

    _provider = _Provider()

    def __init__(self, script):
        self.script = list(script)

    async def chat(self, messages, tools=None, **kwargs):
        return self.script.pop(0)


def test_chat_nlp_end_to_end_renders_normal_transcript(
    tmp_path, monkeypatch, capsys
):
    """The REAL turn pipeline (system prompt -> loop -> finish) in normal mode.

    This pins the wiring the unit tests above cannot: ``chat_nlp`` resolves
    NORMAL from the environment, maps it onto ``DisplayMode.NORMAL`` inside
    ``ToolLoopRunner`` (two distinct enums), executes a real tool, and prints
    the footer — with no tool dump anywhere on screen.
    """
    import agent as agent_mod
    import agent_core.constants as constants_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_DISPLAY_MODE", "normal")
    monkeypatch.setattr(
        agent_mod, "CHAT_HISTORY_JSON_PATH", str(tmp_path / "chat_history.json")
    )
    monkeypatch.setattr(
        agent_mod, "AGENT_MEMORY_JSON_PATH", str(tmp_path / "agent_memory.json")
    )
    monkeypatch.setattr(
        constants_mod, "TURN_LOG_PATH", str(tmp_path / "turn_log.jsonl")
    )

    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    bot = agent_mod.Agent(workspace=str(workspace))
    first_reply = json.dumps(
        {
            "role": "assistant",
            "content": "I need to read the notes before answering.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": "notes.txt"}),
                    },
                }
            ],
        }
    )
    bot.llm = _FakeChatLLM([first_reply, "The notes say alpha and beta."])

    asyncio.run(bot.chat_nlp("what do the notes say?"))

    out = capsys.readouterr().out
    assert "I need to read the notes before answering." in out
    assert "· reading notes.txt" in out
    assert "✓ notes.txt — 2 lines" in out
    assert "The notes say alpha and beta." in out
    assert "✓ done" in out and "1 step" in out
    assert "[tool]" not in out
    assert "[result]" not in out
    assert "read(path=" not in out
    assert bot._chat_history[-1]["content"] == "The notes say alpha and beta."


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"paths": ["a.py", "b.py"]}, "reading a.py, b.py"),
        (
            {"paths": [f"f{i}.py" for i in range(5)]},
            "reading f0.py, f1.py, f2.py (+2 more)",
        ),
        ({"path": "x" * 200}, "reading ..." + "x" * 69),
    ],
)
def test_read_action_phrase_variants(args, expected):
    assert _normal_action_phrase("read", args) == expected
