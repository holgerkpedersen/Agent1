from __future__ import annotations

import importlib

import pytest

from agent_core import colors


@pytest.fixture
def colors_enabled():
    """Give the caller a colors module with color forced ON, then restore.

    ``importlib.reload`` re-evaluates the module-level ``_ENABLED`` flag and
    mutates the *shared* module in ``sys.modules``.  A monkeypatched
    ``isatty``/env pair does not undo that: after the reload the whole test
    session keeps color enabled, leaking ANSI codes into every later test
    that asserts on plain captured output (display-mode tests, shell prompt).
    This fixture snapshots the module state and puts it back afterwards.
    """
    saved_enabled = colors._ENABLED
    try:
        yield colors
    finally:
        colors._ENABLED = saved_enabled


def _reload_enabled(monkeypatch):
    monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("AGENT_NO_COLOR", raising=False)
    return importlib.reload(colors)


def test_colorize_result_does_not_redcode_source_lines_containing_error(
    monkeypatch, colors_enabled
):
    mod = _reload_enabled(monkeypatch)

    source_line = 'raise RuntimeError("Error: not a real tool failure")'
    result = mod.colorize_result(source_line)

    assert result == source_line


def test_colorize_result_redcodes_real_tool_errors_and_verify_failures(
    monkeypatch, colors_enabled
):
    mod = _reload_enabled(monkeypatch)

    red_start = mod._PALETTE["red"]

    assert mod.colorize_result("Error: tool failed").startswith(red_start)
    assert mod.colorize_result("[verify] py_compile ✗").startswith(red_start)


def test_colorize_result_leaves_plain_source_output_unchanged(
    monkeypatch, colors_enabled
):
    mod = _reload_enabled(monkeypatch)

    plain = "print('no issue here')"
    assert mod.colorize_result(plain) == plain
