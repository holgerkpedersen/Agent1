from __future__ import annotations

import importlib

from agent_core import colors


def _reload_enabled(monkeypatch):
    monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("AGENT_NO_COLOR", raising=False)
    return importlib.reload(colors)


def test_colorize_result_does_not_redcode_source_lines_containing_error(monkeypatch):
    mod = _reload_enabled(monkeypatch)

    source_line = 'raise RuntimeError("Error: not a real tool failure")'
    result = mod.colorize_result(source_line)

    assert result == source_line


def test_colorize_result_redcodes_real_tool_errors_and_verify_failures(monkeypatch):
    mod = _reload_enabled(monkeypatch)

    red_start = mod._PALETTE["red"]

    assert mod.colorize_result("Error: tool failed").startswith(red_start)
    assert mod.colorize_result("[verify] py_compile ✗").startswith(red_start)


def test_colorize_result_leaves_plain_source_output_unchanged(monkeypatch):
    mod = _reload_enabled(monkeypatch)

    plain = "print('no issue here')"
    assert mod.colorize_result(plain) == plain
