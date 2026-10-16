"""Regression: a non-UTF-8 text file must be detected, never silently rewritten."""
from __future__ import annotations

import asyncio

import pytest

from agent_core.text_encoding import (
    NonUtf8TextError,
    check_utf8_bytes,
    check_utf8_file,
    read_text_strict,
)


def test_valid_utf8_passes():
    res = check_utf8_bytes("中文 ok\n".encode("utf-8"))
    assert res.ok is True


def test_cp1252_bytes_are_detected_with_location():
    data = "line one\nprice: 5 \xe9uros\n".encode("cp1252")
    res = check_utf8_bytes(data, path="x.md")
    assert res.ok is False
    assert res.line == 2
    assert res.byte_value == 0xE9
    assert "not valid UTF-8" in res.message


def test_read_text_strict_raises_on_non_utf8(tmp_path):
    p = tmp_path / "legacy.txt"
    p.write_bytes("caf\xe9\n".encode("latin-1"))
    with pytest.raises(NonUtf8TextError):
        read_text_strict(p)


def test_read_text_strict_returns_utf8_text(tmp_path):
    p = tmp_path / "ok.txt"
    p.write_bytes("café\n".encode("utf-8"))  # bytes: avoid Windows CRLF translation
    assert read_text_strict(p) == "café\n"


def test_check_utf8_file_reports_path(tmp_path):
    p = tmp_path / "bad.py"
    p.write_bytes(b"x = 1\n\xff\n")
    res = check_utf8_file(p)
    assert res.ok is False
    assert res.line == 2
    assert str(p) == res.path


def _edit(target, old, new):
    from agent import Agent

    agent = Agent.__new__(Agent)
    agent._resolve_nlp_path = lambda p: str(target)
    agent._pending_effects = None  # no trace armed; _note_effect is a no-op
    return asyncio.run(
        agent._nlp_edit({"path": str(target), "old_text": old, "new_text": new})
    )


def test_nlp_edit_keeps_legacy_cp1252_encoding(tmp_path):
    """A legacy cp1252 file is edited and stays cp1252 (no UTF-8 conversion)."""
    target = tmp_path / "legacy.md"
    target.write_bytes("keep \xe9 this\n".encode("cp1252"))

    result = _edit(target, "keep", "KEEP")

    assert result.startswith("Edited")
    assert "kept cp1252" in result
    # Compare newline-normalized: text-mode writes use the platform newline.
    assert target.read_bytes().replace(b"\r\n", b"\n") == "KEEP \xe9 this\n".encode("cp1252")


def test_nlp_edit_refuses_text_unrepresentable_in_legacy_encoding(tmp_path):
    """New text the legacy encoding cannot hold is refused; bytes untouched."""
    target = tmp_path / "legacy.md"
    original = "keep \xe9 this\n".encode("cp1252")
    target.write_bytes(original)

    result = _edit(target, "keep", "\u4e2d\u6587")  # CJK not in cp1252

    assert result.startswith("Edit refused")
    assert target.read_bytes() == original


def test_nlp_edit_undecodable_file_is_refused_and_untouched(tmp_path):
    """Bytes that are neither UTF-8 nor cp1252 are refused, never rewritten."""
    target = tmp_path / "broken.md"
    original = b"keep \x81\x8d\x8f\x90\x9d\n"  # undefined in cp1252
    target.write_bytes(original)

    result = _edit(target, "keep", "KEEP")

    assert result.startswith("Edit refused")
    assert target.read_bytes() == original


def test_nlp_edit_utf8_file_still_edits(tmp_path):
    target = tmp_path / "ok.md"
    target.write_bytes("keep 中文\n".encode("utf-8"))

    result = _edit(target, "keep", "KEEP")

    assert result.startswith("Edited")
    assert target.read_bytes().replace(b"\r\n", b"\n") == "KEEP 中文\n".encode("utf-8")
