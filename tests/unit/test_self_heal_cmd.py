"""Tests for write_cmd per-hunk review (plan 31) and self_heal (plan 29-30)."""
import asyncio
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

from agent_core.commands.self_heal_cmd import SelfHealCommand


class TestWriteReview:
    def _agent(self, tmp_path):
        target = tmp_path / "a.py"
        return target

    def test_yes_skips_review_and_overwrites(self, tmp_path):
        from agent_core.commands.write_cmd import WriteCommand
        target = tmp_path / "a.py"
        target.write_text("old = 1\n", encoding="utf-8")

        class FakeAgent:
            async def read_file(self, path, track_read=True):
                return target.read_text(encoding="utf-8")

            async def write_file(self, path, content):
                target.write_text(content, encoding="utf-8")
                return "written"

        ok = asyncio.run(WriteCommand().execute([str(target), "new = 2\n", "--yes"], FakeAgent()))
        assert ok is True
        assert target.read_text(encoding="utf-8") == "new = 2\n"

    def test_review_rejects_hunks_keeps_original(self, tmp_path):
        from agent_core.commands.write_cmd import WriteCommand
        target = tmp_path / "a.py"
        target.write_text("a = 1\nb = 2\n", encoding="utf-8")

        class FakeAgent:
            async def read_file(self, path, track_read=True):
                return target.read_text(encoding="utf-8")

            async def write_file(self, path, content):
                target.write_text(content, encoding="utf-8")
                return "written"

        # read_choice returns False for every hunk -> nothing applied.
        with patch("agent_core.commands.write_cmd.read_choice", return_value=False):
            ok = asyncio.run(WriteCommand().execute([str(target), "a = 9\nb = 9\n"], FakeAgent()))
        assert ok is True
        assert target.read_text(encoding="utf-8") == "a = 1\nb = 2\n"

    def test_review_approves_selected_hunk(self, tmp_path):
        from agent_core.commands.write_cmd import WriteCommand
        target = tmp_path / "a.py"
        # Two changes separated by >2*context (7) unchanged lines -> two hunks.
        def _body(a: int, c: int) -> str:
            return f"a = {a}\n" + "\n".join(f"k{i}" for i in range(1, 10)) + f"\nc = {c}\n"

        target.write_text(_body(1, 3), encoding="utf-8")

        class FakeAgent:
            async def read_file(self, path, track_read=True):
                return target.read_text(encoding="utf-8")

            async def write_file(self, path, content):
                target.write_text(content, encoding="utf-8")
                return "written"

        answers = iter([True, False])  # approve first hunk, reject second

        with patch("agent_core.commands.write_cmd.read_choice", side_effect=lambda p: next(answers)):
            ok = asyncio.run(WriteCommand().execute(
                [str(target), _body(9, 9)], FakeAgent()
            ))
        assert ok is True
        result = target.read_text(encoding="utf-8")
        assert "a = 9" in result
        assert "c = 3" in result  # second hunk rejected


class TestSelfHeal:
    def test_failing_files_parses_summary(self):
        output = textwrap.dedent("""\
            tests/test_foo.py::test_x FAILED
            ==================== FAILURES ====================
            ____ test_x ____
            E       AssertionError
            ============ short test summary info =============
            FAILED tests/test_foo.py::test_x - AssertionError
            FAILED tests/test_bar.py::test_y - ValueError
        """)
        assert SelfHealCommand._failing_files(output) == ["tests/test_foo.py", "tests/test_bar.py"]

    def test_failing_files_empty_when_green(self):
        assert SelfHealCommand._failing_files("1 passed in 0.01s") == []

    def test_run_pytest_green(self, tmp_path):
        (tmp_path / "test_ok.py").write_text("def test_one():\n    assert 1 == 1\n", encoding="utf-8")
        rc, _ = SelfHealCommand._run_pytest(str(tmp_path))
        assert rc == 0

    def test_heal_loop_green_with_llm_patch(self, tmp_path):
        failing = tmp_path / "test_grow.py"
        failing.write_text(
            "def test_grow():\n    assert grow(1) == 2\n", encoding="utf-8"
        )
        prod = tmp_path / "grow.py"
        prod.write_text("def grow(x):\n    return x\n", encoding="utf-8")

        patched = False

        async def chat(messages, **kwargs):
            nonlocal patched
            patched = True
            return (
                "[PATCH: grow.py]\n"
                "@@ -1,2 +1,2 @@\n"
                " def grow(x):\n"
                "-    return x\n"
                "+    return x + 1\n"
            )

        agent = SimpleNamespace(llm=SimpleNamespace(chat=chat))
        cmd = SelfHealCommand()
        cmd.backups = {}
        ok = asyncio.run(cmd._fix_file(agent, str(prod), "assert grow(1) == 2 failed"))
        assert ok is True
        assert "x + 1" in prod.read_text(encoding="utf-8")

    def test_revert_all_restores_backup(self, tmp_path):
        prod = tmp_path / "grow.py"
        prod.write_text("def grow(x):\n    return x\n", encoding="utf-8")
        cmd = SelfHealCommand()
        cmd.backups = {str(prod): (str(prod) + ".heal_bak", "orig")}
        import shutil
        shutil.copyfile(str(prod), str(prod) + ".heal_bak")
        prod.write_text("def grow(x):\n    return x + 1\n", encoding="utf-8")
        cmd._revert_all()
        assert "return x\n" in prod.read_text(encoding="utf-8")
        assert cmd.backups == {}
