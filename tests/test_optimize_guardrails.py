"""Regression guardrails for the optimizer's mechanical fixers.

Each test pins one defect class from the 2026-08-26 incident where
``optimize . --apply`` corrupted 40+ test fixtures, mis-compiled regex
hoists (flags passed as *pos*), referenced another function's local regex
constant, produced a self-referencing comprehension NameError, emitted
``logger.warning`` before ``logger`` was defined, and deleted live stores
(``nonlocal patched``, class-body stub attributes).
"""
from __future__ import annotations

import textwrap

from agent_core.commands.optimize_cmd import (
    _fix_list_append_join,
    _fix_regex_in_loop,
    _fix_silent_except,
    _tri_mech_fix,
)
from agent_core.patch_utils import apply_patch
from agent_core.patterns import (
    detect_dead_assignment,
    string_literal_rows,
)


class TestStringLiteralRows:
    def test_multiline_fixture_marks_interior_rows_not_start(self):
        src = textwrap.dedent(
            """\
            UNFIXED = textwrap.dedent(\"\"\"\\
                def build(items):
                    out = []
                    for x in items:
                        out.append(str(x))
                \"\"\")
            x = 1
            """
        )
        rows = string_literal_rows(src)
        assert 2 in rows and 5 in rows  # interior content rows
        assert 1 not in rows  # start row carries real code
        assert 7 not in rows  # after the literal

    def test_single_line_escaped_newline_string_is_marked(self):
        src = 'code = "if x == None:\\n    return"\nreal_code = 1\n'
        rows = string_literal_rows(src)
        assert 1 in rows
        assert 2 not in rows


class TestMechanicalFixersRefuseStringLiterals:
    """The central choke point (_tri_mech_fix) must refuse every mechanical
    fix whose target line lies inside a string literal."""

    FIXTURE_FILE = textwrap.dedent(
        '''\
        import textwrap

        SAMPLE = textwrap.dedent("""\\
            def probe(a):
                if a != None:
                    return a
            """)
        '''
    )

    def _lines(self):
        return self.FIXTURE_FILE.split("\n")

    def test_none_eq_refused_inside_fixture(self):
        # line 5 ("    if a != None:") is inside the triple-quoted fixture.
        assert _tri_mech_fix(self._lines(), 5, "none_eq", "t.py", {"suggestion": ""}) is None

    def test_iter_dict_keys_refused_inside_fixture(self):
        src = (
            'SAMPLE = textwrap.dedent("""\\\n'
            "    for k in d.keys():\n"
            '        print(k)\n'
            '""")\n'
        )
        assert _tri_mech_fix(src.split("\n"), 2, "iter_dict_keys", "t.py", {"suggestion": ""}) is None

    def test_dead_assignment_removal_refused_inside_fixture(self):
        src = (
            'CODE = """\n'
            "    ws_path = tmp\n"
            '"""\n'
            "use(CODE)\n"
        )
        finding = {"suggestion": "Variable 'ws_path' assigned at line 2 but never used after. Remove the dead assignment."}
        assert _tri_mech_fix(src.split("\n"), 2, "dead_assignment", "t.py", finding) is None

    def test_regex_hoist_refused_inside_fixture(self):
        src = (
            'CODE = """\\\n'
            "    for f in files:\n"
            '        m = re.search(r"^class", f)\n'
            '""")\n'
        )
        assert _tri_mech_fix(src.split("\n"), 3, "regex_in_loop", "t.py", {"suggestion": ""}) is None


class TestRegexHoistPreservesFlags:
    def test_flags_move_into_compile_and_leave_the_call(self):
        code = textwrap.dedent(
            """\
            import re

            def scan(files):
                hits = []
                for f in files:
                    m = re.match(r"abc", f, re.IGNORECASE)
                    if m:
                        hits.append(m.group(0))
                return hits
            """
        )
        wl = code.split("\n")
        hunk = _fix_regex_in_loop(wl, 5, 6, "t.py", {"suggestion": ""})
        assert hunk is not None
        ok, patched = apply_patch(hunk, wl)
        assert ok, hunk
        joined = patched
        # flags carried into the compile constant...
        assert "re.compile(r\"abc\", re.IGNORECASE)" in joined
        # ...and removed from the call site (no positional-flags-as-pos bug).
        assert ".match(f)" in joined
        assert ".match(f, re.IGNORECASE)" not in joined
        # behavioral proof: the hoisted constant really is case-insensitive
        ns: dict = {}
        exec(joined, ns)
        assert ns["scan"](["ABCxyz"]) == ["ABC"]

    def test_trailing_flags_kwarg_also_carried(self):
        code = textwrap.dedent(
            """\
            import re

            def scan(files):
                for f in files:
                    m = re.match(r"abc", f, flags=re.IGNORECASE)
                    print(m)
            """
        )
        wl = code.split("\n")
        hunk = _fix_regex_in_loop(wl, 4, 5, "t.py", {"suggestion": ""})
        assert hunk is not None
        assert "re.compile(r\"abc\", re.IGNORECASE)" in hunk
        assert "flags=" not in hunk.split("\n")[-2] or ".match(" in hunk

    def test_never_reuses_other_functions_local_constant(self):
        """fix_cmd regression: ``_ERROR_RE`` defined inside one function was
        'reused' from another -> NameError.  The hoist must emit its own
        definition when no same-scope/module-scope constant exists."""
        code = textwrap.dedent(
            """\
            import re

            def parse(stdout):
                _ERROR_RE = re.compile(r'^(.*?):(\\d+): error: (.*)$')
                out = []
                for line in stdout.split("\\n"):
                    m = _ERROR_RE.match(line.strip())
                    if m:
                        out.append(m.groups())
                return out

            def rerun(stdout):
                errs = []
                for line in stdout.split("\\n"):
                    m = re.match(r'^(.*?):(\\d+): error: (.*)$', line.strip())
                    if m:
                        errs.append(line)
                return errs
            """
        )
        wl = code.split("\n")
        target = next(i for i, l in enumerate(wl) if "re.match(r'^" in l)
        hunk = _fix_regex_in_loop(wl, target, target + 1, "t.py", {"suggestion": ""})
        assert hunk is not None
        # A NEW local definition must be added for the second function —
        # never a bare reference to the first function's local.
        assert any(ln.startswith("+") and "= re.compile(" in ln for ln in hunk.split("\n"))
        # ...and applying + executing both functions works (no NameError).
        ok, patched = apply_patch(hunk, wl)
        assert ok
        ns: dict = {}
        exec(patched, ns)
        sample = "f.py:3: error: boom [name-defined]\n"
        assert ns["parse"](sample) and ns["rerun"](sample)


class TestListAppendJoinSelfReference:
    def test_self_referencing_accumulation_bails_out(self):
        """charts.py regression: ``stats.append(f"{len(stats)}")`` became a
        comprehension referencing the list being built -> NameError."""
        code = textwrap.dedent(
            """\
            def render(labels):
                stats: list[str] = []
                for label, val in labels:
                    stats.append(f'<text y="{14 * len(stats)}">{label}={val}</text>')
                return stats
            """
        )
        wl = code.split("\n")
        assert _fix_list_append_join(wl, 3, 4, "t.py", {"suggestion": ""}) is None

    def test_plain_accumulation_still_converts(self):
        code = textwrap.dedent(
            """\
            def f(items):
                out = []
                for x in items:
                    out.append(str(x))
                return ",".join(out)
            """
        )
        wl = code.split("\n")
        assert _fix_list_append_join(wl, 3, 4, "t.py", {"suggestion": ""}) is not None


class TestSilentExceptLoggerPosition:
    def test_print_when_logger_defined_only_below_target(self):
        """agent.py regression: logger at :104, except at :73 -> NameError."""
        code = (
            "try:\n"
            "    stream.reconfigure()\n"
            "except (ValueError, OSError):\n"
            "    pass\n"
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
        )
        wl = code.split("\n")
        hunk = _fix_silent_except(wl, 2, 3, "mod.py", {"suggestion": ""})
        assert hunk is not None
        assert "print(" in hunk
        assert "logger.warning" not in hunk

    def test_logger_warning_when_defined_above_target(self):
        code = (
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "try:\n"
            "    risky()\n"
            "except OSError:\n"
            "    pass\n"
        )
        wl = code.split("\n")
        hunk = _fix_silent_except(wl, 4, 5, "mod.py", {"suggestion": ""})
        assert hunk is not None
        assert "logger.warning(" in hunk


class TestDeadAssignmentScoping:
    def test_nonlocal_store_not_flagged(self):
        src = textwrap.dedent(
            """\
            def outer():
                patched = False

                def inner():
                    nonlocal patched
                    patched = True

                inner()
                return patched
            """
        )
        assert detect_dead_assignment(src) == []

    def test_class_body_attribute_not_flagged(self):
        src = textwrap.dedent(
            """\
            class _FakeProfile:
                temperature = 0.7
                max_tokens = 512
            """
        )
        assert detect_dead_assignment(src) == []

    def test_genuine_local_dead_store_still_flagged(self):
        src = textwrap.dedent(
            """\
            def f():
                x = [i for i in range(5)]
                x = []
                return x
            """
        )
        findings = detect_dead_assignment(src)
        assert len(findings) == 1
