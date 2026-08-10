"""Static pattern detectors for Python code — zero LLM, zero AST, pure regex.

Each detector takes source code as a string and returns a list of
``(line_number, pattern_name, suggestion)`` tuples.  False positives
are possible but kept low by regex anchoring.
"""

import ast
import io
import re
import tokenize


def _docstring_lines(source: str) -> set[int]:
    """Return 1-based line numbers that belong to any docstring in *source*."""
    try:
        tree = ast.parse(source, type_comments=True)
    except SyntaxError:
        return set()

    spans: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(
            first.value, (ast.Constant,)
        ) and isinstance(first.value.value, str):
            spans.append((first.lineno, first.end_lineno or first.lineno))

    lines: set[int] = set()
    for start, end in spans:
        lines.update(range(start, end + 1))
    return lines


def analyze(source: str) -> list[dict]:
    """Run all detectors and return unified findings."""
    all_findings: list[dict] = []
    for detector in DETECTORS:
        for line_no, name, suggestion in detector(source):
            all_findings.append({
                "line": line_no,
                "pattern": name,
                "suggestion": suggestion,
            })
    all_findings.sort(key=lambda f: f["line"])
    return all_findings


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _loop_body_lines(lines: list[str]) -> list[bool]:
    """Return a per-line boolean: whether the line is inside a for/while body.

    A loop opens at a line matching the ``for``/``while`` header.  Its body is
    any *more-indented* non-blank line.  The loop closes as soon as a non-blank
    line dedents to ``<= loop_indent`` (same indentation as the header).

    Shared by the loop-detectors so their scoping is consistent and correct
    inside methods, where indented code after a loop must NOT be mistaken for
    still-being-in-the-loop just because no column-0 line is reached.
    """
    flags = [False] * len(lines)
    in_loop = False
    loop_indent = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            loop_indent = len(line) - len(line.lstrip())
            continue
        if in_loop:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= loop_indent and stripped:
                in_loop = False
        flags[i] = in_loop
    return flags


def _loop_spans(lines: list[str]) -> list[tuple[int, int, int]]:
    """Return ``(start, end, indent)`` for each for/while body.

    ``start`` is the index of the ``for``/``while`` header; ``end`` is the
    index just past the body (exclusive); ``indent`` is the header's indent.
    Nested loops nest naturally.
    """
    spans: list[tuple[int, int, int]] = []
    stack: list[tuple[int, int]] = []  # (header_idx, indent)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            stack.append((i, len(line) - len(line.lstrip())))
            continue
        cur = len(line) - len(line.lstrip())
        while stack and (not stripped or cur <= stack[-1][1]):
            if not stripped:  # blank line: keep current loop open, just skip
                break
            header_idx, indent = stack.pop()
            spans.append((header_idx, i, indent))
    for header_idx, indent in stack:
        spans.append((header_idx, len(lines), indent))
    return spans


def _loop_caches(spans_body: list[str]) -> bool:
    """True if a loop body caches file reads in a dict.

    Recognises the pattern where reads are guarded and memoized, e.g.::

        if full in read_cache:
            ref_content = read_cache[full]
        else:
            ref_content = await agent.read_file(full, track_read=False)
            read_cache[full] = ref_content
    """
    body = "\n".join(spans_body)
    has_guard = re.search(r"\bin\s+(\w+)", body) is not None
    if not has_guard:
        return False
    cache_var = re.search(r"\bin\s+(\w+)", body).group(1)
    # a dict-style write `cache_var[...] = <name>` where that name is the
    # read result, e.g. `read_cache[full] = ref_content`
    has_memoized = re.search(
        rf"{re.escape(cache_var)}\s*\[[^\]]*\]\s*=\s*([A-Za-z_]\w*)", body
    ) is not None
    return has_memoized


def detect_regex_in_loop(source: str) -> list[tuple[int, str, str]]:
    """``re.compile()`` or ``re.match()`` inside a for/while loop body."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop_flags = _loop_body_lines(lines)
    for i, line in enumerate(lines, 1):
        if in_loop_flags[i - 1] and re.search(r"\bre\.(compile|match|search|sub|findall)\(", line) and _line_regex_arg_is_hoistable(line):
            findings.append((i, "regex_in_loop",
                             "Move re.compile() to module level — compiling inside loop wastes cycles"))
    return findings


def _line_regex_arg_is_hoistable(line: str) -> bool:
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(line).readline))
    except tokenize.TokenError:
        return False
    for i, (typ, s, _, _, _) in enumerate(toks):
        if typ == tokenize.NAME and s == "re" and i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt[1] == "." and i + 2 < len(toks):
                after = toks[i + 2]
                if after[0] == tokenize.NAME and after[1] in ("compile", "match", "search", "sub", "findall") and i + 3 < len(toks):
                    if toks[i + 3][1] == "(":
                        j = i + 4
                        while j < len(toks):
                            tj = toks[j]
                            if tj[0] in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE):
                                j += 1
                                continue
                            if tj[0] == tokenize.STRING:
                                val = tj[1]
                                prefix = ""
                                for ch in val:
                                    if ch == "'" or ch == '"':
                                        break
                                    prefix += ch
                                if "{" in val and "f" in prefix.lower():
                                    return False
                                k = j + 1
                                while k < len(toks) and toks[k][0] in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE):
                                    k += 1
                                if k < len(toks) and toks[k][1] == "+":
                                    return False
                                return True
                            return False
    return False


def detect_string_concat_in_loop(source: str) -> list[tuple[int, str, str]]:
    """String ``+=`` inside a for/while loop."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop_flags = _loop_body_lines(lines)
    for i, line in enumerate(lines, 1):
        if in_loop_flags[i - 1] and re.search(r"\w+\s*\+=\s*[\"']", line):
            findings.append((i, "string_concat_in_loop",
                             "Use ''.join() or io.StringIO instead of += in loop — O(n²) becomes O(n)"))
    return findings


def detect_bare_except(source: str) -> list[tuple[int, str, str]]:
    """Bare ``except:`` without exception type."""
    findings: list[tuple[int, str, str]] = []
    for i, line in enumerate(source.split("\n"), 1):
        if re.match(r"^\s*except\s*:", line) and "except:" in line:
            findings.append((i, "bare_except",
                             "Specify exception type — bare except catches KeyboardInterrupt and hides bugs"))
    return findings


def detect_silent_except(source: str) -> list[tuple[int, str, str]]:
    """``except ...: pass`` — silently swallowing errors."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        if re.match(r"^\s*except\b", line):
            ei = i - 1
            e_indent = len(line) - len(line.lstrip())
            body_indent = e_indent + 4
            for j in range(ei + 1, min(ei + 20, len(lines))):
                nxt = lines[j].rstrip("\r")
                if nxt.strip() == "" or nxt.lstrip().startswith("#"):
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if nxt_indent <= e_indent:
                    break  # dedented — end of except body
                if re.match(r"^\s+pass\s*$", nxt):
                    findings.append((i, "silent_except",
                                     "Replace 'pass' with a print/log warning — do NOT re-raise (silent "
                                     "handlers are often intentional fallbacks such as cache reads, "
                                     "optional features, or Ctrl-C handling); re-raising would crash "
                                     "normal operation — preserve the original control flow"))
                    break
                if nxt.strip():
                    break  # non-pass statement — block is no longer silent
    return findings


def detect_duplicate_imports(source: str) -> list[tuple[int, str, str]]:
    """Duplicate ``import X`` or ``from Y import Z`` statements at module
    level only.  Imports inside function bodies are per-function needs and
    are never duplicates of each other."""
    findings: list[tuple[int, str, str]] = []
    seen: dict[str, tuple[int, int]] = {}  # key -> (line, indent)
    for i, line in enumerate(source.split("\n"), 1):
        m = re.match(r"^\s*(?:import\s+(\S+)|from\s+(\S+)\s+import\s+(.+))", line)
        if m:
            key = m.group(0).strip()
            indent = len(line) - len(line.lstrip())
            if key in seen:
                fline, findent = seen[key]
                if indent == 0 and findent == 0:
                    findings.append((i, "duplicate_import",
                                     f"Duplicate import (first at line {fline}). Remove this copy."))
            else:
                seen[key] = (i, indent)
    return findings


def detect_missing_context_manager(source: str) -> list[tuple[int, str, str]]:
    """``open(...).read()`` without ``with`` statement — AST-based, no false positives."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    if _module_shadows_open(tree):
        return []

    exempt_ids: set[int] = set()
    shadowed_func_ids: set[int] = set()

    _BINDING_TARGETS = {
        ast.Assign: lambda n: n.targets,
        ast.AnnAssign: lambda n: [n.target],
        ast.AugAssign: lambda n: [n.target],
        ast.For: lambda n: [n.target],
        ast.AsyncFor: lambda n: [n.target],
        ast.NamedExpr: lambda n: [n.target],
    }

    def _any_target_is_open(node):
        getter = _BINDING_TARGETS.get(type(node))
        if getter is None:
            return False
        for target in getter(node):
            if isinstance(target, ast.Name) and target.id == "open":
                return True
            if isinstance(target, (ast.Tuple, ast.List)):
                for e in target.elts:
                    if isinstance(e, ast.Name) and e.id == "open":
                        return True
        return False

    def _import_binds_open(node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name) == "open":
                    return True
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == "open":
                    return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg == "open":
                    shadowed_func_ids.add(id(node))
                    break
            else:
                for sub in ast.walk(node):
                    if _any_target_is_open(sub) or _import_binds_open(sub):
                        shadowed_func_ids.add(id(node))
                        break

        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                for sub in ast.walk(item.context_expr):
                    exempt_ids.add(id(sub))

        if isinstance(node, ast.Call):
            closing = False
            if isinstance(node.func, ast.Name) and node.func.id == "closing":
                closing = True
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "closing":
                closing = True
            if closing:
                for arg in node.args:
                    for sub in ast.walk(arg):
                        exempt_ids.add(id(sub))

    parent_map: dict[int, int] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = id(node)

    findings: list[tuple[int, str, str]] = []
    seen: set[int] = set()

    for call_node in ast.walk(tree):
        if not isinstance(call_node, ast.Call):
            continue
        if not isinstance(call_node.func, ast.Name):
            continue
        if call_node.func.id != "open":
            continue

        if id(call_node) in exempt_ids:
            continue

        ancestor = parent_map.get(id(call_node))
        inside_shadowed = False
        while ancestor is not None:
            if ancestor in shadowed_func_ids:
                inside_shadowed = True
                break
            ancestor = parent_map.get(ancestor)
        if inside_shadowed:
            continue

        line = call_node.lineno
        if line not in seen:
            seen.add(line)
            findings.append((line, "missing_context_manager",
                             "Use 'with open() as f:' — current code may leak file handles"))
    return findings


def _module_shadows_open(tree) -> bool:
    """True when top-level code binds the name ``'open'`` (assign, import, def, class)."""
    import ast
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "open":
                    return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "open":
                return True
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == "open":
                    return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name) == "open":
                    return True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "open":
                return True
    return False


def _open_write_mode(line: str) -> bool:
    """True when the ``open(`` call on this single line is an explicit write.

    ``open(path, "w")``/``"a"``/``"x"`` (and their binary/plus variants) cannot
    be hoisted or cached — the detector is about *reads*.  Only a determinable
    string mode counts: an unknown/positional-variable mode is treated as a read
    so the detector stays conservative (may over-flag, never under-flag).
    """
    m = re.search(r"\bopen\s*\(([^)]*)\)", line)
    if not m:
        return False
    args = m.group(1)
    pos_args = args.split("=")[0].strip() if "=" in args else args.strip()
    parts = [p.strip() for p in pos_args.split(",")]
    if len(parts) < 2:
        return False  # default mode (r) → read
    mode = parts[1]
    if len(mode) >= 2 and mode[0] in "\"'":
        return mode[1].lower() in "wax"
    return False


_READ_SUGGESTION = (
    "File read inside loop — move open() outside or cache the content"
)
_INVARIANT_SUGGESTION = (
    "Same file re-read on every iteration — read it once before the loop "
    "(or cache the content)"
)


def _for_target_names(header: str) -> set[str]:
    """Identifiers bound by a ``for a, b in ...:`` header (its loop variables)."""
    m = re.search(r"\bfor\s+(.+?)\s+in\s", header)
    if not m:
        return set()
    target_text = m.group(1)
    names: set[str] = set()
    for piece in re.split(r"[(),]", target_text):
        for w in re.findall(r"[A-Za-z_]\w*", piece):
            names.add(w)
    return names


def _open_path_arg(stmt: str) -> str | None:
    """First argument of the ``open(...)``/``read_file(...)`` in *stmt*, or None.

    Prefers the ``with open(X ...) as`` / ``X = open(...)`` line so a separate
    ``.read()`` line can be resolved back to the identifier that holds the path.
    """
    m = re.search(r"\b(?:read_file|open)\s*\(", stmt)
    if not m:
        return None
    pos = m.end()  # index just past the opening '('
    depth = 1
    end = pos
    while end < len(stmt) and depth > 0:
        ch = stmt[end]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        end += 1
    if depth != 0:
        return None  # unbalanced — cannot determine argument
    arg = stmt[pos:end - 1].strip()
    if not arg or arg.startswith("="):
        return None
    # first positional argument — split on commas at depth 0
    parts = []
    depth = 0
    cur: list[str] = []
    for ch in arg:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur).strip())
    return parts[0] if parts else None


def _path_derived(path_expr: str, derived: set[str]) -> bool:
    """True when *path_expr* references an iteration-derived identifier.

    A path that contains the loop variable (``open(fp)``) or a name computed
    from it (``open(full)`` where ``full = os.path.join(root, f)``) changes
    per iteration — each iteration reads a distinct file, so the read cannot
    be hoisted.  Literals and loop-invariant names stay flaggable.
    """
    return any(w in derived for w in re.findall(r"[A-Za-z_]\w*", path_expr))


def _iter_derived_names(body: list[str], loop_vars: set[str]) -> set[str]:
    """Transitive set of names bound to per-iteration values inside a loop.

    Starts from the loop ``for`` target(s) and any nested ``for`` targets, then
    grows fixed-point over simple ``name = expr`` assignments whose right side
    already references a derived name.  e.g. for ``full = os.path.join(root, f)``
    inside ``os.walk`` iteration, ``full`` becomes derived.  A constant
    hand-assigned each iteration (``cfg = "x.yaml"``) stays invariant.
    """
    derived = set(loop_vars)
    assignments: dict[str, str] = {}
    for line in body:
        fm = re.match(r"\s*for\s+(.+?)\s+in\s", line)
        if fm:
            derived |= _for_target_names(line)
        am = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if am:
            name = am.group(1)
            rhs = line.split("=", 1)[1].strip() if "=" in line else ""
            assignments[name] = rhs
    changed = True
    while changed:
        changed = False
        for name, rhs in assignments.items():
            if name not in derived and any(
                w in derived for w in re.findall(r"[A-Za-z_]\w*", rhs)
            ):
                derived.add(name)
                changed = True
    return derived


def _read_receiver_alias(body: list[str]) -> dict[str, str]:
    """Map ``with open(...) as name:`` / ``name = open(...)`` names to path args.

    Lets a later ``.read()`` line resolve back to the identifier that holds the
    opened path so its derivation can be judged.
    """
    result: dict[str, str] = {}
    for line in body:
        with_m = re.match(r"\s*with\s+(.+?)\s+as\s+([A-Za-z_]\w*)\s*:", line)
        if with_m:
            path = _open_path_arg(with_m.group(1))
            if path:
                result[with_m.group(2)] = path
            continue
        asg = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*open\s*\(", line)
        if asg:
            path = _open_path_arg(line)
            if path:
                result[asg.group(1)] = path
    return result


def detect_file_read_in_loop(source: str) -> list[tuple[int, str, str]]:
    """``open()`` or ``read_file()`` inside a for/while loop.

    A read is flagged only when it is a genuinely nested read:
    * the same file is re-read on every iteration (loop-invariant path), or
    * the path cannot be proven to change per iteration.

    Reads of *distinct files per iteration* — ``open(fp)`` for a ``for fp in
    files:`` loop, or ``open(full)`` where ``full = os.path.join(root, f)`` in
    an ``os.walk`` — carry no hoistable work and are NOT flagged.  Loops that
    memoize reads through a cache dict are not flagged either, and write-mode
    ``open()`` calls are excluded: writes cannot be hoisted or cached.

    For reads nested inside multiple loops (e.g. ``while`` inside ``for``),
    derived-names and receiver-aliases from *all* enclosing loop bodies are
    aggregated so that hoistable work can be distinguished from per-iteration
    distinct-file reads.
    """
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    spans = _loop_spans(lines)
    for i, line in enumerate(lines, 1):
        if not ("open(" in line or "read_file(" in line or ".read(" in line):
            continue
        if "open(" in line and _open_write_mode(line):
            continue
        idx = i - 1
        derived: set[str] = set()
        receivers: dict[str, str] = {}
        cached = False
        in_any_loop = False
        for start, end, _ in spans:
            if not (start < idx < end):  # strictly inside this loop's body
                continue
            in_any_loop = True
            body = lines[start + 1:end]
            if _loop_caches(body):
                cached = True
                break
            loop_vars = _for_target_names(lines[start])
            derived |= _iter_derived_names(body, loop_vars)
            receivers.update(_read_receiver_alias(body))
        if not in_any_loop or cached:
            continue
        path = _open_path_arg(line)
        if path is None:
            # bare ``f.read()`` — resolve through its with/as alias
            rm = re.search(r"\.read\s*\(", line)
            if not rm:
                continue
            recv_m = re.search(r"([A-Za-z_]\w*)\s*\.read\s*\(", line)
            rcvr = recv_m.group(1) if recv_m else None
            if not rcvr:
                continue
            path = receivers.get(rcvr, "")
        if _path_derived(path, derived):
            continue  # distinct file per iteration — no hoistable work
        findings.append((i, "file_read_in_loop",
                         _INVARIANT_SUGGESTION if path
                         else _READ_SUGGESTION))
    return findings


def detect_list_append_join(source: str) -> list[tuple[int, str, str]]:
    """``.append()`` / ``.extend()`` / ``+= [x]`` in a loop whose built list
    is consumed by ``''.join()``.

    Only fires when the loop-built list is actually used by a ``.join()`` call
    (matching the pattern's name).  Appends/extensions that feed ``sorted()``,
    ``set()``, ``len()`` or a plain ``return`` are NOT flagged — converting
    those to a comprehension is a style-only change, and the optimizer must not
    be pushed into restructuring loops for no measurable gain.

    Loops whose append expressions reference names *assigned inside the loop
    body* (loop-carried state) are also skipped — they cannot be trivially
    comprehension-converted and would require a structural rewrite.
    """
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    loop_indent = 0
    loop_body_lines: list[str] = []
    body_assigned: set[str] = set()
    loop_candidates: list[tuple[int, str]] = []   # pending until loop closes
    append_lines: list[tuple[int, str]] = []       # filtered survivors
    _APPEND_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.(?:append|extend)\(")
    _IEQ_RE = re.compile(r"([A-Za-z_]\w*)\s*\+=\s*\S")

    def _flush_loop():
        """Filter *loop_candidates* against *body_assigned* and commit
        stateless entries to *append_lines*.  If *any* candidate references
        a body-assigned name (loop-carried state), the entire loop is skipped."""
        nonlocal loop_candidates
        any_stateful = False
        for li, var in loop_candidates:
            ol = lines[li - 1]
            for name in body_assigned:
                if name != var and re.search(rf"\b{re.escape(name)}\b", ol):
                    any_stateful = True
                    break
            if any_stateful:
                break
        if not any_stateful:
            append_lines.extend(loop_candidates)
        loop_candidates = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            _flush_loop()
            in_loop = True
            loop_indent = len(line) - len(line.lstrip())
            loop_body_lines = []
            body_assigned = set()
            loop_candidates = []
            continue
        if in_loop:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= loop_indent and stripped:
                _flush_loop()
                in_loop = False
            else:
                loop_body_lines.append(stripped)
                m = _assignment_match(stripped, compound=False)
                if m:
                    body_assigned.add(m.group(1))
        if in_loop and (".append(" in line or ".extend(" in line
                        or ("+=" in line and "[" in line)):
            has_complex_flow = any(
                re.search(r'\b(if|elif|else|await|break|continue|return|try|except)\b', bl)
                for bl in loop_body_lines
            )
            if not has_complex_flow:
                m = _APPEND_RE.search(line)
                if m:
                    loop_candidates.append((i, m.group(1)))
                else:
                    m = _IEQ_RE.search(line)
                    if m:
                        loop_candidates.append((i, m.group(1)))

    _flush_loop()

    # Only flag lines whose list is consumed by a .join() call.
    for i, var in append_lines:
        if re.search(rf"\.join\s*\(\s*{re.escape(var)}\s*\)", source):
            findings.append((i, "list_append_join",
                             "Build the joined list in one comprehension/generator "
                             "instead of building it per-iteration with .append/.extend/+= — "
                             "currently O(n) per-iteration call overhead"))
    return findings


def _assignment_match(stripped: str, compound: bool = False) -> re.Match | None:
    """Match an assignment statement start, returning the bound name.

    When *compound* is False (default) matches plain ``x = ...`` and annotated
    ``x: Type = ...`` only — used by the adjacent-overwrite pass because a
    compound ``+=`` both reads *and* writes (so it is NOT an overwrite).

    When *compound* is True also matches compound writes
    ``x += ...``, ``x -= ...`` ... — used by the never-used-after pass, where a
    final ``x += `` whose value is never re-read is a dead store.

    Does NOT match ``for`` loops, ``except ... as e``, tuple/unpacking
    (``a, b = ...``) or ``:=`` walrus.
    """
    _compound = r"\+=|-=|\*=|/=|//=|%=|\*\*=|&=|\|=|\^=|<<=|>>="
    target = "=" if not compound else rf"(?:{_compound}|=)"
    return re.match(rf"^([A-Za-z_]\w*)\s*(?::[^=]+)?\s*{target}", stripped)


def _inside_bracket_depth(source: str) -> list[int]:
    """Per-line depth of `(`/`[`/`{` nesting at the START of each line.

    Used to recognize keyword arguments inside multi-line calls and parenthesized
    contexts: a line like ``capture_output=True,`` inside a multi-line
    ``subprocess.run([...], ...)`` call is a kwarg, not an assignment statement.
    ``result[i]`` is the open-bracket depth at the start of source line ``i+1``.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Fall back to all-zero depth (treat nothing as bracketed) on broken
        # input; the dead-assignment detector can then only over-report, never
        # silently miss a real assignment.
        return [0] * (source.count("\n") + 1)

    # Depth at the start of line L = the bracket depth as it stands just before
    # the first real token on that line.
    depth_by_line: dict[int, int] = {}
    depth = 0
    for tok in tokens:
        line_no = tok[2][0]
        # On a blank/comment-only line that has no opening bracket, the depth
        # carried across is its start-of-line depth (state before first token).
        if line_no not in depth_by_line:
            depth_by_line[line_no] = depth
        if tok[0] == tokenize.OP:
            if tok[1] in "([{":
                depth += 1
            elif tok[1] in ")]}":
                depth = max(0, depth - 1)

    # Any line with no tokens at all (all-blank or comment-only) inherits the
    # depth exported below the last tokenized line that preceded it.
    line_count = source.count("\n") + 1
    result = [0] * line_count
    carry = 0
    for ln in range(1, line_count + 1):
        if ln in depth_by_line:
            carry = depth_by_line[ln]
        result[ln - 1] = carry
    return result


def detect_dead_assignment(source: str) -> list[tuple[int, str, str]]:
    """Detect dead assignments: variables that are (a) immediately overwritten
    on the next statement, or (b) assigned at function-body scope but never
    referenced again (dead store).

    For (b) a simple indentation-based heuristic tracks ``def``/``async def``
    nesting without parsing.  Module-level assignments are skipped because the
    name may be imported and used by other modules in the project.
    """
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    bracket_depth = _inside_bracket_depth(source)

    # First pass: adjacent-overwrite detection (kept from the original rule).
    for i, line in enumerate(lines, 1):
        if bracket_depth[i - 1] > 0:
            continue  # inside a call/list/dict — a kwarg, not a statement assignment
        stripped = line.strip()
        m = _assignment_match(stripped)
        if not m:
            continue
        var_name = m.group(1)
        for j in range(i, min(i + 3, len(lines))):
            next_line = lines[j].strip() if j < len(lines) else ""
            if not next_line:
                continue
            m2 = _assignment_match(next_line)
            if m2 and m2.group(1) == var_name:
                # Skip when the overwrite line reads the variable on its RHS
                # (e.g. ``ws = agent.workspace`` then ``ws = to_windows_path(ws)``):
                # the first value is consumed, so the store is NOT dead.
                rhs = next_line[m2.end():]
                if not re.search(rf"\b{re.escape(var_name)}\b", rhs):
                    findings.append((i, "dead_assignment",
                                     f"Variable '{var_name}' assigned at line {i} then immediately overwritten at line {j + 1}. Remove the dead assignment."))
            break

    seen_dead: set[tuple[int, str]] = {(f[0], f[1]) for f in findings}

    # Second pass: assigned-but-never-read-after detection.
    #
    # This pass also considers compound-write opcodes (``x += ...``).  An
    # assignment/compound-write is a dead store when the name is never
    # referenced again on any later line (a compound ``x +=`` reads x once, but
    # leaves a value that is never consumed -> still dead).
    #
    # Track def nesting via indentation.  ``func_depth`` is the number of
    # enclosing ``def`` blocks at the indentation of the assignment.
    for i, line in enumerate(lines, 1):
        if bracket_depth[i - 1] > 0:
            continue  # inside a call/list/dict — a kwarg, not a statement assignment
        stripped = line.strip()
        m = _assignment_match(stripped, compound=True)
        if not m:
            continue
        var_name = m.group(1)
        if var_name.startswith("_"):
            continue  # convention: intentionally unused

        indent = len(line) - len(line.lstrip())
        if _func_depth_at(lines, i, indent) == 0:
            continue  # module-level: may be exported/used elsewhere

        # If already flagged as an adjacent overwrite, don't double-report.
        if (i, "dead_assignment") in seen_dead:
            continue

        # Look for any later reference to the name (excluding this line).
        used_after = False
        for k in range(i, len(lines)):
            if k + 1 == i:  # same line as assignment (1-based i == k+1)
                continue
            if re.search(rf"\b{re.escape(var_name)}\b", lines[k]):
                used_after = True
                break

        # Loop back-edge: a store at the end of a for/while body feeds a read
        # at the top of the *next* iteration, which appears textually BEFORE
        # the store.  Such loop-state variables (e.g. ``prev_error_sigs =
        # dict(...)`` compared against at the top of the body) are live, so a
        # reference anywhere else inside the same loop body keeps the store.
        if not used_after:
            used_after = _loop_body_references(lines, i, var_name)

        if not used_after:
            findings.append((i, "dead_assignment",
                             f"Variable '{var_name}' assigned at line {i} but never used after. Remove the dead assignment."))

    findings.sort(key=lambda f: f[0])
    return findings


def _loop_body_references(lines: list[str], line: int, var_name: str) -> bool:
    """True if *var_name* is referenced on any other line of the enclosing
    for/while body for 1-based *line*.

    Textual scan matching ``detect_dead_assignment``'s style (no AST): walk up
    from ``line`` looking for a for/while header at a smaller indentation;
    then search every line of that body except the assignment line itself.  A
    reference at the top of the body (the next-iteration read) suppresses the
    false dead-store finding.
    """
    indent = len(lines[line - 1]) - len(lines[line - 1].lstrip())
    header: int | None = None
    j = line - 2  # 0-based index one line above the assignment
    while j >= 0:
        lj = lines[j]
        if not lj.strip():
            j -= 1
            continue
        if len(lj) - len(lj.lstrip()) >= indent:
            j -= 1
            continue  # still inside the same or a nested block
        if re.match(r"(?:async\s+)?(?:for|while)\b", lj.strip()):
            header = j
            break
        j -= 1  # a dedented non-loop line (if/def) — keep scanning up
    if header is None:
        return False

    header_indent = len(lines[header]) - len(lines[header].lstrip())
    for k in range(header + 1, len(lines)):
        lk = lines[k]
        if lk.strip() and len(lk) - len(lk.lstrip()) <= header_indent:
            break  # body ended
        if k + 1 == line:
            continue  # the assignment itself
        if re.search(rf"\b{re.escape(var_name)}\b", lk):
            return True
    return False


def detect_walrus_in_comprehension(source: str) -> list[tuple[int, str, str]]:
    """Flag ``:`` assignment expressions (walrus) used inside a comprehension.

    A walrus inside a list/set/dict/generator comprehension binds the target name
    into the *enclosing* scope (PEP 572) rather than the comprehension-local
    scope.  This leaks a name that can mask later re-assignments or shadow an
    existing binding — a classic source of subtle, non-obvious bugs.

    One finding is emitted per comprehension, at the line of the first walrus
    encountered inside it.  A walrus used outside a comprehension (e.g. in a
    plain ``if (m := re.match(...)):``) is NOT flagged here — that is the
    idiomatic use case.
    """
    import ast

    findings: list[tuple[int, str, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        walruses = [sub for sub in ast.walk(node) if isinstance(sub, ast.NamedExpr)]
        if not walruses:
            continue
        first = walruses[0]
        names = ", ".join({ast.unparse(w.target) for w in walruses})
        findings.append((first.lineno, "walrus_in_comprehension",
                         f"Walrus '{names}' inside comprehension leaks into enclosing scope. "
                         f"Bind these as named locals on their own line before the comprehension."))
    findings.sort(key=lambda f: f[0])
    return findings


def _func_depth_at(lines: list[str], line_no: int, assign_indent: int) -> int:
    """Approximate how many ``def``/``async def`` blocks enclose line *line_no*.

    Uses indentation-only tracking: a ``def`` increases depth for lines indented
    strictly more deeply than it.  A def at indent *D* only encloses code with
    indent > D, so module-level statements (indent 0) are never considered
    inside a module-level function body.  The assignment's own *assign_indent*
    disambiguates the final (unprocessed-at-closer) dedent.
    """
    stack: list[int] = []  # indent levels of open def blocks
    for idx in range(line_no - 1):
        raw = lines[idx]
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        while stack and indent <= stack[-1]:
            stack.pop()
        if re.match(r"^\s*(?:async\s+)?def\s+\w+", raw):
            stack.append(indent)
    # A def at indent D only encloses code indented strictly more than D.
    return sum(1 for d in stack if d < assign_indent)


def _strip_strings(line: str) -> str:
    """Remove string literals from *line* so bracket counting ignores parens inside strings."""
    return re.sub(r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', "", line)


def _balance_delta(line: str) -> int:
    """Return net open-minus-close delimiter count for () [] {} on *line* (strings ignored)."""
    clean = _strip_strings(line)
    return clean.count("(") - clean.count(")") + \
           clean.count("[") - clean.count("]") + \
           clean.count("{") - clean.count("}")


def detect_unreachable_code(source: str) -> list[tuple[int, str, str]]:
    """Code after return/break/continue that can never execute.

    Handles multi-line statements (e.g. ``return sorted(set(`` spanning lines)
    by tracking delimiter balance across continuation lines.
    """
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")

    for i, line in enumerate(lines, 1):
        if not re.match(r"^\s*(return|break|continue)\b", line):
            continue

        stmt_indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        # Track multi-line statement: skip continuation lines while brackets are open.
        balance = _balance_delta(line)
        j = i  # pointer to current statement line (0-based: lines[j-1])

        while balance > 0 and j < len(lines):
            j += 1
            if j >= len(lines):
                break
            balance += _balance_delta(lines[j - 1])

        # Now the statement has ended at line j (0-based index j-1).
        # Find the next non-empty, non-comment line.
        for k in range(j, len(lines)):
            nxt = lines[k]
            nxt_stripped = nxt.strip()
            if not nxt_stripped or nxt_stripped.startswith("#"):
                continue
            next_indent = len(nxt) - len(nxt.lstrip())
            if next_indent >= stmt_indent:
                findings.append((k + 1, "unreachable_code",
                                 f"Code after '{stripped}' at line {i} can never execute."))
            break

    return findings


def detect_unused_imports(source: str) -> list[tuple[int, str, str]]:
    """Import statement that is never referenced in the code."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        m = re.match(r"^import\s+(\w+)(?:\s+as\s+(\w+))?", stripped)
        if m:
            name = m.group(2) or m.group(1)
        else:
            m = re.match(r"^from\s+(\S+)\s+import\s+(\w+)(?:\s+as\s+(\w+))?", stripped)
            if m:
                if m.group(1) == "__future__":
                    continue
                name = m.group(3) or m.group(2)
            else:
                continue
        # Check if name appears elsewhere in code (not on import line)
        used = False
        for j, check_line in enumerate(lines, 1):
            if j == i:
                continue
            if re.search(rf"\b{name}\b", check_line):
                used = True
                break
        if not used:
            findings.append((i, "unused_import",
                             f"Imported '{name}' is never used. Remove the import."))
    return findings


# ---------------------------------------------------------------------------
# Cross-file detectors (called from implement --review)
# ---------------------------------------------------------------------------

def detect_module_collisions(generated_files: list[str], existing_files: list[str] | None = None) -> list[dict]:
    """Flag generated modules whose names overlap with existing project modules.

    Checks for near-duplicate filenames (e.g., ``lm_studio_provider.py``
    overlapping with ``lmstudio.py``).  Also verifies via importlib.

    Returns findings with "module_collision" pattern.
    """
    import difflib

    findings: list[dict] = []

    # Collect known project files if not provided
    if existing_files is None:
        existing_files = []
        import os as _os
        for root, dirs, files in _os.walk("."):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
            for f in files:
                if f.endswith(".py"):
                    existing_files.append(_os.path.relpath(_os.path.join(root, f), ".").replace("\\", "/"))

    for fname in generated_files:
        if not fname.endswith(".py") or fname.endswith("__init__.py"):
            continue
        stem = fname.rsplit("/", 1)[-1].replace(".py", "")
        collisions: list[str] = []
        for other in existing_files:
            if other == fname or other in generated_files:
                continue
            other_stem = other.rsplit("/", 1)[-1].replace(".py", "")
            # Check via difflib for similar names (threshold 0.6)
            ratio = difflib.SequenceMatcher(None, stem, other_stem).ratio()
            substring_match = stem in other_stem or other_stem in stem
            prefix_match = other_stem.startswith(stem[:6]) or stem.startswith(other_stem[:6])
            if ratio > 0.6 or substring_match or prefix_match:
                collisions.append(other)
        if collisions:
            findings.append({
                "file": fname,
                "line": 0,
                "pattern": "module_collision",
                "suggestion": f"Module name similar to existing: {', '.join(collisions[:3])}. Consider a more distinct name.",
            })
    return findings


def detect_attribute_errors(source_files: dict[str, str], project_root: str) -> list[dict]:
    """Check attribute access patterns against what imported classes actually export.

    Tracks both direct imports AND annotated variables/functions that reference
    imported types.  E.g., ``p.reasoning_effort`` is checked if ``p: ProfileMetadata``
    and ``ProfileMetadata`` was imported.

    Returns findings with "attribute_error" pattern.
    """
    import ast
    import os as _os

    findings: list[dict] = []

    for fname, source in source_files.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        # Build: {imported_name: (module_path, class_name)}
        imports: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = (node.module, alias.name)

        # Track variable type annotations: {var_name: imported_type_name}
        type_hints: dict[str, str] = {}
        for node in ast.walk(tree):
            # Function parameter annotations: def foo(p: ProfileMetadata)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if arg.annotation and isinstance(arg.annotation, ast.Name):
                        if arg.annotation.id in imports:
                            type_hints[arg.arg] = arg.annotation.id
            # Variable assignments with type annotations: x: ProfileMetadata = ...
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.annotation, ast.Name) and node.annotation.id in imports:
                    if isinstance(node.target, ast.Name):
                        type_hints[node.target.id] = node.annotation.id

        # Collect attribute accesses on tracked variables
        attrs_used: dict[str, set[str]] = {}  # {type_name: {attr, ...}}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    # Direct import access: ProfileMetadata.xyz
                    if var_name in imports:
                        attrs_used.setdefault(var_name, set()).add(node.attr)
                    # Annotated variable access: p.xyz where p: ProfileMetadata
                    if var_name in type_hints:
                        type_name = type_hints[var_name]
                        attrs_used.setdefault(type_name, set()).add(node.attr)

        # Verify each attribute against the imported module
        checked: set[str] = set()
        for obj_name, attrs in attrs_used.items():
            if obj_name not in imports:
                continue
            mod_path, cls_name = imports[obj_name]
            check_key = f"{mod_path}.{cls_name}"
            if check_key in checked:
                continue
            checked.add(check_key)

            mod_file = mod_path.replace(".", _os.sep) + ".py"
            for search_root in [project_root, _os.getcwd()]:
                full = _os.path.normpath(_os.path.join(search_root, mod_file))
                if _os.path.isfile(full):
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            mod_source = f.read()
                        mod_tree = ast.parse(mod_source)
                    except Exception:
                        continue
                    class_fields: set[str] = {"__class__", "__dict__", "__name__", "__init__"}
                    for node in ast.walk(mod_tree):
                        if isinstance(node, ast.ClassDef) and node.name == cls_name:
                            for sn in node.body:
                                if isinstance(sn, (ast.AnnAssign, ast.Assign)):
                                    targets = [sn.target] if isinstance(sn, ast.AnnAssign) else sn.targets
                                    for t in targets:
                                        names = {n.id for n in ast.walk(t) if isinstance(n, ast.Name) and n.id not in ("self", "cls")}
                                        class_fields.update(names)
                            # Also check __init__ parameters
                            for sn in node.body:
                                if isinstance(sn, ast.FunctionDef) and sn.name == "__init__":
                                    for arg in sn.args.args:
                                        if arg.arg != "self":
                                            class_fields.add(arg.arg)
                            # Also check @dataclass fields (ast.Name from annotations)
                            for sn in node.body:
                                if isinstance(sn, ast.AnnAssign):
                                    names = {n.id for n in ast.walk(sn.target) if isinstance(n, ast.Name)}
                                    class_fields.update(names)
                            # body assignments like self.name = ...
                            for sn in ast.walk(node):
                                if isinstance(sn, ast.Assign):
                                    for t in sn.targets:
                                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                                            class_fields.add(t.attr)
                            break
                    missing = attrs - class_fields
                    for attr in sorted(missing):
                        findings.append({
                            "file": fname,
                            "line": 0,
                            "pattern": "attribute_error",
                            "suggestion": f"{obj_name}.{attr} — '{attr}' not found in class '{cls_name}' from {mod_path}. Available: {', '.join(sorted(class_fields)[:12]) or 'none'}",
                        })
                    break
    return findings


def detect_unwired_modules(generated_files: list[str], project_root: str) -> list[dict]:
    """Flag generated modules that are not imported by any existing code.

    Returns findings with "unwired_module" pattern.
    """
    import os as _os
    findings: list[dict] = []

    for fname in generated_files:
        if not fname.endswith(".py") or fname.endswith("__init__.py"):
            continue

        mod_name = fname.replace("/", ".").replace(".py", "")
        # Check if any .py file in the project imports this module
        referenced = False
        for root, dirs, files in _os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "__pycache__")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                fp = _os.path.join(root, f)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        content = fh.read()
                except Exception:
                    continue
                # Check for import references
                if mod_name in content or fname.replace("/", ".").replace(".py", "") in content.replace("\\", "/"):
                    referenced = True
                    break
            if referenced:
                break

        if not referenced:
            findings.append({
                "file": fname,
                "line": 0,
                "pattern": "unwired_module",
                "suggestion": f"Module '{fname}' is not imported by any existing project code. It needs to be wired in (e.g., add 'from {mod_name} import X' in a consumer module).",
            })

    return findings


def detect_class_conflicts(generated_files: list[str], project_root: str) -> list[dict]:
    """Flag generated class/function names that collide with existing production code
    in the *same directory*.  Cross-package name reuse is intentional.
    """
    import os as _os
    import ast

    findings: list[dict] = []
    skip = {"__init__", "__str__", "__repr__", "__eq__", "__hash__", "__call__",
            "__enter__", "__exit__", "__getitem__", "__setitem__", "__iter__", "__len__",
            "get", "set", "run", "main", "execute", "record", "start", "stop"}

    existing: dict[str, tuple[str, str]] = {}  # name → (rel_path, dir)
    for root, dirs, files in _os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "backups")]
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = _os.path.join(root, f)
            rel = _os.path.relpath(fp, project_root).replace("\\", "/")
            if rel in generated_files:
                continue
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                pkg_dir = _os.path.dirname(rel)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name not in skip:
                        existing[node.name] = (rel, pkg_dir)
            except Exception:
                pass

    for fname in generated_files:
        if not fname.endswith(".py"):
            continue
        try:
            with open(_os.path.join(project_root, fname), "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except Exception:
            continue
        gen_dir = _os.path.dirname(fname)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                continue
            if node.name in skip:
                continue
            if node.name in existing:
                ex_rel, ex_dir = existing[node.name]
                if ex_dir == gen_dir:
                    findings.append({
                        "file": fname, "line": node.lineno, "pattern": "class_conflict",
                        "suggestion": f"'{node.name}' already defined in {ex_rel} (same directory). Rename in {fname} to avoid import collision.",
                    })

    return findings


# ---------------------------------------------------------------------------
# New detectors (added 2026-08-09)
# ---------------------------------------------------------------------------

_NONE_EQ_RE = re.compile(r"(\b[A-Za-z_]\w*)\s*(==|!=)\s*None\b")
_FOR_IN_KEYS_RE = re.compile(r"\bfor\s+\w+\s+in\s+(\w+)\.keys\(\)")
_IN_KEYS_RE = re.compile(r"in\s+(\w+)\.keys\(\)")


def _fstring_has_interpolation(content: str) -> bool:
    """True when *content* contains an unescaped ``{`` (not ``{{``)."""
    i = 0
    while i < len(content):
        if content[i] == "{":
            if i + 1 < len(content) and content[i + 1] == "{":
                i += 2
            else:
                return True
        else:
            i += 1
    return False


def _fstring_lines_without_interpolation(source: str) -> list[tuple[int, int]]:
    """Return ``(line, column)`` for every f‑string that has NO interpolation.

    Uses the tokenizer to read the ``FSTRING_START`` … ``FSTRING_END`` stream.
    An ``OP {`` between start and end signals interpolation — only f‑strings
    without one are reported.
    """
    findings: list[tuple[int, int]] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return findings
    lines = source.split("\n")
    fstring_start: tuple[int, int] | None = None
    has_interp = False
    for tok in tokens:
        if tok.type == tokenize.FSTRING_START:
            fstring_start = tok.start
            has_interp = False
            continue
        if fstring_start is None:
            continue
        if tok.type == tokenize.FSTRING_END:
            if not has_interp:
                line_no, col = fstring_start
                if line_no <= len(lines) and col < len(lines[line_no - 1]):
                    if lines[line_no - 1][col] in ('f', 'F'):
                        findings.append((line_no, col))
            fstring_start = None
            continue
        if tok.type == tokenize.OP and tok.string == "{":
            has_interp = True
    return findings


def detect_none_eq(source: str) -> list[tuple[int, str, str]]:
    """``x == None`` or ``x != None`` — should be ``is None``/``is not None``."""
    findings: list[tuple[int, str, str]] = []
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        m = _NONE_EQ_RE.search(stripped)
        if m:
            findings.append((i, "none_eq",
                             f"Use '{m.group(1)} is None' / '{m.group(1)} is not None' "
                             f"instead of '{m.group(1)} {m.group(2)} None'"))
    return findings


def detect_fstring_without_placeholder(source: str) -> list[tuple[int, str, str]]:
    """f‑string literal whose content contains NO interpolation expression."""
    findings: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for line_no, col in _fstring_lines_without_interpolation(source):
        if line_no in seen:
            continue
        seen.add(line_no)
        findings.append((line_no, "fstring_without_placeholder",
                         "Remove the 'f' prefix — no interpolation expressions inside"))
    return findings


def detect_iter_dict_keys(source: str) -> list[tuple[int, str, str]]:
    """``for k in d.keys():`` or ``k in d.keys()`` — drop the redundant ``.keys()``."""
    findings: list[tuple[int, str, str]] = []
    _ds = _docstring_lines(source)
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#") or i in _ds:
            continue
        m = _FOR_IN_KEYS_RE.search(stripped) or _IN_KEYS_RE.search(stripped)
        if not m:
            continue
        findings.append((i, "iter_dict_keys",
                         f"Drop redundant .keys(): use '{m.group(1)} in d' or "
                         f"'for {m.group(1)} in d:' instead"))
    return findings


def detect_type_comparison(source: str) -> list[tuple[int, str, str]]:
    """``type(x) == SomeType`` / ``type(x) in (A, B)`` — use ``isinstance``."""
    findings: list[tuple[int, str, str]] = []
    _ds = _docstring_lines(source)
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#") or i in _ds:
            continue
        m = re.search(r"type\((\w+)\)\s*==\s*(\w+)", stripped)
        if m:
            findings.append((i, "type_comparison",
                             f"Use isinstance({m.group(1)}, {m.group(2)}) instead of type() == check"))
            continue
        m = re.search(r"type\((\w+)\)\s*!=\s*(\w+)", stripped)
        if m:
            findings.append((i, "type_comparison",
                             f"Use 'not isinstance({m.group(1)}, {m.group(2)})' instead of type() != check"))
            continue
        m = re.search(r"type\((\w+)\)\s+in\s+\(([^)]+)\)", stripped)
        if m:
            findings.append((i, "type_comparison",
                             f"Use isinstance({m.group(1)}, ({m.group(2)})) instead of type() in (...) check"))
    return findings


def detect_mutable_default_arg(source: str) -> list[tuple[int, str, str]]:
    """``def f(x=[], y={})`` — mutable default argument persists across calls."""
    findings: list[tuple[int, str, str]] = []
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith(("def ", "async def ")):
            continue
        m = re.search(r"=\s*(\[\]|\{\})", stripped)
        if m:
            findings.append((i, "mutable_default_arg",
                             f"Mutable default argument {m.group(1)}: use None and "
                             f"guard with '{m.group(1)} or param is None' in function body"))
    return findings


def detect_redundant_bool_expr(source: str) -> list[tuple[int, str, str]]:
    """``return True if cond else False`` — replace with ``bool(cond)``."""
    findings: list[tuple[int, str, str]] = []
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = re.search(r"return\s+True\s+if\s+(.+?)\s+else\s+False\s*$", stripped)
        if m:
            findings.append((i, "redundant_bool_expr",
                             f"Use 'return bool({m.group(1).strip()})' instead of True-if-else-False"))
            continue
        m = re.search(r"return\s+False\s+if\s+(.+?)\s+else\s+True\s*$", stripped)
        if m:
            findings.append((i, "redundant_bool_expr",
                             f"Use 'return not ({m.group(1).strip()})' or "
                             f"'return not bool({m.group(1).strip()})' instead"))
    return findings


DETECTORS = [
    detect_regex_in_loop,
    detect_string_concat_in_loop,
    detect_bare_except,
    detect_silent_except,
    detect_duplicate_imports,
    detect_missing_context_manager,
    detect_file_read_in_loop,
    detect_list_append_join,
    detect_dead_assignment,
    detect_unreachable_code,
    detect_walrus_in_comprehension,
    detect_unused_imports,
    detect_none_eq,
    detect_fstring_without_placeholder,
    detect_iter_dict_keys,
    detect_type_comparison,
    detect_mutable_default_arg,
    detect_redundant_bool_expr,
]
