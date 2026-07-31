"""Static pattern detectors for Python code — zero LLM, zero AST, pure regex.

Each detector takes source code as a string and returns a list of
``(line_number, pattern_name, suggestion)`` tuples.  False positives
are possible but kept low by regex anchoring.
"""

import re
from collections import Counter


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

def detect_regex_in_loop(source: str) -> list[tuple[int, str, str]]:
    """``re.compile()`` or ``re.match()`` inside a for/while loop body."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and re.search(r"\bre\.(compile|match|search|sub|findall)\(", line):
            findings.append((i, "regex_in_loop",
                             "Move re.compile() to module level — compiling inside loop wastes cycles"))
    return findings


def detect_string_concat_in_loop(source: str) -> list[tuple[int, str, str]]:
    """String ``+=`` inside a for/while loop."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and re.search(r"\w+\s*\+=\s*[\"']", line):
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
            # Look at next line for pass
            if i < len(lines) and re.match(r"^\s+pass\s*$", lines[i]):
                findings.append((i, "silent_except",
                                 "Replace 'pass' with logging or re-raise — error is silently swallowed"))
    return findings


def detect_duplicate_imports(source: str) -> list[tuple[int, str, str]]:
    """Duplicate ``import X`` or ``from Y import Z`` statements."""
    findings: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}
    for i, line in enumerate(source.split("\n"), 1):
        m = re.match(r"^\s*(?:import\s+(\S+)|from\s+(\S+)\s+import\s+(.+))", line)
        if m:
            key = m.group(0).strip()
            if key in seen:
                findings.append((i, "duplicate_import",
                                 f"Duplicate import (first at line {seen[key]}). Remove this copy."))
            else:
                seen[key] = i
    return findings


def detect_missing_context_manager(source: str) -> list[tuple[int, str, str]]:
    """``open(...).read()`` without ``with`` statement."""
    findings: list[tuple[int, str, str]] = []
    for i, line in enumerate(source.split("\n"), 1):
        if "open(" in line and "with " not in line and " with " not in line and "with(" not in line:
            # Avoid false positives: comments, in strings, or in already-correct patterns
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "contextlib" in stripped or "closing" in stripped:
                continue
            findings.append((i, "missing_context_manager",
                             "Use 'with open() as f:' — current code may leak file handles"))
    return findings


def detect_file_read_in_loop(source: str) -> list[tuple[int, str, str]]:
    """``open()`` or ``read_file()`` inside a for/while loop."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and ("open(" in line or "read_file(" in line or ".read(" in line):
            findings.append((i, "file_read_in_loop",
                             "File read inside loop — move open() outside or cache the content"))
    return findings


def detect_list_append_join(source: str) -> list[tuple[int, str, str]]:
    """``.append()`` in a loop → ``''.join()`` — not a bug, just an optimization hint."""
    findings: list[tuple[int, str, str]] = []
    lines = source.split("\n")
    in_loop = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^\s*(for|while)\s", line):
            in_loop = True
            continue
        if in_loop and not line.startswith((" ", "\t")) and stripped:
            in_loop = False
        if in_loop and ".append(" in line:
            # Only flag if we also see a join later in the same function
            findings.append((i, "list_append_join",
                             "If this builds a string list, use list comprehension or generator — currently O(n) memory"))
    return findings


# ---------------------------------------------------------------------------
#  Cross-file detectors (called from implement --review)
# ---------------------------------------------------------------------------

def detect_module_collisions(generated_files: list[str], existing_files: list[str] | None = None) -> list[dict]:
    """Flag generated modules whose names overlap with existing project modules.

    Checks for near-duplicate filenames (e.g., ``lm_studio_provider.py``
    overlapping with ``lmstudio.py``).  Also verifies via importlib.

    Returns findings with "module_collision" pattern.
    """
    import importlib.util
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


DETECTORS = [
    detect_regex_in_loop,
    detect_string_concat_in_loop,
    detect_bare_except,
    detect_silent_except,
    detect_duplicate_imports,
    detect_missing_context_manager,
    detect_file_read_in_loop,
    detect_list_append_join,
]
