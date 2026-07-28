"""Fix command for agent interactive mode."""
import os
import re
from datetime import datetime
from pathlib import Path

from .base import Command
from agent_core import to_windows_path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


def extract_signatures(source: str) -> dict:
    """Extract function/class signatures from Python source."""
    sigs = {}
    for m in re.finditer(r'^class\s+(\w+)\s*(?:\((.*?)\))?\s*:', source, re.MULTILINE):
        cls_name = m.group(1)
        bases = m.group(2).strip() if m.group(2) else ""
        sigs[cls_name] = f"class {cls_name}({bases})" if bases else f"class {cls_name}"

    for m in re.finditer(r'^\s+def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*(.+?))?\s*:', source, re.MULTILINE):
        func_name = m.group(1)
        params = m.group(2).strip() if m.group(2) else ""
        returns = m.group(3).strip() if m.group(3) else ""
        sig = f"{func_name}({params})"
        if returns:
            sig += f" -> {returns}"
        sigs[func_name] = sig
    return sigs


class FixCommand(Command):
    """Fix code errors from traceback or description."""

    @property
    def name(self) -> str:
        return "fix"

    @property
    def help_text(self) -> str:
        return 'fix "<traceback>" | <file> --desc "issue" - Fix code errors'

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        parts = args

        if "--desc" in parts:
            di = parts.index("--desc")
            desc_text = parts[di + 1].strip('"') if di + 1 < len(parts) else ""

            target_file = None
            for p in parts:
                if not p.startswith("--") and p != desc_text:
                    target_file = p
                    break

            if not desc_text:
                self.error('Usage: fix <file> --desc "describe what\'s wrong"')
                return True

            if target_file:
                target_file = os.path.abspath(target_file)
                if not os.path.exists(target_file):
                    self.error(f"Target not found: {target_file}")
                    return True
                if os.path.isdir(target_file):
                    ws_dir = target_file
                    target_file = None
                else:
                    ws_dir = str(Path(target_file).parent)
            else:
                ws_dir = os.path.abspath(".")

            print(f"\nAnalyzing project in {ws_dir}...")
            print(f"Problem: {desc_text[:120]}...")

            candidate_files = set()

            if target_file and os.path.isfile(target_file):
                candidate_files.add(target_file)
            elif not target_file:
                for f in os.listdir(ws_dir):
                    fp = os.path.join(ws_dir, f)
                    if f.endswith(".py") and os.path.isfile(fp):
                        candidate_files.add(fp)
                        print(f"  Seed: {f}")

            def get_imported_files(filepath):
                result = set()
                try:
                    with open(filepath, "r") as f:
                        content = f.read()
                    for match in re.finditer(r'from\s+(\S+)\s+import\s+', content):
                        module = match.group(1)
                        path = module.replace('.', os.sep) + '.py'
                        for search_dir in [ws_dir, str(Path(filepath).parent)]:
                            full = os.path.join(search_dir, path)
                            if os.path.isfile(full):
                                result.add(os.path.normpath(full))
                                break
                except Exception:
                    pass
                return result

            for f in list(candidate_files):
                candidate_files |= get_imported_files(f)
            for f in list(candidate_files):
                candidate_files |= get_imported_files(f)

            keywords = {w.lower() for w in re.findall(r'\w+', desc_text) if len(w) > 3} - {'this', 'that', 'with', 'from', 'they', 'have', 'what', 'when', 'then', 'than', 'show', 'just', 'like'}

            resp_matched = set()
            tasks_md_path = os.path.join(ws_dir, "project_tasks.md")
            if os.path.exists(tasks_md_path):
                current_file = None
                with open(tasks_md_path, "r", encoding="utf-8") as tf:
                    for line in tf:
                        m = re.search(r'`([^`]+\.py)`', line)
                        if m:
                            current_file = m.group(1)
                        elif current_file and line.strip().startswith('-'):
                            task_text = line.strip('- ').strip().lower()
                            if any(kw in task_text for kw in keywords):
                                fp = os.path.normpath(os.path.join(ws_dir, current_file))
                                if os.path.isfile(fp):
                                    resp_matched.add(fp)
                                    print(f"  Responsibility match: {current_file} → '{task_text[:80]}'")
                if resp_matched:
                    candidate_files |= resp_matched
                    print(f"  + {len(resp_matched)} files from project_tasks.md responsibility matching")

            plan_md_path = os.path.join(ws_dir, "project_plan.md")
            if os.path.exists(plan_md_path):
                plan_matched = set()
                with open(plan_md_path, "r", encoding="utf-8") as pf:
                    plan_text = pf.read()
                for root, dirs, files in os.walk(ws_dir):
                    for f in files:
                        if f.endswith(".py") and f != "__init__.py":
                            fp = os.path.normpath(os.path.join(root, f))
                            rel = os.path.relpath(fp, ws_dir).replace("\\", "/")
                            idx = plan_text.find(rel)
                            if idx > 0:
                                snippet = plan_text[max(0, idx-100):idx+200].lower()
                                if any(kw in snippet for kw in keywords):
                                    plan_matched.add(fp)
                                    print(f"  Plan match: {rel}")
                if plan_matched:
                    candidate_files |= plan_matched

            print(f"  Tracing imports: {len(candidate_files)} relevant files")

            all_source = "## Project structure\n\n"
            py_files = []
            sig_map = {}

            for root, dirs, files in os.walk(ws_dir):
                if ".git" in root or "__pycache__" in root:
                    continue
                for f in files:
                    if not f.endswith(".py") or f == "__init__.py":
                        continue
                    fp = os.path.normpath(os.path.join(root, f))
                    rel = os.path.relpath(fp, ws_dir).replace("\\", "/")
                    py_files.append(fp)

                    if fp in candidate_files:
                        with open(fp, "r", encoding="utf-8") as sf:
                            all_source += f"\n\n# === {fp} ===\n{sf.read()}"
                    else:
                        try:
                            with open(fp, "r", encoding="utf-8") as sf:
                                sigs = extract_signatures(sf.read())
                            if sigs:
                                sig_map[rel] = ", ".join(f"{n}" for n in sorted(sigs.keys())[:8])
                        except Exception:
                            pass

            all_source += f"\n\n## Other project files (signatures only, {len(sig_map)} total)\n\n"
            for rel, sigs in sorted(sig_map.items()):
                all_source += f"  {rel}: {sigs}\n"

            print(f"  Collected {len(py_files)} Python files ({len(all_source)} bytes)")

            msgs = [
                {"role": "system", "content": "You are an expert Python debugger. Analyze the codebase below. Fix ALL files needed. Keep code concise. NEVER create duplicate functions or classes (_v1, _v2, _clean, _final variants). One implementation per concept.\n\nOutput each fixed file as:\n[FILE: absolute/path/to/file.py]\n```python\n# complete fixed code\n```"},
                {"role": "user", "content": f"The user reports this issue:\n\n{desc_text}\n\nFull project codebase:\n\n{all_source}\n\nAnalyze the issue, find the root cause, and fix ALL affected files. Output each fixed file with its full path."}
            ]

            print("Sending to LLM for deep analysis...")
            response = await agent.llm.chat(msgs)

            if response.startswith("[Error") or response.startswith("[LM Studio"):
                self.error(f"LLM error: {response[:200]}")
                return True

            file_pattern = r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```'
            fixes = re.findall(file_pattern, response, re.DOTALL)

            if not fixes:
                self.error("Could not parse fixes.")
                print(response[:1000])
                return True

            fixed_count = 0
            for fpath, new_code in fixes:
                fpath = fpath.strip()
                new_code = new_code.strip()
                if not os.path.exists(fpath):
                    print(f"  Skipping {fpath} (not found)")
                    continue
                if len(new_code) < 50 or "import" not in new_code:
                    print(f"  Skipping {fpath} (invalid content)")
                    continue

                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_code)
                fixed_count += 1
                print(f"  Fixed: {fpath} ({len(new_code)} bytes)")

                changelog_path = os.path.join(ws_dir, "CHANGES.md")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                entry = f"\n## {timestamp} — fix --desc\n\n**Change**: Modified `{os.path.basename(fpath)}`\n**Reason**: {desc_text[:200]}\n"
                with open(changelog_path, "a", encoding="utf-8") as cl:
                    cl.write(entry)

            print(f"\nFixed {fixed_count}/{len(fixes)} files.")
            return True

        traceback_text = ""
        if parts:
            traceback_text = " ".join(parts)

        if traceback_text.startswith('"') and traceback_text.endswith('"'):
            traceback_text = traceback_text[1:-1]

        if not traceback_text or "File \"" not in traceback_text:
            print("Paste the full traceback, then press Enter on an empty line:")
            lines = []
            if traceback_text:
                lines.append(traceback_text)
            while True:
                try:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                except EOFError:
                    break
            traceback_text = "\n".join(lines)

        if not traceback_text or "File \"" not in traceback_text:
            self.error("Could not find file references in traceback.")
            return True

        print(f"\nParsing traceback ({len(traceback_text)} chars)...")

        file_pattern = r'File "(.*?)", line (\d+)'
        matches = re.findall(file_pattern, traceback_text)

        if not matches:
            self.error("Could not find file/line references in traceback.")
            return True

        fpath, line_num = matches[-1]

        if fpath.startswith("<"):
            self.error(f"Cannot fix built-in module: {fpath}")
            return True

        if not os.path.exists(fpath):
            self.error(f"File not found: {fpath}")
            return True

        error_lines = traceback_text.strip().split('\n')
        error_msg = error_lines[-1] if error_lines else "Unknown error"

        print(f"\nError in {fpath}:{line_num}")
        print(f"  {error_msg}")

        is_import_error = "ImportError" in error_msg or "ModuleNotFoundError" in error_msg or "cannot import" in error_msg
        root_path = fpath
        root_line = line_num

        if is_import_error:
            all_files_in_trace = matches
            if len(all_files_in_trace) > 1:
                print(f"\n  Cascade detected! {len(all_files_in_trace)} files in trace:")
                for i, (fp, ln) in enumerate(all_files_in_trace):
                    marker = " → ROOT" if i == 0 else ""
                    print(f"    {i+1}. {fp}:{ln}{marker}")

                root_file = all_files_in_trace[0][0]
                root_ln = all_files_in_trace[0][1]

                if root_file != fpath and os.path.exists(root_file):
                    print(f"\n  Root cause is in {root_file}:{root_ln}, not in {fpath}")
                    print(f"  Fixing {root_file} instead...")
                    fpath = root_file
                    line_num = root_ln
                    error_msg = f"Cascading ImportError from {fpath}"

        with open(fpath, "r", encoding="utf-8") as f:
            current_code = f.read()

        lines_list = current_code.split('\n')
        line_idx = int(line_num) - 1
        start = max(0, line_idx - 3)
        end = min(len(lines_list), line_idx + 4)
        print(f"\n  Context (lines {start+1}-{end}):")
        for i in range(start, end):
            marker = ">>>" if i == line_idx else "   "
            print(f"  {marker} {i+1}: {lines_list[i][:120]}")

        project_dir = str(Path(fpath).parent)
        export_map = {}
        for root, dirs, files in os.walk(project_dir):
            if ".git" in root or "__pycache__" in root:
                continue
            for fp_name in files:
                if fp_name.endswith(".py"):
                    full = os.path.join(root, fp_name)
                    try:
                        with open(full, "r") as pf:
                            src = pf.read()
                        exports = set()
                        for m in re.finditer(r'^(?:class|def)\s+(\w+)', src, re.MULTILINE):
                            exports.add(m.group(1))
                        if exports:
                            rel = os.path.relpath(full, project_dir).replace("\\", "/")
                            export_map[rel] = exports
                    except Exception:
                        pass

        all_broken = []
        for match in re.finditer(r'from\s+(\S+)\s+import\s+(.+?)(?:\s*#|\s*$)', current_code):
            src_module = match.group(1)
            imported_names = [n.strip().split(' as ')[0].strip() for n in match.group(2).strip('()').split(',')]
            src_file = src_module.replace('.', '/') + '.py'
            if src_file not in export_map:
                continue
            src_exports = export_map.get(src_file, set())
            for name in imported_names:
                if name.isupper() or name.startswith('_'):
                    continue
                if name not in src_exports:
                    all_broken.append((src_module, name, src_file, sorted(src_exports)))

        if all_broken:
            print(f"\n  Found {len(all_broken)} broken imports in {fpath}:")
            for mod, name, src_file, avail in all_broken:
                print(f"    '{name}' from '{mod}' not found. Available: {', '.join(avail[:5])}")

        print(f"\nSending to LLM for fix...")
        import subprocess
        fix_msgs = [
            {"role": "system", "content": "Fix ALL broken imports in this file. Use ONLY imports that exist in the project. Keep stdlib/third-party imports unchanged. No duplicate functions. No _v1/_v2 variants.\n\nOutput as: [FILE: filename.py]\n```python\n# complete fixed code\n```"},
            {"role": "user", "content": f"Fix ALL errors in {fpath}:\n\nError from traceback at line {line_num}:\n{error_msg}\n\nAll broken imports in this file (must fix ALL):\n" + "\n".join([f"  import '{n}' from '{m}' — not found. Available in {s}: {', '.join(a[:8])}" for m, n, s, a in all_broken]) + f"\n\nFull traceback:\n{traceback_text}\n\nCurrent code:\n```python\n{current_code}\n```"}
        ]
        fixed = await agent.llm.chat(fix_msgs)

        if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
            self.error(f"LLM error: {fixed[:200]}")
            return True

        match = re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', fixed, re.DOTALL)
        if match:
            new_code = match.group(2).strip()
            if len(new_code) > len(current_code) * 0.1 and 'import' in new_code:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_code)
                print(f"\nFixed: {fpath} ({len(new_code)} bytes)")

                changelog_path = os.path.join(str(Path(fpath).parent), "CHANGES.md")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                entry = f"\n## {timestamp} — fix\n\n**Change**: Modified `{os.path.basename(fpath)}`\n**Reason**: {error_msg[:200]}\n"
                with open(changelog_path, "a", encoding="utf-8") as cl:
                    cl.write(entry)

                result = subprocess.run(
                    ["python", "-m", "py_compile", fpath],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("Compiled OK!")
                else:
                    print(f"Still has errors:\n{result.stderr[:300]}")
            else:
                print("LLM returned invalid fix (too short or not code)")
        else:
            print("Could not parse fix from LLM response")
            print(f"Raw: {fixed[:300]}")

        return True
