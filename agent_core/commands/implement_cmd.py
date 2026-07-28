"""Implement command for agent interactive mode."""
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .base import Command
from agent_core import to_windows_path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class ImplementCommand(Command):
    """Implement files from task plan using LLM."""

    @property
    def name(self) -> str:
        return "implement"

    @property
    def help_text(self) -> str:
        return "implement <taskplan.md> [--keep|--force|--fix] - Implement files from task plan"

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        parts = args

        if len(parts) < 1:
            self.error("Usage: implement <taskplan.md> [analysis.md] [plan.md] [entities.md] [--keep] [--refresh] [--force] [--fix] [--workspace <path>]")
            return True

        keep_mode = "--keep" in parts
        refresh_cache = "--refresh" in parts
        force_mode = "--force" in parts
        fix_mode = "--fix" in parts

        target_workspace = agent.workspace
        if "--workspace" in parts:
            ws_idx = parts.index("--workspace")
            if ws_idx + 1 < len(parts):
                target_workspace = parts[ws_idx + 1].strip('"')

        skip_tokens = ["--keep", "--refresh", "--force", "--fix", "--workspace", target_workspace]
        filtered_parts = [p for p in parts if p not in skip_tokens]

        taskplan_file = filtered_parts[0] if filtered_parts else ""
        analysis_file = filtered_parts[1] if len(filtered_parts) > 1 else "analysis.md"
        plan_file = filtered_parts[2] if len(filtered_parts) > 2 else "plan.md"
        entities_file = filtered_parts[3] if len(filtered_parts) > 3 else "entities.md"

        cache_file = os.path.join(os.path.dirname(os.path.realpath(taskplan_file)) if os.path.isabs(taskplan_file) else ".", ".implement_cache.json")
        if not os.path.isabs(taskplan_file):
            cache_file = os.path.join(".", ".implement_cache.json")

        analysis_content = ""
        plan_content = ""
        entities_content = ""
        taskplan_content = ""

        try:
            with open(taskplan_file, "r", encoding="utf-8") as f:
                taskplan_content = f.read()
        except FileNotFoundError:
            self.error(f"File not found: {taskplan_file}")
            return True

        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_content = f.read()
        except FileNotFoundError:
            print(f"Warning: {analysis_file} not found")

        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_content = f.read()
        except FileNotFoundError:
            print(f"Warning: {plan_file} not found")

        try:
            with open(entities_file, "r", encoding="utf-8") as f:
                entities_content = f.read()
        except FileNotFoundError:
            print(f"Warning: {entities_file} not found")

        all_files = None

        if keep_mode and not refresh_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data.get("taskplan") == taskplan_file:
                    all_files = cache_data.get("files", [])
                    print(f"Using cached file list ({len(all_files)} files): {', '.join(all_files)}")
            except Exception:
                pass

        if all_files is None:
            print("Analyzing task plan to identify all files...")

            list_messages = [
                {"role": "system", "content": "List ALL files that need to be implemented from the task plan. Reply with ONLY filenames, one per line. No explanations."},
                {"role": "user", "content": f"List every file that needs to be created or modified from this task plan:\n\n## Task Plan:\n{taskplan_content}\n\n## Analysis:\n{analysis_content if analysis_content else 'N/A'}\n\n## Plan:\n{plan_content if plan_content else 'N/A'}\n\n## Entities:\n{entities_content if entities_content else 'N/A'}"}
            ]

            file_list_response = await agent.llm.chat(list_messages)

            if not file_list_response or file_list_response.startswith("[Error") or file_list_response.startswith("[LM Studio"):
                self.error(f"LM Studio API not responding or returned an error: {file_list_response}")
                return True

            file_lines = [line.strip() for line in file_list_response.strip().split('\n') if line.strip() and not line.startswith('#')]
            all_files = [f for f in file_lines if f.endswith(('.py', '.json', '.yaml', '.yml', '.env', '.md', '.txt', '.cfg', '.ini', '.toml'))]

            if not all_files:
                all_files = re.findall(r'`([^`]+\.(?:py|json|yaml|yml|env|txt|cfg|ini|toml))`', file_list_response)

            cache_data = {"taskplan": taskplan_file, "files": all_files}
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f)
                print(f"Cached file list to {cache_file}")
            except Exception:
                pass

        print(f"Found {len(all_files)} files to implement: {', '.join(all_files)}")

        protected_files = set()
        if os.path.exists(".protected"):
            with open(".protected", "r", encoding="utf-8") as pf:
                for line in pf:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        protected_files.add(line)
        print(f"Protected files: {protected_files}")

        analyzed_file = ""
        match = re.search(r'# Analysis of (\S+)', analysis_content)
        if match:
            analyzed_file = match.group(1)
            print(f"Analyzed file from analysis.md: {analyzed_file}")

        def file_needs_generation(fname):
            from pathlib import Path
            raw_ws = target_workspace
            if raw_ws.startswith('/c/') or raw_ws.startswith('/C/'):
                raw_ws = 'C:' + raw_ws[2:]
            fpath = Path(raw_ws) / fname
            if not fpath.exists():
                return True, "not found"
            if fpath.stat().st_size == 0:
                return True, "empty"
            if fname.endswith(".py"):
                result = subprocess.run(
                    ["python", "-m", "py_compile", os.path.realpath(fpath)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    return True, f"compile failed: {result.stderr.strip()}"
                return False, "OK"
            return False, "OK"

        implemented = []
        if keep_mode and not force_mode:
            files_to_skip = []
            for fname in all_files:
                is_analyzed = analyzed_file and fname == analyzed_file
                if is_analyzed:
                    files_to_skip.append(f"{fname}: force-regenerate (was analyzed)")
                    continue
                needs_gen, reason = file_needs_generation(fname)
                if needs_gen:
                    files_to_skip.append(f"{fname}: {reason}")
                else:
                    files_to_skip.append(f"{fname}: already exists, compile OK")
                    if fname not in implemented:
                        implemented.append(fname)

            print(f"\nFiles to skip (already exist and compile): {len(files_to_skip)}")
            for f in files_to_skip:
                print(f"  - {f}")

            files_to_generate = [fname for fname in all_files if fname not in implemented]
            print(f"\nFiles to generate: {len(files_to_generate)}: {', '.join(files_to_generate)}")

            if not files_to_generate:
                print("All files already exist and compile. Nothing to do.")
                if not fix_mode:
                    return True
                print("\n[fix] Running validation on existing files...")
                implemented = [f for f in all_files if f.endswith(".py")]
                all_files = implemented

            all_files = files_to_generate

        errors = []
        batch_size = 1

        pre_snapshot = set()
        ws = target_workspace
        ws = to_windows_path(ws)
        for fp in Path(ws).rglob("*"):
            if fp.is_file() and ".git" not in str(fp) and "__pycache__" not in str(fp):
                pre_snapshot.add(str(fp.relative_to(Path(ws))).replace("\\", "/"))

        def extract_signatures(source: str) -> dict:
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

            for td_match in re.finditer(r'class\s+(\w+)\s*\(\s*(?:TypedDict|Protocol)\b', source):
                td_name = td_match.group(1)
                pos = td_match.end()
                paren_depth = 0
                while pos < len(source) and source[pos] != ':':
                    if source[pos] == '(': paren_depth += 1
                    elif source[pos] == ')': paren_depth -= 1
                    pos += 1
                if pos < len(source) and source[pos] == ':':
                    body_start = pos + 1
                    fields = []
                    lines = source[body_start:].split('\n')
                    for line in lines:
                        stripped = line.strip()
                        if not stripped or stripped.startswith('#') or stripped.startswith('"""'):
                            continue
                        if not line.startswith('    ') and stripped:
                            break
                        if ':' in stripped and not stripped.startswith('def ') and not stripped.startswith('class '):
                            field_name = stripped.split(':')[0].strip()
                            field_type = ':'.join(stripped.split(':')[1:]).strip()
                            fields.append(f"{field_name}: {field_type}")
                    if fields:
                        sigs[td_name] = f"{td_name} fields: {{{', '.join(fields[:8])}}}"

            return sigs

        export_map = {}
        for fname in list(all_files):
            fp = Path(ws) / fname
            if fp.exists():
                try:
                    existing = fp.read_text(encoding="utf-8")
                    sigs = extract_signatures(existing)
                    if sigs:
                        export_map[fname] = sigs
                except Exception:
                    pass

        if export_map:
            total_exports = sum(len(v) for v in export_map.values())
            print(f"Initial export map: {total_exports} signatures from {len(export_map)} existing files")

            broken_existing = []
            for fname in export_map:
                fp = Path(ws) / fname
                if not fp.exists() or not fname.endswith(".py"):
                    continue
                r = subprocess.run(
                    ["python", "-c", f"import py_compile; py_compile.compile(r'{os.path.realpath(fp)}', doraise=True)"],
                    capture_output=True, text=True, cwd=str(Path(ws))
                )
                if r.returncode != 0:
                    broken_existing.append((fname, r.stderr.strip()[-150:]))

            if broken_existing:
                print(f"\n  WARNING: {len(broken_existing)} existing files fail py_compile:")
                for fname, err in broken_existing:
                    print(f"    {fname}: {err[:100]}")
                print(f"  These exports may be incomplete. Consider running --fix first.")

        generated_content = {}

        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            print(f"\nGenerating batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}: {batch}")

            batch_files_md = "\n".join([f"- {f}" for f in batch])

            export_context = ""
            if export_map:
                export_lines = []
                for mod, sigs in sorted(export_map.items()):
                    if sigs:
                        sig_list = ", ".join(f"{name}: {sig}" for name, sig in sorted(sigs.items()))
                        export_lines.append(f"  {mod} → {sig_list}")
                if export_lines:
                    export_context = "\n\nAvailable project modules (use only these names with these exact signatures):\n" + "\n".join(export_lines)

            impl_messages = [
                {"role": "system", "content": "You are an expert Python developer. Implement the specified files concisely.\n\nRULES:\n0. NEVER use <tool_call>, <function_call>, or XML tags. Respond in plain text with [FILE:] blocks only.\n1. All code MUST pass mypy strict type checking and py_compile.\n2. Use ONLY imports that match the available exports listed below. Do not invent names.\n3. NEVER create duplicate functions or classes. One implementation per concept. No _v1, _v2, _clean, _final variants.\n4. Keep files under 200 lines. Refactor if longer.\n\nFormat each file as:\n[FILE: filename.py]\n```python\n# code\n```"},
                {"role": "user", "content": f"Files to implement:\n{batch_files_md}\n{export_context}\n\n## Task Plan:\n{taskplan_content}\n\n## Analysis:\n{analysis_content if analysis_content else 'N/A'}\n\nImplement these files using imports from the available modules listed above."}
            ]

            impl_response = None
            for attempt in range(3):
                try:
                    impl_response = await agent.llm.chat(impl_messages)
                    if impl_response and not impl_response.startswith("[Error:"):
                        break
                    print(f"  Attempt {attempt + 1} failed, retrying...")
                except Exception as e:
                    print(f"  Attempt {attempt + 1} error: {e}, retrying...")
                    if attempt == 2:
                        impl_response = None

            if not impl_response or impl_response.startswith("[Error:"):
                print(f"  Failed after 3 attempts, skipping batch")
                continue

            patterns = [
                r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```\s*$',
                r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```',
                r'\[FILE:\s*([^\]]+)\]\s*\n+(.*?)(?=\[FILE:|$)',
            ]

            matches = []
            for pattern in patterns:
                matches = list(re.findall(pattern, impl_response, re.DOTALL))
                if matches:
                    break

            if not matches and ("<tool_call" in impl_response or "</tool_call>" in impl_response):
                print(f"  Detected tool calls, retrying with plain text instruction...")
                impl_messages.append({"role": "user", "content": "Respond ONLY in [FILE: filename.py] format. No <tool_call> tags."})
                impl_response = await agent.llm.chat(impl_messages)
                for pattern in patterns:
                    matches = list(re.findall(pattern, impl_response, re.DOTALL))
                    if matches:
                        break

            if not matches:
                print(f"  Warning: Could not parse files from batch response")
                print(f"  Raw response: {impl_response[:500]}")
                continue

            for filename, content in matches:
                content = content.strip()
                generated_content[filename] = content
                print(f"  Generated: {filename} ({len(content)} bytes)")

                sigs = extract_signatures(content)
                if sigs:
                    export_map[filename] = sigs

        if generated_content:
            for fname in all_files:
                if fname not in export_map:
                    fp = Path(ws) / fname
                    if fp.exists():
                        try:
                            existing = fp.read_text(encoding="utf-8")
                            sigs = extract_signatures(existing)
                            if sigs:
                                export_map[fname] = sigs
                        except Exception:
                            pass

            print(f"\nExport map: {sum(len(v) for v in export_map.values())} exports across {len(export_map)} modules")

            broken_imports = {}
            stdlib_modules = set()

            for fname, content in generated_content.items():
                missing = []
                for match in re.finditer(r'from\s+(\S+)\s+import\s+(.+?)(?:\s*#|\s*$)', content):
                    src_module = match.group(1)
                    imported_names = [n.strip().split(' as ')[0].strip() for n in match.group(2).strip('()').split(',')]
                    src_file = src_module.replace('.', '/') + '.py'

                    if src_file not in export_map:
                        stdlib_modules.add(src_module)
                        continue

                    for name in imported_names:
                        if name.isupper() or name.startswith('_'):
                            continue
                        src_exports = export_map.get(src_file, {})
                        if name not in src_exports:
                            missing.append((src_module, name))

                if missing:
                    broken_imports[fname] = missing

            if stdlib_modules:
                print(f"  Skipped stdlib/third-party: {', '.join(sorted(stdlib_modules))}")

            if broken_imports:
                print(f"\n  Found {sum(len(v) for v in broken_imports.values())} broken imports in {len(broken_imports)} files:")
                for fname, missing in broken_imports.items():
                    for mod, name in missing:
                        print(f"    {fname}: '{name}' from '{mod}' not found")
            else:
                print(f"  All imports verified!")
        else:
            print("No content generated.")

        for filename, content in generated_content.items():
            raw_workspace = target_workspace
            if raw_workspace.startswith('/c/') or raw_workspace.startswith('/C/'):
                raw_workspace = 'C:' + raw_workspace[2:]
            workspace = Path(raw_workspace)
            filepath = workspace / filename

            filepath.parent.mkdir(parents=True, exist_ok=True)

            skip_reason = None
            is_analyzed_file = analyzed_file and filename == analyzed_file

            if not force_mode and not is_analyzed_file and filepath.exists() and filepath.stat().st_size > 0:
                if filename.endswith(".py"):
                    filepath_str = os.path.realpath(filepath)
                    result = subprocess.run(
                        ["python", "-m", "py_compile", filepath_str],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        skip_reason = "Already exists and compiles OK"

            if skip_reason:
                print(f"  Skipping {filename}: {skip_reason}")
                if filename not in implemented:
                    implemented.append(filename)
                continue

            if filename.endswith(".py"):
                func_names = re.findall(r'def\s+(\w+)', content)
                if len(func_names) > 20:
                    from collections import Counter
                    counts = Counter(func_names)
                    similar_prefixes = {}
                    for name in func_names:
                        prefix = re.sub(r'_\d+$|_v\d+$|_clean$|_final$', '', name)
                        similar_prefixes[prefix] = similar_prefixes.get(prefix, 0) + 1
                    max_dupes = max(similar_prefixes.values()) if similar_prefixes else 1
                    if max_dupes > 10:
                        print(f"  REJECTED: {filename} has {max_dupes} near-duplicate functions")
                        continue
                if len(content) > 50000:
                    print(f"  REJECTED: {filename} is {len(content)} bytes (max 50KB)")
                    continue

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            if filename.endswith(".py"):
                filepath_str = os.path.realpath(filepath)
                r = subprocess.run(
                    ["python", "-m", "py_compile", filepath_str],
                    capture_output=True, text=True
                )

                if r.returncode != 0:
                    stripped = content.rstrip()
                    lines = stripped.splitlines() if stripped else []
                    last_line = lines[-1].strip() if lines else ""

                    ends_mid = last_line.endswith(('(', '[', '{', ':', ',', '+', '-', '*', '/', '=', '\\'))
                    incomplete = (last_line and last_line[-1].isalnum()
                        and not last_line.endswith(('pass', 'return', 'break', 'continue', 'True', 'False', 'None')))
                    opens = sum(last_line.count(c) for c in '({[')
                    closes = sum(last_line.count(c) for c in ')}]')
                    unbalanced = opens > closes

                    is_truncated = ends_mid or (incomplete and unbalanced) or (incomplete and len(content) < 500)

                    if is_truncated and "unterminated" in r.stderr or "unexpected EOF" in r.stderr or "SyntaxError" in r.stderr:
                        print(f"  WARNING: {filename} appears truncated, re-requesting...")
                        retry_msgs = [
                            {"role": "system", "content": "Generate ONLY the complete code for this file. Output as:\n[FILE: filename.py]\n```python\n# complete code here\n```"},
                            {"role": "user", "content": f"Generate complete code for {filename}."}
                        ]
                        retry_content = await agent.llm.chat(retry_msgs)
                        if not retry_content.startswith("[Error"):
                            match = re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', retry_content, re.DOTALL)
                            if match:
                                new_content = match.group(2).strip()
                                if len(new_content) > len(content) * 0.5:
                                    with open(filepath, "w", encoding="utf-8") as f:
                                        f.write(new_content)
                                    content = new_content
                                    print(f"  Re-written: {filename} ({len(content)} bytes)")
                                    r = subprocess.run(
                                        ["python", "-m", "py_compile", filepath_str],
                                        capture_output=True, text=True
                                    )

                    if r.returncode != 0:
                        errors.append(f"{filename}: {r.stderr}")
                        print(f"  Compile error in {filename}")
                    else:
                        print(f"  Compiled OK: {filename}")
                else:
                    print(f"  Compiled OK: {filename}")
            implemented.append(filename)
            print(f"  Written: {filename}")

            manifest_file = Path(ws) / ".generated_manifest.json"
            manifest = {}
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
            manifest[filename] = {"date": str(datetime.now()), "size": len(content)}
            manifest_file.write_text(json.dumps(manifest, indent=2))

            if filename.endswith(".py"):
                filepath_str = os.path.realpath(filepath)
                result = subprocess.run(
                    ["python", "-m", "py_compile", filepath_str],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    errors.append(f"{filename}: {result.stderr}")
                    print(f"  Compile error in {filename}")
                else:
                    print(f"  Compiled OK: {filename}")

        print(f"\n{'='*50}")
        print(f"Implementation complete: {len(implemented)}/{len(all_files)} files")

        if implemented:
            print(f"\nImplemented files:")
            for f in implemented:
                print(f"  - {f}")

        if errors:
            print(f"\nCompilation errors ({len(errors)}):")
            for err in errors:
                print(f"  - {err}")
        else:
            print("\nAll Python files compiled successfully!")

        missing_files = set(all_files) - set(implemented)
        if missing_files:
            print(f"\nMissing files ({len(missing_files)}):")
            for f in missing_files:
                print(f"  - {f}")

        post_snapshot = set()
        for fp in Path(ws).rglob("*"):
            if fp.is_file() and ".git" not in str(fp) and "__pycache__" not in str(fp):
                post_snapshot.add(str(fp.relative_to(Path(ws))).replace("\\", "/"))

        new_files = post_snapshot - pre_snapshot
        removed_files = pre_snapshot - post_snapshot
        if new_files:
            print(f"\n  New files created: {len(new_files)}")
            for f in sorted(new_files):
                print(f"    + {f}")
        if removed_files:
            print(f"\n  Files no longer present: {len(removed_files)}")
            for f in sorted(removed_files):
                print(f"    - {f}")

        if fix_mode and implemented:
            print(f"\n{'='*50}")
            print(f"[fix] Deep validation: checking imports + class instantiation...")
            print(f"{'='*50}")

            for fix_attempt in range(3):
                errors_found = []

                for fname in implemented:
                    if not fname.endswith(".py"):
                        continue
                    ws = target_workspace
                    if ws.startswith("/c/") or ws.startswith("/C/"):
                        ws = "C:" + ws[2:]
                    fp = Path(ws) / fname
                    fpath_str = os.path.realpath(fp)

                    r = subprocess.run(["python", "-m", "py_compile", fpath_str], capture_output=True, text=True)
                    if r.returncode != 0:
                        errors_found.append((fname, fpath_str, f"COMPILE: {r.stderr.strip()}"))
                        continue

                    import tempfile
                    mod_name = fname[:-3].replace('\\', '.').replace('/', '.')
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
                        tf.write(f"import sys; sys.path.insert(0, r'{ws}')\n")
                        tf.write(f"import {mod_name}\n")
                        tf.write("print('OK')\n")
                        tfpath = tf.name
                    r = subprocess.run(["python", tfpath], capture_output=True, text=True, cwd=str(Path(ws)))
                    os.unlink(tfpath)
                    if r.returncode != 0:
                        errors_found.append((fname, fpath_str, f"IMPORT: {r.stderr.strip()}"))
                        continue

                    r = subprocess.run(
                        ["python", "-m", "mypy", fpath_str, "--ignore-missing-imports"],
                        capture_output=True, text=True, cwd=str(Path(ws))
                    )
                    if r.returncode != 0 and "No module named" not in r.stderr:
                        type_errors = [l.strip() for l in r.stdout.split('\n') if l.strip() and ':' in l and not l.startswith('Found')]
                        if type_errors:
                            errors_found.append((fname, fpath_str, f"TYPE: {'; '.join(type_errors[:5])}"))

                    with open(fpath_str, "r", encoding="utf-8") as f:
                        source = f.read()
                    class_names = re.findall(r'^class\s+(\w+)', source, re.MULTILINE)
                    for cn in class_names:
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
                            tf.write(f"import sys; sys.path.insert(0, r'{ws}')\n")
                            tf.write(f"import {mod_name}\n")
                            tf.write(f"c={mod_name}.{cn}\n")
                            tf.write(f"import inspect\n")
                            tf.write(f"try:\n    sig=inspect.signature(c)\n    print(f'OK: {cn}'+str(list(sig.parameters.keys())))\n")
                            tf.write(f"except (ValueError, TypeError):\n    print(f'OK: {cn} (builtin/Protocol/TypedDict)')\n")
                            tfpath = tf.name
                        r = subprocess.run(["python", tfpath], capture_output=True, text=True, cwd=str(Path(ws)))
                        os.unlink(tfpath)
                        if r.returncode != 0:
                            errors_found.append((fname, fpath_str, f"CLASS {cn}: {r.stderr.strip()}"))
                        else:
                            print(f"  {fname}: {r.stdout.strip()}")

                if not errors_found:
                    print("\n[fix] All files pass deep validation!")
                    print("\n[fix] Running smoke tests (instantiate + call methods)...")

                    smoke_errors = []

                    for fname in implemented:
                        if not fname.endswith(".py"):
                            continue
                        fp = Path(ws) / fname
                        fpath_str = os.path.realpath(fp)

                        with open(fpath_str, "r", encoding="utf-8") as sf:
                            source = sf.read()

                        for cn_match in re.finditer(r'class\s+(\w+)\s*(?:\(.*?\))?\s*:', source):
                            cn = cn_match.group(1)
                            if cn.startswith('_') or 'Protocol' in source[cn_match.start():cn_match.start()+200]:
                                continue

                            init_match = re.search(rf'class\s+{cn}.*?\n(\s+)def\s+__init__\s*\((.*?)\)\s*:', source[cn_match.start():], re.DOTALL)
                            params = init_match.group(2) if init_match else ""

                            required = []
                            for p in params.split(','):
                                p = p.strip()
                                if not p or p == 'self':
                                    continue
                                if '=' not in p:
                                    required.append(p.split(':')[0].strip())

                            mod_name = fname[:-3].replace('\\', '.').replace('/', '.')

                            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
                                tf.write(f"import sys; sys.path.insert(0, r'{ws}')\n")
                                tf.write(f"import {mod_name}\n")
                                tf.write(f"print(f'Testing {cn}...')\n")
                                tf.write("try:\n")

                                if not required:
                                    tf.write(f"    obj = {mod_name}.{cn}()\n")
                                    tf.write(f"    print(f'  OK: {cn}() works')\n")
                                elif len(required) <= 2:
                                    args = ", ".join(f'"mock_{r.split(":")[0].strip()}"' if ':' in r else f'"mock_{r.strip()}"' for r in required)
                                    tf.write(f"    obj = {mod_name}.{cn}({args})\n")
                                    tf.write(f"    print(f'  OK: {cn}() works')\n")
                                else:
                                    tf.write(f"    print(f'  SKIP: {cn} needs {len(required)} args')\n")

                                tf.write("except Exception as e:\n")
                                tf.write(f"    print(f'  FAIL: {cn}: {{type(e).__name__}}: {{e}}')\n")
                                tfpath = tf.name

                            r = subprocess.run(["python", tfpath], capture_output=True, text=True, cwd=str(Path(ws)))
                            os.unlink(tfpath)

                            if r.returncode != 0:
                                smoke_errors.append((fname, fpath_str, r.stderr.strip()[-300:]))
                            else:
                                output = r.stdout.strip()
                                if "FAIL:" in output:
                                    smoke_errors.append((fname, fpath_str, output))
                                    print(f"  {fname}: {output}")
                                elif "SKIP:" in output:
                                    pass
                                elif "OK:" in output:
                                    print(f"  {output}")

                    if not smoke_errors:
                        print("[fix] Smoke tests all passed!")
                        break

                    print(f"\n[fix] {len(smoke_errors)} smoke test failures:")
                    for fname, fpath, err in smoke_errors:
                        errors_found.append((fname, fpath, f"SMOKE: {err[:200]}"))
                        print(f"  - {fname}: {err[:100]}")

                err_root = errors_found[0][0]
                for fname, _, err in errors_found:
                    if "COMPILE:" in err:
                        err_root = fname
                        break
                print(f"\n[fix] Root error file: {err_root} (fixing this first)")

                print(f"\n[fix] Attempt {fix_attempt + 1}: {len(errors_found)} errors")
                for fname, fpath, err in errors_found:
                    print(f"  - {fname}:")
                    print(f"    {err}")

                if fix_attempt >= 2:
                    print("[fix] Max attempts reached.")
                    break

                for fname, fpath, err in errors_found:
                    print(f"\n[fix] Fixing {fname}...")
                    with open(fpath, "r", encoding="utf-8") as f:
                        current_code = f.read()

                    fix_msgs = [
                        {"role": "system", "content": "Fix the error. Output ONLY the corrected file. Start with [FILE: filename.py] immediately. No explanations. No duplicate functions. No _v1/_v2 variants."},
                        {"role": "user", "content": f"Error in {fname}:\n{err}\n\nCurrent code:\n```python\n{current_code}\n```\n\nOutput the fixed file."}
                    ]
                    fixed = await agent.llm.chat(fix_msgs)
                    if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
                        print(f"  LLM error: {fixed}")
                        continue

                    match = re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', fixed, re.DOTALL)
                    if match:
                        new_code = match.group(2).strip()
                        if not re.search(r'\b(import|def |class )\b', new_code):
                            print(f"  WARNING: Fix for {fname} is not valid Python code, skipping")
                            continue
                        if fname != err_root and len(new_code) < 100:
                            print(f"  Skipping cascade fix for {fname} (root issue is in {err_root})")
                            continue
                        if len(new_code) < len(current_code) * 0.1:
                            print(f"  WARNING: Fix is {len(new_code)} bytes vs original {len(current_code)} bytes, skipping")
                            continue
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(new_code)
                        print(f"  Fixed: {fname} ({len(new_code)} bytes)")

            print(f"\n[fix] Complete")

        return True
