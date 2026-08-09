"""Workflow command for agent interactive mode."""
import os
import re
from pathlib import Path

from .base import Command, read_stdin
from agent_core import to_windows_path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


_AGENT_SPEC_KEYWORDS = frozenset([
    "agent", "self-improvement", "self improvement", "vulnerability",
    "safe", "safety", "security", "hardening", "remediation",
    "llm", "prompt injection", "reward hacking", "capability escalation",
])

_PRIORITIZED_LLM_FILES = [
    "agent.py", "lmstudio.py", "meta_policy.py", "metrics_tracker.py",
    "prompt_cache.py", "orchestrator.py", "refinement_voter.py",
    "sanitizer.py", "retry.py", "provider.py", "tool_loop.py",
    "llm_client.py", "config.py", "constants.py",
]


def _detect_subpackages(workspace: str) -> list[str]:
    """Find existing subpackage directories (containing __init__.py) in the workspace.

    Returns a sorted list of relative paths like ``['agent_core', 'agent1', 'src/agent1']``.
    Used to generate workspace-agnostic path rules for LLM prompts.
    """
    if not os.path.isdir(workspace):
        return []
    pkgs = set()
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', '.pytest_cache')]
        if '__init__.py' in files:
            rel = os.path.relpath(root, workspace).replace('\\', '/')
            if rel == '.':
                pkgs.add(root.rsplit(os.sep, 1)[-1])  # root package name
            else:
                pkgs.add(rel)
    return sorted(pkgs)


def _collect_existing_names(workspace: str) -> tuple[dict[str, list[str]], list[str]]:
    """Scan workspace for class/function names + filenames grouped by directory.

    Returns ``(names_by_dir, existing_filenames)``.
    """
    taken: dict[str, list[str]] = {}
    filenames: list[str] = []
    if not os.path.isdir(workspace):
        return taken, filenames
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "backups")]
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, workspace).replace("\\", "/")
            filenames.append(rel)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    src = fh.read()
            except Exception:
                continue
            names = {m.group(1) for m in re.finditer(r'^(?:class|def)\s+(\w+)', src, re.MULTILINE)
                     if not m.group(1).startswith("__")}
            if names:
                rel_dir = os.path.relpath(root, workspace).replace("\\", "/").rstrip(".")
                taken.setdefault(rel_dir, []).extend(sorted(names)[:20])
    return taken, filenames


def _collision_warning(taken: dict[str, list[str]], filenames: list[str]) -> str:
    """Build a collision warning string for LLM prompts."""
    parts = []
    if taken:
        parts.append("CRITICAL: DO NOT create new files defining these class/function names.\n"
                      "They already exist — modify the existing file instead:")
        parts += [f"  {d or 'root'}: {', '.join(names[:30])}" for d, names in sorted(taken.items())[:12]]
    if filenames:
        # Flag similar filenames per directory
        by_dir: dict[str, list[str]] = {}
        for f in filenames:
            d = os.path.dirname(f).replace("\\", "/")
            by_dir.setdefault(d or "root", []).append(os.path.basename(f))
        parts.append("CRITICAL: Do NOT create filenames that overlap with existing ones in the same directory.")
        parts.append("Use distinct names. Example: use llm_config.py not config.py, retry_adapter.py not retry_policy.py.")
        parts += [f"  {d}: {', '.join(sorted(names)[:12])}" for d, names in sorted(by_dir.items())[:12]]
    return "\n\n".join(parts) if parts else ""


def _spec_mentions_agent(spec_content: str) -> bool:
    """Return True if the spec references agent/self-improvement/security concepts."""
    text = spec_content.lower()
    return any(kw in text for kw in _AGENT_SPEC_KEYWORDS)


def _scan_workspace_context(ws_path: Path, spec_content: str) -> tuple[bool, str]:
    """Scan target workspace for relevant Python files when spec mentions agent/self-improvement.

    Prioritises LLM-layer files first so they fit within the context budget.
    Returns (used_any, combined_code_text).
    """
    if not _spec_mentions_agent(spec_content):
        return False, ""
    ws = str(ws_path)
    all_py: list[str] = []
    for dp, dn, filenames in os.walk(ws):
        if ".git" in dp or "__pycache__" in dp or ".pytest_cache" in dp:
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                all_py.append(os.path.join(dp, fn))

    # Prioritise LLM-layer files first (they're most relevant to self-improvement)
    prioritised: list[str] = []
    rest: list[str] = []
    for fp in all_py:
        basename = os.path.basename(fp).lower()
        if any(p.lower() == basename for p in _PRIORITIZED_LLM_FILES):
            prioritised.append(fp)
        else:
            rest.append(fp)
    ordered = prioritised + rest

    combined = ""
    lines_used = 0
    max_lines = 6000
    for fp in ordered:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception:
            continue
        rel = os.path.relpath(fp, ws).replace("\\", "/")
        entry = f"\n\n# ---- {rel} ----\n{content}"
        if lines_used + len(entry.splitlines()) > max_lines:
            remaining = max_lines - lines_used
            if remaining <= 0:
                break
            lines += entry.splitlines()[:remaining]
            combined = "".join([combined, "\n".join(lines)])
            break
        combined += entry
        lines_used += len(content.splitlines())
    return bool(combined), combined


class WorkflowCommand(Command):
    """Full pipeline: analyze, plan, entities, taskplan."""

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def help_text(self) -> str:
        return (
            "workflow <target> [--from spec.md] [--stdin] [--brainstorm] "
            "[--features spec.md] [--workspace <path>] — Full pipeline\n"
            "  Greenfield analyze auto-scans the target workspace when the spec references\n"
            "  agent/self-improvement/security. Produces a structured 8-section analysis and\n"
            "  halts at an ambiguity gate (print clarifying questions) unless --force is given."
        )

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        parts = args

        if len(parts) < 1:
            self.error("Usage: workflow <target> [--from <spec.md>] [--stdin] [--force] [--brainstorm] [--workspace <path>]")
            self.error("  target: .  |  --desc/-from spec | --features spec (file or inline)")
            return True

        force = "--force" in parts
        brainstorm = "--brainstorm" in parts

        spec_file = None
        greenfield = False
        features_file = None

        # Use agent's workspace as default — --workspace overrides after parsing
        def _ws_dir() -> Path:
            tw = agent.workspace
            idx = parts.index("--workspace") if "--workspace" in parts else -1
            if idx >= 0 and idx + 1 < len(parts):
                tw = parts[idx + 1]
            if tw.startswith('/c/') or tw.startswith('/C/'):
                tw = 'C:' + tw[2:]
            return Path(tw)

        _ws_dir().mkdir(parents=True, exist_ok=True)

        if "--desc" in parts:
            di = parts.index("--desc")
            if di + 1 < len(parts):
                desc_text = parts[di + 1].strip('"')
                spec_file = str(_ws_dir() / "project_spec.md")
                with open(spec_file, "w", encoding="utf-8") as f:
                    f.write(f"# Project Specification\n\n{desc_text}")
                greenfield = True
                print(f"\n[desc] {desc_text[:120]}...")
        elif "--stdin" in parts:
            text = read_stdin("Paste spec or description. Type --- on its own line when done:")
            parts = [p for p in parts if p != "--stdin"]
            if text.strip():
                spec_file = str(_ws_dir() / "project_spec.md")
                with open(spec_file, "w", encoding="utf-8") as f:
                    f.write(f"# Project Specification\n\n{text}")
                greenfield = True
                print(f"\n[stdin] {len(text)} chars")
        elif "--from" in parts:
            fi = parts.index("--from")
            end = fi + 1
            while end < len(parts) and not parts[end].startswith("--"):
                end += 1
            spec_file = " ".join(parts[fi + 1:end])
            greenfield = True

        if "--features" in parts:
            fi = parts.index("--features")
            feat_val = None
            if fi + 1 < len(parts):
                feat_val = parts[fi + 1]
            if feat_val is not None and os.path.isfile(feat_val):
                features_file = feat_val
            elif feat_val is not None:
                features_file = str(_ws_dir() / "project_features.md")
                with open(features_file, "w", encoding="utf-8") as f:
                    f.write(f"# Feature Requirements\n\n{feat_val}")
                print(f"\n[features] {feat_val}")

        filtered = [p for p in parts if not p.startswith("--") and p not in [spec_file, features_file]]
        target = filtered[0] if filtered else "."

        target_workspace = agent.workspace
        if "--workspace" in parts:
            ws_idx = parts.index("--workspace")
            if ws_idx + 1 < len(parts):
                target_workspace = parts[ws_idx + 1]

        if spec_file and not os.path.isabs(spec_file):
            spec_file = os.path.join(target_workspace, spec_file)
        spec_file = to_windows_path(spec_file) if spec_file else None

        if features_file and not os.path.isabs(features_file):
            features_file = os.path.join(target_workspace, features_file)
        features_file = to_windows_path(features_file) if features_file else None

        ws_path = _ws_dir()
        ws_path.mkdir(parents=True, exist_ok=True)
        print(f"Workspace: {ws_path}")

        # Pre-scan existing names to warn about collisions
        taken_names, existing_filenames = _collect_existing_names(str(ws_path))
        collision_warning = _collision_warning(taken_names, existing_filenames) if taken_names or existing_filenames else ""

        # Dynamic path rules based on workspace structure
        pkgs = _detect_subpackages(str(ws_path))
        if not pkgs:
            pkgs = ["agent_core", "agent1", "src/agent1"]  # fallback
        pkg_list = "`, `".join(pkgs[:5])
        path_rule = f"\n\nPATH RULES: New files MUST use a sub-package prefix (`{pkg_list}/`). BAD: bare filenames or `src/` without subdirectory."
        prompt_context = collision_warning + path_rule + "\n\nSIZE RULES: New files max 150 lines (SRP). Split large concepts. Modifying: minimal changes only."

        analysis_md = str(ws_path / "project_analysis.md")
        plan_md = str(ws_path / "project_plan.md")
        entities_md = str(ws_path / "project_entities.md")
        tasks_md = str(ws_path / "project_tasks.md")

        def step_ok(result):
            return not (result.startswith("[Error") or result.startswith("[LM Studio"))

        if spec_file:
            with open(spec_file, "r", encoding="utf-8") as f:
                spec_content = f.read()
            print(f"\n[spec] Loaded from {spec_file}")

            if not force and os.path.exists(plan_md):
                print(f"\n[Skipping plan] exists")
            else:
                if not force and os.path.exists(analysis_md):
                    print(f"\n[Skipping analyze] exists")
                else:
                    print(f"\n[analyze] Analyzing spec...")

                    # Scan target workspace for code context when spec references agent/self-improvement
                    ws_ctx_found, ws_code_context = _scan_workspace_context(ws_path, spec_content)
                    if ws_ctx_found:
                        print(f"  [analyze] Included workspace code context (self-improvement detected)")
                    trace_header = f"# Analysis for workspace {ws_path}\n"
                    if spec_file:
                        trace_header += f"| Spec file: {os.path.basename(spec_file)}\n\n"
                    else:
                        trace_header += "\n"

                    analyze_system = (
                        "You are an expert software analyst and security reviewer. Produce a structured analysis.\n"
                        "Use exactly these section headers, in this order:\n\n"
                        "## 1. SCOPE\n"
                        "  What is being built, key deliverables, boundaries of the system.\n"
                        "## 2. ASSUMPTIONS\n"
                        "  What the spec assumes but does not state explicitly.\n"
                        "## 3. RISKS\n"
                        "  Ambiguity, missing details, technical challenges, and recursive risks\n"
                        "  (e.g. hardening one path may create another).\n"
                        "## 4. DEPENDENCIES\n"
                        "  External systems, libraries, tooling, compliance requirements.\n"
                        "## 5. THREAT MODEL & ATTACK SURFACE\n"
                        "  Identify likely attack vectors: prompt injection, reward hacking,\n"
                        "  capability escalation, sandbox escapes, LLM-output trust boundaries.\n"
                        "  Reference specific code files from the provided context where relevant.\n"
                        "## 6. MISSING INFORMATION (BLOCKERS)\n"
                        "  Enumerate every piece of required information that is absent from\n"
                        "  the spec and must be supplied before implementation can proceed.\n"
                        "  If none, write: 'No blockers — spec is sufficient to proceed.'\n"
                        "## 7. CLARIFYING QUESTIONS\n"
                        "  List concrete questions for the user. Each question must be answerable\n"
                        "  via --desc or an answers file.\n"
                        "## 8. SUCCESS METRICS & OVERSIGHT\n"
                        "  Define measurable safety criteria (e.g. % prompt-injection blocked,\n"
                        "  rollback capability, human-approval gates, audit trail).\n"
                        "  Specify oversight and recovery: who approves changes to safety mechanisms,\n"
                        "  how to revert self-modification.\n\n"
                        "Rules:\n"
                        "- Be concise. Max 3 bullet points per section unless detail is needed.\n"
                        "- Cite file paths from the workspace context when identifying risks.\n"
                        "- End your output with exactly one of these lines (no extra text after):\n"
                        "  **BLOCKED:** yes   (when section 6 has blockers)\n"
                        "  **BLOCKED:** no\n"
                        "- Never use <tool_call> or XML tags."
                    )

                    user_prompt = f"Analyze this specification for the agent in workspace {ws_path}:\n\n{spec_content}"
                    if ws_ctx_found:
                        user_prompt += (
                            "\n\n## Workspace code context (target agent being hardened):\n"
                            f"{ws_code_context}\n"
                        )
                    user_prompt += f"\n\nSpec file reference: {os.path.basename(spec_file) if spec_file else 'inline'}"

                    r = await agent.llm.chat([
                        {"role": "system", "content": analyze_system},
                        {"role": "user", "content": user_prompt},
                    ])
                    if not step_ok(r):
                        print(f"[analyze] FAILED: {r[:200]}")
                        return True

                    # Write analysis with traceability header
                    with open(analysis_md, "w", encoding="utf-8") as f:
                        f.write(trace_header)
                        f.write(r)
                    print(f"[analyze] Written to {analysis_md}")

                    # Ambiguity gate — halt before plan if blockers exist and --force not given
                    blocked_match = re.search(r"\*\*BLOCKED:\*\*\s*(yes|no)", r, re.IGNORECASE)
                    if blocked_match and blocked_match.group(1).lower() == "yes" and not force:
                        print("\n[analyze] BLOCKED — spec requires clarification before proceeding.")
                        # Extract and print the blocker / questions sections
                        for section_marker in ["## 6. MISSING INFORMATION", "## 7. CLARIFYING QUESTIONS"]:
                            idx = r.find(section_marker)
                            if idx == -1:
                                continue
                            next_section = len(r)
                            for later in ["## 8.", "## Refinement"]:
                                later_idx = r.find(later, idx + 1)
                                if later_idx != -1 and later_idx < next_section:
                                    next_section = later_idx
                            print(f"\n{r[idx:next_section].strip()}")
                        print(f"\nNext steps:")
                        print(f"  1. Answer the questions above, then run again with --desc \"<answers>\"")
                        print(f"     or create a file and pass it via --from <file>")
                        print(f"  2. Or use --force to skip this gate and proceed anyway.")
                        return True

                    # Self-critique refinement pass — only if analysis was not blocked
                    print("  [analyze] Running self-critique refinement...")
                    if ws_ctx_found:
                        critique_user = (
                            f"## Spec:\n{spec_content}\n\n"
                            f"## Workspace context:\n{ws_code_context}\n\n"
                            f"## Analysis to critique:\n{r}"
                        )
                    else:
                        critique_user = (
                            f"## Spec:\n{spec_content}\n\n"
                            f"## Analysis to critique:\n{r}"
                        )
                    critique_r = await agent.llm.chat([
                        {"role": "system", "content": (
                            "You are a senior security engineer. Critique the following analysis.\n"
                            "List concrete gaps, missed attack vectors, missing metrics, or overlooked\n"
                            "code references. Be specific — cite file paths and line-level concerns.\n"
                            "If the analysis is already thorough, say so concisely."
                        )},
                        {"role": "user", "content": critique_user},
                    ])
                    if step_ok(critique_r):
                        with open(analysis_md, "a", encoding="utf-8") as f:
                            f.write("\n\n---\n\n## Refinement (self-critique)\n")
                            f.write(critique_r)
                        print(f"  [analyze] Appended refinement pass")
                    else:
                        print(f"  [analyze] Critique failed (non-blocking): {critique_r[:100]}")

                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()

                print(f"\n[plan] Creating plan...")
                r = await agent.llm.chat([
                    {"role": "system", "content": "You are an expert software architect. Create a detailed coding plan with ALL files needed. Follow SOLID principles — each file has a single responsibility. New files should be small and focused (max 150 lines). If a concept needs more code, split across multiple files. Ensure all Python code passes mypy strict type checking. No unbound TypeVars, no type mismatches. Never use <tool_call> or XML tags."},
                    {"role": "user", "content": f"Create coding plan:\n\n## Spec:\n{spec_content}\n\n## Analysis:\n{analysis}"}
                ])
                if not step_ok(r):
                    print(f"[plan] FAILED: {r[:200]}")
                    return True
                with open(plan_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[plan] Written")

            if not force and os.path.exists(entities_md):
                print(f"\n[Skipping entities] exists")
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read() if os.path.exists(analysis_md) else ""
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": "Extract shared classes/types. Output ONLY Python code — no intro text. Start with ```python. All types must be valid — no unbound TypeVars, no forward-ref errors. Must pass mypy strict. Avoid circular imports. Never use <tool_call> or XML tags."},
                    {"role": "user", "content": f"Extract entities:\n\n## Spec:\n{spec_content}\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}"}
                ])
                if not step_ok(r):
                    print(f"[entities] FAILED: {r[:200]}")
                    return True
                with open(entities_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[entities] Written")

            if not force and os.path.exists(tasks_md):
                print(f"\n[Skipping taskplan] exists")
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read() if os.path.exists(analysis_md) else ""
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                with open(entities_md, "r", encoding="utf-8") as f:
                    entities = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": "Create task plan. List files in dependency order. Format: 'Task N: `file.py` — what to do'. Include type-checking validation. No intro text. No code blocks. Never use <tool_call>, XML tags, or function-calling syntax.\n\nFollow the PATH RULES and SIZE RULES in the prompt below."},
                    {"role": "user", "content": f"Create task plan:\n\n## Spec:\n{spec_content}\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}\n\n## Entities:\n{entities}{prompt_context}"}
                ])
                if not step_ok(r):
                    print(f"[taskplan] FAILED: {r[:200]}")
                    return True
                with open(tasks_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[taskplan] Written")

            print(f"\nNext: implement {tasks_md} {plan_md} {entities_md} --workspace {target_workspace} --force")

        elif features_file:
            with open(features_file, "r", encoding="utf-8") as f:
                features = f.read()
            print(f"\n[features] Loaded from {features_file}")

            if not force and os.path.exists(analysis_md):
                print(f"\n[Skipping analyze] exists")
            else:
                print(f"\n[analyze] Scanning existing py files...")
                py_files = []
                for dp, dn, filenames in os.walk(ws_path):
                    if ".git" in dp or "__pycache__" in dp:
                        continue
                    for fn in filenames:
                        if fn.endswith(".py"):
                            py_files.append(os.path.join(dp, fn))
                if not py_files:
                    self.error("[analyze] No py files. Use --from for greenfield.")
                    return True
                combined = ""
                for pf in py_files:
                    try:
                        with open(pf, "r", encoding="utf-8") as f:
                            combined += f"\n\n# ---- {pf} ----\n{f.read()}"
                    except Exception:
                        print(f"Warning: silenced exception in workflow_cmd.py:490")
                r = await agent.llm.chat([
                    {"role": "system", "content": "You are an expert code reviewer. Analyze the existing code AND these new features. Find bugs, gaps, and what needs to change."},
                    {"role": "user", "content": f"## Existing Code:\n{combined}\n\n## New Features:\n{features}\n\nAnalyze both existing issues and what must change for the new features."}
                ])
                if not step_ok(r):
                    print(f"[analyze] FAILED: {r[:200]}")
                    return True
                with open(analysis_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[analyze] Written")

            if not force and os.path.exists(plan_md):
                print(f"\n[Skipping plan] exists")
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()

                existing_plan = ""
                if os.path.exists(plan_md):
                    with open(plan_md, "r", encoding="utf-8") as f:
                        existing_plan = f.read()[:3000]

                r = await agent.llm.chat([
                    {"role": "system", "content": "You are an expert software architect. Create a plan that extends the EXISTING codebase with these new features.\n\nIMPORTANT:\n- Start with '## Feature Addition: <summary>'\n- List NEW files to create\n- List EXISTING files to modify and what minimal changes are needed\n- Explain WHY each change is needed\n- Preserve existing architecture"},
                    {"role": "user", "content": f"## Existing code analysis:\n{analysis}\n\n## Existing plan:\n{existing_plan if existing_plan else 'No existing plan'}\n\n## New features to add:\n{features}\n\nCreate a plan that integrates these features into the existing codebase."}
                ])
                if not step_ok(r):
                    print(f"[plan] FAILED: {r[:200]}")
                    return True
                with open(plan_md, "a", encoding="utf-8") as f:
                    f.write(f"\n\n---\n\n{r}")
                print(f"[plan] Appended to {plan_md}")

            if not force and os.path.exists(entities_md):
                print(f"\n[Skipping entities] exists")
            else:
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                entities_existing = ""
                if os.path.exists(entities_md):
                    with open(entities_md, "r", encoding="utf-8") as f:
                        entities_existing = f.read()[:3000]
                r = await agent.llm.chat([
                    {"role": "system", "content": "Extract ONLY NEW shared entities. Output ONLY Python code — no intro text. Start with ```python. All types must pass mypy strict."},
                    {"role": "user", "content": f"## Plan:\n{plan}\n\n## Existing entities:\n{entities_existing}\n\nExtract only new entities needed."}
                ])
                if not step_ok(r):
                    print(f"[entities] FAILED: {r[:200]}")
                    return True
                with open(entities_md, "a", encoding="utf-8") as f:
                    f.write(f"\n\n---\n\n{r}")
                print(f"[entities] Appended")

            if not force and os.path.exists(tasks_md):
                print(f"\n[Skipping taskplan] exists")
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": "Create task plan for adding these features. Format: mark file as [NEW] or [MODIFY], then '— what to do'. Include type-checking validation. No intro text. Never use <tool_call> or XML tags.\n\nFollow the rules in the prompt below."},
                    {"role": "user", "content": f"## Analysis:\n{analysis}\n\n## Plan:\n{plan}\n\nCreate implementation tasks.{prompt_context}"}
                ])
                if not step_ok(r):
                    print(f"[taskplan] FAILED: {r[:200]}")
                    return True
                with open(tasks_md, "a", encoding="utf-8") as f:
                    f.write(f"\n\n---\n\n{r}")
                print(f"[taskplan] Appended")

            print(f"\nNext: implement {tasks_md} {analysis_md} {plan_md} {entities_md} --workspace {target_workspace} --keep")

        else:
            if not force and os.path.exists(analysis_md):
                print(f"\n[Skipping analyze] exists")
            else:
                print(f"\n[analyze] Scanning py files...")
                py_files = []
                for dp, dn, filenames in os.walk(ws_path):
                    if ".git" in dp or "__pycache__" in dp:
                        continue
                    for fn in filenames:
                        if fn.endswith(".py"):
                            py_files.append(os.path.join(dp, fn))
                if not py_files:
                    self.error("[analyze] No py files. Use --from spec.md for greenfield.")
                    return True
                combined = ""
                for pf in py_files:
                    try:
                        with open(pf, "r", encoding="utf-8") as f:
                            combined += f"\n\n# ---- {pf} ----\n{f.read()}"
                    except Exception:
                        print(f"Warning: silenced exception in workflow_cmd.py:585")
                analyze_system = (
                    "You are an expert software architect. Evaluate this codebase across 5 dimensions. "
                    "For each dimension, limit to 3 bullet points max 50 words each. Be concise."
                    "\n\n1. CODE QUALITY — bugs, edge cases, type safety, error handling gaps"
                    "\n2. COMPLETENESS — missing tests, missing docs, missing features"
                    "\n3. ARCHITECTURE — DRY violations, coupling, SRP breaks"
                    "\n4. INNOVATION — new capabilities that would make this system more useful"
                    "\n5. PRODUCTION — logging, monitoring, config, security gaps"
                    + ("\n6. BRAINSTORMING — bold creative features (3 bullets max)"
                       if brainstorm else "")
                )
                r = await agent.llm.chat([
                    {"role": "system", "content": analyze_system},
                    {"role": "user", "content": combined}
                ])
                if not step_ok(r):
                    print(f"[analyze] FAILED: {r[:200]}")
                    return True
                with open(analysis_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[analyze] Written")

            if not force and os.path.exists(plan_md):
                print(f"\n[Skipping plan] exists")
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": (
                        "Create a prioritized implementation plan. "
                        "Categorize: [FIX], [FEATURE], [ARCH], [OPS]. "
                        "Priorities: MUST, SHOULD, COULD. Max 3 items per category. "
                        "Be concise — one line per item."
                    )},
                    {"role": "user", "content": f"Create plan:\n\n{analysis}"}
                ])
                if not step_ok(r):
                    print(f"[plan] FAILED: {r[:200]}")
                    return True
                with open(plan_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[plan] Written")

            if not force and os.path.exists(entities_md):
                print(f"\n[Skipping entities] exists")
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": "Extract shared entities. Output ONLY Python code. Start with ```python. No intro text. Include only new/modified types — skip types that already exist unchanged. Avoid circular imports. Never use <tool_call> or XML tags."},
                    {"role": "user", "content": f"Extract entities:\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}"}
                ])
                if not step_ok(r):
                    print(f"[entities] FAILED: {r[:200]}")
                    return True
                with open(entities_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[entities] Written")

            if not force and os.path.exists(tasks_md):
                print(f"\n[Skipping taskplan] exists")
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": "Create task plan. Format: 'Task N: `file.py` [TAG] — what to do'. List in dependency order. Be concise — one line per task. No intro text. Never use <tool_call> or XML tags.\n\nFollow the rules in the prompt below."},
                    {"role": "user", "content": f"Create task plan:\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}{prompt_context}"}
                ])
                if not step_ok(r):
                    print(f"[taskplan] FAILED: {r[:200]}")
                    return True
                with open(tasks_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[taskplan] Written")

            print(f"\nNext: implement {tasks_md} {analysis_md} {plan_md} {entities_md} --workspace {target_workspace} --keep")

        return True