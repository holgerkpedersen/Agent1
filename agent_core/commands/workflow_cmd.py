"""Workflow command for agent interactive mode."""
import os
from pathlib import Path

from .base import Command, read_stdin
from agent_core import to_windows_path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class WorkflowCommand(Command):
    """Full pipeline: analyze, plan, entities, taskplan."""

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def help_text(self) -> str:
        return "workflow <target> [--from spec.md] [--stdin] [--brainstorm] [--features spec.md] — Full pipeline"

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

        if "--desc" in parts:
            di = parts.index("--desc")
            if di + 1 < len(parts):
                desc_text = parts[di + 1].strip('"')
                spec_file = str(ws_path / "project_spec.md")
                with open(spec_file, "w", encoding="utf-8") as f:
                    f.write(f"# Project Specification\n\n{desc_text}")
                greenfield = True
                print(f"\n[desc] {desc_text[:120]}...")
        elif "--stdin" in parts:
            text = read_stdin("Paste spec or description. Type --- on its own line when done:")
            parts = [p for p in parts if p != "--stdin"]
            if text.strip():
                spec_file = str(ws_path / "project_spec.md")
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
                features_file = str(ws_path / "project_features.md")
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

        ws = to_windows_path(target_workspace)
        ws_path = Path(ws)
        ws_path.mkdir(parents=True, exist_ok=True)
        print(f"Workspace: {ws_path}")

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
                print(f"\n[plan] Creating plan...")
                r = await agent.llm.chat([
                    {"role": "system", "content": "You are an expert software architect. Create a detailed coding plan with ALL files needed. Ensure all Python code passes mypy strict type checking. No unbound TypeVars, no type mismatches."},
                    {"role": "user", "content": f"Create coding plan:\n\n{spec_content}"}
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
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": "Extract shared classes/types. Output ONLY Python code — no intro text. Start with ```python. All types must be valid — no unbound TypeVars, no forward-ref errors. Must pass mypy strict. Avoid circular imports."},
                    {"role": "user", "content": f"Extract entities:\n\n## Spec:\n{spec_content}\n\n## Plan:\n{plan}"}
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
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                with open(entities_md, "r", encoding="utf-8") as f:
                    entities = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": "Create task plan. List files in dependency order. Format: 'Task N: `file.py` — what to do'. Include type-checking validation. No intro text. No code blocks.\n\nCRITICAL RULE: Every file path MUST have a directory prefix. Good: `agent1/logger.py`, `src/agent1/memory.py`. BAD: `logger.py`, `memory.py`, `utils.py`. Never emit bare filenames at workspace root — they will be REJECTED."},
                    {"role": "user", "content": f"Create task plan:\n\n## Spec:\n{spec_content}\n\n## Plan:\n{plan}\n\n## Entities:\n{entities}"}
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
                        pass
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
                    {"role": "system", "content": "Create task plan for adding these features. Format: mark file as [NEW] or [MODIFY], then '— what to do'. Include type-checking validation. No intro text.\n\nCRITICAL RULE: Every file path MUST have a directory prefix. Good: `agent1/logger.py`. BAD: `logger.py`. Never emit bare filenames at workspace root."},
                    {"role": "user", "content": f"## Analysis:\n{analysis}\n\n## Plan:\n{plan}\n\nCreate implementation tasks."}
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
                        pass
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
                    {"role": "system", "content": "Extract shared entities. Output ONLY Python code. Start with ```python. No intro text. Include only new/modified types — skip types that already exist unchanged. Avoid circular imports."},
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
                    {"role": "system", "content": "Create task plan. Format: 'Task N: `file.py` [TAG] — what to do'. List in dependency order. Be concise — one line per task. No intro text.\n\nCRITICAL RULE: Every file path MUST have a directory prefix. Good: `agent1/logger.py`. BAD: `logger.py`. Never emit bare filenames at workspace root."},
                    {"role": "user", "content": f"Create task plan:\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}"}
                ])
                if not step_ok(r):
                    print(f"[taskplan] FAILED: {r[:200]}")
                    return True
                with open(tasks_md, "w", encoding="utf-8") as f:
                    f.write(r)
                print(f"[taskplan] Written")

            print(f"\nNext: implement {tasks_md} {analysis_md} {plan_md} {entities_md} --workspace {target_workspace} --keep")

        return True