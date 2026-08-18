"""Workflow command for agent interactive mode.

Invocation (inside the REPL)::

    workflow . --desc <feature description> [--auto]
    workflow . --continue

Stages
------
The workflow runs the greenfield pipeline and writes each phase into a
timestamped ``.docs/<ts>/`` folder (git-ignored; the newest folder is the
fallback lookup target for ``implement``):

1. **analyze** — scans the codebase and produces ``project_analysis.md``
   with verification of code claims (``analysis_verifier``).
2. **plan** — derives ``project_plan.md`` (FIX/FEATURE/ARCH/OPS priorities).
3. **entities** — emits ``project_entities.py`` with the dataclass/enum
   contract shared by the plan documents.
4. **taskplan** — writes ``project_tasks.md``, one backticked file path per
   task, the input for ``implement``.

Configuration
-------------
- ``--desc <text>`` — the feature/change description that seeds the pipeline.
- ``--auto`` — autonomous mode: skips confirmations, runs the tailored
  ``implement`` command inline at the end.
- ``--continue`` — resumes from the newest ``.docs/<ts>/`` run folder.
- Past decisions (``.decisions.json``) are injected as design constraints in
  the analysis prompt; reasoning output is stripped from LLM responses.

Return
------
``execute`` returns ``True`` after printing the next-command hint
(``implement ... --workspace <ws>``) unless ``--auto`` runs it inline.
"""
import os
import re
import shutil
import sys
from pathlib import Path

from .analysis_verifier import verify_analysis_claims
from .base import (
    Command,
    FlowStopped,
    auto_choice,
    chat_stoppable,
    clear_stop,
    read_input,
    read_stdin,
    set_autonomous,
    stop_requested,
)
from .doc_paths import latest_run_dir, new_run_dir
from .reasoning_strip import strip_reasoning
from agent_core import to_windows_path
from agent_core.decisions import add_decision, annotate_candidates, extract_from_analysis

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from agent import Agent

_AGENT_SPEC_KEYWORDS = frozenset([
    "agent", "self-improvement", "self improvement", "vulnerability",
    "safe", "safety", "security", "hardening", "remediation",
    "llm", "prompt injection", "reward hacking", "capability escalation",
    # Internal identifiers: specs about the agent's own loop internals must also
    # trigger the workspace context scan, otherwise the model has no path
    # knowledge and (per the PATH VERIFICATION rule) BLOCKs asking where things
    # are instead of answering (observed: "chat_nlp ... stuck detection" spec
    # with no context → BLOCKED).
    "chat_nlp", "auto-continue", "auto continue", "stuck detection",
    "_looks_incomplete", "heuristics", "untested", "test coverage",
    "semantic index", "tool loop", "orchestrator", "decisions", "trace",
])

_PRIORITIZED_LLM_FILES = [
    "agent.py", "lmstudio.py", "meta_policy.py", "metrics_tracker.py",
    "prompt_cache.py", "orchestrator.py", "refinement_voter.py",
    "sanitizer.py", "retry.py", "provider.py", "tool_loop.py",
    "llm_client.py", "config.py", "constants.py",
]

_HIGH_RISK_STDLIB_NAMES = frozenset({
    "logging", "json", "types", "os", "sys", "io", "re", "http",
    "email", "xml", "html", "csv", "sqlite3", "pickle", "socket",
    "unittest", "argparse", "configparser", "pathlib", "subprocess",
    "asyncio", "collections", "functools", "itertools", "typing",
    "importlib", "inspect", "threading", "multiprocessing", "signal",
})

#: Prompt rule injected into every analysis-adjacent system prompt (decision #035).
#: Models like laguna-s-2.1 fabricate file paths and symbol locations instead of
#: checking what exists; this rule forces them to ground citations in the workspace
#: context listing rather than inventing paths, which previously led to a stuck
#: state where the model kept retrying non-existent paths.
_PATH_VERIFICATION_RULE = (
    "PATH VERIFICATION (CRITICAL): Every file path you cite MUST come from the "
    "workspace context listing provided above. If a path is NOT in that listing, do "
    "NOT cite it — write `UNVERIFIED` instead of guessing a location or line number. "
    "Never invent paths: if unsure whether a symbol exists in a file, first list the "
    "directory and read the real file before stating where it is defined."
)


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


def _specs_match(prev_spec_path: "str | Path", current_spec_text: str) -> bool:
    """True when the previous run's spec equals the current one.

    Carry-over of plan/entities/taskplan is only safe when the TASK is the
    same — a different spec must regenerate everything from scratch.
    """
    try:
        prev = Path(prev_spec_path).read_text(encoding="utf-8")
    except OSError:
        return False
    return prev.strip() == current_spec_text.strip()


def _planned_files_from_taskplan(taskplan_content: str) -> list[str]:
    """Backticked file paths from the taskplan (same extraction as implement)."""
    files = re.findall(r'`([^`]+\.py)`', taskplan_content)
    if not files:
        files = re.findall(r'`([^`\s]+\.[a-z]{2,4})`', taskplan_content)
    return list(dict.fromkeys(files))


def _tailored_implement_parts(
    tasks_md: str,
    plan_md: str,
    entities_md: str,
    analysis_md: str | None,
    ws: str,
    taskplan_content: str,
) -> tuple[list[str], str]:
    """Build the 'implement' parts + a codebase-aware hint for this run.

    Tailoring (deterministic, precision-first):
    - positionals in implement's expected order (tasks, analysis, plan, entities)
    - duplicate pre-check via the existing gate → warning, and NO ``--force``
      so the Layer-1 gate can offer the MODIFY filter
    - ``--modify`` when any planned file already exists (brownfield): without
      it implement skips existing compile-OK files, so [MODIFY] tasks silently
      do nothing
    - ``--keep`` only when a matching implement cache exists (true resume);
      a stale cache is called out so the user can add ``--refresh``
    """
    from .implement_cmd import _check_planned_duplicates

    parts = ["implement", tasks_md, plan_md, entities_md]
    if analysis_md and os.path.exists(analysis_md):
        parts.insert(2, analysis_md)
    parts += ["--workspace", ws]

    hints: list[str] = []
    planned = _planned_files_from_taskplan(taskplan_content)
    if planned:
        dup_reasons = _check_planned_duplicates(planned, ws, taskplan_content)
        if dup_reasons:
            example = dup_reasons[0].split(" — ", 1)[-1]
            example = example.replace("duplicates existing module(s): ", "")
            hints.append(
                f"{len(dup_reasons)} planned file(s) duplicate existing modules "
                f"(e.g. {example}) — implement will offer to drop them; no --force is used."
            )

        # Brownfield: planned files that already exist would be skipped by
        # default ("Already exists and compiles OK") — --modify turns them
        # into reviewed diff-applications instead.
        existing_planned = [
            f for f in planned if (Path(ws) / f).is_file()
        ]
        if existing_planned:
            parts += ["--modify"]
            hints.append(
                f"{len(existing_planned)} planned file(s) already exist — "
                "--modify applies them as reviewed diffs (else they are skipped). "
                "Add --allow-rewrite only if a wholesale rewrite is truly wanted."
            )

    cache_file = os.path.join(
        os.path.dirname(os.path.realpath(tasks_md)), ".implement_cache.json"
    )
    if os.path.exists(cache_file):
        try:
            import hashlib as _hashlib
            import json as _json

            with open(cache_file, "r", encoding="utf-8") as f:
                cache = _json.load(f)
            cached_hash = cache.get("taskplan_hash", "")
            current_hash = _hashlib.md5(taskplan_content.encode()).hexdigest()[:8]
            if cache.get("taskplan") == tasks_md and cached_hash == current_hash:
                parts += ["--keep"]
                hints.append("--keep: matching implement cache found (true resume).")
            else:
                hints.append(
                    "Stale implement cache detected — it will be ignored and rebuilt "
                    "on the next run (no --keep was added); add --keep --refresh to "
                    "rebuild the file list and resume."
                )
        except Exception:
            pass

    hint = (
        "\n".join(f"  {h}" for h in hints)
        if hints
        else "  No known conflicts — the plan matches the current codebase."
    )
    hint += "\n  Optional verification: --fix (auto-fix loop on syntax/mypy failures) or --review (deep LLM pass)."
    return parts, hint


def _module_inventory(workspace: str, limit: int = 150) -> str:
    """Compact list of existing modules (rel path + first docstring line).

    Fed into the plan/taskplan prompts so the LLM sees the ground-truth
    module names instead of inventing near-duplicates (e.g. it cannot propose
    ``shell_allowlist.py`` when ``security/allowlist.py`` is listed).
    """
    entries: list[str] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", "backups")]
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, workspace).replace("\\", "/")
            purpose = ""
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    src = fh.read()
                for line in src.splitlines()[:8]:
                    stripped = line.strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        purpose = stripped.strip('"\'').strip()[:70]
                        break
            except Exception:
                continue
            entries.append(f"{rel}{(' — ' + purpose) if purpose else ''}")
            if len(entries) >= limit:
                break
    return "\n".join(entries)


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
            combined += "\n".join(entry.splitlines()[:remaining])
            break
        combined += entry
        lines_used += len(content.splitlines())
    return bool(combined), combined


async def _write_verified_analysis(text: str, analysis_md: str, ws_path: Path) -> tuple[str, int, int]:
    """Verify code claims in *text* against *ws_path*, then write to *analysis_md*.

    Unverifiable file paths, symbol names, line numbers, and code snippets are
    flagged in an appended ``## Verification Report`` instead of being trusted
    silently. Returns ``(verified_content, checked_count, flagged_count)``.
    """
    result = await verify_analysis_claims(text, ws_path)
    if result.text != text:
        with open(analysis_md, "w", encoding="utf-8") as f:
            f.write(result.text)
    if result.checked:
        status = "clean" if not result.flagged else f"{result.flagged} flagged"
        print(f"  [analyze] Verified {result.checked} code claims ({status})")
    return result.text, result.checked, result.flagged


def _analysis_flag_gate(checked: int, flagged: int, force: bool, report_text: str = "") -> bool:
    """Gate on the analysis verification report before planning.

    A single unverifiable claim (fabricated file path, symbol, or line) can
    poison the whole downstream plan, so ANY flagged claim pauses the run for
    confirmation — unless ``--force`` was given, which only warns.
    """
    if flagged <= 0:
        return True
    if force:
        print(
            f"  [analyze] WARNING: {flagged} of {checked} code claim(s) could not be "
            "verified — continuing (--force)."
        )
        return True
    print(
        f"\n  [analyze] {flagged} of {checked} code claims could not be verified "
        "(fabricated paths/symbols/lines would poison the downstream plan):"
    )
    report = report_text.split("## Verification Report", 1)[-1].strip()
    for line in report.splitlines()[:10]:
        print(f"    {line}")
    # Autonomous mode auto-DENIES (safe default); only an explicit answer or
    # --force may continue past unverifiable claims.
    answer = auto_choice("  Continue to planning anyway? (y/N): ", default="n", auto_default="n").strip().lower()
    return answer in ("y", "yes")


async def _repair_analysis(
    agent: "Agent", analysis_text: str, report_text: str, ws_path_str: str
) -> tuple[str | None, int, int]:
    """One-shot LLM repair round for flagged claims (decision #036).

    Feeds the original analysis + its Verification Report + the ground-truth module
    inventory back to the model and asks it to rewrite ONLY the flagged lines with
    correct paths/symbols/lines. Re-verifies; returns ``(repaired_text, checked,
    flagged)`` — repaired_text is None when repair did not clear all flags (caller
    falls back to today's gate). The inventory teaches the model what actually
    exists so it stops inventing paths that laguna-s-2.1-style models fabricate.
    """
    inventory = _module_inventory(ws_path_str) or "(no modules found)"
    report = report_text.split("## Verification Report", 1)[-1].strip()
    repair_system = (
        _PATH_VERIFICATION_RULE + "\n\n"
        "You are a precise code auditor. The analysis below has UNVERIFIED claims — paths,\n"
        "symbols, or line numbers that could not be confirmed against the workspace.\n"
        "Rewrite ONLY those flagged lines with correct values drawn from the module inventory\n"
        "and file listings given above; leave every verified claim unchanged. Output the full\n"
        "corrected analysis text — no intro, no tags."
    )
    repair_user = (
        f"## Module inventory (ground truth):\n{inventory}\n\n"
        f"## Verification Report:\n{report}\n\n"
        f"## Analysis to repair:\n{analysis_text}"
    )
    try:
        repaired = await agent.llm.chat([
            {"role": "system", "content": repair_system},
            {"role": "user", "content": repair_user},
        ], max_tokens=8000)
    except Exception as exc:  # pragma: no cover - defensive, never raises
        print(f"  [analyze] Repair round failed (non-blocking): {exc}")
        return None, 0, 0
    if not repaired or repaired.startswith("[Error") or repaired.startswith("File not found"):
        return None, 0, 0
    result = await verify_analysis_claims(repaired, Path(ws_path_str))
    print(
        f"  [analyze] Repair round re-verified {result.checked} claims "
        f"({result.flagged if result.flagged else 'clean'})"
    )
    return (repaired if result.flagged == 0 else None), result.checked, result.flagged


async def _verify_or_repair_gate(
    agent: "Agent", analysis_text: str, analysis_md: str, ws_path: Path | str, checked: int,
    flagged: int, report_text: str, force: bool,
) -> bool:
    """Verify gate with an automatic path-existence repair pass (decision #036).

    On any unverifiable claim (and not ``--force``), run one LLM repair round; if it
    clears all flags the analysis is rewritten in *analysis_md* and planning proceeds
    without a pause. Otherwise today's confirmation gate applies (``--force`` only
    warns). Returns True when the run may continue past verification.
    """
    ws_path_str = str(ws_path)
    v_text = report_text
    v_flagged = flagged
    if v_flagged > 0 and not force:
        print("  [analyze] Running path-existence repair pass for flagged claims...")
        repaired, rep_checked, rep_flagged = await _repair_analysis(
            agent, analysis_text, v_text, ws_path_str
        )
        if repaired is not None:
            with open(analysis_md, "w", encoding="utf-8") as f:
                f.write(repaired)
            print(f"  [analyze] Repair re-verified {rep_checked} claims ({rep_flagged})")
            v_text = repaired
            v_flagged = rep_flagged
    return _analysis_flag_gate(checked, v_flagged, force, report_text=v_text)


async def _extract_decisions_if_any(agent: "Agent", analysis_md: str, ws_path: str | Path) -> None:
    """Read analysis, extract decision candidates, prompt to record. Non-blocking.

    Candidates are fact-checked against the workspace before the record prompt
    (negative-coverage claims, nonexistent affected files, and repeated
    unverified-claim tokens become warnings); recording any warned candidate
    requires an explicit ``y`` and is auto-denied in autonomous mode. The
    analysis' own Verification Report (remaining unverifiable claims) is echoed
    into the extraction prompt so candidates are not based on it.
    """
    try:
        with open(analysis_md, "r", encoding="utf-8") as f:
            analysis = f.read()
        ws_str = str(ws_path)
        report = ""
        if "## Verification Report" in analysis:
            report = analysis.split("## Verification Report", 1)[-1].strip()
        candidates = await extract_from_analysis(
            agent,
            analysis,
            inventory=_module_inventory(ws_str),
            verification_report=report,
        )
        if not candidates:
            return
        candidates = annotate_candidates(candidates, ws_path, verification_report=report)
        print(f"\n[decide] Extracted {len(candidates)} decision candidates:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c.get('title', 'Untitled')}")
            ctx = c.get('context', '')
            if ctx:
                print(f"     {ctx}")
            for w in c.get("warnings", []):
                print(f"     ⚠ {w}")
        if not sys.stdin.isatty():
            return
        print("\n  Record? (1,2/all/N, press Enter to skip): ", end="")
        choice = read_input().strip().lower()
        if stop_requested():
            return
        if choice and choice != "n":
            selected: list[int] = []
            if choice == "all":
                selected = list(range(len(candidates)))
            else:
                for part in choice.replace(" ", "").split(","):
                    try:
                        selected.append(int(part) - 1)
                    except ValueError:
                        pass
            warned = [
                i for i in selected
                if 0 <= i < len(candidates) and candidates[i].get("warnings")
            ]
            if warned and not auto_choice(
                f"  {len(warned)} candidate(s) carry unverified claims — "
                "record them anyway? (y/N): ",
                default="n", auto_default="n",
            ).strip().lower().startswith("y"):
                for i in warned:
                    print(f"  Skipped: {candidates[i].get('title', 'Untitled')} (unverified claims)")
                selected = [i for i in selected if i not in warned]
            for idx in selected:
                if 0 <= idx < len(candidates):
                    c = candidates[idx]
                    record = add_decision(
                        ws_str,
                        c.get("title", "Untitled"),
                        context=c.get("context", ""),
                        decision=c.get("decision", ""),
                        rationale=c.get("rationale", ""),
                        affected_files=c.get("affected_files", []),
                        tags=c.get("tags", []),
                        warnings=c.get("warnings"),
                    )
                    print(f"  Recorded #{record['id']}: {record['title']}")
    except Exception:
        pass


class WorkflowCommand(Command):
    """Full pipeline: analyze, plan, entities, taskplan.

    Invocation: ``workflow . --desc <feature description> [--auto]`` or
    ``workflow . --continue`` (see the module docstring for stages and
    configuration).  Returns ``True`` and prints the tailored next
    ``implement`` command unless ``--auto`` runs it inline.
    """

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
            "  halts at an ambiguity gate (print clarifying questions) unless --force is given.\n"
            "  Docs (spec/analysis/plan/entities/tasks) are written to .docs/<timestamp>/\n"
            "  (git-ignored) — one folder per run; the newest run is found by readers."
        )

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        parts = args

        if len(parts) < 1:
            self.error("Usage: workflow <target> [--from <spec.md>] [--stdin] [--force] [--brainstorm] [--workspace <path>]")
            self.error("  target: .  |  --desc/-from spec | --features spec (file or inline)")
            return True

        force = "--force" in parts
        brainstorm = "--brainstorm" in parts
        auto_flag = "--auto" in parts
        if auto_flag:
            set_autonomous(True)
            print("\n[auto] Autonomous mode — prompts auto-select safe defaults.")

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

        # Every generated doc for this run goes into a fresh .docs/<timestamp>/
        # folder (git-ignored) instead of polluting the workspace root.  The
        # previous run folder is captured BEFORE creating the new one so the
        # skip/carry-over logic below can resume from it.
        prev_run = latest_run_dir(_ws_dir())
        run_dir = new_run_dir(_ws_dir())
        print(f"Run folder: {run_dir}")

        if "--desc" in parts:
            di = parts.index("--desc")
            if di + 1 < len(parts):
                desc_text = parts[di + 1].strip('"')
                spec_file = str(run_dir / "project_spec.md")
                with open(spec_file, "w", encoding="utf-8") as f:
                    f.write(f"# Project Specification\n\n{desc_text}")
                greenfield = True
                print(f"\n[desc] {desc_text[:120]}...")
        elif "--stdin" in parts:
            text = read_stdin("Paste spec or description. Type --- on its own line when done:")
            parts = [p for p in parts if p != "--stdin"]
            if text.strip():
                spec_file = str(run_dir / "project_spec.md")
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
                features_file = str(run_dir / "project_features.md")
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

        # Probe LM Studio before spending time on the pipeline
        # (disable_thinking: a reasoning model spends the 8-token probe budget
        # on reasoning_content and returns empty content — the probe must get
        # a real answer, not a truncated reasoning fragment)
        probe = await agent.llm.chat([
            {"role": "system", "content": "Respond with exactly one word: ready"},
            {"role": "user", "content": "ready"}
        ], max_tokens=8, disable_thinking=True)
        if probe.startswith("[Error") or probe.startswith("[LM Studio"):
            self.error(f"LM Studio is not reachable: {probe[:200]}")
            self.error("Start LM Studio and ensure a model is loaded, then retry.")
            return True

        # Pre-scan existing names to warn about collisions
        taken_names, existing_filenames = _collect_existing_names(str(ws_path))
        collision_warning = _collision_warning(taken_names, existing_filenames) if taken_names or existing_filenames else ""

        # Dynamic path rules based on workspace structure
        pkgs = _detect_subpackages(str(ws_path))
        if not pkgs:
            pkgs = ["agent_core", "agent1", "src/agent1"]  # fallback
        pkg_list = "`, `".join(pkgs[:5])
        path_rule = f"\n\nPATH RULES: New files MUST use a sub-package prefix (`{pkg_list}/`). BAD: bare filenames or `src/` without subdirectory."
        stdlib_shadow = (
            "\n\nSTDLIB SHADOWING (CRITICAL): Never create a package or directory whose"
            " name matches one of these Python standard-library modules: "
            + ", ".join(sorted(_HIGH_RISK_STDLIB_NAMES)[:20])
            + ", etc. A package named `logging` will BREAK `import logging`."
            "  Use distinct names (e.g. `log_utils/` not `logging/`)."
        )
        prompt_context = collision_warning + path_rule + "\n\nSIZE RULES: New files max 150 lines (SRP). Split large concepts. Modifying: minimal changes only." + stdlib_shadow
        prompt_context += (
            "\n\nMODULE POLICY (CRITICAL): This is an established repository. "
            "Prefer [MODIFY] over [NEW]: if an existing module already covers the "
            "concern, extend it — never create a near-duplicate module (e.g. do not "
            "propose an allow-list module when security/allowlist.py exists). Only "
            "propose a [NEW] file when NO existing module addresses the concern, and "
            "name the closest existing module in the task description."
        )
        inventory = _module_inventory(str(ws_path))
        if inventory:
            prompt_context += (
                "\n\nEXISTING MODULES (never create near-duplicates of these):\n"
                + inventory
            )

        analysis_md = str(run_dir / "project_analysis.md")
        plan_md = str(run_dir / "project_plan.md")
        entities_md = str(run_dir / "project_entities.md")
        tasks_md = str(run_dir / "project_tasks.md")

        def step_ok(result: str) -> bool:
            return not (result.startswith("[Error") or result.startswith("[LM Studio"))

        def _reuse(name: str, label: str) -> bool:
            """True when *label* can be skipped because its doc already exists.

            Checks the current run folder first, then the previous run — a
            doc carried over from the previous run is copied into this run
            folder so each run folder stays self-contained.  Carry-over only
            happens when the previous run's spec matches the current one (same
            task); a different spec must be regenerated from scratch.
            """
            cur = run_dir / name
            if cur.is_file():
                print(f"\n[Skipping {label}] exists")
                return True
            if prev_run is not None:
                cand = prev_run / name
                if cand.is_file():
                    if _specs_match(prev_run / "project_spec.md", spec_content):
                        shutil.copy2(cand, cur)
                        print(f"\n[Skipping {label}] exists — carried over from {prev_run.name}")
                        return True
                    print(
                        f"\n[analyze] Previous run {prev_run.name} has a DIFFERENT spec — "
                        f"regenerating {label} (no carry-over)."
                    )
            return False

        async def _run_next(parts_next: list[str]) -> None:
            """Run the tailored 'implement' command inline — exactly the same
            command the REPL would execute, with the same flow-stop wrapping.

            ``parts_next[0]`` is the command name ("implement"); it must be
            stripped before execute() — otherwise it lands in the taskplan
            positional slot and resolve to "<workspace>/implement" → missing.
            """
            from .implement_cmd import ImplementCommand

            print(f"\n[workflow] Running next step: implement {parts_next[1] if len(parts_next) > 1 else ''} ...")
            clear_stop()
            _llm = getattr(agent, "llm", None)
            _chat = getattr(_llm, "chat", None)
            if _llm is not None and _chat is not None:
                _llm.chat = chat_stoppable(_chat)
            try:
                try:
                    await ImplementCommand().execute(parts_next[1:], agent)
                except FlowStopped:
                    print("  Flow stopped by user.")
            finally:
                if _llm is not None and _chat is not None:
                    _llm.chat = _chat

        async def _offer_next(tasks_md: str, plan_md: str, entities_md: str, analysis_md: str | None, target_ws: str) -> None:
            """Print the tailored next command and (interactively or
            autonomously) run it inline."""
            try:
                taskplan_content = Path(tasks_md).read_text(encoding="utf-8")
            except OSError:
                taskplan_content = ""
            parts_next, hint = _tailored_implement_parts(
                tasks_md, plan_md, entities_md, analysis_md,
                target_ws, taskplan_content,
            )
            print(f"\nNext: {' '.join(parts_next)}")
            print(hint)
            choice = auto_choice("  Run this command now? (y/N): ", default="n", auto_default="y")
            if stop_requested():
                return
            if choice in ("y", "yes"):
                await _run_next(parts_next)

        if spec_file:
            with open(spec_file, "r", encoding="utf-8") as f:
                spec_content = f.read()
            print(f"\n[spec] Loaded from {spec_file}")

            if not force and _reuse("project_plan.md", "plan"):
                pass  # skipped — message printed by _reuse
            else:
                if not force and _reuse("project_analysis.md", "analyze"):
                    pass  # skipped — message printed by _reuse
                else:
                    print("\n[analyze] Analyzing spec...")

                    # Scan target workspace for code context when spec references agent/self-improvement
                    ws_ctx_found, ws_code_context = _scan_workspace_context(ws_path, spec_content)
                    if ws_ctx_found:
                        print("  [analyze] Included workspace code context (self-improvement detected)")
                    module_inventory = _module_inventory(str(ws_path))
                    trace_header = f"# Analysis for workspace {ws_path}\n"
                    if spec_file:
                        trace_header += f"| Spec file: {os.path.basename(spec_file)}\n\n"
                    else:
                        trace_header += "\n"

                    analyze_system = (
                        _PATH_VERIFICATION_RULE + "\n"
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
                    # Always provide the ground-truth module listing: the PATH
                    # VERIFICATION rule requires every cited path to come from the
                    # workspace context, so without this the model must either
                    # fabricate (forbidden) or BLOCK asking for locations.
                    if module_inventory:
                        user_prompt += (
                            "\n\n## Workspace module inventory (ground truth — only these modules exist):\n"
                            f"{module_inventory}\n"
                        )
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
                        f.write(strip_reasoning(r, mode="analysis"))
                    print("  [analyze] First pass generated")

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
                            section_text = r[idx:next_section].strip()
                            # Models occasionally repeat a section header (observed
                            # with the "## 7." heading); show each section once.
                            seen: list[str] = []
                            parts = section_text.splitlines()
                            for line in parts:
                                stripped = line.strip()
                                if stripped == section_marker and section_marker in seen:
                                    continue
                                if stripped == section_marker:
                                    seen.append(section_marker)
                                print(line)
                            print("")
                        print("\nNext steps:")
                        print("  1. Answer the questions above, then run again with --desc \"<answers>\"")
                        print("     or create a file and pass it via --from <file>")
                        print("  2. Or use --force to skip this gate and proceed anyway.")
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
                            _PATH_VERIFICATION_RULE + "\n"
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
                            f.write(strip_reasoning(critique_r, mode="analysis"))
                        print("  [analyze] Appended refinement pass")
                    else:
                        print(f"  [analyze] Critique failed (non-blocking): {critique_r[:100]}")

                    with open(analysis_md, "r", encoding="utf-8") as f:
                        final_analysis = f.read()
                    v_text, v_checked, v_flagged = await _write_verified_analysis(final_analysis, analysis_md, ws_path)
                    if not await _verify_or_repair_gate(agent, final_analysis, analysis_md, ws_path, v_checked, v_flagged, v_text, force):
                        print("\n[analyze] Stopped at the verification gate — fix the spec or re-run with --force.")
                        return True

                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()

                # Validate analysis has expected sections
                expected = ["## 1. SCOPE", "## 2. ASSUMPTIONS", "## 3. RISKS"]
                missing = [h for h in expected if h not in analysis]
                if missing:
                    print(f"  [analyze] WARNING: missing expected sections: {', '.join(missing)}")
                    print("  The analysis format may be non-standard. Plan/task quality may be reduced.")

                await _extract_decisions_if_any(agent, analysis_md, ws_path)

                print("\n[plan] Creating plan...")
                r = await agent.llm.chat([
                    {"role": "system", "content": "You are an expert software architect. Create a detailed coding plan with ALL files needed. Follow SOLID principles — each file has a single responsibility. New files should be small and focused (max 150 lines). If a concept needs more code, split across multiple files. Ensure all Python code passes mypy strict type checking. No unbound TypeVars, no type mismatches. Never use <tool_call> or XML tags." + prompt_context},
                    {"role": "user", "content": f"Create coding plan:\n\n## Spec:\n{spec_content}\n\n## Analysis:\n{analysis}"}
                ])
                if not step_ok(r):
                    print(f"[plan] FAILED: {r[:200]}")
                    return True
                with open(plan_md, "w", encoding="utf-8") as f:
                    f.write(strip_reasoning(r, mode="light"))
                print("[plan] Written")

            if not force and _reuse("project_entities.md", "entities"):
                pass  # skipped — message printed by _reuse
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
                    f.write(strip_reasoning(r, mode="light"))
                print("[entities] Written")

            if not force and _reuse("project_tasks.md", "taskplan"):
                pass  # skipped — message printed by _reuse
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read() if os.path.exists(analysis_md) else ""
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                with open(entities_md, "r", encoding="utf-8") as f:
                    entities = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": (
                        _PATH_VERIFICATION_RULE + "\n\n"
                        "Create a numbered task plan. Output ONLY tasks — no reasoning, no planning notes, no self-talk.\n\n"
                        "FORMAT — exactly one task per line:\n"
                        "  N. `path/to/file.py` — short description\n\n"
                        "RULES:\n"
                        "- Every file path MUST be backtick-wrapped\n"
                        "- One file per line, numbered sequentially (1. 2. 3.)\n"
                        "- No bullet markers (-, *), no [TAGS], no brackets around filenames\n"
                        "- If modifying an existing file, use its EXACT current path\n"
                        "- List in dependency order (configs first, then utilities, then consumers)\n"
                        + prompt_context
                    )},
                    {"role": "user", "content": f"## Spec:\n{spec_content}\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}\n\n## Entities:\n{entities}"}
                ])
                if not step_ok(r):
                    print(f"[taskplan] FAILED: {r[:200]}")
                    return True
                with open(tasks_md, "w", encoding="utf-8") as f:
                    f.write(strip_reasoning(r, mode="light"))
                print("[taskplan] Written")

            await _offer_next(tasks_md, plan_md, entities_md, analysis_md, str(target_workspace))

        elif features_file:
            with open(features_file, "r", encoding="utf-8") as f:
                features = f.read()
            print(f"\n[features] Loaded from {features_file}")

            if not force and _reuse("project_analysis.md", "analyze"):
                pass  # skipped — message printed by _reuse
            else:
                print("\n[analyze] Scanning existing py files...")
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
                        print("Warning: silenced exception in workflow_cmd.py")
                r = await agent.llm.chat([
                    {"role": "system", "content": "You are an expert code reviewer. Analyze the existing code AND these new features. Find bugs, gaps, and what needs to change."},
                    {"role": "user", "content": f"## Existing Code:\n{combined}\n\n## New Features:\n{features}\n\nAnalyze both existing issues and what must change for the new features."}
                ])
                if not step_ok(r):
                    print(f"[analyze] FAILED: {r[:200]}")
                    return True
                v_text, v_checked, v_flagged = await _write_verified_analysis(r, analysis_md, ws_path)
                if not await _verify_or_repair_gate(agent, r, analysis_md, ws_path, v_checked, v_flagged, v_text, force):
                    print("\n[analyze] Stopped at the verification gate — fix the spec or re-run with --force.")
                    return True
                await _extract_decisions_if_any(agent, analysis_md, ws_path)

            if not force and _reuse("project_plan.md", "plan"):
                pass  # skipped — message printed by _reuse
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()

                existing_plan = ""
                if os.path.exists(plan_md):
                    with open(plan_md, "r", encoding="utf-8") as f:
                        existing_plan = f.read()[:3000]

                print("\n[plan] Creating plan...")
                r = await agent.llm.chat([
                    {"role": "system", "content": "You are an expert software architect. Create a plan that extends the EXISTING codebase with these new features.\n\nIMPORTANT:\n- Start with '## Feature Addition: <summary>'\n- List NEW files to create\n- List EXISTING files to modify and what minimal changes are needed\n- Explain WHY each change is needed\n- Preserve existing architecture"},
                    {"role": "user", "content": f"## Existing code analysis:\n{analysis}\n\n## Existing plan:\n{existing_plan if existing_plan else 'No existing plan'}\n\n## New features to add:\n{features}\n\nCreate a plan that integrates these features into the existing codebase."}
                ])
                if not step_ok(r):
                    print(f"[plan] FAILED: {r[:200]}")
                    return True
                with open(plan_md, "a", encoding="utf-8") as f:
                    f.write(f"\n\n---\n\n{strip_reasoning(r, mode="light")}")
                print(f"[plan] Appended to {plan_md}")

            if not force and _reuse("project_entities.md", "entities"):
                pass  # skipped — message printed by _reuse
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
                    f.write(f"\n\n---\n\n{strip_reasoning(r, mode="light")}")
                print("[entities] Appended")

            if not force and _reuse("project_tasks.md", "taskplan"):
                pass  # skipped — message printed by _reuse
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": (
                        _PATH_VERIFICATION_RULE + "\n\n"
                        "Create a numbered task plan for adding these features. Output ONLY tasks — no reasoning, no planning notes, no self-talk.\n\n"
                        "FORMAT — exactly one task per line:\n"
                        "  N. [NEW] `path/to/new.py` — short description\n"
                        "  N. [MODIFY] `existing/file.py` — short description\n\n"
                        "RULES:\n"
                        "- Every file path MUST be backtick-wrapped\n"
                        "- Prefix each task with [NEW] or [MODIFY]\n"
                        "- One file per line, numbered sequentially\n"
                        "- No bullet markers, no extra [TAGS], no brackets around filenames\n"
                        "- If modifying an existing file, use its EXACT current path\n"
                        + prompt_context
                    )},
                    {"role": "user", "content": f"## Analysis:\n{analysis}\n\n## Plan:\n{plan}"}
                ])
                if not step_ok(r):
                    print(f"[taskplan] FAILED: {r[:200]}")
                    return True
                with open(tasks_md, "a", encoding="utf-8") as f:
                    f.write(f"\n\n---\n\n{strip_reasoning(r, mode="light")}")
                print("[taskplan] Appended")

            await _offer_next(tasks_md, plan_md, entities_md, analysis_md, str(target_workspace))

        else:
            if not force and _reuse("project_analysis.md", "analyze"):
                pass  # skipped — message printed by _reuse
            else:
                print("\n[analyze] Scanning py files...")
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
                lines_used = 0
                max_lines = 5000
                for pf in py_files:
                    try:
                        with open(pf, "r", encoding="utf-8") as f:
                            entry = f"\n\n# ---- {pf} ----\n{f.read()}"
                    except Exception:
                        continue
                    entry_lines = len(entry.splitlines())
                    if lines_used + entry_lines > max_lines:
                        remaining = max_lines - lines_used
                        if remaining > 0:
                            combined += "\n".join(entry.splitlines()[:remaining])
                        break
                    combined += entry
                    lines_used += entry_lines
                analyze_system = (
                    "You are an expert software architect. Evaluate this codebase.\n"
                    "Output ONLY the numbered sections below, exactly in this order.\n\n"
                    "## 1. CODE QUALITY — bugs, edge cases, type safety, error handling gaps\n"
                    "## 2. COMPLETENESS — missing tests, missing docs, missing features\n"
                    "## 3. ARCHITECTURE — DRY violations, coupling, SRP breaks\n"
                    "## 4. INNOVATION — new capabilities that would make this system more useful\n"
                    "## 5. PRODUCTION — logging, monitoring, config, security gaps\n"
                    + ("## 6. BRAINSTORMING — bold creative features\n" if brainstorm else "")
                    + "RULES:\n"
                    "- Max 3 bullet points per section, max 50 words per bullet\n"
                    "- Cite file paths from the code when identifying issues\n"
                    "- Total output under 800 words — no intro, no conclusion, no repetition of the code\n"
                )
                analyze_user = (
                    "Evaluate this codebase. Follow the numbered-section FORMAT from the "
                    "system prompt exactly.\n\n"
                    f"## Codebase:\n{combined}"
                )
                r = await agent.llm.chat([
                    {"role": "system", "content": analyze_system},
                    {"role": "user", "content": analyze_user},
                ])
                if not step_ok(r):
                    print(f"[analyze] FAILED: {r[:200]}")
                    return True
                v_text, v_checked, v_flagged = await _write_verified_analysis(r, analysis_md, ws_path)
                if not await _verify_or_repair_gate(agent, r, analysis_md, ws_path, v_checked, v_flagged, v_text, force):
                    print("\n[analyze] Stopped at the verification gate — fix the spec or re-run with --force.")
                    return True
                await _extract_decisions_if_any(agent, analysis_md, ws_path)

            if not force and _reuse("project_plan.md", "plan"):
                pass  # skipped — message printed by _reuse
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()
                print("\n[plan] Creating plan...")
                r = await agent.llm.chat([
                    {"role": "system", "content": (
                        _PATH_VERIFICATION_RULE + "\n"
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
                    f.write(strip_reasoning(r, mode="light"))
                print("[plan] Written")

            if not force and _reuse("project_entities.md", "entities"):
                pass  # skipped — message printed by _reuse
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
                    f.write(strip_reasoning(r, mode="light"))
                print("[entities] Written")

            if not force and _reuse("project_tasks.md", "taskplan"):
                pass  # skipped — message printed by _reuse
            else:
                with open(analysis_md, "r", encoding="utf-8") as f:
                    analysis = f.read()
                with open(plan_md, "r", encoding="utf-8") as f:
                    plan = f.read()
                r = await agent.llm.chat([
                    {"role": "system", "content": (
                        _PATH_VERIFICATION_RULE + "\n\n"
                        "Create a numbered task plan. Output ONLY tasks — no reasoning, no planning notes, no self-talk.\n\n"
                        "FORMAT — exactly one task per line:\n"
                        "  N. `path/to/file.py` — short description\n\n"
                        "RULES:\n"
                        "- Every file path MUST be backtick-wrapped\n"
                        "- One file per line, numbered sequentially (1. 2. 3.)\n"
                        "- No bullet markers (-, *), no [TAGS], no brackets around filenames\n"
                        "- If modifying an existing file, use its EXACT current path\n"
                        "- List in dependency order (configs first, then utilities, then consumers)\n"
                        + prompt_context
                    )},
                    {"role": "user", "content": f"## Analysis:\n{analysis}\n\n## Plan:\n{plan}"}
                ])
                if not step_ok(r):
                    print(f"[taskplan] FAILED: {r[:200]}")
                    return True
                with open(tasks_md, "w", encoding="utf-8") as f:
                    f.write(strip_reasoning(r, mode="light"))
                print("[taskplan] Written")

            await _offer_next(tasks_md, plan_md, entities_md, analysis_md, str(target_workspace))

        if auto_flag:
            set_autonomous(False)
        return True