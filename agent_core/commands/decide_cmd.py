"""Decide command — record, search, and manage design decisions.

Usage:
    decide "title" --why "context" --what "decision" --tags t1,t2 --files f1.py,f2.py
    decide list [--tag t] [--file f.py] [--search "keyword"]
    decide show <id>
    decide check --text "new decision idea"
    decide resolve <id1> <id2>
    decide link <id1> <id2> --why "reason"
    decide extract [--from analysis.md]
"""

from pathlib import Path
from typing import TYPE_CHECKING

from .base import Command, auto_choice, read_input, stop_requested
from .doc_paths import find_input
from .workflow_cmd import _module_inventory
from agent_core.decisions import (
    add_decision,
    annotate_candidates,
    check_contradictions,
    extract_from_analysis,
    find_decisions,
    find_overlaps,
    load_decisions,
    resolve_contradictions,
    save_decisions,
)

if TYPE_CHECKING:
    from agent import Agent


_DECIDE_HELP = """decide — Track design decisions for this workspace

  decide "title" --why "..." --what "..." [--tags t1,t2] [--files f1.py]
      Record a new decision

  decide list [--tag t] [--file f.py] [--search "keyword"]
      List matching decisions

  decide show <id>
      Show full decision record

  decide check --text "new decision idea"
      Check if an idea contradicts any past decisions (LLM-powered)

  decide resolve <id1> <id2>
      Resolve a contradiction between two decisions (LLM-powered)

  decide link <id1> <id2> --why "reason"
      Link two decisions as related

  decide extract [--from analysis.md]
      Auto-extract decision candidates from a project analysis file"""


class DecideCommand(Command):
    @property
    def name(self) -> str:
        return "decide"

    @property
    def help_text(self) -> str:
        return _DECIDE_HELP

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            self.error("Usage: decide <subcommand> [...]")
            print(_DECIDE_HELP)
            return True

        sub = args[0].lower()

        if sub == "list":
            return await self._cmd_list(args[1:], agent)
        elif sub == "show":
            return await self._cmd_show(args[1:], agent)
        elif sub == "check":
            return await self._cmd_check(args[1:], agent)
        elif sub == "resolve":
            return await self._cmd_resolve(args[1:], agent)
        elif sub == "link":
            return await self._cmd_link(args[1:], agent)
        elif sub == "extract":
            return await self._cmd_extract(args[1:], agent)
        else:
            return await self._cmd_add(args, agent)

    # ── add ─────────────────────────────────────────────────────────────

    async def _cmd_add(self, args: list[str], agent: "Agent") -> bool:
        title = args[0] if args else ""
        context = _extract_flag(args, "--why", "--context")
        decision = _extract_flag(args, "--what", "--decision")
        tags = _extract_list(args, "--tags")
        files = _extract_list(args, "--files")

        if not title:
            self.error("Title required: decide \"title\" --why \"...\" --what \"...\"")
            return True

        ws = str(Path(agent.workspace).resolve())
        record = add_decision(
            ws, title,
            context=context,
            decision=decision,
            affected_files=files,
            tags=tags,
        )
        print(f"Recorded decision #{record['id']}: {record['title']}")
        return True

    # ── list ────────────────────────────────────────────────────────────

    async def _cmd_list(self, args: list[str], agent: "Agent") -> bool:
        ws = str(Path(agent.workspace).resolve())
        tag = _extract_flag(args, "--tag")
        file = _extract_flag(args, "--file")
        keyword = _extract_flag(args, "--search")
        tags = [tag] if tag else None
        files = [file] if file else None

        results = find_decisions(ws, tags=tags, files=files, keyword=keyword)
        if not results:
            print("No decisions found.")
            return True

        print(f"{len(results)} decision(s):")
        print("-" * 60)
        for d in results:
            files_str = ", ".join(d.get("affected_files", [])[:3])
            tags_str = ", ".join(d.get("tags", [])[:5])
            print(
                f"  #{d['id']}  {d['date'][:10]}  {d['title']}\n"
                f"         files: {files_str or '-'}\n"
                f"         tags:  {tags_str or '-'}"
            )
        return True

    # ── show ────────────────────────────────────────────────────────────

    async def _cmd_show(self, args: list[str], agent: "Agent") -> bool:
        if not args:
            self.error("Usage: decide show <id>")
            return True
        decision_id = args[0]
        ws = str(Path(agent.workspace).resolve())
        decisions = load_decisions(ws)
        record = next((d for d in decisions if d["id"] == decision_id), None)
        if not record:
            print(f"Decision #{decision_id} not found.")
            return True
        for key in ["id", "date", "title", "context", "decision", "rationale",
                     "affected_files", "tags", "contradictions", "resolved_by"]:
            val = record.get(key, "")
            if isinstance(val, list):
                val = ", ".join(val)
            if val:
                print(f"  {key}: {val}")
        return True

    # ── check ───────────────────────────────────────────────────────────

    async def _cmd_check(self, args: list[str], agent: "Agent") -> bool:
        text = _extract_flag(args, "--text")
        if not text:
            print("Paste your decision idea, then press Enter on an empty line:")
            lines = []
            while True:
                line = read_input()
                if line == "":
                    break
                lines.append(line)
            text = "\n".join(lines)
            if stop_requested():
                return True
        if not text.strip():
            self.error("No decision text provided.")
            return True

        ws = str(Path(agent.workspace).resolve())
        decisions = load_decisions(ws)
        if not decisions:
            print("No existing decisions. No contradictions possible.")
            return True

        # Instant overlap check
        overlaps = find_overlaps(
            {"tags": [], "affected_files": _extract_file_refs(text)},
            decisions,
            ws,
        )
        if overlaps:
            ids = ", ".join(f"#{d['id']}" for d in overlaps)
            print(f"Tag/file overlap detected with: {ids}")

        # Always LLM-powered deep check
        print("\nChecking for contradictions (LLM)...")
        result = await check_contradictions(agent, decisions, text)
        print(result)
        return True

    # ── resolve ─────────────────────────────────────────────────────────

    async def _cmd_resolve(self, args: list[str], agent: "Agent") -> bool:
        if len(args) < 2:
            self.error("Usage: decide resolve <id1> <id2>")
            return True
        id1, id2 = args[0], args[1]
        ws = str(Path(agent.workspace).resolve())
        decisions = load_decisions(ws)
        d1 = next((d for d in decisions if d["id"] == id1), None)
        d2 = next((d for d in decisions if d["id"] == id2), None)
        if not d1 or not d2:
            print(f"Decision #{id1 if not d1 else id2} not found.")
            return True

        print(f"Resolving #{id1} vs #{id2}...")
        result = await resolve_contradictions(agent, d1, d2)
        print(result)
        return True

    # ── link ────────────────────────────────────────────────────────────

    async def _cmd_link(self, args: list[str], agent: "Agent") -> bool:
        if len(args) < 2:
            self.error("Usage: decide link <id1> <id2> --why \"reason\"")
            return True
        id1, id2 = args[0], args[1]
        reason = _extract_flag(args[2:], "--why")
        ws = str(Path(agent.workspace).resolve())
        decisions = load_decisions(ws)
        d1 = next((d for d in decisions if d["id"] == id1), None)
        d2 = next((d for d in decisions if d["id"] == id2), None)
        if not d1 or not d2:
            print(f"Decision #{id1 if not d1 else id2} not found.")
            return True
        d1.setdefault("contradictions", []).append(
            {"id": id2, "reason": reason, "status": "linked"}
        )
        d2.setdefault("contradictions", []).append(
            {"id": id1, "reason": reason, "status": "linked"}
        )
        save_decisions(ws, decisions)
        print(f"Linked #{id1} -> #{id2}: {reason if reason else 'no reason given'}")
        return True

    # ── extract ────────────────────────────────────────────────────────

    async def _cmd_extract(self, args: list[str], agent: "Agent") -> bool:
        source = _extract_flag(args, "--from") or "project_analysis.md"
        # Relative inputs are resolved against the WORKSPACE (never the
        # process CWD); analysis docs live in .docs/<timestamp>/, so the
        # newest run folder (then the workspace root) is used as fallback.
        source = find_input(str(agent.workspace), source)
        try:
            analysis = Path(source).read_text(encoding="utf-8")
        except OSError:
            self.error(f"Cannot read {source}")
            return True

        print(f"Extracting decisions from {source}...")
        ws = str(Path(agent.workspace).resolve())
        report = ""
        if "## Verification Report" in analysis:
            report = analysis.split("## Verification Report", 1)[-1].strip()
        candidates = await extract_from_analysis(
            agent,
            analysis,
            inventory=_module_inventory(ws),
            verification_report=report,
        )
        if not candidates:
            print("No decision candidates found.")
            return True
        candidates = annotate_candidates(candidates, ws, verification_report=report)

        for i, c in enumerate(candidates, 1):
            print(f"\n  {i}. {c.get('title', 'Untitled')}")
            print(f"     Context: {c.get('context', '-')}")
            print(f"     Decision: {c.get('decision', '-')}")
            print(f"     Tags: {', '.join(c.get('tags', []))}")
            print(f"     Files: {', '.join(c.get('affected_files', []))}")
            for w in c.get("warnings", []):
                print(f"     ⚠ {w}")

        print("\nRecord these decisions? (1,2/all/N): ", end="")
        choice = read_input().strip().lower()
        if stop_requested():
            return True

        if choice == "all":
            selected = list(range(len(candidates)))
        elif choice == "n" or choice == "":
            print("No decisions recorded.")
            return True
        else:
            selected = []
            for part in choice.replace(" ", "").split(","):
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    selected.extend(range(int(lo) - 1, int(hi)))
                else:
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
                rationale = input(f"  Rationale for '{c['title']}' (optional): ").strip()
                record = add_decision(
                    ws,
                    c["title"],
                    context=c.get("context", ""),
                    decision=c.get("decision", ""),
                    rationale=rationale or c.get("rationale", ""),
                    affected_files=c.get("affected_files", []),
                    tags=c.get("tags", []),
                    warnings=c.get("warnings"),
                )
                print(f"  Recorded #{record['id']}")

        return True


# ── helpers ──────────────────────────────────────────────────────────────


def _extract_flag(args: list[str], *names: str) -> str:
    for name in names:
        if name in args:
            idx = args.index(name)
            if idx + 1 < len(args):
                return args[idx + 1]
    return ""


def _extract_list(args: list[str], flag: str) -> list[str]:
    val = _extract_flag(args, flag)
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def _extract_file_refs(text: str) -> list[str]:
    import re
    return re.findall(r"[\w/\\-]+\.py", text)
