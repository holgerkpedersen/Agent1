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
    CATEGORIES,
    STATUS_ACTIVE,
    _STATUSES,
    add_decision,
    annotate_candidates,
    check_contradictions,
    count_by_category,
    count_by_status,
    extract_from_analysis,
    find_decisions,
    find_meta_warnings,
    find_open_contradictions,
    find_overlaps,
    find_stale_decisions,
    ledger_health,
    load_decisions,
    resolve_contradictions,
    save_decisions,
)

if TYPE_CHECKING:
    from agent import Agent


_DECIDE_HELP = """decide [add|list|show|check|resolve|link|extract|review|set-status|categories] - Track design decisions for this workspace

  decide "title" --why "..." --what "..." [--tags t1,t2] [--files f1.py] [--category cat]
      Record a new decision

  decide list [--tag t] [--file f.py] [--search "keyword"]
              [--status active|superseded|archived] [--category cat]
              [--open] [--summary]
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
      Auto-extract decision candidates from a project analysis file

  decide review [--strict]
      Health check on the ledger: stale affected_files, open
      contradictions, meta_warnings (unverified claims).  --strict exits
      non-zero when stale or open-contradiction issues are found.

  decide set-status <id> <status>
      Change a decision's status (active/superseded/archived)

  decide categories
      Show decision counts by category"""


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
        elif sub == "review":
            return await self._cmd_review(args[1:], agent)
        elif sub == "set-status":
            return await self._cmd_set_status(args[1:], agent)
        elif sub == "categories":
            return await self._cmd_categories(args[1:], agent)
        else:
            return await self._cmd_add(args, agent)

    # ── add ─────────────────────────────────────────────────────────────

    async def _cmd_add(self, args: list[str], agent: "Agent") -> bool:
        title = args[0] if args else ""
        context = _extract_flag(args, "--why", "--context")
        decision = _extract_flag(args, "--what", "--decision")
        tags = _extract_list(args, "--tags")
        files = _extract_list(args, "--files")
        category = _extract_flag(args, "--category") or None

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
            category=category,
        )
        print(f"Recorded decision #{record['id']}: {record['title']}")
        return True

    # ── list ────────────────────────────────────────────────────────────

    async def _cmd_list(self, args: list[str], agent: "Agent") -> bool:
        ws = str(Path(agent.workspace).resolve())
        tag = _extract_flag(args, "--tag")
        file = _extract_flag(args, "--file")
        keyword = _extract_flag(args, "--search")
        status = _extract_flag(args, "--status") or None
        category = _extract_flag(args, "--category") or None
        show_summary = "--summary" in args
        show_open = "--open" in args
        tags = [tag] if tag else None
        files = [file] if file else None

        results = find_decisions(ws, tags=tags, files=files, keyword=keyword,
                                 status=status, category=category)
        if not results:
            print("No decisions found.")
            return True

        # Summary header
        if show_summary:
            all_decisions = load_decisions(ws)
            by_status = count_by_status(all_decisions)
            by_cat = count_by_category(all_decisions)
            print(f"Total: {len(all_decisions)} decisions")
            print(f"  by status: {', '.join(f'{k}={v}' for k, v in by_status.items())}")
            print(f"  by category: {', '.join(f'{k}={v}' for k, v in by_cat.items())}")
            print()

        # --open: active decisions with unresolved contradictions
        if show_open:
            results = [d for d in results
                       if d.get("status", STATUS_ACTIVE) == STATUS_ACTIVE
                       and any(c.get("status") not in ("resolved", "superseded")
                               for c in d.get("contradictions", []))]

        print(f"{len(results)} decision(s):")
        print("-" * 60)
        for d in results:
            dfiles = d.get("affected_files") or []
            dtags = d.get("tags") or []
            files_str = ", ".join(dfiles[:3])
            tags_str = ", ".join(dtags[:5])
            date_str = (d.get("date") or "")[:10]
            status_str = d.get("status", STATUS_ACTIVE)
            cat_str = d.get("category") or ""
            extra = ""
            if status_str != STATUS_ACTIVE:
                extra += f"  [{status_str}]"
            if cat_str:
                extra += f"  {{{cat_str}}}"
            print(
                f"  #{d['id']}  {date_str}  {d['title']}{extra}\n"
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
                     "affected_files", "tags", "contradictions", "resolved_by",
                     "status", "category"]:
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
                        print("Silenced exception in decide_cmd.py:315")

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


# ── review ───────────────────────────────────────────────────────────

    async def _cmd_review(self, args: list[str], agent: "Agent") -> bool:
        """Ledger health check: stale affected_files, open contradictions,
        meta_warnings (unverified claims), and unresolved candidates
        (decision #054, #080, #082, #084, #086, #087)."""
        strict = "--strict" in args
        ws = str(Path(agent.workspace).resolve())
        decisions = load_decisions(ws)
        if not decisions:
            print("No decisions recorded.")
            return True

        report = ledger_health(ws, decisions)
        stale = report["stale"]
        open_contras = report["open_contradictions"]
        meta_flags = report["meta_warnings"]

        print(f"{len(decisions)} decision(s) recorded.")

        error_count = 0

        if stale:
            print(f"\n{len(stale)} decision(s) reference missing files:")
            for d in stale:
                print(
                    f"  #{d['id']}  {d['title']}\n"
                    f"         missing: {', '.join(d['_missing_files'])}"
                )
            error_count += len(stale)
        else:
            print("\nAll recorded affected_files exist on disk.")

        if open_contras:
            print(f"\n{len(open_contras)} decision(s) with open contradictions:")
            for d in open_contras:
                print(
                    f"  #{d['id']}  {d['title']}"
                    f"  -> open vs {', '.join(d['_open_contradiction_ids'])}"
                )
            error_count += len(open_contras)
        else:
            print("No open contradictions.")

        if meta_flags:
            print(f"\n{len(meta_flags)} decision(s) carry meta_warnings:")
            for d in meta_flags:
                for w in d["_meta_warnings"]:
                    print(f"  #{d['id']}  {d['title']}\n         {w}")
        else:
            print("No meta_warnings.")

        if strict and error_count:
            print(f"\nFAIL: {error_count} issue(s) found (stale + open contradictions).")
            return True

        if strict and not error_count and not meta_flags:
            print("\nOK (strict mode, all checks passed).")
        elif not strict:
            print(
                "\nTip: `decide show <id>` for details; "
                "`decide resolve <id1> <id2>` for open contradictions."
            )
        return True

    # ── set-status ──────────────────────────────────────────────────────────

    async def _cmd_set_status(self, args: list[str], agent: "Agent") -> bool:
        if len(args) < 2:
            self.error("Usage: decide set-status <id> <status>")
            return True
        decision_id, new_status = args[0], args[1]
        if new_status not in _STATUSES:
            self.error(f"Invalid status '{new_status}'. Use: active, superseded, archived")
            return True
        ws = str(Path(agent.workspace).resolve())
        decisions = load_decisions(ws)
        record = next((d for d in decisions if d["id"] == decision_id), None)
        if not record:
            print(f"Decision #{decision_id} not found.")
            return True
        old_status = record.get("status", STATUS_ACTIVE)
        record["status"] = new_status
        save_decisions(ws, decisions)
        print(f"Decision #{decision_id}: {old_status} -> {new_status}")
        return True

    # ── categories ──────────────────────────────────────────────────────────

    async def _cmd_categories(self, args: list[str], agent: "Agent") -> bool:
        ws = str(Path(agent.workspace).resolve())
        decisions = load_decisions(ws)
        if not decisions:
            print("No decisions recorded.")
            return True
        by_cat = count_by_category(decisions)
        print(f"{len(decisions)} decision(s) across {len(by_cat)} categories:")
        print("-" * 40)
        for cat, count in by_cat.items():
            print(f"  {cat:25s} {count}")
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
