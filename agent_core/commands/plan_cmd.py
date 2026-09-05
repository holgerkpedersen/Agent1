"""Plan command for agent interactive mode.

Two subcommands:

* ``plan <analysis.md> <plan.md>`` — generate a coding plan from an analysis
  file (LLM-backed; output is regression-checked for path claims).
* ``plan exec [<planfile>] [--tasks <json>] [--dry-run] [--yes]`` — execute a
  proposed plan through isolated, role-gated subagents.  This is the executor
  half of the plan-mode workflow: a planner produces ``.docs/<ts>/plan_proposed.md``
  (persisted by ``Agent._persist_plan_answer``), then ``mode build`` +
  ``plan exec`` runs it.  Plan mode is read-only, so ``plan exec`` refuses to
  run while the session is in plan mode.
"""
from .base import Command, auto_choice, stop_requested
from .doc_paths import find_input, resolve_output, new_run_dir
from .plan_verifier import check_doc, apply_report, summarize, PlanVerifier
from .plan_schema import validate_plan_markdown
from .plan_lifecycle import PlanLifecycleManager
from .plan_dry_run import PlanDryRunner
from .plan_decision_gate import PlanDecisionGate

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent

from agent_core.plan_execution import (
    PlanTask,
    build_and_validate_graph,
    parse_plan_tasks,
    run_plan,
)
from agent_core.orchestration.dependency_graph import CycleError
from agent_core.subagent_roles import get_role, role_names


class PlanCommand(Command):
    """Generate coding plan from analysis file / execute a proposed plan."""

    @property
    def name(self) -> str:
        return "plan"

    @property
    def help_text(self) -> str:
        return (
            "plan <analysis.md> <plan.md> - Generate coding plan from analysis\n"
            "  Output is regression-checked (paths exist or are marked new);\n"
            "  flagged claims pause for confirmation unless --force.\n"
            "plan exec [<planfile>] [--tasks <json>] [--dry-run] [--yes] - Run a plan\n"
            "  via isolated subagents. Requires build mode (run 'mode build' first)."
        )

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if not args:
            self.error(self.help_text)
            return True

        sub = args[0].lower()
        rest = args[1:]

        if sub == "exec":
            return await self._cmd_exec(rest, agent)
        if sub in ("generate", "gen"):
            return await self._cmd_generate(rest, agent)

        # Default: treat the whole arg list as the generate form.
        return await self._cmd_generate(args, agent)

    # ── plan generate ────────────────────────────────────────────────────
    async def _cmd_generate(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 2:
            self.error("Usage: plan <analysis.md> <plan.md>")
            return True

        force = "--force" in args
        clean_args = [a for a in args if a != "--force"]
        if not clean_args:
            self.error("Usage: plan <analysis.md> <plan.md>")
            return True

        # Bare input filenames fall back to the newest .docs run folder.
        analysis_file = find_input(agent.workspace, clean_args[0])
        # Bare output filenames go to .docs/<timestamp>/ (the input's run
        # folder when it has one) — explicit paths are kept.
        plan_file = resolve_output(agent.workspace, clean_args[1], sibling_of=analysis_file)

        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis_content = f.read()
        except FileNotFoundError:
            self.error(f"File not found: {analysis_file}")
            return True

        messages = [
            {"role": "system", "content": "You are an expert software architect. Based on the code analysis provided, create a detailed coding plan with specific implementation steps, prioritized by impact and dependencies."},
            {"role": "user", "content": f"Create a coding plan based on this analysis:\n\n{analysis_content}"}
        ]
        plan = await agent.llm.chat(messages)

        content = f"# Coding Plan\n\n{plan}"
        result = check_doc("plan", content, agent.workspace)
        summarize(result, "plan")
        if not result.clean and not force:
            answer = auto_choice(
                "  Plan references unverifiable paths — write anyway? (y/N): ",
                default="n", auto_default="n",
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("[plan] Halted — regenerate with corrected paths "
                      "(or rerun with --force).")
                return True
        # Schema validation on proposal.
        ok, errs = validate_plan_markdown(content)
        if not ok and not force:
            self.error("Plan schema validation failed:\n" + "\n".join(f"  - {e}" for e in errs))
            answer = auto_choice(
                "  Write anyway? (y/N): ", default="n", auto_default="n",
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("[plan] Halted — fix schema issues and regenerate.")
                return True

        content = apply_report(content, result)

        with open(plan_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Coding plan written to {plan_file}")

        return True

    # ── plan exec ────────────────────────────────────────────────────────
    async def _cmd_exec(self, args: list[str], agent: 'Agent') -> bool:
        if agent.is_plan_mode():
            print(
                "[plan exec] Refused: plan mode is read-only. "
                "Run 'mode build' first, then 'plan exec'."
            )
            return True

        force = "--force" in args
        dry_run = "--dry-run" in args
        yes = "--yes" in args
        clean = [a for a in args if a not in ("--force", "--dry-run", "--yes")]

        tasks_file = clean[0] if clean else None
        json_file = None
        if "--tasks" in clean:
            idx = clean.index("--tasks")
            if idx + 1 < len(clean):
                json_file = clean[idx + 1]

        # Resolve the plan text.
        if json_file is not None:
            try:
                with open(find_input(agent.workspace, json_file), "r", encoding="utf-8") as f:
                    plan_text = f.read()
            except FileNotFoundError:
                self.error(f"Tasks file not found: {json_file}")
                return True
            tasks = parse_plan_tasks(plan_text, fmt="json")
        else:
            plan_path = None
            if tasks_file is not None:
                plan_path = find_input(agent.workspace, tasks_file)
            else:
                from .doc_paths import latest_run_dir
                run = latest_run_dir(agent.workspace)
                if run is not None and (run / "plan_proposed.md").is_file():
                    plan_path = str(run / "plan_proposed.md")
            if plan_path is None:
                self.error(
                    "No plan found. Pass a plan file (with a '## Tasks' block) "
                    "or --tasks <json>, or run 'mode plan' first to produce one."
                )
                return True
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_text = f.read()
            tasks = parse_plan_tasks(plan_text, fmt="md")

        if not tasks:
            print(
                "[plan exec] No tasks found. Add a '## Tasks' block to the plan "
                "(e.g. '- [T1] do the thing (role: implementer)') or pass --tasks <json>."
            )
            return True

        # Validate roles + acyclicity before touching the workspace.
        for t in tasks:
            if get_role(t.role) is None:
                print(
                    f"[plan exec] Unknown role '{t.role}' for task '{t.id}'. "
                    f"Available: {', '.join(role_names())}"
                )
                return True

        try:
            graph, order = build_and_validate_graph(tasks)
        except CycleError as exc:
            cycle = " -> ".join(exc.cycle) if exc.cycle else "<unknown>"
            print(f"[plan exec] Dependency cycle detected: {cycle}. Aborting.")
            return True
        except ValueError as exc:
            print(f"[plan exec] {exc}")
            return True

        print(f"[plan exec] {len(tasks)} task(s), topological order:")
        for i, tid in enumerate(order, 1):
            t = next((x for x in tasks if x.id == tid), None)
            deps = f" (deps: {', '.join(t.depends_on)})" if t and t.depends_on else ""
            print(f"  {i}. [{tid}] {t.role if t else '?'}: {t.description if t else ''}{deps}")

        # ── Pre-execution gates ───────────────────────────────────────
        # Dry-run safety gate
        dry_runner = PlanDryRunner()
        dry_result = dry_runner.validate(plan_text)
        if not dry_result.valid:
            print("[plan exec] Dry-run safety gate FAILED:")
            for err in dry_result.errors:
                print(f"  ✗ {err}")
            return True
        if dry_result.warnings:
            print(f"[plan exec] Dry-run warnings ({len(dry_result.warnings)}):")
            for w in dry_result.warnings[:5]:
                print(f"  ⚠ {w}")
            if len(dry_result.warnings) > 5:
                print(f"  ... and {len(dry_result.warnings) - 5} more")

        # Decision gate
        decision_gate = PlanDecisionGate(agent.workspace)
        gate_result = decision_gate.validate(plan_text)
        if not gate_result.passed:
            print("[plan exec] Decision gate FAILED — plan violates architectural constraints:")
            for v in gate_result.violations:
                print(f"  ✗ {v}")
            if not force:
                return True
            print("[plan exec] Proceeding anyway (--force).")

        if dry_run:
            print("[plan exec] --dry-run: no subagents spawned.")
            return True

        # Lifecycle: transition proposed → executing
        from .doc_paths import latest_run_dir
        run_dir = latest_run_dir(agent.workspace)
        lifecycle = None
        if run_dir is not None:
            try:
                lifecycle = PlanLifecycleManager(run_dir, agent.workspace)
                lifecycle.start_plan()
            except Exception as exc:  # noqa: BLE001
                print(f"[plan exec] (lifecycle start note: {exc})")

        if not yes:
            answer = auto_choice(
                "  Execute this plan with isolated subagents? (y/N): ",
                default="n", auto_default="n",
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("[plan exec] Aborted — no changes made.")
                return True

        # Honour a stop request between scheduling and dispatch.
        if stop_requested():
            print("[plan exec] Stop requested before execution.")
            return True

        snapshot = await run_plan(agent, tasks)

        # Write a durable execution report next to the plan.
        report_lines = ["# Plan Execution Report", ""]
        for t in snapshot["tasks"]:
            node = graph.get_node(t["task_id"])
            role = getattr(node, "task_type", "?") or "?"
            inputs = t.get("input_data") or {}
            dep_note = f" (consumed {len(inputs)} dependency result(s))" if inputs else ""
            report_lines.append(
                f"- [{t['task_id']}] {t['status']} (role: {role}){dep_note}"
            )
        report = "\n".join(report_lines) + "\n"
        try:
            out = new_run_dir(agent.workspace) / "plan_execution_report.md"
            out.write_text(report, encoding="utf-8")
            print(f"[plan exec] Report written to {out}")
        except Exception as exc:  # noqa: BLE001 - best-effort artifact
            print(f"[plan exec] (report not written: {exc})")

        failed = [t for t in snapshot["tasks"] if t["status"] != "completed"]
        if failed:
            print(
                f"[plan exec] Done with {len(failed)} non-completed task(s): "
                f"{', '.join(t['task_id'] for t in failed)}"
            )
        else:
            print("[plan exec] All tasks completed.")

        # Lifecycle: transition executing → executed
        if lifecycle is not None:
            try:
                lifecycle.finish_plan()
            except Exception as exc:  # noqa: BLE001
                print(f"[plan exec] (lifecycle finish note: {exc})")

        return True
