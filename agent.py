def _persist_plan_answer(self, plan_text: str) -> None:
        """Save a plan-mode final answer to ``.docs/<ts>/plan_proposed.md``.

        Plan-mode answers used to evaporate as terminal text; persisting them
        gives the user a durable artifact to hand to ``implement``/``fix``
        later (plan item D-#14).  Best-effort: a write failure must never
        lose the already-printed answer or crash the turn.
        """
        with _suppress_and_log("could not persist plan-mode answer"):
            from agent_core.commands.doc_paths import new_run_dir

            out = new_run_dir(self.workspace) / "plan_proposed.md"
            out.write_text(
                "# Proposed plan\n\n"
                f"_Generated in plan mode on {datetime.now():%Y-%m-%d %H:%M}. "
                "Review before applying — switch to build mode to implement._\n\n"
                f"{plan_text}\n",
                encoding="utf-8",
            )
            print(yellow(f"\n  [plan] Saved to {out}"))