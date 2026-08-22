"""multillm command — fire the same prompt at MULTIPLE LLMs simultaneously.

Usage::

    multillm "question" [--models m1,m2,...] [--max-tokens N]
             [--thinking] [--concurrency N] [--template <id>]
             [--role model:system-prompt] [--role-file path.json]

Default models: the current agent model plus the configured opencode model
(``laguna-s-2.1`` + ``opencode-go/deepseek-v4-flash`` when nothing is
persisted) — one local LM Studio model and one hosted opencode model answer
the same prompt in parallel, so their answers can be compared directly.

Each model gets its OWN provider instance (separate opencode session, no
shared state), all ``chat`` calls are fired with ``asyncio.gather``, and the
results are printed side by side with a consensus summary.  The dormant
``ConsensusVoter``/``RefinementVoter`` machinery records each model's verdict
under the template id.

Tools: each model runs through the SAME tool loop the agent uses — it can
READ files, search, list directories, run tests, etc. instead of answering
from the prompt alone.  Each model gets its OWN loop instance (no
cross-model tool-state contamination), all running concurrently.

Roles: ``--role model:prompt`` (repeatable, quote multi-word prompts) or
``--role-file file.json`` (``{"model": "system prompt"}``) give DIFFERENT
LLMs DIFFERENT expert personas — each role is prepended as that model's own
``system`` message while the question stays the same, e.g.::

    multillm "review this code" --models laguna-s-2.1,opencode-go/deepseek-v4-flash \
        --role "laguna-s-2.1:You are a security auditor. Focus on vulnerabilities." \
        --role "opencode-go/deepseek-v4-flash:You are a performance engineer. Focus on hot paths."

Inline ``--role`` is the primary way to set roles — no roles.json required.
The REPL splits input with ``shlex.split(posix=False)``, which keeps the
literal quotes on a quoted value; the command strips them (same convention
as ``analyze_cmd``/``fix_cmd``/``implement_cmd``), so multi-word prompts
work inline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Command

if TYPE_CHECKING:
    from agent import Agent


def _default_models(agent: "Agent") -> list[str]:
    """Current agent model + the configured opencode model (deduped)."""
    current = agent.llm.model_name
    try:
        from agent_core.config import load_agent_settings
        settings = load_agent_settings()
        opencode_model = settings.opencode_model
    except Exception:
        opencode_model = "opencode-go/deepseek-v4-flash"
    seen: list[str] = []
    for m in (current, opencode_model):
        if m and m not in seen:
            seen.append(m)
    return seen


class MultiLlmCommand(Command):
    """Ask multiple LLMs the same question simultaneously."""

    @property
    def name(self) -> str:
        return "multillm"

    @property
    def help_text(self) -> str:
        return (
            'multillm "question" [--models laguna-s-2.1,opencode-go/...] '
            "[--max-tokens N] [--thinking] [--concurrency N] "
            "[--role model:prompt] [--role-file path.json] — ask multiple "
            "LLMs the same question in parallel, each with its own role"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        from agent_core.config import load_agent_settings
        from agent_core.llm.parallel import run_parallel, summarize
        from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

        parts = list(args)
        models: list[str] = []
        max_tokens: int | None = None
        disable_thinking = True
        concurrency: int | None = None
        template_id = "parallel"
        roles: dict[str, str] = {}

        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "--models" and i + 1 < len(parts):
                models = [m.strip() for m in parts[i + 1].split(",") if m.strip()]
                i += 2
                continue
            if p == "--max-tokens" and i + 1 < len(parts):
                try:
                    max_tokens = max(1, int(parts[i + 1]))
                except ValueError:
                    self.error("--max-tokens expects a number.")
                    return True
                i += 2
                continue
            if p == "--thinking":
                disable_thinking = False
                i += 1
                continue
            if p == "--concurrency" and i + 1 < len(parts):
                try:
                    concurrency = max(1, int(parts[i + 1]))
                except ValueError:
                    self.error("--concurrency expects a number.")
                    return True
                i += 2
                continue
            if p == "--template" and i + 1 < len(parts):
                template_id = parts[i + 1]
                i += 2
                continue
            if p == "--role" and i + 1 < len(parts):
                role_spec = parts[i + 1].strip('"')
                if ":" not in role_spec:
                    self.error('--role expects "model:system prompt".')
                    return True
                role_model, _, role_text = role_spec.partition(":")
                role_model = role_model.strip()
                if role_model:
                    roles[role_model] = role_text.strip()
                i += 2
                continue
            if p == "--role-file" and i + 1 < len(parts):
                role_path = parts[i + 1]
                try:
                    with open(role_path, encoding="utf-8") as f:
                        import json
                        file_roles = json.load(f)
                    if not isinstance(file_roles, dict):
                        self.error("--role-file must contain a JSON object.")
                        return True
                    for k, v in file_roles.items():
                        roles[str(k)] = str(v)
                except OSError as exc:
                    self.error(f"Could not read role file: {exc}")
                    return True
                except ValueError as exc:
                    self.error(f"Invalid JSON in role file: {exc}")
                    return True
                i += 2
                continue
            i += 1

        # Everything that is not a flag is the question (quoted at the REPL).
        question_parts = [
            p for p in parts
            if p not in ("--thinking",)
            and not p.startswith("--")
        ]
        # Drop flag values from the question.
        skip_values = {
            "--models", "--max-tokens", "--concurrency", "--template",
            "--role", "--role-file",
        }
        flag_values: set[str] = set()
        for j, p in enumerate(parts):
            if p in skip_values and j + 1 < len(parts):
                flag_values.add(parts[j + 1])
        question_parts = [p for p in question_parts if p not in flag_values]

        question = " ".join(question_parts).strip()
        if not question:
            self.error('Usage: multillm "question" [--models m1,m2]')
            return True

        if not models:
            models = _default_models(agent)
        if len(models) < 2:
            self.error(
                "multillm needs at least two models — pass --models "
                "laguna-s-2.1,opencode-go/deepseek-v4-flash"
            )
            return True

        messages = [{"role": "user", "content": question}]
        print(f"\n  [multillm] {len(models)} model(s) in parallel: {', '.join(models)}")
        print(f"  [multillm] question: {question[:200]}")

        try:
            settings = load_agent_settings()
        except Exception as exc:
            self.error(f"Could not load agent settings: {exc}")
            return True

        run = await run_parallel(
            messages,
            models,
            settings,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            template_id=template_id,
            concurrency=concurrency,
            roles=roles,
            # Give the models the SAME tools the agent uses — each model can
            # read/search/list files, run tests, etc. instead of answering
            # from the prompt alone (2026-08-21: models asked for the file
            # content instead of reading it).  Degrades gracefully when the
            # agent does not expose the tool executor (tests / minimal hosts).
            tools=list(NLP_TOOL_SCHEMAS) if getattr(agent, "_execute_tool_call", None) else None,
            execute_tool_fn=getattr(agent, "_execute_tool_call", None),
        )

        print()
        for r in run.results:
            head = f"  [{r.provider}] {r.model}"
            if roles.get(r.model):
                head += f"  (role: {roles[r.model][:60]})"
            print(f"{'=' * 60}")
            print(head)
            print(f"{'=' * 60}")
            if r.ok:
                print(r.text)
            else:
                print(f"  [Error] {r.error}")
            print()
        print(f"{'=' * 60}")
        print(summarize(run))
        return True