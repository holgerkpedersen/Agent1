"""Clear command for agent interactive mode."""

from .base import Command

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


class ClearCommand(Command):
    """Show memory state and clear agent memory."""

    @property
    def name(self) -> str:
        return "clear"

    @property
    def help_text(self) -> str:
        return "clear [stats|--force] - Show/clear agent memory"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        force = "--force" in args
        stats_only = "stats" in args

        stats = agent.memory_stats()
        kg_sections = stats.get("knowledge_graph", 0)
        kg_entries = stats.get("knowledge_graph_entries", kg_sections)

        print("Memory state:")
        print(f"  chat history:       {stats['chat_history']} messages")
        print(f"  files read:         {stats['files_read']} tracked")
        if stats["stale_files"] > 0:
            print(f"  stale files:        {stats['stale_files']} (changed externally)")
        if stats["working_memory"] > 0:
            print(f"  working memory:     {stats['working_memory']} items")
        if stats["semantic_index"] > 0:
            print(f"  semantic index:     {stats['semantic_index']} terms")
        print(
            f"  knowledge graph:    {kg_entries} entries "
            f"in {kg_sections} sections"
        )
        if kg_entries == 0:
            print(
                "    (empty by design until you add entries: 'kg add fact "
                "<key> <value>', or seed curated ones with "
                "'python scripts/seed_kg.py')"
            )

        if stats_only:
            return True

        if (
            stats["chat_history"] == 0
            and stats["files_read"] == 0
            and stats["working_memory"] == 0
            and stats["semantic_index"] == 0
            and kg_entries == 0
        ):
            print("Nothing to clear.")
            return True

        if not force:
            confirm = input("Clear all memory? (y/n): ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Cancelled.")
                return True

        agent.clear_history()
        print("Agent memory cleared.")
        return True
