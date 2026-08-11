import logging
from typing import Optional

from agent_core.config import load_agent_settings, _validate_settings
from ..session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class ClearCommand:
    """Clear command implementation with confirmation prompt."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager
        # Validate configuration paths on initialization
        settings = load_agent_settings()
        _validate_settings(settings)

    def display_stats(self) -> None:
        """Display statistics/summary of conversation context before clearing."""
        stats = self._session_manager.get_session_stats()
        logger.info("Session stats:")
        logger.info(f"  Messages: {stats['message_count']}")
        logger.info(f"  Tokens used: {stats['token_usage']}")
        logger.info(f"  Last activity: {stats['last_activity']}")

    def prompt_user_confirmation(self) -> bool:
        """Prompt user for explicit confirmation before proceeding.

        Re-prompts until a valid y/n response is received. Returns True to
        proceed with clear, False to abort and leave conversation intact.
        """
        while True:
            response = input("Are you sure? [y/n]: ").strip().lower()
            if response == "y":
                logger.info("User confirmed session clear.")
                return True  # Proceed with clear
            elif response == "n":
                logger.warning("Clear aborted. Conversation context left intact.")
                return False  # Abort, leave conversation intact
            else:
                logger.warning("Invalid input. Please respond with 'y' or 'n'.")

    def run(self) -> Optional[str]:
        """Execute the clear command flow.

        Displays stats, prompts for confirmation, and only clears session if
        user confirms. Returns a status message describing outcome.
        """
        self.display_stats()
        if not self.prompt_user_confirmation():
            return "Clear aborted by user."
        self._session_manager.clear_session()
        logger.info("Session cleared successfully.")
        return "Session cleared successfully."


def create_clear_command(session_manager: SessionManager) -> ClearCommand:
    """Factory function to construct a ClearCommand instance."""
    return ClearCommand(session_manager)