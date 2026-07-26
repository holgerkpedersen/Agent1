from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol


class CommandHandler(Protocol):
    """Structural protocol describing the command handler interface."""

    @abstractmethod
    async def handle(self, args: List[str]) -> int: ...  # Returns exit code


class BaseCommandHandler(ABC):
    """Abstract base class for all concrete command handlers.

    Subclasses must provide a human-readable ``name`` and implement
    the asynchronous :meth:`handle` entry point which returns an exit code.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def handle(self, args: List[str]) -> int: ...