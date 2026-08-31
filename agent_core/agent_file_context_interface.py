"""
Agent file context interface — bridges /file and @file detection with content retrieval.

Parameters
----------
retriever:
    A :class:`FileContextRetriever` used to look up file contents.
optimizer:
    A :class:`ContextOptimizer` used to decide which content to return.
"""

from typing import Optional
from agent_core.file_context_retriever import FileContextRetriever
from agent_core.context_optimizer import ContextOptimizer


class AgentFileContextInterface:
    """Detects ``/file`` or ``@file`` keywords and returns retrieved context snippets.

    Parameters
    ----------
    retriever:
        A :class:`FileContextRetriever` used to look up file contents.
    optimizer:
        A :class:`ContextOptimizer` used to decide which content to return.
    """

    def __init__(
        self,
        retriever: FileContextRetriever,
        optimizer: Optional[ContextOptimizer] = None,
    ) -> None:
        self._retriever = retriever
        self._optimizer = optimizer

    def process_request(self, message: str) -> list[str]:
        """Return a list of context snippets for every ``/file`` or ``@file`` in *message*.

        If an optimizer is provided, it will be used to select the most appropriate
        context (full, skeleton, or snippet). Otherwise, the full content is returned.
        """
        filenames = self._retriever.extract_filenames(message)
        if self._optimizer:
            return self._optimizer.optimize(message, filenames)

        contexts: list[str] = []
        for fn in filenames:
            content = self._retriever.retrieve(fn)
            if content is not None:
                contexts.append(content)
        return contexts
