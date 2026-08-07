"""Agent file context interface — bridges /file and @file detection with content retrieval."""

from agent_core.file_context_retriever import FileContextRetriever


class AgentFileContextInterface:
    """Detects ``/file`` / ``@file`` keywords and returns retrieved context snippets.

    Parameters
    ----------
    retriever:
        A :class:`FileContextRetriever` used to look up file contents.
    """

    def __init__(self, retriever: FileContextRetriever) -> None:
        self._retriever = retriever

    def process_request(self, message: str) -> list[str]:
        """Return a list of context snippets for every ``/file`` / ``@file`` in *message*.

        Each snippet is the file's text content.  Files that cannot be read are
        silently skipped so the caller always gets a (possibly empty) list.
        """
        filenames = self._retriever.extract_filenames(message)
        contexts: list[str] = []
        for fn in filenames:
            content = self._retriever.retrieve(fn)
            if content is not None:
                contexts.append(content)
        return contexts