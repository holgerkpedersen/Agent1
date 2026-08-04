import os
from typing import Optional


class FileContextRetriever:
    def __init__(self, base_path: str = ".") -> None:
        self.base_path = base_path

    def retrieve(self, query: str) -> Optional[str]:
        filename = query.strip()
        full_path = os.path.join(self.base_path, filename)
        if not os.path.isfile(full_path):
            return None
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._sanitize_content(content)

    def _sanitize_content(self, content: str) -> str:
        max_chars = 4096
        if len(content) > max_chars:
            return content[:max_chars]
        return content


class FileRequestDetector:
    KEYWORDS = ("/file:", "@file:")

    def detect(self, message: str) -> Optional[str]:
        for keyword in self.KEYWORDS:
            index = message.find(keyword)
            if index != -1:
                filename = message[index + len(keyword):].strip()
                return filename
        return None


class TempFileMessageManager:
    def __init__(self) -> None:
        self._temp_messages: list[str] = []

    def add(self, content: str) -> None:
        self._temp_messages.append(content)

    def get_all(self) -> list[str]:
        return list(self._temp_messages)

    def clear(self) -> None:
        self._temp_messages.clear()


class AgentFileContextInterface:
    def __init__(self, retriever: FileContextRetriever) -> None:
        self.retriever = retriever
        self.detector = FileRequestDetector()
        self.temp_manager = TempFileMessageManager()

    def process(self, message: str) -> Optional[str]:
        filename = self.detector.detect(message)
        if filename is None:
            return None
        content = self.retriever.retrieve(filename)
        if content is None:
            return None
        temp_msg = f"[FILE CONTEXT] {filename}:\n{content}"
        self.temp_manager.add(temp_msg)
        return temp_msg

    def get_temp_messages(self) -> list[str]:
        return self.temp_manager.get_all()

    def clear_temp_messages(self) -> None:
        self.temp_manager.clear()