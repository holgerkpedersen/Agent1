from pathlib import Path
from src.agent1.file_context import FileContextRetriever


def test_retrieve_valid_filename(tmp_path: Path) -> None:
    file = tmp_path / "sample.txt"
    file.write_text("hello world")
    retriever = FileContextRetriever(base_path=str(tmp_path))
    result = retriever.retrieve("sample.txt")
    assert result == "hello world"


def test_retrieve_invalid_filename(tmp_path: Path) -> None:
    retriever = FileContextRetriever(base_path=str(tmp_path))
    result = retriever.retrieve("nonexistent.txt")
    assert result is None


def test_sanitize_truncation() -> None:
    retriever = FileContextRetriever()
    content = "x" * 5000
    sanitized = retriever._sanitize_content(content)
    assert len(sanitized) == 4096