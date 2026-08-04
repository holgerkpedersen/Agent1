import html
import re
from typing import Any, Dict, List, Union


class Sanitizer:
    """
    Provides utilities to sanitize inputs against common injection attacks,
    specifically targeting LLM-based code execution and XSS vulnerabilities.
    """

    # Patterns for shell injection and python code execution keywords
    _FORBIDDEN_PATTERNS = [
        r"eval\(",
        r"exec\(",
        r"__import__",
        r"subprocess\.",
        r"os\.system",
        r"os\.popen",
        r"getattr\(",
        r"setattr\(",
        r";\s*",
        r"\|\|\s*",
        r"&&\s*",
        r"\|\\s*",
        r"\$\(.*\)",
        r"`.*`",
    ]

    def __init__(self, allow_html: bool = False) -> None:
        """
        Initialize the sanitizer.

        :param allow_html: If False, HTML tags will be escaped or stripped.
        """
        self.allow_html = allow_html
        self._combined_pattern = re.compile(
            "|".join(self._FORBIDDEN_PATTERNS), 
            re.IGNORECASE
        )

    def sanitize_string(self, text: str) -> str:
        """
        Sanitizes a single string by removing dangerous patterns and escaping HTML.

        :param text: The raw input string.
        :return: The sanitized string.
        """
        if not text:
            return text

        # Remove shell/code injection patterns
        sanitized = self._combined_pattern.sub("", text)

        # Handle HTML
        if not self.allow_html:
            sanitized = html.escape(sanitized)

        return sanitized.strip()

    def sanitize(self, data: Any) -> Any:
        """
        Recursively sanitizes complex data structures (dict, list, etc.).

        :param data: The data to sanitize.
        :return: The sanitized version of the input data.
        """
        if isinstance(data, str):
            return self.sanitize_string(data)
        
        if isinstance(data, dict):
            return {
                str(k): self.sanitize(v) 
                for k, v in data.items()
            }
        
        if isinstance(data, list):
            return [self.sanitize(item) for item in data]
        
        if isinstance(data, tuple):
            return tuple(self.sanitize(item) for item in data)

        return data