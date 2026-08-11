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

    # Patterns for common secrets and credentials
    _SECRET_PATTERNS = [
        r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+",
        r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+",
        r"(?i)(token|auth_token|access_token)\s*[:=]\s*\S+",
        r"(?i)Bearer\s+\S+",
    ]

    _CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

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

        # Strip non-printable control characters for consistent sanitization
        cleaned = self._CONTROL_CHARS.sub("", text)

        # Remove shell/code injection patterns
        sanitized = self._combined_pattern.sub("", cleaned)

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

    def mask_secrets(self, text: str) -> str:
        """
        Masks sensitive information like API keys, passwords, and tokens.

        :param text: The raw input string potentially containing secrets.
        :return: The string with secret values masked.
        """
        if not text:
            return text
        
        masked = text
        for pattern in self._SECRET_PATTERNS:
            masked = re.sub(pattern, lambda m: f"{m.group(0)[:12]}****", masked)
        return masked

    def sanitize_and_mask(self, data: Any) -> Any:
        """
        Recursively sanitizes and masks secrets in complex data structures.

        :param data: The data to process.
        :return: The sanitized and masked version of the input data.
        """
        if isinstance(data, str):
            return self.mask_secrets(self.sanitize_string(data))
        
        if isinstance(data, dict):
            return {
                str(k): self.sanitize_and_mask(v) 
                for k, v in data.items()
            }
        
        if isinstance(data, list):
            return [self.sanitize_and_mask(item) for item in data]
        
        if isinstance(data, tuple):
            return tuple(self.sanitize_and_mask(item) for item in data)

        return data