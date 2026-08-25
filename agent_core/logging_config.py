"""Canonical structured logging module — JSON pipeline with correlation injection.

This is the single source of truth for agent logging (plan ARCH item 7):

- :class:`CorrelationIdContext` / :func:`get_correlation_id` — async-safe
  correlation ID scoping (shared with ``agent_core.context_management``).
- :class:`JsonFormatter` / :class:`HumanReadableFormatter` — machine-parseable
  JSON (prod) and ANSI-colored console output (dev).
- :func:`setup_logging` — one-call configuration with optional JSON/file
  output and correlation filtering.

The root-level ``logging_config.py`` is a deprecated re-export of this
module; new code must import from ``agent_core.logging_config``.
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .context_management import CORRELATION_ID_CTX
except ImportError:
    # Fallback for direct execution or isolated testing outside package structure
    import contextvars
    CORRELATION_ID_CTX = contextvars.ContextVar("correlation_id", default="")

NO_CORRELATION_ID = "no-correlation-id"


def get_correlation_id() -> str:
    """Return the current async-safe correlation ID (or the no-id marker)."""
    return CORRELATION_ID_CTX.get() or NO_CORRELATION_ID


class CorrelationIdContext:
    """Context manager scoping a correlation ID to the current async task.

    Usage::

        with CorrelationIdContext("run-42") as corr_id:
            ...  # every log record carries correlation_id=run-42
    """

    def __init__(self, corr_id: str | None = None) -> None:
        self._corr_id = corr_id or str(uuid.uuid4())
        self._token: Any = None

    def __enter__(self) -> str:
        self._token = CORRELATION_ID_CTX.set(self._corr_id)
        return self._corr_id

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token is not None:
            CORRELATION_ID_CTX.reset(self._token)


class SafeJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that safely serializes common agent framework types."""

    def default(self, obj: Any) -> str | dict[str, Any]:
        if isinstance(obj, (datetime, Path)):
            return str(obj)
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if is_dataclass(obj):
            try:
                return asdict(obj)  # type: ignore[arg-type]
            except Exception:
                # Fallback for frozen or complex dataclasses that fail asdict()
                return getattr(obj, "__dict__", repr(obj))
        if isinstance(obj, Exception):
            return {
                "type": type(obj).__name__,
                "message": str(obj),
            }
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items()}
        # Ultimate fallback to prevent logging crashes on unknown payloads
        return str(obj)


class CorrelationIdFilter(logging.Filter):
    """Injects the current async-safe correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON strings using SafeJsonEncoder."""

    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", get_correlation_id()),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        # Attach custom extra fields (e.g. allowlist event metadata)
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "relativeCreated",
                "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "pathname", "filename", "module", "levelno", "levelname",
            ):
                log_data[key] = value

        return json.dumps(log_data, cls=SafeJsonEncoder)


class HumanReadableFormatter(logging.Formatter):
    """ANSI-colored console format for local development."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        corr_id = getattr(record, "correlation_id", get_correlation_id())[:8]
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        msg = record.getMessage()
        exc_str = ""
        if record.exc_info and record.exc_info[0] is not None:
            exc_str = "\n" + self.formatException(record.exc_info)
        return (
            f"{color}[{ts}] [{record.levelname:<8}] [{corr_id}] "
            f"{record.name}:{record.funcName} | {msg}{exc_str}{reset}"
        )


def setup_logging(
    level: int = logging.INFO,
    json_format: bool = True,
    mode: str | None = None,
    log_file: str | Path | None = None,
    propagate: bool = False,
) -> None:
    """
    Configure the root logger with structured output and correlation tracking.

    Args:
        level: Logging threshold (default: INFO).
        json_format: If True, uses SafeJsonEncoder for machine-parseable logs.
                     If False, falls back to standard human-readable formatting.
        mode: ``"prod"`` forces JSON, ``"dev"`` forces the ANSI human-readable
              formatter (overrides *json_format*).
        log_file: Optional path — appends a JSON file handler (always JSON so
                  files stay parseable by aggregators).
        propagate: Whether the root logger propagates to parent handlers.
    """
    use_json = json_format if mode is None else (mode == "prod")
    console_formatter = "json" if use_json else "human"

    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": console_formatter,
            "filters": ["correlation"],
            "stream": "ext://sys.stdout",
        },
    }
    if log_file:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "json",
            "filters": ["correlation"],
            "filename": str(file_path),
            "mode": "a",
            "encoding": "utf-8",
        }

    log_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JsonFormatter},
            "human": {"()": HumanReadableFormatter},
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        },
        "filters": {
            "correlation": {"()": CorrelationIdFilter},
        },
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": propagate,
        },
    }

    logging.config.dictConfig(log_config)

    if mode == "dev":
        # Suppress noisy third-party libraries in dev mode
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_framework_logger(name: str = __name__) -> logging.Logger:
    """Return a pre-configured logger for agent modules."""
    return logging.getLogger(name)


# =============================================================================
# Async Compatibility Notes
# =============================================================================
# Python's built-in `logging` module is thread-safe for basic emit operations.
# However, context variables (like CORRELATION_ID_CTX) do NOT automatically propagate
# to threads spawned via `concurrent.futures.ThreadPoolExecutor` or
# `loop.run_in_executor()`. To maintain correlation tracking across executors,
# explicitly copy and run the context:
#
#   import asyncio
#   from .context_management import copy_correlation_context
#
#   ctx = copy_correlation_context()
#   loop.run_in_executor(executor, ctx.run, target_function)
