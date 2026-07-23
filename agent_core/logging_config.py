"""Structured JSON logging pipeline with correlation injection for agent observability."""

import json
import logging
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .context_management import CORRELATION_ID_CTX
except ImportError:
    # Fallback for direct execution or isolated testing outside package structure
    import contextvars
    CORRELATION_ID_CTX = contextvars.ContextVar("correlation_id", default="")


class SafeJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that safely serializes common agent framework types."""

    def default(self, obj: Any) -> str | dict[str, Any]:  # type: ignore[override]
        if isinstance(obj, (datetime, Path)):
            return str(obj)
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if is_dataclass(obj):
            try:
                return asdict(obj)
            except Exception:
                # Fallback for frozen or complex dataclasses that fail asdict()
                return getattr(obj, "__dict__", repr(obj))
        if isinstance(obj, Exception):
            return {
                "type": type(obj).__name__,
                "message": str(obj),
            }
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        # Ultimate fallback to prevent logging crashes on unknown payloads
        return str(obj)


class CorrelationIdFilter(logging.Filter):
    """Injects the current async-safe correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = CORRELATION_ID_CTX.get() or "no-correlation-id"
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
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "unknown"),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, cls=SafeJsonEncoder)


def setup_logging(level: int = logging.INFO, json_format: bool = True) -> None:
    """
    Configure the root logger with structured JSON output and correlation tracking.

    Args:
        level: Logging threshold (default: INFO).
        json_format: If True, uses SafeJsonEncoder for machine-parseable logs.
                     If False, falls back to standard human-readable formatting.
    """
    log_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JsonFormatter},
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        },
        "filters": {
            "correlation": {"()": CorrelationIdFilter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "json" if json_format else "standard",
                "filters": ["correlation"],
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"],
            "propagate": False,
        },
    }

    logging.config.dictConfig(log_config)


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