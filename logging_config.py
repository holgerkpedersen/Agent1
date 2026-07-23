"""
logging_config.py
Production-ready structured logging configuration with async-safe correlation IDs.
Supports JSON (prod) and human-readable (dev) formatters, context injection, and graceful fallbacks.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


# ---------------------------------------------------------------------------
# 🆔 Async-Safe Correlation ID Context
# ---------------------------------------------------------------------------

# Each async task/thread gets its own correlation ID via contextvars (thread-safe & await-safe)
CORRELATION_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="00000000-0000-0000-0000-000000000000")


def get_correlation_id() -> str:
    """Retrieve the current correlation ID from async context."""
    return CORRELATION_ID_CTX.get()


class CorrelationIdContext:
    """Context manager to scope correlation IDs per request/tool call."""
    
    def __init__(self, corr_id: Optional[str] = None):
        self._corr_id = corr_id or str(uuid.uuid4())
        self._token: Optional[contextvars.Token[str]] = None
        
    def __enter__(self) -> str:
        self._token = CORRELATION_ID_CTX.set(self._corr_id)
        return self._corr_id
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            CORRELATION_ID_CTX.reset(self._token)


# ---------------------------------------------------------------------------
# 📝 Custom Formatters
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Outputs log records as structured JSON for aggregation pipelines (ELK, Datadog, etc.)."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", get_correlation_id()),
        }
        
        # Attach exception traceback if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Attach custom extra fields
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "created", "relativeCreated", 
                           "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                           "pathname", "filename", "module", "levelno", "levelname"):
                log_data[key] = value
                
        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Color-friendly, readable format for local development."""
    
    # ANSI escape codes (disabled on Windows if not supported, but Python 3.8+ handles them well)
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
        "RESET": "\033[0m"
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
            
        return f"{color}[{ts}] [{record.levelname:<8}] [{corr_id}] {record.name}:{record.funcName} | {msg}{exc_str}{reset}"


# ---------------------------------------------------------------------------
# 🔍 Correlation ID Filter (Injects into LogRecord)
# ---------------------------------------------------------------------------

class CorrelationIdFilter(logging.Filter):
    """Automatically attaches the current async context correlation ID to every log record."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


# ---------------------------------------------------------------------------
# 🛠️ Configuration API
# ---------------------------------------------------------------------------

def setup_logging(
    level: Union[str, int] = "INFO",
    mode: str = "dev",
    log_file: Optional[Union[str, Path]] = None,
    propagate: bool = False
) -> logging.Logger:
    """
    Configure root and framework loggers with structured or human-readable output.
    
    Args:
        level: Logging severity threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        mode: "dev" for console-friendly colors, "prod" for JSON streamlines.
        log_file: Optional path to write logs to disk.
        propagate: Whether root logger should propagate to parent handlers.
        
    Returns:
        Configured root logger instance.
    """
    
    formatter = JsonFormatter() if mode == "prod" else HumanReadableFormatter()
    
    # Clear existing handlers to prevent duplicates on re-init
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root_logger.propagate = propagate
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIdFilter())
    root_logger.addHandler(console_handler)
    
    # File Handler (optional)
    if log_file:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        # Always use JSON for files to ensure parseability by log aggregators
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(file_handler)
        
    # Suppress noisy third-party libraries in dev mode
    if mode == "dev":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
    return root_logger


def get_framework_logger(name: str = __name__) -> logging.Logger:
    """Retrieve a pre-configured logger for agent modules."""
    return logging.getLogger(name)