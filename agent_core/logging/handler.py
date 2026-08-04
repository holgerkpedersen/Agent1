import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional

from agent_core.config.schema import ConfigManager


class JSONLogHandler(logging.Handler):
    """
    A logging handler that outputs log records as structured JSON strings.
    Useful for observability in REPLs and async agent loops.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stream = sys.stderr

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record into a JSON string.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }

        # Include extra attributes passed via the 'extra' parameter in logging calls
        standard_attrs = {
            "args", "asctime", "created", "exc_info", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "message",
            "name", "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName"
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_entry[key] = value

        # Handle exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emits the formatted log message to the stream.
        """
        try:
            msg = self.format(record)
            self.stream.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(config_manager: Optional[ConfigManager] = None) -> None:
    """
    Configures the global logging system with a JSONLogHandler.
    If a ConfigManager is provided, it can be used to determine log levels or other settings.
    """
    root_logger = logging.getLogger()
    
    # Default level; could be extended if ConfigManager provides specific keys
    log_level = logging.INFO
    if config_manager:
        # Logic for extracting log level from ConfigManager would go here
        pass

    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    json_handler = JSONLogHandler()
    # The format method is overridden, so we don't need a specific Formatter object
    root_logger.addHandler(json_handler)