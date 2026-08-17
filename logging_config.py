"""Deprecated re-export of the canonical logging configuration.

The single source of truth is :mod:`agent_core.logging_config` — structured
JSON logging with the shared async-safe :class:`CorrelationIdContext` (plan
ARCH item 8).  This root-level module exists only so legacy imports keep
working; new code must import from ``agent_core.logging_config``.
"""
from agent_core.logging_config import (  # noqa: F401
    CORRELATION_ID_CTX,
    CorrelationIdContext,
    CorrelationIdFilter,
    HumanReadableFormatter,
    JsonFormatter,
    SafeJsonEncoder,
    get_correlation_id,
    get_framework_logger,
    setup_logging,
)
