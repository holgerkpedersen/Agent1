"""Canonical logging config tests (plan ARCH items 7-8): the root-level
logging_config is a pure re-export of agent_core.logging_config, and the
correlation context scopes async-safe IDs."""
import logging

import pytest


def test_root_module_is_pure_reexport():
    import agent_core.logging_config as canonical
    import logging_config as root

    assert root.CorrelationIdContext is canonical.CorrelationIdContext
    assert root.JsonFormatter is canonical.JsonFormatter
    assert root.HumanReadableFormatter is canonical.HumanReadableFormatter
    assert root.CorrelationIdFilter is canonical.CorrelationIdFilter
    assert root.SafeJsonEncoder is canonical.SafeJsonEncoder
    assert root.CORRELATION_ID_CTX is canonical.CORRELATION_ID_CTX
    assert root.setup_logging is canonical.setup_logging
    assert root.get_correlation_id is canonical.get_correlation_id


def test_correlation_id_scoping():
    from agent_core.logging_config import (
        CorrelationIdContext,
        get_correlation_id,
    )

    assert get_correlation_id() == "no-correlation-id"
    with CorrelationIdContext("run-42") as corr_id:
        assert corr_id == "run-42"
        assert get_correlation_id() == "run-42"
    assert get_correlation_id() == "no-correlation-id"


def test_setup_logging_dev_human_formatter():
    from agent_core.logging_config import HumanReadableFormatter, setup_logging

    setup_logging(level=logging.DEBUG, mode="dev")
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, HumanReadableFormatter)


def test_setup_logging_prod_json_formatter():
    from agent_core.logging_config import JsonFormatter, setup_logging

    setup_logging(level=logging.INFO, mode="prod")
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)


def test_log_file_handler_writes_json(tmp_path):
    from agent_core.logging_config import JsonFormatter, setup_logging

    log_file = tmp_path / "agent.log"
    setup_logging(level=logging.DEBUG, mode="prod", log_file=log_file)
    file_handlers = [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
    ]
    assert len(file_handlers) == 1
    assert isinstance(file_handlers[0].formatter, JsonFormatter)

    logging.getLogger("test.module").warning("hello %s", "world")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello world" in content
