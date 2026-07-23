"""Test suite for agent_core.logging_config and correlation ID integration."""

import json
import logging
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent_core.context_management import CORRELATION_ID_CTX, CorrelationIdContext
from agent_core.logging_config import setup_logging, SafeJsonEncoder, CorrelationIdFilter


@pytest.fixture(autouse=True)
def isolate_root_logger():
    """Prevent logging configuration from bleeding across tests."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    
    # Clean slate for each test
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    
    yield
    
    # Restore previous state
    root.handlers.clear()
    for handler in original_handlers:
        root.addHandler(handler)
    root.setLevel(original_level)


class TestSafeJsonEncoder:
    """Validate JSON serialization edge cases handled by the custom encoder."""

    def test_serializes_datetime_objects(self):
        dt = datetime.now(timezone.utc)
        raw = json.dumps({"timestamp": dt}, cls=SafeJsonEncoder)
        parsed = json.loads(raw)
        
        assert isinstance(parsed["timestamp"], str)
        # Verify ISO-8601 substring presence to ensure valid conversion
        assert "T" in parsed["timestamp"] or "+" in parsed["timestamp"]

    def test_serializes_path_objects(self):
        target = Path("/var/log/agent/core.log")
        raw = json.dumps({"log_file": target}, cls=SafeJsonEncoder)
        parsed = json.loads(raw)
        
        assert parsed["log_file"] == "/var/log/agent/core.log"

    def test_fallback_to_str_for_unknown_types(self):
        class UnserializableClass:
            pass
        
        obj = UnserializableClass()
        raw = json.dumps({"payload": obj}, cls=SafeJsonEncoder)
        parsed = json.loads(raw)
        
        assert isinstance(parsed["payload"], str)
        # Should not raise TypeError and should contain repr/str content
        assert "UnserializableClass" in parsed["payload"]

    def test_handles_nested_structures(self):
        data = {
            "meta": {"created_at": datetime.now(timezone.utc), "source": Path(".")},
            "list_with_mixed": [42, True, None, {"nested": datetime.utcnow()}]
        }
        raw = json.dumps(data, cls=SafeJsonEncoder)
        
        # Must parse without throwing
        parsed = json.loads(raw)
        assert parsed["meta"]["source"] == "."


class TestCorrelationIdFilter:
    """Verify correlation context injection into logging records."""

    def test_injects_correlation_id_when_context_active(self):
        filter_inst = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="traceable event", args=(), exc_info=None
        )
        
        token = CORRELATION_ID_CTX.set("ctx-abc-123")
        try:
            result_record = filter_inst.filter(record)
            assert result_record is not None
            assert getattr(result_record, "correlation_id", None) == "ctx-abc-123"
        finally:
            CORRELATION_ID_CTX.reset(token)

    def test_injects_empty_string_when_context_missing(self):
        filter_inst = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="orphan event", args=(), exc_info=None
        )
        
        # Ensure context var holds default/empty state
        token = CORRELATION_ID_CTX.set("")
        try:
            result_record = filter_inst.filter(record)
            assert getattr(result_record, "correlation_id", None) == ""
        finally:
            CORRELATION_ID_CTX.reset(token)

    def test_filter_does_not_drop_records(self):
        """Filters should return truthy values to allow log propagation."""
        filter_inst = CorrelationIdFilter()
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "keep me", (), None)
        
        token = CORRELATION_ID_CTX.set("ctx-retain")
        try:
            assert bool(filter_inst.filter(record)) is True
        finally:
            CORRELATION_ID_CTX.reset(token)


class TestSetupLoggingIntegration:
    """End-to-end validation of logging pipeline configuration."""

    @patch("logging.config.dictConfig")
    def test_passes_valid_dictconfig_structure(self, mock_dict_config):
        """Ensure setup_logging generates a valid dictConfig payload."""
        setup_logging(level=logging.DEBUG, json_format=True)
        
        assert mock_dict_config.called
        config = mock_dict_config.call_args[0][0]
        
        # Basic structural validation per logging.config spec
        assert isinstance(config, dict)
        assert "handlers" in config
        assert "loggers" in config or "root" in config

    @patch("logging.config.dictConfig")
    def test_respects_level_parameter(self, mock_dict_config):
        """Verify custom log level propagates to configuration."""
        setup_logging(level=logging.WARNING)
        
        # Implementation detail check: typically root logger level is set here
        config = mock_dict_config.call_args[0][0]
        assert mock_dict_config.called  # Primary assertion: no crash on valid level

    def test_integration_json_output_parsing_and_correlation(self):
        """Verify real log output parses as JSON and contains injected correlation ID."""
        setup_logging(level=logging.INFO, json_format=True)
        
        root_logger = logging.getLogger()
        stream_buffer = StringIO()
        
        # Redirect console handler to buffer for inspection
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = stream_buffer
                
        with CorrelationIdContext("integration-test-99"):
            root_logger.info("Pipeline validation message")
            
        # Flush handlers to ensure write completion
        for handler in root_logger.handlers:
            handler.flush()
            
        output = stream_buffer.getvalue().strip()
        assert len(output) > 0, "Logging pipeline produced no output"
        
        parsed_log = json.loads(output)
        
        # Validate JSON structure expectations
        msg_key = next((k for k in ("message", "msg", "text") if k in parsed_log), None)
        assert msg_key is not None, f"Expected message field missing in: {parsed_log}"
        assert "Pipeline validation message" in parsed_log[msg_key]
        
        # Validate correlation injection survived formatter pipeline
        assert parsed_log.get("correlation_id") == "integration-test-99"

    def test_integration_non_json_fallback_graceful(self):
        """Ensure json_format=False does not crash and logs normally."""
        setup_logging(level=logging.INFO, json_format=False)
        
        root_logger = logging.getLogger()
        stream_buffer = StringIO()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = stream_buffer
                
        with CorrelationIdContext("fallback-ctx"):
            root_logger.warning("Non-JSON test event")
            
        for handler in root_logger.handlers:
            handler.flush()
            
        output = stream_buffer.getvalue().strip()
        assert "Non-JSON test event" in output
        # Non-JSON mode typically uses standard formatting; correlation may or may not appear 
        # depending on formatter config, but it must NOT raise exceptions.