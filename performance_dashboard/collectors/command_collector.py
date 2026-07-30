"""Collectors that capture command execution statistics as CommandMetric records."""

import re
import resource
import subprocess
import time

from performance_dashboard.models import CommandMetric, PerformanceRecord
from performance_dashboard.utils.time_utils import align_to_bucket, get_current_timestamp
from performance_dashboard.utils.validation import MetricSchemaValidator, validate_command_metric


_MEMORY_PATTERN = re.compile(
    r"Memory\s*usage:\s*(?P<value>[\d.]+)\s*(?P<unit>MB|KiB)", re.IGNORECASE
)
_TIME_PATTERN = re.compile(r"Execution\s*time:\s*(?P<value>[\d.]+)\s*ms", re.IGNORECASE)
_CPU_PATTERN = re.compile(
    r"(?:CPU\s*utilization|cpu):\s*(?P<value>[\d.]+)%", re.IGNORECASE
)
_COMMAND_PATTERN = re.compile(r"Command:\s*(?P<name>\S+)", re.IGNORECASE)
_RETURN_CODE_PATTERN = re.compile(
    r"Return\s*code:\s*(?P<code>-?\d+)", re.IGNORECASE
)


def _convert_memory_unit(value: float, unit: str) -> float:
    """Normalize memory quantities reported in KiB into MiB."""
    if unit.startswith("k"):
        return value / 1024.0
    return value


def parse_execution_logs(log_text: str) -> dict[str, object]:
    """Extract timing/memory/CPU/command fields from structured command logs."""
    extracted: dict[str, object] = {}

    time_match = _TIME_PATTERN.search(log_text)
    if time_match is not None and isinstance(time_match.group("value"), str):
        extracted["execution_time_ms"] = float(time_match.group("value"))

    memory_match = _MEMORY_PATTERN.search(log_text)
    if memory_match is not None:
        value_str = memory_match.group("value")
        unit_str = memory_match.group("unit")
        if isinstance(value_str, str) and isinstance(unit_str, str):
            extracted["memory_usage_mb"] = _convert_memory_unit(
                float(value_str), unit_str.lower()
            )

    cpu_match = _CPU_PATTERN.search(log_text)
    if cpu_match is not None and isinstance(cpu_match.group("value"), str):
        extracted["cpu_utilization_percent"] = float(cpu_match.group("value"))

    command_match = _COMMAND_PATTERN.search(log_text)
    if command_match is not None and isinstance(command_match.group("name"), str):
        extracted["command_name"] = str(command_match.group("name"))

    return_code_match = _RETURN_CODE_PATTERN.search(log_text)
    if return_code_match is not None and isinstance(return_code_match.group("code"), str):
        extracted["return_code"] = int(return_code_match.group("code"))

    return extracted


def normalize_metric_formats(raw: dict[str, object] | list[object]) -> CommandMetric | None:
    """Convert arbitrary metric representations into a validated CommandMetric.

    Uses isinstance narrowing guards to reject malformed inputs (wrong field
    count in positional lists, non-numeric timing values, boolean return codes).
    """
    validator = MetricSchemaValidator()

    if isinstance(raw, list):
        if len(raw) != 5:
            return None
        exec_time = raw[0]
        memory = raw[1]
        cpu = raw[2]
        command_name = raw[3]
        return_code = raw[4]
    else:
        # Remaining union member narrows to dict[str, object].
        exec_time = raw.get("execution_time_ms")
        memory = raw.get("memory_usage_mb")
        cpu = raw.get("cpu_utilization_percent")
        command_name = raw.get("command_name")
        return_code = raw.get("return_code")

    if not isinstance(exec_time, (int, float)):
        return None
    if not isinstance(memory, (int, float)):
        return None
    if not isinstance(cpu, (int, float)):
        return None
    if not isinstance(command_name, str):
        return None
    # bool is a subtype of int; reject it as a malformed exit status.
    if not isinstance(return_code, int) or isinstance(return_code, bool):
        return None

    metric: CommandMetric = {
        "execution_time_ms": float(exec_time),
        "memory_usage_mb": float(memory),
        "cpu_utilization_percent": float(cpu),
        "command_name": str(command_name),
        "return_code": int(return_code),
    }

    if not validate_command_metric(validator, metric):
        return None
    return metric


def collect_command_metrics(
    command_name: str,
    command_args: list[str],
    timeout_seconds: float | None = None,
) -> CommandMetric:
    """Run a command and measure runtime, memory footprint, CPU usage, exit code."""
    start_perf = time.perf_counter()

    completed = subprocess.run(
        command_args, capture_output=True, timeout=timeout_seconds
    )

    elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    memory_mb = float(usage.ru_maxrss) / 1024.0
    cpu_seconds = float(usage.ru_utime + usage.ru_stime)
    wall_seconds = elapsed_ms / 1000.0
    cpu_percent = (cpu_seconds / wall_seconds * 100.0) if wall_seconds > 0 else 0.0

    metric: CommandMetric = {
        "execution_time_ms": float(elapsed_ms),
        "memory_usage_mb": memory_mb,
        "cpu_utilization_percent": cpu_percent,
        "command_name": str(command_name),
        "return_code": int(completed.returncode),
    }

    validator = MetricSchemaValidator()
    if not validate_command_metric(validator, metric):
        raise ValueError("collected command metrics failed schema validation")

    return metric


def build_performance_record(
    metric: CommandMetric,
    source_identifier: str,
    extra_metadata: dict[str, str] | None = None,
) -> PerformanceRecord:
    """Wrap a collected CommandMetric into a timestamped PerformanceRecord."""
    timestamp = get_current_timestamp()
    bucketed = align_to_bucket(timestamp, 60)

    metadata: dict[str, str] = {"timestamp_bucket": str(int(bucketed))}
    if extra_metadata is not None:
        metadata.update(extra_metadata)

    record: PerformanceRecord = {
        "timestamp": float(timestamp),
        "record_type": metric,
        "source_identifier": str(source_identifier),
        "metadata": metadata,
    }
    return record