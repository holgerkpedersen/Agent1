"""Temporal utility functions for timezone normalization, duration computation, and display formatting."""

from datetime import datetime, timedelta, timezone
from typing import Optional


def normalize_timezone(timestamp: float) -> datetime:
    """Convert a Unix timestamp to a UTC-normalized datetime.

    Args:
        timestamp: A Unix epoch timestamp in seconds.

    Returns:
        A datetime object normalized to UTC timezone.
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def compute_duration_spans(start_time: float, end_time: Optional[float]) -> timedelta:
    """Compute the elapsed time between two timestamps.

    Args:
        start_time: The starting Unix timestamp in seconds.
        end_time: The ending Unix timestamp in seconds, or None if still ongoing.

    Returns:
        A timedelta representing the duration span. If end_time is None, returns zero duration.
    """
    if end_time is None:
        return timedelta(0)
    start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
    return end_dt - start_dt


def format_display_label(dt: Optional[datetime], locale_format: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """Format a datetime into a human-readable display label.

    Args:
        dt: A datetime object to format, or None if unavailable.
        locale_format: The strftime format string for the desired output style.

    Returns:
        A formatted date string, or 'N/A' if dt is None.
    """
    if dt is None:
        return "N/A"
    return dt.strftime(locale_format)


def align_to_bucket(timestamp: float, bucket_size_seconds: int = 60) -> float:
    """Align a timestamp to the nearest lower time bucket boundary.

    Args:
        timestamp: A Unix epoch timestamp in seconds.
        bucket_size_seconds: The size of each time bucket in seconds (default: 60 for minute-level).

    Returns:
        A Unix timestamp aligned to the start of its containing bucket.
    """
    if bucket_size_seconds <= 0:
        raise ValueError("bucket_size_seconds must be a positive integer")
    return float(int(timestamp) // bucket_size_seconds * bucket_size_seconds)


def compute_bucket_range(
    start_time: float, end_time: Optional[float], bucket_size_seconds: int = 60
) -> list[float]:
    """Generate all bucket boundaries within a time range.

    Args:
        start_time: The starting Unix timestamp in seconds.
        end_time: The ending Unix timestamp in seconds, or None for open-ended ranges.
        bucket_size_seconds: The size of each time bucket in seconds (default: 60).

    Returns:
        A list of aligned bucket boundary timestamps spanning from start to end.
        If end_time is None, returns a single-element list with the aligned start.
    """
    if bucket_size_seconds <= 0:
        raise ValueError("bucket_size_seconds must be a positive integer")
    start_bucket = align_to_bucket(start_time, bucket_size_seconds)
    if end_time is None:
        return [start_bucket]
    end_bucket = align_to_bucket(end_time, bucket_size_seconds)
    buckets: list[float] = []
    current = start_bucket
    while current <= end_bucket:
        buckets.append(current)
        current += float(bucket_size_seconds)
    return buckets


def is_within_interval(
    timestamp: float, interval_start: float, interval_end: Optional[float]
) -> bool:
    """Check whether a timestamp falls within a specified time interval.

    Args:
        timestamp: The Unix epoch timestamp to check in seconds.
        interval_start: The start boundary of the interval in seconds.
        interval_end: The end boundary of the interval in seconds, or None for open-ended intervals.

    Returns:
        True if the timestamp is within [interval_start, interval_end], False otherwise.
        If interval_end is None, only checks against interval_start (timestamp >= start).
    """
    if timestamp < interval_start:
        return False
    if interval_end is not None and timestamp > interval_end:
        return False
    return True


def humanize_duration(duration: timedelta) -> str:
    """Convert a timedelta into a compact, human-readable duration string.

    Args:
        duration: A timedelta object representing an elapsed time span.

    Returns:
        A formatted string like '1h 30m' or '45s', with components omitted if zero.
    """
    total_seconds = int(duration.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def get_current_timestamp() -> float:
    """Get the current Unix timestamp normalized to UTC.

    Returns:
        The current time as a Unix epoch timestamp in seconds (float).
    """
    return datetime.now(tz=timezone.utc).timestamp()