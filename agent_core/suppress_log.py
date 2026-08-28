"""Shared best-effort exception suppressor.

Centralizes the "log and swallow" pattern so inline
``except Exception: logger.<level>(..., traceback.format_exc())`` blocks can
collapse to a single ``with _suppress_and_log(label):``. The definition lives in
this stdlib-only module to avoid a circular import between ``agent`` and the rest
of ``agent_core``. The issues detector deliberately skips this function's own
body (this module is the sink, not a finding).
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from typing import Iterator

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_and_log(label: str) -> Iterator[None]:
    """Run the block, logging (not propagating) any exception.

    Use as a context manager around best-effort work that must not abort the
    surrounding flow even if it fails::

        with _suppress_and_log("could not flush telemetry"):
            send_telemetry()

    The exception is logged at WARNING with its traceback and then swallowed.
    """
    try:
        yield
    except Exception:  # noqa: BLE001 - deliberate best-effort swallow
        logger.warning("%s\n%s", label, traceback.format_exc())
