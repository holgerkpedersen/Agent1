"""Best-effort suppression context manager extracted from agent.py (step t10-suppress).

One function::

    _suppress_and_log(label)  — run block, log (never propagate) any exception

Exported from agent_core.bootstrap.__all__ so existing call sites in agent.py
keep working without adding their own import.
"""

from __future__ import annotations

import logging
import traceback
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_and_log(label: str) -> Iterator[None]:
    """Run the block, logging (not propagating) any exception.

    Centralises the "best-effort: log and continue" pattern so call sites have
    a single, duplication-proof handler instead of an inline ``except Exception``
    that an edit can accidentally clone.
    """
    try:
        yield
    except Exception:  # noqa: BLE001 - deliberate best-effort swallow
        logger.warning("%s\n%s", label, traceback.format_exc())
