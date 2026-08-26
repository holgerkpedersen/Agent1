"""API package for the performance dashboard system.

This module exposes public API components used to serve dashboard data,
including endpoint handlers and response serializers.
"""

from typing import Any

__version__: str = "1.0.0"

# Public API exports - modules providing endpoints and serialization logic
public_modules: list[str] = ["endpoints", "serializers"]

# Re-export commonly used types for convenience in downstream consumers
APIResponseEnvelope: type = __import__(
    "performance_dashboard.models", fromlist=["APIResponseEnvelope"]
).APIResponseEnvelope

__all__: list[str] = [
    "__version__",
    "public_modules",
    "APIResponseEnvelope",
    "endpoints",
    "serializers",
]


def get_api_version() -> str:
    """Return the current API version string."""
    return __version__


def list_public_modules() -> list[str]:
    """List available public submodules within the api package."""
    return public_modules[:]


# Lazy-loaded submodule references to avoid circular import issues
endpoints: Any = None
serializers: Any = None

try:
    endpoints = __import__(
        "performance_dashboard.api.endpoints", fromlist=["*"]
    )
except ImportError:
    print("Silenced exception in __init__.py:46")

try:
    serializers = __import__(
        "performance_dashboard.api.serializers", fromlist=["*"]
    )
except ImportError:
    print("Silenced exception in __init__.py:53")


def is_module_available(module_name: str) -> bool:
    """Check whether a named submodule has been successfully imported."""
    return module_name in public_modules and globals().get(
        module_name, None
    ) is not None