"""Centralized timeout constants for the agent framework.

All timeout values used across the codebase are defined here.
Provider-specific overrides via environment variables are handled
in the respective provider modules, but the defaults live here.
"""

# HTTP connection timeout (seconds) — for establishing TCP connections
HTTP_CONNECT_TIMEOUT: float = 5.0

# HTTP read/request timeout (seconds) — for waiting on responses
HTTP_READ_TIMEOUT: float = 30.0

# Management/health-check timeouts (seconds) — quick probes
HEALTH_CHECK_TIMEOUT: float = 10.0
MODEL_REFRESH_TIMEOUT: float = 15.0

# Chat/LLM API timeout (seconds) — default for all providers
# Override via provider-specific env vars (LMSTUDIO_CHAT_TIMEOUT, etc.)
DEFAULT_CHAT_TIMEOUT: float = 600.0

# Model loading timeout (seconds) — loading large models can be slow
MODEL_LOAD_TIMEOUT: float = 300.0

# Command execution timeouts (seconds)
SEARCH_COMMAND_TIMEOUT: float = 30.0
COMPILATION_CHECK_TIMEOUT: float = 30.0

__all__ = [
    "HTTP_CONNECT_TIMEOUT",
    "HTTP_READ_TIMEOUT",
    "HEALTH_CHECK_TIMEOUT",
    "MODEL_REFRESH_TIMEOUT",
    "DEFAULT_CHAT_TIMEOUT",
    "MODEL_LOAD_TIMEOUT",
    "SEARCH_COMMAND_TIMEOUT",
    "COMPILATION_CHECK_TIMEOUT",
]
