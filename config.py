"""
config.py
Centralized typed configuration management using pydantic-settings.
Loads environment variables from .env, validates constraints at startup, 
and exposes a singleton instance for dependency injection across modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore[import-untyped]
except ImportError:
    # Fallback for older Pydantic v1 environments where settings are built-in
    from pydantic import BaseSettings
    
    class _LegacyBaseSettings(BaseSettings):
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"

    class SettingsConfigDict(dict):
        pass


class AgentConfig(BaseSettings):
    """
    Typed configuration model for the agent runtime.
    
    Attributes:
        model_name: Identifier for the LLM to use (e.g., 'gpt-4o', 'claude-3-opus').
        workspace_root: Absolute path to the sandboxed workspace directory.
        http_timeout_connect: Max seconds to wait for initial TCP handshake.
        http_timeout_read: Max seconds to wait for data transfer after connection.
        max_index_size: Maximum number of entries allowed in the semantic index before eviction.
        chunk_overlap_bytes: Number of bytes to overlap between search chunks to prevent boundary misses.
        log_level: Logging verbosity level (DEBUG, INFO, WARNING, ERROR).
    """
    
    # Pydantic v2 settings configuration
    if hasattr(BaseSettings, 'model_config'):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False
        )

    # --- LLM Configuration ---
    model_name: str = "gpt-4o"
    api_key: str = ""
    
    # --- Workspace & File I/O ---
    workspace_root: Path = Path("/c/Dev/Agent1")
    chunk_overlap_bytes: int = 512
    
    @property
    def is_workspace_valid(self) -> bool:
        """Ensure workspace root is an absolute path."""
        return self.workspace_root.is_absolute()

    # --- HTTP Timeouts (seconds) ---
    http_timeout_connect: float = 5.0
    http_timeout_read: float = 30.0
    
    # --- Indexing & Memory Management ---
    max_index_size: int = 5000
    
    # --- Logging Configuration ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# ---------------------------------------------------------------------------
# Singleton Instance Initialization & Validation
# ---------------------------------------------------------------------------

def _load_config() -> AgentConfig:
    """Load configuration and perform startup validation checks."""
    cfg = AgentConfig()
    
    if not cfg.is_workspace_valid:
        raise ValueError(
            f"Configuration Error: 'workspace_root' must be an absolute path. "
            f"Received relative path: '{cfg.workspace_root}'. "
            f"Please update your .env file or environment variables."
        )
        
    return cfg


# Global singleton instance exported for DI across modules
config = _load_config()