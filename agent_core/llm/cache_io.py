import json
from pathlib import Path

from .types import RetryParams

CACHE_FILENAME = ".implement_cache.json"


def _cache_path(workspace: str | None = None) -> Path:
    if workspace:
        return Path(workspace) / CACHE_FILENAME
    return Path(CACHE_FILENAME)


def load_retry_params(workspace: str | None = None) -> RetryParams:
    path = _cache_path(workspace)
    if not path.exists():
        return {
            "base_delay": 1.0,
            "max_retries": 3,
            "timeout_multiplier": 2.0,
            "token_limit_multiplier": 1.5,
        }
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("cache content is not a dict")
        return {
            "base_delay": float(data.get("base_delay", 1.0)),
            "max_retries": int(data.get("max_retries", 3)),
            "timeout_multiplier": float(data.get("timeout_multiplier", 2.0)),
            "token_limit_multiplier": float(data.get("token_limit_multiplier", 1.5)),
        }
    except (json.JSONDecodeError, ValueError, OSError):
        return {
            "base_delay": 1.0,
            "max_retries": 3,
            "timeout_multiplier": 2.0,
            "token_limit_multiplier": 1.5,
        }


def save_retry_params(params: RetryParams, workspace: str | None = None) -> None:
    path = _cache_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(params))
    except OSError:
        pass