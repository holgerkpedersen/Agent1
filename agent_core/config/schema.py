import json
import os
from pathlib import Path
from typing import Any, Optional, List, Dict
from pydantic import BaseModel

class Config(BaseModel):
    log_level: str = "INFO"
    debug: bool = False
    timeout: int = 30

class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "admin"
    password: str = ""
    name: str = "db"

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4

class AppConfig(BaseModel):
    config: Config
    database: DatabaseConfig
    server: ServerConfig

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path: Optional[str] = config_path
        self.app_config: Optional[AppConfig] = None
        self._watchers: List[Path] = []

    def _convert_value(self, value: str) -> Any:
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def _set_nested(self, data: Dict[str, Any], key: str, value: str) -> None:
        converted = self._convert_value(value)
        parts = key.split("__")
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = converted

    def config(self, value: Config) -> None:
        if self.app_config is None:
            self.app_config = AppConfig(
                config=value,
                database=DatabaseConfig(),
                server=ServerConfig()
            )
        else:
            self.app_config.config = value

    def load_from_env(self, prefix: str = "AGENT_") -> None:
        data: Dict[str, Any] = {}
        for env_key, env_val in os.environ.items():
            if env_key.startswith(prefix):
                clean_key = env_key[len(prefix):].lower()
                self._set_nested(data, clean_key, env_val)
        self._apply_data(data)

    def load_from_file(self, path: Optional[str] = None) -> None:
        path_to_load = path or self._config_path
        if not path_to_load or not os.path.exists(path_to_load):
            return
        with open(path_to_load, "r") as f:
            data = json.load(f)
            self._apply_data(data)

    def reload(self) -> None:
        self.load_from_file()
        self.load_from_env()

    def _apply_data(self, data: Dict[str, Any]) -> None:
        if self.app_config is None:
            # Create from dict if none exists
            self.app_config = AppConfig(**data)
        else:
            # Update existing model fields
            for key, value in data.items():
                if hasattr(self.app_config, key):
                    setattr(self.app_config, key, value)

    @classmethod
    def validate_log_level(cls, value: str) -> str:
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        return normalized if normalized in levels else "INFO"

    def watchers(self) -> List[Path]:
        return self._watchers