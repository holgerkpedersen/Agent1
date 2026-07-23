"""
Type definitions for the application.

This module contains all shared type aliases, dataclasses, TypedDicts,
and enumerations used throughout the codebase to ensure type safety
and consistency across modules.
"""

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        # Fallback for older versions without typing_extensions
        TypeAlias = type


# ─────────────────────────────────────────────
# Generic Type Variables
# ─────────────────────────────────────────────

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
K = TypeVar("K")
V = TypeVar("V")


# ─────────────────────────────────────────────
# Common Type Aliases
# ─────────────────────────────────────────────

JSONPrimitive: TypeAlias = Union[str, int, float, bool, None]
JSONObject: TypeAlias = Dict[str, Any]
JSONArray: TypeAlias = List[Any]
JSONValue: TypeAlias = Union[JSONPrimitive, JSONObject, JSONArray]

StringDict: TypeAlias = Dict[str, str]
IntStrMap: TypeAlias = Dict[int, str]
PathLike: TypeAlias = Union[str, Path]

Callback: TypeAlias = Callable[..., Any]
AsyncCallback: TypeAlias = Callable[..., None]


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class LogLevel(Enum):
    """Standard logging severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Status(Enum):
    """Generic operational status codes."""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    QUEUED = auto()


class HTTPMethod(Enum):
    """Supported HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class Role(Enum):
    """User role permissions."""

    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"


class SortOrder(Enum):
    """Sorting direction for query results."""

    ASCENDING = "asc"
    DESCENDING = "desc"


# ─────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Point:
    """Immutable 2D coordinate point."""

    x: float
    y: float

    def distance(self, other: Point) -> float:
        """Calculate Euclidean distance to another point."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box defined by two corner points."""

    top_left: Point
    bottom_right: Point

    @property
    def width(self) -> float:
        return abs(self.bottom_right.x - self.top_left.x)

    @property
    def height(self) -> float:
        return abs(self.bottom_right.y - self.top_left.y)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class PaginationInfo:
    """Metadata for paginated query results."""

    page: int = 1
    per_page: int = 20
    total_items: int = 0
    total_pages: int = field(init=False)

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("Page must be at least 1")
        if self.per_page < 1:
            raise ValueError("per_page must be at least 1")
        object.__setattr__(
            self, "total_pages", max(1, -(-self.total_items // self.per_page))
        )


@dataclass
class Result(Generic[T]):
    """Generic container for operation results with success/failure tracking."""

    data: Optional[T] = None
    status: Status = Status.PENDING
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, data: T, message: str = "OK") -> Result[T]:
        """Create a successful result instance."""
        return cls(data=data, status=Status.COMPLETED, message=message)

    @classmethod
    def failure(cls, message: str = "Error", metadata: Optional[Dict[str, Any]] = None) -> Result[T]:
        """Create a failed result instance."""
        return cls(
            data=None,
            status=Status.FAILED,
            message=message,
            metadata=metadata or {},
        )

    @property
    def is_success(self) -> bool:
        return self.status == Status.COMPLETED

    @property
    def is_failure(self) -> bool:
        return self.status == Status.FAILED


@dataclass
class ConfigValue:
    """Represents a single configuration key-value pair with metadata."""

    key: str
    value: Any
    description: str = ""
    default: Any = None
    required: bool = False
    encrypted: bool = False

    def __repr__(self) -> str:
        display_value = "****" if self.encrypted else repr(self.value)
        return f"ConfigValue(key={self.key!r}, value={display_value})"


@dataclass
class ConfigSection:
    """A group of related configuration values."""

    name: str
    description: str = ""
    values: List[ConfigValue] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key from this section."""
        for val in self.values:
            if val.key == key:
                return val.value
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert the section's values into a plain dictionary."""
        return {v.key: v.value for v in self.values}


@dataclass
class AppConfig:
    """Top-level application configuration container."""

    app_name: str = "application"
    version: str = "0.1.0"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    sections: List[ConfigSection] = field(default_factory=list)

    def get_section(self, name: str) -> Optional[ConfigSection]:
        """Retrieve a configuration section by name."""
        for section in self.sections:
            if section.name == name:
                return section
        return None


@dataclass
class HTTPRequest:
    """Represents an incoming or outgoing HTTP request."""

    method: HTTPMethod = HTTPMethod.GET
    path: str = "/"
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Union[str, List[str]]] = field(default_factory=dict)
    body: Optional[bytes] = None
    content_type: Optional[str] = None

    @property
    def url(self) -> str:
        """Construct the full URL path with query string."""
        qs = "&".join(
            f"{k}={v}" if isinstance(v, str) else "&".join(f"{k}={item}" for item in v)
            for k, v in self.query_params.items()
        )
        return f"{self.path}?{qs}" if qs else self.path


@dataclass
class HTTPResponse:
    """Represents an HTTP response."""

    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    content_type: str = "application/json"

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_redirect(self) -> bool:
        return 300 <= self.status_code < 400

    @property
    def is_client_error(self) -> bool:
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        return 500 <= self.status_code < 600


@dataclass
class ValidationError:
    """Represents a single field validation error."""

    field_name: str
    message: str
    code: Optional[str] = None
    severity: Literal["warning", "error"] = "error"


@dataclass
class ValidationResult:
    """Aggregated result of validating a data structure."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.valid


@dataclass
class SortKey:
    """Describes a sort operation for query builders."""

    field_name: str
    order: SortOrder = SortOrder.ASCENDING


@dataclass
class QueryFilter:
    """Represents a filter condition for data queries."""

    field_name: str
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "contains"]
    value: Any


# ─────────────────────────────────────────────
# TypedDict Definitions (for structured dicts)
# ─────────────────────────────────────────────

try:
    from typing import TypedDict
except ImportError:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        # Minimal fallback if neither is available
        class TypedDict(dict):  # type: ignore[no-redef, misc]
            pass


class UserRecord(TypedDict, total=False):
    """Dictionary representing a user record."""

    id: int
    username: str
    email: str
    role: Role
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]


class DatabaseConfig(TypedDict, total=False):
    """Dictionary for database connection configuration."""

    host: str
    port: int
    name: str
    user: str
    password: str
    driver: str
    ssl_enabled: bool
    max_connections: int
    timeout_seconds: int


class CacheConfig(TypedDict, total=False):
    """Dictionary for cache subsystem configuration."""

    backend: Literal["memory", "redis", "memcached"]
    ttl_seconds: int
    max_size_mb: int
    prefix: str


# ─────────────────────────────────────────────
# Utility Type Helpers
# ─────────────────────────────────────────────

def flatten_result(result: Result[Result[T]]) -> Result[T]:
    """Flatten a nested Result into a single Result."""
    if not result.is_success or result.data is None:
        return Result[T](data=None, status=result.status, message=result.message)
    inner = result.data
    return Result[T](
        data=inner.data,
        status=inner.status,
        message=inner.message,
        timestamp=inner.timestamp,
        metadata={**result.metadata, **inner.metadata},
    )


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass or dict-like object into a plain dictionary."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Cannot convert {type(obj).__name__} to dict")


def merge_dicts(base: Dict[K, V], override: Dict[K, V]) -> Dict[K, V]:
    """Deep-merge two dictionaries (override takes precedence)."""

    def _deep_merge(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            merged = a.copy()
            for k, v in b.items():
                merged[k] = _deep_merge(merged.get(k), v)
            return merged
        return b

    return {k: _deep_merge(base.get(k), override.get(k)) for k in set(base) | set(override)}


# ─────────────────────────────────────────────
# Literal Type Shorthands
# ─────────────────────────────────────────────

EnvironmentLiteral: TypeAlias = Literal["development", "staging", "production"]
StorageBackendLiteral: TypeAlias = Literal["local", "s3", "gcs", "azure"]
AuthSchemeLiteral: TypeAlias = Literal["bearer", "basic", "api_key", "oauth2"]


@dataclass(frozen=True)
class Environment:
    """Immutable environment descriptor."""

    name: EnvironmentLiteral
    is_production: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_production", self.name == "production")


@dataclass
class HealthCheckResult:
    """Outcome of a service health check."""

    service_name: str
    healthy: bool
    status_code: int = 200
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSample:
    """A single metric data point."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)