from enum import Enum
from typing import TypedDict


class TaskType(Enum):
    IMPLEMENT = "implement"
    FIX = "fix"
    ANALYZE = "analyze"
    WORKFLOW = "workflow"
    OPTIMIZE = "optimize"
    PERF_TUNING = "perf_tuning"


class FailureCategory(Enum):
    TIMEOUT = "timeout"
    TOKEN_LIMIT = "token_limit"
    API_ERROR = "api_error"


class RetryParams(TypedDict):
    base_delay: float
    max_retries: int
    timeout_multiplier: float
    token_limit_multiplier: float


class ProfileType(Enum):
    FAST_CODEGEN = "fast_codegen"
    DEEP_ANALYSIS = "deep_analysis"
    PRECISE = "precise"


class VoteResult(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class PromptMetrics(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float