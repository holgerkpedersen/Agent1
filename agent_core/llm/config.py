from enum import Enum

from agent_core.llm.llm_types import ProfileType as ProfileType
from agent_core.llm.llm_types import TaskType

__all__ = ["ProfileType", "TaskType", "TASK_PROFILE_MAP", "DEFAULT_RETRY_PARAMS"]


TASK_PROFILE_MAP: dict[TaskType, ProfileType] = {
    TaskType.IMPLEMENT: ProfileType.FAST_CODEGEN,
    TaskType.FIX: ProfileType.FAST_CODEGEN,
    TaskType.ANALYZE: ProfileType.DEEP_ANALYSIS,
    TaskType.WORKFLOW: ProfileType.DEEP_ANALYSIS,
    TaskType.OPTIMIZE: ProfileType.PRECISE,
    TaskType.PERF_TUNING: ProfileType.PRECISE,
}


DEFAULT_RETRY_PARAMS = {
    "base_delay": 1.0,
    "max_retries": 3,
    "timeout_multiplier": 2.0,
    "token_limit_multiplier": 1.5,
}