"""Type constants shared across agent modules.

Extracted from agent.py as the t00 types concern (step t01).
"""

#: Threshold for the cross-session goal relevance check.
#: When a new user prompt overlaps less than this fraction of content words
#: with the recent user-prompt window, the session goal is considered
#: unrelated and the history is compacted (decision #094).
_GOAL_RELATED_THRESHOLD: float = 0.25
