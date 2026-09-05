"""Plan lifecycle manager for state transitions and JSONL audit logging.

This module handles the atomic transitions of a plan through its lifecycle:
proposed -> executing -> executed.

It provides:
* :class:`PlanLifecycleManager`: Handles state transitions and file renaming.
* :func:`append_log`: Appends audit entries to a `.plans.jsonl` file.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .plan_schema import PlanStatus, PlanTransition, PlanLogEntry

class PlanLifecycleManager:
    """Manages the state transitions and physical file layout of a plan."""
    
    def __init__(self, plan_dir: Path, workspace_root: Path):
        self.plan_dir = plan_dir
        self.workspace_root = workspace_root
        self.log_file = plan_dir / ".plans.jsonl"
        
    def start_plan(self) -> Path:
        """Transition a plan from PROPOSED to EXECUTING.
        
        Renames plan_proposed.md to plan_executing.md.
        Returns the new path.
        """
        src = self.plan_dir / "plan_proposed.md"
        dst = self.plan_dir / "plan_executing.md"
        
        if not src.exists():
            raise FileNotFoundError(f"No proposed plan found at {src}")
            
        shutil.move(str(src), str(dst))
        self._log_transition(src, PlanTransition.START, PlanStatus.EXECUTING)
        return dst
        
    def finish_plan(self) -> Path:
        """Transition a plan from EXECUTING to EXECUTED.
        
        Renames plan_executing.md to plan_executed_<timestamp>.md.
        Returns the new path.
        """
        src = self.plan_dir / "plan_executing.md"
        if not src.exists():
            # Fallback for plans skipped through starting phase
            src = self.plan_dir / "plan_proposed.md"
            if not src.exists():
                raise FileNotFoundError(f"No executing plan found in {self.plan_dir}")
                
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dst = self.plan_dir / f"plan_executed_{ts}.md"
        
        shutil.move(str(src), str(dst))
        self._log_transition(dst, PlanTransition.FINISH, PlanStatus.EXECUTED)
        return dst
        
    def _log_transition(self, path: Path, transition: PlanTransition, status: PlanStatus):
        entry = PlanLogEntry(
            plan_id=path.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            transition=transition,
            status=status,
            details=f"File: {path}"
        )
        append_log(self.log_file, entry)

def append_log(log_path: Path, entry: PlanLogEntry):
    """Append a structured log entry to the JSONL file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict()) + "\n")
