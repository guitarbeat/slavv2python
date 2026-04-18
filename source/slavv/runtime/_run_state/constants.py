from __future__ import annotations

PREPROCESS_STAGE = "preprocess"
PIPELINE_STAGES = ["energy", "vertices", "edges", "network"]
TRACKED_RUN_STAGES = [PREPROCESS_STAGE, *PIPELINE_STAGES]
STAGE_WEIGHTS = {
    PREPROCESS_STAGE: 0.05,
    "energy": 0.35,
    "vertices": 0.15,
    "edges": 0.30,
    "network": 0.15,
}
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_TARGET = "completed_target"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "resume_blocked"
