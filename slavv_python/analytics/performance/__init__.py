"""Phase 2 performance helpers (profiling baselines; no unwind)."""

from __future__ import annotations

from slavv_python.analytics.performance.energy_n_jobs import (
    parse_n_jobs_cli_value,
    recommend_energy_n_jobs,
    recommend_energy_n_jobs_from_host,
    resolve_cli_n_jobs,
)
from slavv_python.analytics.performance.phase2_baseline import (
    PIPELINE_STAGES,
    StageTiming,
    baseline_payload,
    bottleneck_measured,
    parse_stage_metrics,
)

__all__ = [
    "PIPELINE_STAGES",
    "StageTiming",
    "baseline_payload",
    "bottleneck_measured",
    "parse_n_jobs_cli_value",
    "parse_stage_metrics",
    "recommend_energy_n_jobs",
    "recommend_energy_n_jobs_from_host",
    "resolve_cli_n_jobs",
]
