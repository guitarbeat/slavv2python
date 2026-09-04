"""Phase 2 performance helpers (profiling baselines; no unwind)."""

from __future__ import annotations

from slavv_python.analytics.performance.edge_timing import (
    SCHEMA_VERSION as EDGE_TIMING_SCHEMA_VERSION,
)
from slavv_python.analytics.performance.edge_timing import (
    EdgeTimingRecord,
    build_edge_timing_payload,
    write_edge_timing,
)
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
    "EDGE_TIMING_SCHEMA_VERSION",
    "PIPELINE_STAGES",
    "EdgeTimingRecord",
    "StageTiming",
    "baseline_payload",
    "bottleneck_measured",
    "build_edge_timing_payload",
    "parse_n_jobs_cli_value",
    "parse_stage_metrics",
    "recommend_energy_n_jobs",
    "recommend_energy_n_jobs_from_host",
    "resolve_cli_n_jobs",
    "write_edge_timing",
]
