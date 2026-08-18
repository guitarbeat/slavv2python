"""CI-safe tests for Phase 2 profiling baseline extraction."""

from __future__ import annotations

import json
from pathlib import Path

from slavv_python.analytics.parity.constants import LIVE_DEST_NAMES
from slavv_python.analytics.performance.phase2_baseline import (
    CARRIED_REASON,
    ENERGY_HISTORICAL_NOTE,
    baseline_payload,
    bottleneck_measured,
    is_measured_elapsed,
    parse_stage_metrics,
    peak_memory_mib,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRACKED = _REPO_ROOT / "docs" / "reference" / "core" / "phase2-profiling-baseline.json"


def test_zero_elapsed_is_not_measured() -> None:
    assert is_measured_elapsed(0.0) is False
    assert is_measured_elapsed(5534.0) is True


def test_parse_ignores_missing_and_names_carried_energy() -> None:
    records = parse_stage_metrics(
        {
            "energy": {"elapsed_seconds": 0.0, "status": "completed", "peak_memory_bytes": 10},
            "vertices": {"elapsed_seconds": 0.0, "status": "completed"},
            "edges": {
                "elapsed_seconds": 5534.0,
                "status": "completed",
                "peak_memory_bytes": 1574629376,
            },
            "network": {"elapsed_seconds": 416.0, "status": "completed"},
        }
    )
    by_name = {item.name: item for item in records}
    assert by_name["energy"].measured is False
    assert CARRIED_REASON in by_name["energy"].to_dict()["reason"]
    assert by_name["edges"].measured is True
    assert bottleneck_measured(records) == "edges"
    assert peak_memory_mib(1574629376) == 1574629376 / (1024.0 * 1024.0)


def test_payload_refuses_unwind_and_stretch_claims() -> None:
    records = parse_stage_metrics(
        {"edges": {"elapsed_seconds": 10.0, "status": "completed"}},
    )
    payload = baseline_payload(records=records, n_jobs=6)
    assert payload["not_unwind"] is True
    assert payload["not_stretch"] is True
    assert payload["bottleneck_measured_on_dest"] == "edges"
    assert payload["bottleneck_full_pipeline_historical"] == "energy"
    assert payload["do_not_overwrite"] == list(LIVE_DEST_NAMES)
    assert ENERGY_HISTORICAL_NOTE in payload["energy_historical_note"]
    assert "auto is implemented" in payload["next_allowed"]
    assert "Edges/Network profiling" in payload["next_allowed"]


def test_tracked_baseline_schema_if_present() -> None:
    if not _TRACKED.is_file():
        return
    payload = json.loads(_TRACKED.read_text(encoding="utf-8"))
    assert payload["phase"] == 2
    assert payload["workstream"] == "profiling_baseline"
    assert payload["not_unwind"] is True
    assert payload["stages"]["energy"]["measured"] is False
    assert payload["stages"]["edges"]["measured"] is True
    assert payload["bottleneck_measured_on_dest"] == "edges"
    assert payload["do_not_overwrite"] == list(LIVE_DEST_NAMES)
    assert "auto is implemented" in payload["next_allowed"]
    assert "Edges/Network profiling" in payload["next_allowed"]
