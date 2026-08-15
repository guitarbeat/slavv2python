"""Unit tests for synthetic complexity ladder report / stop orchestration."""

from __future__ import annotations

import pytest

from slavv_python.analytics.parity.probes.synthetic_ladder_report import (
    NON_CERTIFICATION_NOTE,
    assemble_ladder_report,
    orchestrate_from_rung_results,
    soft_cap_blocks_next_rung,
)


@pytest.mark.unit
def test_ae1_first_break_on_edges_stops_without_later_rungs():
    rungs = [
        {
            "rung_id": "y_junction_32",
            "status": "match",
            "executed": True,
            "matlab_wall_sec": 10.0,
            "python_wall_sec": 12.0,
        },
        {
            "rung_id": "double_junction_32",
            "status": "first_break",
            "first_break_surface": "edges",
            "executed": True,
            "matlab_wall_sec": 11.0,
            "python_wall_sec": 13.0,
        },
    ]
    report = orchestrate_from_rung_results(rungs, created_utc="2026-08-14T00:00:00Z")
    assert report["outcome"] == "first_break"
    assert report["first_break_rung"] == "double_junction_32"
    assert report["first_break_surface"] == "edges"
    assert len(report["ladder_rungs"]) == 2
    assert "NOT Certification" in report["note"]
    assert "Phase 1" in report["note"]


@pytest.mark.unit
def test_ae2_soft_cap_full_match_banner():
    rungs = [
        {
            "rung_id": rid,
            "status": "match",
            "executed": True,
            "matlab_wall_sec": 5.0,
            "python_wall_sec": 6.0,
        }
        for rid in (
            "y_junction_32",
            "double_junction_32",
            "asymmetric_y_48",
            "y_junction_64",
        )
    ]
    report = orchestrate_from_rung_results(rungs, created_utc="2026-08-14T00:00:00Z")
    assert report["outcome"] == "soft_cap_full_match"
    assert report["soft_cap_reason"] == "end_of_ladder"
    assert report["note"] == NON_CERTIFICATION_NOTE


@pytest.mark.unit
def test_ae3_no_invented_extra_rung_after_soft_cap():
    rungs = [
        {
            "rung_id": "y_junction_64",
            "status": "match",
            "executed": True,
            "matlab_wall_sec": 5.0,
            "python_wall_sec": 6.0,
            "soft_cap_blocked": "end_of_ladder",
        }
    ]
    report = orchestrate_from_rung_results(
        rungs,
        created_utc="2026-08-14T00:00:00Z",
        planned_rung_ids=("y_junction_64",),
    )
    assert report["outcome"] == "soft_cap_full_match"
    assert report["soft_cap_reason"] == "end_of_ladder"
    assert len(report["ladder_rungs"]) == 1


@pytest.mark.unit
def test_size_soft_cap_blocks_oversized_next_rung():
    reason = soft_cap_blocks_next_rung(
        next_rung_id="y_junction_64",
        prior_matlab_wall_sec=10.0,
        prior_python_wall_sec=10.0,
        soft_size_max_dim=48,
    )
    assert reason == "size"


@pytest.mark.unit
def test_time_soft_cap_blocks_after_slow_prior_side():
    reason = soft_cap_blocks_next_rung(
        next_rung_id="double_junction_32",
        prior_matlab_wall_sec=200.0,
        prior_python_wall_sec=10.0,
        soft_time_sec=180.0,
    )
    assert reason == "time"


@pytest.mark.unit
def test_null_wall_sec_skips_that_side_time_budget():
    reason = soft_cap_blocks_next_rung(
        next_rung_id="double_junction_32",
        prior_matlab_wall_sec=None,
        prior_python_wall_sec=10.0,
        soft_time_sec=180.0,
    )
    assert reason is None


@pytest.mark.unit
def test_matlab_unavailable_is_inconclusive_not_match():
    rungs = [
        {
            "rung_id": "y_junction_32",
            "status": "inconclusive",
            "executed": True,
            "error": "MATLAB unavailable",
        }
    ]
    report = orchestrate_from_rung_results(rungs, created_utc="2026-08-14T00:00:00Z")
    assert report["outcome"] == "inconclusive"
    assert report["soft_cap_reason"] is None


@pytest.mark.unit
def test_assemble_always_includes_non_cert_note():
    report = assemble_ladder_report(
        rung_results=[],
        outcome="failed",
        created_utc="2026-08-14T00:00:00Z",
    )
    assert "NOT Certification" in report["note"]
