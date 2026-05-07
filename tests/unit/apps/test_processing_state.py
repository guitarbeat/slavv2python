from __future__ import annotations

from slavv_python.apps.state.processing import (
    build_processing_run_dir,
    load_processing_snapshot,
    store_processing_session_state,
    summarize_processing_metrics,
)
from slavv_python.runtime.run_state import RunSnapshot
from tests.support.payload_builders import build_processing_results


def test_build_processing_run_dir_varies_with_validated_parameters():
    upload_bytes = b"same-uploaded-file"

    first = build_processing_run_dir(
        upload_bytes,
        {"radius_of_smallest_vessel_in_microns": 1.5},
    )
    second = build_processing_run_dir(
        upload_bytes,
        {"radius_of_smallest_vessel_in_microns": 2.0},
    )
    repeated = build_processing_run_dir(
        upload_bytes,
        {"radius_of_smallest_vessel_in_microns": 1.5},
    )

    assert first != second
    assert first == repeated


def test_load_processing_snapshot_returns_none_without_run_dir():
    assert load_processing_snapshot({}, snapshot_loader=lambda run_dir: run_dir) is None


def test_load_processing_snapshot_uses_snapshot_loader():
    seen_run_dirs: list[str] = []

    snapshot = load_processing_snapshot(
        {"current_run_dir": "run-dir"},
        snapshot_loader=lambda run_dir: (
            seen_run_dirs.append(run_dir) or RunSnapshot(run_id="run-123", stages={})
        ),
    )

    assert seen_run_dirs == ["run-dir"]
    assert snapshot is not None
    assert snapshot.run_id == "run-123"


def test_summarize_processing_metrics_counts_normalized_payload_parts():
    metrics = summarize_processing_metrics(build_processing_results())

    assert metrics == {
        "vertices": 3,
        "edges": 2,
        "strands": 1,
        "bifurcations": 0,
    }


def test_store_processing_session_state_persists_processing_outputs():
    session_state = {
        "curation_baseline_counts": {"vertices": 3},
        "last_curation_mode": "interactive",
        "share_report_prepared_signature": "abc123",
    }
    snapshot = RunSnapshot(run_id="run-123", stages={})
    processing_results = build_processing_results(
        overrides={"metadata": {"slavv_python": "processing"}}
    )

    store_processing_session_state(
        session_state,
        results=processing_results,
        validated_params={"radius_of_smallest_vessel_in_microns": 1.5},
        image_shape=(6, 5, 4),
        dataset_name="sample.tif",
        run_dir="run-dir",
        final_snapshot=snapshot,
    )

    assert session_state["processing_results"]["metadata"] == {"slavv_python": "processing"}
    assert session_state["parameters"] == {"radius_of_smallest_vessel_in_microns": 1.5}
    assert session_state["image_shape"] == (6, 5, 4)
    assert session_state["dataset_name"] == "sample.tif"
    assert session_state["current_run_dir"] == "run-dir"
    assert session_state["run_snapshot"]["run_id"] == "run-123"
    assert "curation_baseline_counts" not in session_state
    assert "last_curation_mode" not in session_state
    assert "share_report_prepared_signature" not in session_state
