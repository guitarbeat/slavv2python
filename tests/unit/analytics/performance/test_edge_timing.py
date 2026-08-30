from __future__ import annotations

import json

from slavv_python.analytics.performance.edge_timing import (
    SCHEMA_VERSION,
    build_edge_timing_payload,
    write_edge_timing,
)


def test_edge_timing_payload_splits_discovery_and_selection() -> None:
    payload = build_edge_timing_payload(
        discovery_seconds=2.5,
        selection_seconds=0.75,
        candidate_count=12,
        edge_count=4,
        exact_route=True,
        writer_authorized=True,
        started_at="2026-08-30T01:00:00Z",
        completed_at="2026-08-30T01:00:04Z",
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["profile"] == "exact-route"
    assert payload["discovery_strategy"] == "watershed"
    assert payload["precision"] == "float64"
    assert payload["candidate_count"] == 12
    assert payload["edge_count"] == 4
    assert payload["total_seconds"] == 3.25
    assert payload["spans"] == {
        "watershed_discovery_seconds": 2.5,
        "edge_selection_seconds": 0.75,
    }


def test_edge_timing_persists_hash_sidecar(tmp_path) -> None:
    path = tmp_path / "phase2_edges_split.json"
    payload = build_edge_timing_payload(
        discovery_seconds=1,
        selection_seconds=2,
        candidate_count=3,
        edge_count=2,
        exact_route=False,
        writer_authorized=True,
    )

    assert write_edge_timing(path, payload) == path
    assert json.loads(path.read_text(encoding="utf-8"))["profile"] == "paper"
    sidecar = path.with_name(f"{path.name}.sha256")
    assert sidecar.is_file()
    assert len(sidecar.read_text(encoding="utf-8").strip()) == 64
