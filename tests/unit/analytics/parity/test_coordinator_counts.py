"""Tests for canonical parity count helpers."""

from __future__ import annotations

import numpy as np
import pytest
from tests.support.run_state_builders import materialize_checkpoint_surface

from slavv_python.analytics.parity.counts import (
    extract_matlab_counts,
    extract_source_python_counts,
    read_python_counts_from_run,
)
from slavv_python.schema.results import EdgeSet, NetworkResult, VertexSet


@pytest.mark.unit
def test_extract_matlab_counts_prefers_nested_matlab_block() -> None:
    counts = extract_matlab_counts(
        {
            "matlab": {"vertices_count": 4, "edges_count": 5, "strand_count": 3},
            "matlab_counts": {"vertices": 99, "edges": 99, "strands": 99},
        }
    )
    assert counts.vertices == 4
    assert counts.edges == 5
    assert counts.strands == 3


@pytest.mark.unit
def test_extract_matlab_counts_legacy_flat_keys() -> None:
    counts = extract_matlab_counts({"matlab_counts": {"vertices": 7, "edges": 8, "strands": 9}})
    assert counts.vertices == 7
    assert counts.edges == 8
    assert counts.strands == 9


@pytest.mark.unit
def test_read_python_counts_from_run_uses_typed_checkpoints(tmp_path) -> None:
    run_root = tmp_path / "run"
    materialize_checkpoint_surface(run_root, stages=("vertices", "edges", "network"))

    lumen_radius_pixels = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    lumen_radius_microns = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    vertices = VertexSet.create(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([1], dtype=np.int16),
        np.array([-1.0], dtype=np.float32),
        lumen_radius_pixels,
        lumen_radius_microns,
    )
    edges = EdgeSet.create(
        traces=[],
        connections=np.array([[0, 1]], dtype=np.int32),
        energies=np.array([-1.0], dtype=np.float32),
    )
    network = NetworkResult.create(
        strands=[np.array([0, 1], dtype=np.int32)],
        bifurcations=np.empty((0, 3), dtype=np.int32),
        vertex_degrees=np.array([1, 1], dtype=np.int32),
    )

    checkpoints = run_root / "02_Output" / "python_results" / "checkpoints"
    vertices.save(checkpoints / "checkpoint_vertices.pkl")
    edges.save(checkpoints / "checkpoint_edges.pkl")
    network.save(checkpoints / "checkpoint_network.pkl")

    counts = read_python_counts_from_run(run_root)
    assert counts.vertices == 1
    assert counts.edges == 1
    assert counts.strands == 1


@pytest.mark.unit
def test_extract_source_python_counts_nested_and_legacy() -> None:
    nested = extract_source_python_counts(
        {
            "python": {
                "vertices_count": 2,
                "edges_count": 3,
                "network_strands_count": 4,
            }
        }
    )
    assert nested.strands == 4

    legacy = extract_source_python_counts(
        {"python_counts": {"vertices": 1, "edges": 2, "strands": 3}}
    )
    assert legacy.strands == 3
