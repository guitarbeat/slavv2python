"""Unit tests for edge candidate diagnostic counters (Phase 1).

Verifies that frontier and watershed supplement diagnostics are populated
in the candidate payload and comparison report.
"""

import numpy as np
import pytest

from slavv.core.tracing import (
    _append_candidate_unit,
    _build_edge_candidate_audit,
    _empty_edge_diagnostics,
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates_matlab_frontier,
    _supplement_matlab_frontier_candidates_with_watershed_joins,
    _trace_local_geodesic_between_vertices,
    paint_vertex_center_image,
)
from slavv.parity.metrics import compare_edges


def _make_small_volume(size: int = 11):
    """Create a tiny negative-energy tube for deterministic edge tracing."""
    energy = np.full((size, size, size), -0.5, dtype=np.float64)
    coords = np.indices((size, size, size))
    x = coords[1] - size // 2
    z = coords[2] - size // 2
    energy -= (x**2 + z**2).astype(np.float64) * 0.1
    return energy


@pytest.mark.unit
class TestWatershedSupplementDiagnostics:
    """Diagnostics from the watershed supplement step."""

    def test_parity_all_contacts_mode_adds_negative_watershed_pair_without_frontier_seed(self):
        """The candidate-stage parity mode should admit valid watershed contacts directly."""
        energy = np.full((9, 9, 9), 1.0, dtype=np.float64)
        energy[4, 1:8, 4] = -1.0
        vertex_positions = np.array([[4.0, 1.0, 4.0], [4.0, 7.0, 4.0]], dtype=np.float32)
        candidates = {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "metrics": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "connection_sources": [],
            "diagnostics": _empty_edge_diagnostics(),
        }

        result = _finalize_matlab_parity_candidates(
            candidates,
            energy,
            None,
            vertex_positions,
            -1.0,
            {
                "number_of_edges_per_vertex": 1,
                "parity_watershed_candidate_mode": "all_contacts",
            },
        )

        assert result["connections"].tolist() == [[0, 1]]
        assert result["connection_sources"] == ["watershed"]
        assert result["diagnostics"]["watershed_join_supplement_count"] == 1
        assert result["diagnostics"]["watershed_accepted"] == 1

    def test_parity_geodesic_salvage_adds_candidate_for_frontier_deficit_origin(
        self,
        monkeypatch,
    ):
        """Frontier-deficit origins should be able to contribute local geodesic candidates."""
        energy = np.full((9, 9, 9), 1.0, dtype=np.float64)
        energy[4, 1:8, 4] = -1.0
        candidates = {
            "traces": [np.array([[4, 1, 4], [4, 2, 4]], dtype=np.float32)],
            "connections": np.array([[0, 1]], dtype=np.int32),
            "metrics": np.array([-1.0], dtype=np.float32),
            "energy_traces": [np.array([-1.0, -1.0], dtype=np.float32)],
            "scale_traces": [np.zeros(2, dtype=np.int16)],
            "origin_indices": np.array([0], dtype=np.int32),
            "connection_sources": ["frontier"],
            "diagnostics": {
                **_empty_edge_diagnostics(),
                "frontier_per_origin_candidate_counts": {0: 1},
            },
        }
        vertex_positions = np.array(
            [
                [4.0, 1.0, 4.0],
                [2.0, 3.0, 4.0],
                [4.0, 7.0, 4.0],
            ],
            dtype=np.float32,
        )

        def fake_geodesic(*_args, **_kwargs):
            start = np.asarray(_args[1], dtype=np.float32)
            end = np.asarray(_args[2], dtype=np.float32)
            if np.allclose(start, vertex_positions[0]) and np.allclose(end, vertex_positions[2]):
                return np.array(
                    [[4, 1, 4], [4, 2, 4], [4, 3, 4], [4, 4, 4], [4, 5, 4], [4, 6, 4], [4, 7, 4]],
                    dtype=np.float32,
                )
            return None

        monkeypatch.setattr(
            "slavv.core.tracing._trace_local_geodesic_between_vertices",
            fake_geodesic,
        )

        result = _finalize_matlab_parity_candidates(
            candidates,
            energy,
            None,
            vertex_positions,
            -1.0,
            {
                "number_of_edges_per_vertex": 4,
                "parity_watershed_candidate_mode": "legacy_supplement",
                "parity_candidate_salvage_mode": "frontier_deficit_geodesic",
                "parity_geodesic_salvage_k_nearest": 2,
            },
        )

        assert [0, 2] in result["connections"].tolist()
        assert "geodesic" in result["connection_sources"]
        assert result["diagnostics"]["watershed_accepted"] == 0
        assert result["diagnostics"]["geodesic_join_supplement_count"] >= 1
        assert result["diagnostics"]["geodesic_accepted"] >= 1

    def test_trace_local_geodesic_between_vertices_returns_voxel_path(self):
        """The bounded geodesic tracer should return a simple negative-energy path."""
        energy = np.full((7, 7, 7), 1.0, dtype=np.float64)
        energy[3, 1:6, 3] = -1.0

        trace = _trace_local_geodesic_between_vertices(
            energy,
            np.array([3.0, 1.0, 3.0], dtype=np.float32),
            np.array([3.0, 5.0, 3.0], dtype=np.float32),
            -1.0,
            box_margin_voxels=2,
        )

        assert trace is not None
        assert trace.shape[1] == 3
        assert trace[0].tolist() == [3.0, 1.0, 3.0]
        assert trace[-1].tolist() == [3.0, 5.0, 3.0]

    def test_supplement_diagnostics_populated_when_all_rejected(self):
        """Even when no supplements pass, rejection counters must be set."""
        energy = np.full((5, 5, 5), 1.0, dtype=np.float64)  # all positive → all rejected
        candidates = {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "metrics": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "diagnostics": _empty_edge_diagnostics(),
        }
        vertex_positions = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32)
        result = _supplement_matlab_frontier_candidates_with_watershed_joins(
            candidates, energy, None, vertex_positions, -1.0
        )
        diag = result["diagnostics"]
        assert "watershed_total_pairs" in diag
        assert "watershed_already_existing" in diag
        assert "watershed_energy_rejected" in diag
        assert "watershed_endpoint_degree_rejected" in diag
        assert "watershed_accepted" in diag
        assert diag["watershed_accepted"] == 0

    def test_supplement_diagnostics_count_accepted(self):
        """Verify accepted count matches supplement count."""
        energy = _make_small_volume(9)
        vertex_positions = np.array([[4.0, 1.0, 4.0], [4.0, 7.0, 4.0]], dtype=np.float32)
        candidates = {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "metrics": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "diagnostics": _empty_edge_diagnostics(),
        }
        result = _supplement_matlab_frontier_candidates_with_watershed_joins(
            candidates, energy, None, vertex_positions, -1.0
        )
        diag = result["diagnostics"]
        assert diag["watershed_accepted"] == diag["watershed_join_supplement_count"]

    def test_supplement_metric_threshold_can_reject_weak_watershed_traces(self):
        """A parity-only metric threshold should reject weak watershed supplements."""
        energy = _make_small_volume(9)
        vertex_positions = np.array([[4.0, 1.0, 4.0], [4.0, 7.0, 4.0]], dtype=np.float32)
        candidates = {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "metrics": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "diagnostics": _empty_edge_diagnostics(),
        }
        result = _supplement_matlab_frontier_candidates_with_watershed_joins(
            candidates,
            energy,
            None,
            vertex_positions,
            -1.0,
            enforce_frontier_reachability=False,
            parity_watershed_metric_threshold=-1.1,
        )
        diag = result["diagnostics"]
        assert diag["watershed_metric_threshold_rejected"] >= 1
        assert diag["watershed_accepted"] == 0


@pytest.mark.unit
class TestFrontierCandidateDiagnostics:
    """Diagnostics from the frontier candidate generator."""

    def test_append_candidate_unit_preserves_connection_sources(self):
        aggregate = {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "metrics": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "connection_sources": [],
            "diagnostics": _empty_edge_diagnostics(),
        }
        unit_payload = {
            "candidate_source": "watershed",
            "traces": [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)],
            "connections": [[0, 1]],
            "metrics": [-1.0],
            "energy_traces": [np.array([-1.0, -1.0], dtype=np.float32)],
            "scale_traces": [np.zeros(2, dtype=np.int16)],
            "origin_indices": [0],
            "connection_sources": ["watershed"],
            "diagnostics": _empty_edge_diagnostics(),
        }

        _append_candidate_unit(aggregate, unit_payload)

        assert aggregate["connection_sources"] == ["watershed"]

    def test_per_origin_summary_populated(self):
        """Per-origin candidate counts must appear after frontier generation."""
        energy = _make_small_volume(9)
        vertex_positions = np.array([[4.0, 2.0, 4.0], [4.0, 6.0, 4.0]], dtype=np.float32)
        vertex_scales = np.array([0, 0], dtype=np.int16)
        lumen_radius_microns = np.array([1.0], dtype=np.float32)
        microns_per_voxel = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
        params = {
            "number_of_edges_per_vertex": 4,
            "max_edge_length_per_origin_radius": 20.0,
        }

        candidates = _generate_edge_candidates_matlab_frontier(
            energy,
            np.zeros_like(energy, dtype=np.int16),
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            params,
        )

        diag = candidates["diagnostics"]
        assert "frontier_origins_with_candidates" in diag
        assert "frontier_origins_without_candidates" in diag
        assert "frontier_per_origin_candidate_counts" in diag
        assert diag["frontier_origins_with_candidates"] + diag[
            "frontier_origins_without_candidates"
        ] == len(vertex_positions)

    def test_candidate_audit_summarizes_origin_and_source_counts(self):
        candidate_edges = {
            "connections": np.array([[0, 1], [0, 3], [1, 2], [2, 4]], dtype=np.int32),
            "origin_indices": np.array([0, 0, 1, 2], dtype=np.int32),
            "connection_sources": ["frontier", "frontier", "watershed", "fallback"],
            "traces": [
                np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float32),
                np.array([[1, 1, 1], [1.5, 1.5, 1.5]], dtype=np.float32),
                np.array([[2, 2, 2], [2.5, 2.5, 2.5]], dtype=np.float32),
                np.array([[3, 3, 3], [3.5, 3.5, 3.5]], dtype=np.float32),
            ],
            "diagnostics": {
                "candidate_traced_edge_count": 4,
                "terminal_edge_count": 4,
                "chosen_edge_count": 2,
                "watershed_metric_threshold_rejected": 3,
            },
        }

        candidate_audit = _build_edge_candidate_audit(
            candidate_edges,
            vertex_count=5,
            use_frontier_tracer=True,
            frontier_origin_counts={0: 2, 1: 1},
            supplement_origin_counts={1: 1},
        )

        assert candidate_audit["schema_version"] == 1
        assert candidate_audit["candidate_connection_count"] == 4
        assert candidate_audit["source_breakdown"]["frontier"]["candidate_connection_count"] == 3
        assert candidate_audit["source_breakdown"]["watershed"]["candidate_connection_count"] == 1
        assert candidate_audit["source_breakdown"]["frontier"]["candidate_endpoint_pair_count"] == 2
        assert candidate_audit["source_breakdown"]["frontier"][
            "candidate_endpoint_pair_samples"
        ] == [
            (0, 1),
            (0, 3),
        ]
        assert candidate_audit["source_breakdown"]["watershed"][
            "candidate_endpoint_pair_samples"
        ] == [(1, 2)]
        assert candidate_audit["pair_source_breakdown"]["frontier_only_pair_count"] == 2
        assert candidate_audit["pair_source_breakdown"]["watershed_only_pair_count"] == 1
        assert candidate_audit["pair_source_breakdown"]["fallback_only_pair_count"] == 1
        assert candidate_audit["pair_source_breakdown"]["watershed_only_endpoint_pair_samples"] == [
            (1, 2)
        ]
        assert candidate_audit["frontier_per_origin_candidate_counts"] == {0: 2, 1: 1}
        assert candidate_audit["diagnostic_counters"]["watershed_metric_threshold_rejected"] == 3
        assert isinstance(candidate_audit["per_origin_summary"], list)
        assert len(candidate_audit["per_origin_summary"]) == 3
        origin_zero = next(
            entry for entry in candidate_audit["per_origin_summary"] if entry["origin_index"] == 0
        )
        assert origin_zero["candidate_endpoint_pair_count"] == 2
        assert origin_zero["candidate_endpoint_pair_samples"] == [(0, 1), (0, 3)]
        assert origin_zero["frontier_endpoint_pair_samples"] == [(0, 1), (0, 3)]


@pytest.mark.unit
class TestCoverageReportSplitBySource:
    """The compare_edges report should include frontier vs supplement split."""

    def test_coverage_includes_source_split(self):
        matlab_edges = {
            "connections": np.array([[0, 1]], dtype=np.int32),
            "traces": [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)],
            "energies": np.array([-1.0], dtype=np.float32),
        }
        python_edges = {
            "connections": np.array([[0, 1]], dtype=np.int32),
            "traces": [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)],
            "energies": np.array([-1.0], dtype=np.float32),
        }
        candidate_edges = {
            "connections": np.array([[0, 1]], dtype=np.int32),
            "diagnostics": {
                "watershed_join_supplement_count": 0,
                "watershed_total_pairs": 5,
                "watershed_already_existing": 3,
                "watershed_endpoint_degree_rejected": 1,
                "watershed_energy_rejected": 2,
                "watershed_accepted": 0,
                "frontier_origins_with_candidates": 1,
                "frontier_origins_without_candidates": 0,
            },
        }

        comparison = compare_edges(matlab_edges, python_edges, candidate_edges)
        coverage = comparison["diagnostics"]["candidate_endpoint_coverage"]
        assert "frontier_candidate_endpoint_pair_count" in coverage
        assert "supplement_candidate_endpoint_pair_count" in coverage
        assert "watershed_total_pairs" in coverage
        assert "watershed_endpoint_degree_rejected" in coverage
        assert "frontier_origins_with_candidates" in coverage
