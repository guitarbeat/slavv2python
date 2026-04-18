"""Unit tests for frontier tracing semantics (Phase 3).

Tests frontier ordering tiebreaks, parent/child resolution, terminal
pruning, and path backtracking to verify MATLAB alignment.
"""

import numpy as np
import pytest

from slavv.core.edge_candidates import (
    _prune_frontier_indices_beyond_found_vertices,
    _resolve_frontier_edge_connection,
    _trace_origin_edges_matlab_frontier,
)
from slavv.core.vertices import (
    paint_vertex_center_image,
)


@pytest.mark.unit
class TestFrontierOrdering:
    """Verify that equal-energy frontier voxels break ties by linear index."""

    def test_symmetric_corridor_produces_deterministic_result(self):
        """Two equal-energy corridors leading to two vertices should produce
        deterministic, repeatable results via linear-index tiebreaking."""
        energy = np.full((7, 7, 7), 1.0, dtype=np.float32)
        # Create two symmetric corridors from vertex 0
        for x in range(1, 6):
            energy[3, x, 3] = -5.0  # horizontal corridor
            energy[x, 3, 3] = -5.0  # vertical corridor

        vertex_positions = np.array(
            [[3.0, 0.0, 3.0], [3.0, 5.0, 3.0], [5.0, 3.0, 3.0]],
            dtype=np.float32,
        )
        vertex_scales = np.array([0, 0, 0], dtype=np.int16)
        center_image = paint_vertex_center_image(vertex_positions, energy.shape)

        result_1 = _trace_origin_edges_matlab_frontier(
            energy,
            np.zeros_like(energy, dtype=np.int16),
            vertex_positions,
            vertex_scales,
            np.array([1.0], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            center_image,
            0,
            {
                "number_of_edges_per_vertex": 4,
                "space_strel_apothem": 1,
                "max_edge_length_per_origin_radius": 10.0,
            },
        )

        result_2 = _trace_origin_edges_matlab_frontier(
            energy,
            np.zeros_like(energy, dtype=np.int16),
            vertex_positions,
            vertex_scales,
            np.array([1.0], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            center_image,
            0,
            {
                "number_of_edges_per_vertex": 4,
                "space_strel_apothem": 1,
                "max_edge_length_per_origin_radius": 10.0,
            },
        )

        # Must be bit-for-bit identical across runs
        assert result_1["connections"] == result_2["connections"]
        for t1, t2 in zip(result_1["traces"], result_2["traces"]):
            assert np.array_equal(t1, t2)

    def test_source_fanout_respects_requested_frontier_budget(self):
        """The frontier tracer should honor MATLAB's full per-origin edge budget."""
        energy = np.full((9, 9, 9), 1.0, dtype=np.float32)
        for x in range(1, 8):
            energy[4, x, 4] = -5.0
        for y in range(1, 8):
            energy[y, 4, 4] = -5.0

        vertex_positions = np.array(
            [
                [4.0, 4.0, 4.0],
                [4.0, 1.0, 4.0],
                [4.0, 7.0, 4.0],
                [1.0, 4.0, 4.0],
                [7.0, 4.0, 4.0],
            ],
            dtype=np.float32,
        )
        vertex_scales = np.zeros(len(vertex_positions), dtype=np.int16)
        center_image = paint_vertex_center_image(vertex_positions, energy.shape)

        payload = _trace_origin_edges_matlab_frontier(
            energy,
            np.zeros_like(energy, dtype=np.int16),
            vertex_positions,
            vertex_scales,
            np.array([1.0], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            center_image,
            0,
            {
                "number_of_edges_per_vertex": 4,
                "space_strel_apothem": 1,
                "max_edge_length_per_origin_radius": 20.0,
            },
        )

        assert payload["connections"] == [[0, 1], [0, 3], [0, 4], [0, 2]]


@pytest.mark.unit
class TestParentChildResolution:
    """Verify that child paths worse than parent are rejected."""

    def test_child_with_better_energy_is_invalidated(self):
        """A child path with lower max energy than its parent is rejected
        (MATLAB semantics: child is 'stealing' parent's good voxels)."""
        energy = np.full((5, 5, 5), 1.0, dtype=np.float32)
        shape = energy.shape

        def li(coord):
            return int(coord[0] + coord[1] * shape[0] + coord[2] * shape[0] * shape[1])

        # Set up energy values
        root = (2, 2, 2)
        parent_mid = (2, 3, 2)
        parent_end = (2, 4, 2)
        child_end = (3, 2, 2)

        energy[root] = -6.0
        energy[parent_mid] = -5.0
        energy[parent_end] = -4.0
        energy[child_end] = -7.0  # child has BETTER energy

        root_li = li(root)
        parent_mid_li = li(parent_mid)
        parent_end_li = li(parent_end)
        child_end_li = li(child_end)

        current_path = [child_end_li, root_li]
        parent_path = [parent_end_li, parent_mid_li, root_li]

        pointer_index_map = {
            root_li: -1,
            parent_end_li: -1,
            parent_mid_li: -1,
        }

        origin_idx, terminal_idx = _resolve_frontier_edge_connection(
            current_path,
            terminal_vertex_idx=2,
            seed_origin_idx=0,
            edge_paths_linear=[parent_path],
            edge_pairs=[(1, 0)],
            pointer_index_map=pointer_index_map,
            energy=energy,
            shape=shape,
        )

        assert origin_idx is None
        assert terminal_idx is None

    def test_child_with_worse_energy_is_accepted(self):
        """A child path with higher (worse) max energy than its parent is accepted."""
        energy = np.full((5, 5, 5), 1.0, dtype=np.float32)
        shape = energy.shape

        def li(coord):
            return int(coord[0] + coord[1] * shape[0] + coord[2] * shape[0] * shape[1])

        root = (2, 2, 2)
        parent_mid = (2, 3, 2)
        parent_end = (2, 4, 2)
        child_end = (3, 2, 2)

        energy[root] = -6.0
        energy[parent_mid] = -5.0
        energy[parent_end] = -4.0
        energy[child_end] = -3.0  # child has WORSE energy

        root_li = li(root)
        parent_mid_li = li(parent_mid)
        parent_end_li = li(parent_end)
        child_end_li = li(child_end)

        current_path = [child_end_li, root_li]
        parent_path = [parent_end_li, parent_mid_li, root_li]

        pointer_index_map = {
            root_li: -1,
            parent_end_li: -1,
            parent_mid_li: -1,
        }

        _origin_idx, terminal_idx = _resolve_frontier_edge_connection(
            current_path,
            terminal_vertex_idx=2,
            seed_origin_idx=0,
            edge_paths_linear=[parent_path],
            edge_pairs=[(1, 0)],
            pointer_index_map=pointer_index_map,
            energy=energy,
            shape=shape,
        )

        # Should be accepted (child worse than parent → valid)
        assert terminal_idx == 2


@pytest.mark.unit
class TestTerminalPruning:
    """Verify displacement-based pruning of frontier voxels."""

    def test_voxels_beyond_found_vertex_are_pruned(self):
        """Voxels that lie beyond an already-found vertex direction are removed."""
        candidates = np.array([[2, 3, 2], [2, 4, 2], [2, 1, 2]], dtype=np.int32)
        pruned = _prune_frontier_indices_beyond_found_vertices(
            candidates,
            origin_position_microns=np.array([2.0, 2.0, 2.0]),
            displacement_vectors=[np.array([0.0, 1.0, 0.0])],
            microns_per_voxel=np.ones(3, dtype=np.float32),
        )

        # [2, 4, 2] is beyond the found vertex direction, should be removed
        assert pruned.tolist() == [[2, 3, 2], [2, 1, 2]]

    def test_no_displacement_vectors_skips_pruning(self):
        """With no displacement vectors, all candidates pass through."""
        candidates = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
        pruned = _prune_frontier_indices_beyond_found_vertices(
            candidates,
            origin_position_microns=np.array([0.0, 0.0, 0.0]),
            displacement_vectors=[],
            microns_per_voxel=np.ones(3, dtype=np.float32),
        )
        assert len(pruned) == 2

    def test_empty_candidates_returns_empty(self):
        """Empty input returns empty output."""
        pruned = _prune_frontier_indices_beyond_found_vertices(
            np.zeros((0, 3), dtype=np.int32),
            origin_position_microns=np.array([0.0, 0.0, 0.0]),
            displacement_vectors=[np.array([1.0, 0.0, 0.0])],
            microns_per_voxel=np.ones(3, dtype=np.float32),
        )
        assert len(pruned) == 0


@pytest.mark.unit
class TestPathBacktracking:
    """Verify that terminal hit path backtracking produces correct traces."""

    def test_frontier_tracer_backtracks_correctly(self):
        """Terminal hit should produce a trace from origin to terminal vertex."""
        energy = np.full((7, 7, 7), 1.0, dtype=np.float32)
        energy[3, 1:6, 3] = -5.0  # corridor

        vertex_positions = np.array([[3.0, 1.0, 3.0], [3.0, 5.0, 3.0]], dtype=np.float32)
        vertex_scales = np.array([0, 0], dtype=np.int16)
        center_image = paint_vertex_center_image(vertex_positions, energy.shape)

        payload = _trace_origin_edges_matlab_frontier(
            energy,
            np.zeros_like(energy, dtype=np.int16),
            vertex_positions,
            vertex_scales,
            np.array([1.0], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            center_image,
            0,
            {
                "number_of_edges_per_vertex": 1,
                "space_strel_apothem": 1,
                "max_edge_length_per_origin_radius": 10.0,
            },
        )

        assert len(payload["traces"]) >= 1
        trace = payload["traces"][0]
        # Trace should start near origin and end near terminal
        assert np.allclose(trace[-1], vertex_positions[1], atol=1.0) or np.allclose(
            trace[0], vertex_positions[1], atol=1.0
        )
        assert payload["connections"][0] == [0, 1]

    def test_terminal_hit_clears_new_frontier_voxels(self):
        """A terminal hit should not leave newly revealed terminal-adjacent voxels queued."""
        energy = np.ones((9, 9, 9), dtype=np.float32)
        corridor = {
            (3, 1, 4): -5.0,
            (3, 2, 4): -4.0,
            (3, 3, 4): -3.0,
            (3, 4, 4): -2.0,
            (3, 5, 4): -1.0,
            (3, 6, 4): -0.5,
            (3, 7, 4): -0.5,
        }
        for coord, value in corridor.items():
            energy[coord] = value

        vertex_positions = np.array(
            [[3.0, 1.0, 4.0], [3.0, 5.0, 4.0], [3.0, 7.0, 4.0]],
            dtype=np.float32,
        )
        vertex_scales = np.zeros(3, dtype=np.int16)
        center_image = paint_vertex_center_image(vertex_positions, energy.shape)

        payload = _trace_origin_edges_matlab_frontier(
            energy,
            np.zeros_like(energy, dtype=np.int16),
            vertex_positions,
            vertex_scales,
            np.array([1.0], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            center_image,
            0,
            {
                "number_of_edges_per_vertex": 2,
                "space_strel_apothem": 1,
                "max_edge_length_per_origin_radius": 20.0,
            },
        )

        assert payload["connections"] == [[0, 1]]
        assert payload["diagnostics"]["stop_reason_counts"]["terminal_frontier_hit"] == 1

    def test_frontier_tracer_records_child_better_than_parent_rejection_before_valid_connection(
        self, monkeypatch
    ):
        """A rejected child-better-than-parent hit should still allow the next valid hit.

        This mirrors the active neighborhood diagnosis: the frontier hits two
        terminals, but only one survives into the candidate manifest because the
        first is rejected during local parent/child resolution.
        """

        energy = np.full((9, 9, 9), 1.0, dtype=np.float32)
        energy[4, 4, 4] = -9.0
        energy[4, 5, 4] = -8.0
        energy[4, 6, 4] = -7.0
        energy[5, 4, 4] = -6.0
        energy[6, 4, 4] = -5.0
        energy[4, 3, 4] = -4.0
        energy[4, 2, 4] = -3.0

        vertex_positions = np.array(
            [
                [4.0, 4.0, 4.0],
                [4.0, 6.0, 4.0],
                [6.0, 4.0, 4.0],
                [4.0, 2.0, 4.0],
            ],
            dtype=np.float32,
        )
        vertex_scales = np.zeros(4, dtype=np.int16)
        center_image = paint_vertex_center_image(vertex_positions, energy.shape)

        resolve_calls = {"count": 0}

        def fake_resolve(*_args, **_kwargs):
            resolve_calls["count"] += 1
            if resolve_calls["count"] == 1:
                return (
                    None,
                    None,
                    "rejected_child_better_than_parent",
                    {
                        "parent_path_max_energy": -4.0,
                        "child_path_max_energy": -6.0,
                        "resolution_note": "child path stays local but is rejected before manifest emission",
                    },
                )
            return (
                0,
                2,
                "accepted_parent_origin_half",
                {
                    "bifurcation_choice": "parent_origin_half",
                    "resolution_note": "second terminal survives into the candidate manifest",
                },
            )

        monkeypatch.setattr(
            "slavv.core.edge_candidates._resolve_frontier_edge_connection_details",
            fake_resolve,
        )

        payload = _trace_origin_edges_matlab_frontier(
            energy,
            np.zeros_like(energy, dtype=np.int16),
            vertex_positions,
            vertex_scales,
            np.array([1.0], dtype=np.float32),
            np.ones(3, dtype=np.float32),
            center_image,
            0,
            {
                "number_of_edges_per_vertex": 2,
                "space_strel_apothem": 1,
                "max_edge_length_per_origin_radius": 8.0,
            },
        )

        assert resolve_calls["count"] == 2
        assert payload["connections"] == [[0, 2]]
        assert payload["diagnostics"]["stop_reason_counts"]["terminal_frontier_hit"] == 2
        assert payload["diagnostics"]["frontier_per_origin_terminal_rejections"] == {"0": 1}
        assert payload["diagnostics"]["frontier_per_origin_terminal_accepts"] == {"0": 1}
        assert (
            payload["diagnostics"]["frontier_terminal_resolution_counts"][
                "rejected_child_better_than_parent"
            ]
            == 1
        )
        assert (
            payload["diagnostics"]["frontier_terminal_resolution_counts"][
                "accepted_parent_origin_half"
            ]
            == 1
        )

        lifecycle_events = payload["frontier_lifecycle_events"]
        assert len(lifecycle_events) == 2

        rejected_event = lifecycle_events[0]
        assert rejected_event["resolution_reason"] == "rejected_child_better_than_parent"
        assert rejected_event["survived_candidate_manifest"] is False
        assert rejected_event["rejection_reason"] == "rejected_child_better_than_parent"
        assert rejected_event["parent_child_outcome"] == "child_better_than_parent"
        assert rejected_event["final_survival_stage"] == "pre_manifest_rejection"
        assert rejected_event["resolution_debug"]["parent_path_max_energy"] == -4.0
        assert rejected_event["resolution_debug"]["child_path_max_energy"] == -6.0

        accepted_event = lifecycle_events[1]
        assert accepted_event["resolution_reason"] == "accepted_parent_origin_half"
        assert accepted_event["survived_candidate_manifest"] is True
        assert accepted_event["rejection_reason"] is None
        assert accepted_event["bifurcation_choice"] == "parent_origin_half"
        assert accepted_event["final_survival_stage"] == "candidate_manifest"
