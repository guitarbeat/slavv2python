from __future__ import annotations

import math

import numpy as np

from slavv_python.core.edges.common import (
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_frontier_directional_suppression_factors,
    _matlab_frontier_insert_available_location,
    _matlab_frontier_pop_best_available_location,
    _matlab_frontier_scale_offsets,
    _matlab_frontier_select_seed_moves,
    _matlab_frontier_size_tolerance,
)


def test_matlab_frontier_size_tolerance_matches_radius_tolerance_formula():
    lumen_radius_microns = np.array([1.0, 2.0, 4.0], dtype=np.float32)

    tolerance = _matlab_frontier_size_tolerance(lumen_radius_microns)

    assert math.isclose(tolerance, math.log(1.5) / math.log(2.0))


def test_matlab_frontier_size_tolerance_uses_first_two_radii_like_matlab():
    lumen_radius_microns = np.array([1.0, 2.0, 5.0], dtype=np.float32)

    tolerance = _matlab_frontier_size_tolerance(lumen_radius_microns)

    assert math.isclose(tolerance, math.log(1.5) / math.log(2.0))


def test_matlab_frontier_scale_offsets_include_full_27_neighborhood_at_small_scale():
    offsets, distances = _matlab_frontier_scale_offsets(
        0,
        np.array([1.0], dtype=np.float32),
        np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    assert offsets.shape == (27, 3)
    assert distances.shape == (27,)
    assert any(np.array_equal(offset, np.array([1, 1, 1], dtype=np.int32)) for offset in offsets)


def test_matlab_frontier_scale_offsets_grow_with_scale():
    small_offsets, _small_distances = _matlab_frontier_scale_offsets(
        0,
        np.array([1.0, 2.0], dtype=np.float32),
        np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )
    large_offsets, _large_distances = _matlab_frontier_scale_offsets(
        1,
        np.array([1.0, 2.0], dtype=np.float32),
        np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    assert len(large_offsets) > len(small_offsets)
    assert np.max(np.abs(large_offsets), axis=0).tolist() == [2, 2, 2]


def test_matlab_frontier_adjusted_neighbor_energies_penalizes_scale_mismatch():
    adjusted = _matlab_frontier_adjusted_neighbor_energies(
        np.array([-5.0, -5.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32),
        neighbor_r_over_R=np.array([1.0, 1.0], dtype=np.float32) / 2.0,
        neighbor_scale_indices=np.array([0, 2], dtype=np.int16),
        propagated_scale_index=0,
        current_d_over_r=0.0,
        origin_radius_microns=2.0,
        current_forward_unit=None,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0, 4.0], dtype=np.float32),
    )

    assert adjusted[0] < adjusted[1]


def test_matlab_frontier_adjusted_neighbor_energies_penalizes_reverse_direction():
    adjusted = _matlab_frontier_adjusted_neighbor_energies(
        np.array([-5.0, -5.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.int32),
        neighbor_r_over_R=np.array([1.0, 1.0], dtype=np.float32) / 2.0,
        neighbor_scale_indices=np.array([0, 0], dtype=np.int16),
        propagated_scale_index=0,
        current_d_over_r=0.0,
        origin_radius_microns=2.0,
        current_forward_unit=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
    )

    assert adjusted[0] < 0.0
    assert adjusted[1] == 0.0


def test_matlab_frontier_adjusted_neighbor_energies_penalizes_long_total_distance():
    near = _matlab_frontier_adjusted_neighbor_energies(
        np.array([-5.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0]], dtype=np.int32),
        neighbor_r_over_R=np.array([1.0], dtype=np.float32) / 1.0,
        neighbor_scale_indices=np.array([0], dtype=np.int16),
        propagated_scale_index=0,
        current_d_over_r=0.0,
        origin_radius_microns=1.0,
        current_forward_unit=None,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
    )
    far = _matlab_frontier_adjusted_neighbor_energies(
        np.array([-5.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0]], dtype=np.int32),
        neighbor_r_over_R=np.array([1.0], dtype=np.float32) / 1.0,
        neighbor_scale_indices=np.array([0], dtype=np.int16),
        propagated_scale_index=0,
        current_d_over_r=3.0 / 1.0,
        origin_radius_microns=1.0,
        current_forward_unit=None,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
    )

    assert near[0] < far[0]


def test_matlab_frontier_directional_suppression_zeros_chosen_direction():
    suppression = _matlab_frontier_directional_suppression_factors(
        np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.int32),
        selected_index=0,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
    )

    assert suppression[0] == 0.0
    assert suppression[1] == 0.5
    assert suppression[2] == 1.0


def test_matlab_frontier_select_seed_moves_prefers_opposite_directions_for_source():
    selected = _matlab_frontier_select_seed_moves(
        np.array([-5.0, -4.0, -4.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.int32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        current_is_source=True,
        edge_budget=2,
        current_branch_order=0,
    )

    assert selected == [(0, 0), (2, 1)]


def test_matlab_frontier_select_seed_moves_limits_continuation_to_one_direction():
    selected = _matlab_frontier_select_seed_moves(
        np.array([-5.0, -4.0, -4.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.int32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        current_is_source=False,
        edge_budget=2,
        current_branch_order=0,
    )

    assert selected == [(0, 0)]


def test_matlab_frontier_available_locations_pop_best_from_sorted_tail():
    available_entries: list[tuple[float, int]] = []
    available_map = {11: -1.0, 22: -3.0, 33: -2.0}
    _matlab_frontier_insert_available_location(available_entries, linear_index=11, energy=-1.0)
    _matlab_frontier_insert_available_location(available_entries, linear_index=22, energy=-3.0)
    _matlab_frontier_insert_available_location(available_entries, linear_index=33, energy=-2.0)

    assert available_entries == [(-1.0, 11), (-2.0, 33), (-3.0, 22)]
    assert _matlab_frontier_pop_best_available_location(available_entries, available_map) == (
        -3.0,
        22,
    )


def test_matlab_frontier_available_locations_skip_stale_entries():
    available_entries = [(-2.0, 33), (-3.0, 22)]
    available_map = {33: -2.0}

    assert _matlab_frontier_pop_best_available_location(available_entries, available_map) == (
        -2.0,
        33,
    )
