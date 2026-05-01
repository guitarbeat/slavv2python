from __future__ import annotations

import numpy as np
import pytest
from source.core._edge_candidates import global_watershed as global_watershed_module
from source.core._edge_candidates.generate import _finalize_matlab_parity_candidates
from source.core._edge_candidates.global_watershed import (
    _generate_edge_candidates_matlab_global_watershed,
    _initialize_matlab_global_watershed_state,
    _matlab_global_watershed_border_locations,
    _matlab_global_watershed_current_strel,
    _matlab_global_watershed_finalize_edge_trace,
    _matlab_global_watershed_insert_available_location,
    _matlab_global_watershed_reset_join_locations,
    _matlab_global_watershed_reveal_unclaimed_strel,
    _matlab_global_watershed_scale_pointer_map,
    _matlab_global_watershed_seed_index_range,
    _matlab_global_watershed_tolerance_mask,
    _matlab_global_watershed_trace_half,
)


def test_matlab_global_watershed_border_locations_cover_all_boundary_voxels():
    border_locations = _matlab_global_watershed_border_locations((3, 3, 3))

    assert len(border_locations) == 26
    assert 13 not in border_locations


def test_initialize_matlab_global_watershed_state_matches_shared_map_layout():
    energy = np.arange(27, dtype=np.float32).reshape((3, 3, 3), order="F")
    vertex_positions = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)

    state = _initialize_matlab_global_watershed_state(energy, vertex_positions)
    vertex_index_map = state["vertex_index_map"]
    energy_map_temp = state["energy_map_temp"]
    adjacency = state["vertex_adjacency_matrix"].toarray()

    center_linear = 13
    border_vertex_linear = 12

    assert state["vertex_locations"].tolist() == [center_linear, border_vertex_linear]
    assert state["available_locations"].tolist() == [border_vertex_linear, center_linear]
    assert state["vertex_energies"].tolist() == [13.0, 12.0]
    assert state["pointer_map"].dtype == np.uint64
    assert state["d_over_r_map"].dtype == np.float64
    assert np.isneginf(energy_map_temp.ravel(order="F")[center_linear])
    assert np.isneginf(energy_map_temp.ravel(order="F")[border_vertex_linear])
    assert vertex_index_map.ravel(order="F")[center_linear] == 1
    assert vertex_index_map.ravel(order="F")[border_vertex_linear] == 3
    assert vertex_index_map.ravel(order="F")[0] == 3
    assert adjacency.shape == (3, 3)
    assert np.array_equal(adjacency, np.eye(3, dtype=bool))


def test_matlab_global_watershed_current_strel_filters_to_in_bounds_coords():
    strel = _matlab_global_watershed_current_strel(
        0,
        current_scale_label=1,
        shape=(3, 3, 3),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    assert np.all(strel["coords"] >= 0)
    assert np.all(strel["coords"] < np.array([3, 3, 3], dtype=np.int32))
    assert len(strel["coords"]) < 27
    assert strel["pointer_indices"].dtype == np.uint64
    assert strel["pointer_indices"].tolist() == [14, 15, 17, 18, 23, 24, 26, 27]


def test_matlab_global_watershed_reveal_unclaimed_strel_only_claims_zero_vertex_voxels():
    vertex_index_map = np.zeros((3, 3, 3), dtype=np.uint32, order="F")
    energy_map = np.full((3, 3, 3), 99.0, dtype=np.float32, order="F")
    pointer_map = np.zeros((3, 3, 3), dtype=np.uint64, order="F")
    d_over_r_map = np.zeros((3, 3, 3), dtype=np.float32, order="F")
    size_map = np.full((3, 3, 3), -1, dtype=np.int16, order="F")

    vertex_index_map[1, 1, 1] = 7
    energy_map[1, 1, 1] = -9.0
    pointer_map[1, 1, 1] = 55
    d_over_r_map[1, 1, 1] = 2.5
    size_map[1, 1, 1] = 4

    result = _matlab_global_watershed_reveal_unclaimed_strel(
        current_vertex_index=2,
        current_scale_label=3,
        current_d_over_r=1.25,
        valid_linear=np.array([13, 22], dtype=np.int64),
        strel_pointer_indices=np.array([9, 10], dtype=np.uint64),
        strel_distance_microns=np.array([0.0, 0.5], dtype=np.float32),
        strel_adjusted_energies=np.array([-5.0, -6.0], dtype=np.float32),
        vertex_index_map_flat=vertex_index_map.ravel(order="F"),
        energy_map_flat=energy_map.ravel(order="F"),
        pointer_map_flat=pointer_map.ravel(order="F"),
        d_over_r_map_flat=d_over_r_map.ravel(order="F"),
        size_map_flat=size_map.ravel(order="F"),
        lut_size=27,
    )

    assert result["vertices_of_current_strel"].tolist() == [7, 0]
    assert result["is_without_vertex_in_strel"].tolist() == [False, True]
    assert vertex_index_map[1, 1, 1] == 7
    assert energy_map[1, 1, 1] == -9.0
    assert pointer_map[1, 1, 1] == 55
    assert d_over_r_map[1, 1, 1] == 2.5
    assert size_map[1, 1, 1] == 4
    assert vertex_index_map[1, 1, 2] == 2
    assert energy_map[1, 1, 2] == -6.0
    assert pointer_map[1, 1, 2] == 10
    assert d_over_r_map[1, 1, 2] == np.float32(1.75)
    assert size_map[1, 1, 2] == 3


def test_matlab_global_watershed_reveal_unclaimed_strel_raises_for_invalid_claim_pointers():
    with pytest.raises(AssertionError, match="invalid claim pointers"):
        _matlab_global_watershed_reveal_unclaimed_strel(
            current_vertex_index=1,
            current_scale_label=2,
            current_d_over_r=0.5,
            valid_linear=np.array([1], dtype=np.int64),
            strel_pointer_indices=np.array([99], dtype=np.uint64),
            strel_distance_microns=np.array([0.25], dtype=np.float32),
            strel_adjusted_energies=np.array([-1.0], dtype=np.float32),
            vertex_index_map_flat=np.zeros((8,), dtype=np.uint32),
            energy_map_flat=np.zeros((8,), dtype=np.float32),
            pointer_map_flat=np.zeros((8,), dtype=np.uint64),
            d_over_r_map_flat=np.zeros((8,), dtype=np.float64),
            size_map_flat=np.zeros((8,), dtype=np.int16),
            lut_size=4,
        )


def test_matlab_global_watershed_tolerance_mask_tracks_suppressed_energies():
    initial = _matlab_global_watershed_tolerance_mask(
        np.array([-2.0, -1.0], dtype=np.float32),
        current_vertex_energy=-1.0,
        energy_tolerance=1.0,
    )
    suppressed = _matlab_global_watershed_tolerance_mask(
        np.array([0.0, -1.0], dtype=np.float32),
        current_vertex_energy=-1.0,
        energy_tolerance=1.0,
    )

    assert initial.tolist() == [True, True]
    assert suppressed.tolist() == [False, True]


def test_matlab_global_watershed_seed_index_range_matches_origin_only_fanout():
    assert list(
        _matlab_global_watershed_seed_index_range(
            current_pointer_value=0,
            edge_number_tolerance=4,
        )
    ) == [1, 2, 3, 4]
    assert list(
        _matlab_global_watershed_seed_index_range(
            current_pointer_value=7,
            edge_number_tolerance=4,
        )
    ) == [1]


def test_matlab_global_watershed_insert_available_location_primary_seed_keeps_sorted_order():
    updated = _matlab_global_watershed_insert_available_location(
        [11, 22, 33],
        next_location=44,
        next_energy=-5.0,
        energy_lookup={11: -1.0, 22: -3.0, 33: -6.0},
        seed_idx=1,
        is_current_location_clear=True,
    )

    assert updated == [11, 22, 44, 33]


def test_matlab_global_watershed_insert_available_location_secondary_seed_replaces_uncleared_tail():
    updated = _matlab_global_watershed_insert_available_location(
        [11, 22, 33],
        next_location=44,
        next_energy=-5.0,
        energy_lookup={11: -1.0, 22: -3.0, 33: -6.0},
        seed_idx=2,
        is_current_location_clear=False,
    )

    assert updated == [11, 22, 44]


def test_matlab_global_watershed_reset_join_locations_matches_indexed_matlab_removal():
    updated, is_clear = _matlab_global_watershed_reset_join_locations(
        [11, 22, 22, 33, 44],
        next_vertex_locations=np.array([22, 44, 44], dtype=np.int64),
        is_current_location_clear=False,
    )

    assert is_clear is True
    assert updated == [11, 22, 33]


def test_matlab_global_watershed_reset_join_locations_keeps_cleared_tail_behavior():
    updated, is_clear = _matlab_global_watershed_reset_join_locations(
        [11, 22, 33, 44],
        next_vertex_locations=np.array([22, 44], dtype=np.int64),
        is_current_location_clear=True,
    )

    assert is_clear is True
    assert updated == [11, 33]


def test_generate_edge_candidates_matlab_global_watershed_recovers_simple_bridge():
    energy = np.ones((5, 5, 5), dtype=np.float32)
    energy[1, 2, 2] = -1.0
    energy[2, 2, 2] = -2.0
    energy[3, 2, 2] = -1.0
    scale_indices = np.zeros((5, 5, 5), dtype=np.int16)
    vertex_positions = np.array([[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)

    candidates = _generate_edge_candidates_matlab_global_watershed(
        energy,
        scale_indices,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        _vertex_center_image=np.zeros((5, 5, 5), dtype=np.int32),
        params={"comparison_exact_network": True},
    )

    assert candidates["matlab_global_watershed_exact"] is True
    assert candidates["candidate_source"] == "global_watershed"
    assert len(candidates["connections"]) == 1
    assert sorted(candidates["connections"].tolist()[0]) == [0, 1]
    assert candidates["connection_sources"] == ["global_watershed"]
    assert len(candidates["traces"]) == 1
    endpoints = {
        tuple(candidates["traces"][0][0].tolist()),
        tuple(candidates["traces"][0][-1].tolist()),
    }
    assert endpoints == {
        (1.0, 2.0, 2.0),
        (3.0, 2.0, 2.0),
    }
    assert candidates["diagnostics"]["candidate_traced_edge_count"] == 1
    assert candidates["pointer_map"].shape == energy.shape
    assert candidates["raw_pointer_map"].shape == energy.shape
    assert candidates["raw_pointer_map"].dtype == np.uint64
    assert candidates["d_over_r_map"].dtype == np.float64
    assert candidates["vertex_index_map"].shape == energy.shape


def test_generate_edge_candidates_matlab_global_watershed_uses_configured_step_size(
    monkeypatch,
):
    observed_step_sizes: list[float] = []

    def fake_current_strel(*_args, **kwargs):
        observed_step_sizes.append(float(kwargs["step_size_per_origin_radius"]))
        raise RuntimeError("captured step size")

    monkeypatch.setattr(
        global_watershed_module,
        "_matlab_global_watershed_current_strel",
        fake_current_strel,
    )

    with pytest.raises(RuntimeError, match="captured step size"):
        _generate_edge_candidates_matlab_global_watershed(
            np.full((3, 3, 3), -1.0, dtype=np.float32),
            np.zeros((3, 3, 3), dtype=np.int16),
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([0], dtype=np.int16),
            lumen_radius_microns=np.array([1.0], dtype=np.float32),
            microns_per_voxel=np.ones((3,), dtype=np.float32),
            _vertex_center_image=np.zeros((3, 3, 3), dtype=np.int32),
            params={"step_size_per_origin_radius": 2.5},
        )

    assert observed_step_sizes == [2.5]


def test_matlab_global_watershed_scale_pointer_map_matches_final_matlab_formula():
    pointer_map = np.zeros((3, 3, 3), dtype=np.uint64)
    size_map = np.ones((3, 3, 3), dtype=np.int16)
    pointer_map[1, 1, 1] = 14
    pointer_map[1, 1, 2] = 27

    scaled = _matlab_global_watershed_scale_pointer_map(
        pointer_map,
        size_map,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    assert scaled[1, 1, 1] == np.float32(1000.0 / 27.0 * 14.0)
    assert scaled[1, 1, 2] == np.float32(1000.0 / 27.0 * 27.0)


def test_matlab_global_watershed_trace_half_follows_linear_lut_offsets():
    shape = (5, 5, 5)
    pointer_map = np.zeros(shape, dtype=np.uint64)
    size_map = np.ones(shape, dtype=np.int16)
    strel = _matlab_global_watershed_current_strel(
        63,
        current_scale_label=1,
        shape=shape,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )
    offsets = strel["offsets"]
    pointer_indices = strel["pointer_indices"]
    forward_x_idx = int(pointer_indices[np.all(offsets == np.array([1, 0, 0]), axis=1)][0])

    pointer_map[2, 2, 2] = np.uint64(forward_x_idx)
    pointer_map[3, 2, 2] = np.uint64(forward_x_idx)

    traced = _matlab_global_watershed_trace_half(
        63,
        pointer_map=pointer_map,
        size_map=size_map,
        shape=shape,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    assert traced == [63, 62, 61]


def test_matlab_global_watershed_finalize_edge_trace_samples_linear_trace_directly():
    shape = (3, 3, 3)
    energy_map = np.arange(27, dtype=np.float32).reshape(shape, order="F")
    scale_image = (100 + np.arange(27, dtype=np.int16)).reshape(shape, order="F")

    trace, energy_trace, scale_trace = _matlab_global_watershed_finalize_edge_trace(
        [13, 12],
        [14, 17],
        shape=shape,
        energy_map=energy_map,
        scale_image=scale_image,
    )

    assert trace.tolist() == [
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
    ]
    assert energy_trace.tolist() == [12.0, 13.0, 14.0, 17.0]
    assert scale_trace.tolist() == [112, 113, 114, 117]


def test_finalize_matlab_parity_candidates_is_noop_for_exact_global_watershed_payload():
    candidates = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "diagnostics": {},
        "matlab_global_watershed_exact": True,
    }

    finalized = _finalize_matlab_parity_candidates(
        candidates,
        energy=np.zeros((3, 3, 3), dtype=np.float32),
        scale_indices=np.zeros((3, 3, 3), dtype=np.int16),
        vertex_positions=np.zeros((0, 3), dtype=np.float32),
        energy_sign=-1.0,
        params={"comparison_exact_network": True},
        microns_per_voxel=np.ones((3,), dtype=np.float32),
    )

    assert finalized is candidates
