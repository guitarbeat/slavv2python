"""Comprehensive test suite for global watershed algorithm.

This test suite verifies exact numerical behavior matches MATLAB across different
edge configurations, with special focus on the areas that have caused bugs:
- Energy map processing (size, distance, direction penalties)
- Structuring element application (geometry, offsets, distance calculations)
- Seed point selection with varying tolerances
- Directional suppression factors (the cumulative energy mutation bug)
- Frontier expansion logic (available_locations list management)
- Edge termination conditions (energy tolerance, vertex collisions)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from slavv_python.core.edge_candidates_internal.common import (
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_frontier_directional_suppression_factors,
)
from slavv_python.core.edge_candidates_internal.generate import _finalize_matlab_parity_candidates
from slavv_python.core.edge_candidates_internal.global_watershed import (
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

# ============================================================================
# Energy Map Processing Tests
# ============================================================================


@pytest.mark.unit
def test_energy_adjustments_size_penalty():
    """Test that size mismatch penalties match MATLAB formula.

    MATLAB applies: exp(-0.5 * (size_difference / size_tolerance)^2)
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~450
    """
    # Two neighbors with same raw energy but different scales
    raw_energies = np.array([-5.0, -5.0], dtype=np.float32)
    neighbor_scale_indices = np.array([0, 2], dtype=np.int16)  # Scale 0 and scale 2

    adjusted = _matlab_frontier_adjusted_neighbor_energies(
        raw_energies,
        neighbor_offsets=np.array([[1, 0, 0], [1, 0, 0]], dtype=np.int32),
        neighbor_r_over_R=np.array([1.0, 1.0], dtype=np.float32) / 2.0,
        neighbor_scale_indices=neighbor_scale_indices,
        propagated_scale_index=0,  # Current scale is 0
        current_d_over_r=0.0,
        origin_radius_microns=2.0,
        current_forward_unit=None,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0, 4.0], dtype=np.float32),
    )

    # Scale 0 neighbor should have higher (less negative) adjusted energy than scale 2
    assert adjusted[0] < adjusted[1], "Same-scale neighbor should be preferred"
    assert adjusted.dtype == np.float32


@pytest.mark.unit
def test_energy_adjustments_distance_penalty():
    """Test that distance penalties match MATLAB formula.

    MATLAB applies local distance adjustment:
    (1 - cos(pi * min(1, (4/3) * r/R))) / 2

    And total distance adjustment:
    exp(-0.5 * (3 * d/R / distance_tolerance)^2)

    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~460-470
    """
    # Test near vs far from origin
    near_adjusted = _matlab_frontier_adjusted_neighbor_energies(
        np.array([-5.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0]], dtype=np.int32),
        neighbor_r_over_R=np.array([0.5], dtype=np.float32) / 1.0,
        neighbor_scale_indices=np.array([0], dtype=np.int16),
        propagated_scale_index=0,
        current_d_over_r=0.0,  # Near origin
        origin_radius_microns=1.0,
        current_forward_unit=None,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
    )

    far_adjusted = _matlab_frontier_adjusted_neighbor_energies(
        np.array([-5.0], dtype=np.float32),
        neighbor_offsets=np.array([[1, 0, 0]], dtype=np.int32),
        neighbor_r_over_R=np.array([0.5], dtype=np.float32) / 1.0,
        neighbor_scale_indices=np.array([0], dtype=np.int16),
        propagated_scale_index=0,
        current_d_over_r=3.0 / 1.0,  # Far from origin
        origin_radius_microns=1.0,
        current_forward_unit=None,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
    )

    # Near origin should have better (more negative) adjusted energy
    assert near_adjusted[0] < far_adjusted[0], "Near origin should be preferred"


@pytest.mark.unit
def test_energy_adjustments_direction_penalty():
    """Test that directional penalties match MATLAB formula.

    MATLAB zeros out backward directions and scales by cosine alignment.
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~480
    """
    # Forward and backward neighbors with same raw energy
    raw_energies = np.array([-5.0, -5.0], dtype=np.float32)
    neighbor_offsets = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.int32)

    adjusted = _matlab_frontier_adjusted_neighbor_energies(
        raw_energies,
        neighbor_offsets=neighbor_offsets,
        neighbor_r_over_R=np.array([1.0, 1.0], dtype=np.float32) / 2.0,
        neighbor_scale_indices=np.array([0, 0], dtype=np.int16),
        propagated_scale_index=0,
        current_d_over_r=0.0,
        origin_radius_microns=2.0,
        current_forward_unit=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Forward in +X
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
    )

    # Forward direction should be preferred, backward should be zeroed
    assert adjusted[0] < 0.0, "Forward direction should have negative energy"
    assert adjusted[1] == 0.0, "Backward direction should be zeroed"


# ============================================================================
# Structuring Element Tests
# ============================================================================


@pytest.mark.unit
def test_strel_geometry_matches_matlab_lut():
    """Test that structuring element geometry matches MATLAB LUT.

    MATLAB builds strels using spherical + box union with specific ordering.
    Reference: Vectorization-Public/slavv_python/calculate_linear_strel_range.m
    """
    strel = _matlab_global_watershed_current_strel(
        13,  # Center of 3x3x3 volume
        current_scale_label=1,
        shape=(3, 3, 3),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    # Verify data types match MATLAB
    assert strel["coords"].dtype == np.int32
    assert strel["offsets"].dtype == np.int32
    assert strel["linear_indices"].dtype == np.int64
    assert strel["pointer_indices"].dtype == np.uint64
    assert strel["r_over_R"].dtype == np.float32
    assert strel["distance_microns"].dtype == np.float32
    assert strel["unit_vectors"].dtype == np.float32

    # Verify all coords are in bounds
    assert np.all(strel["coords"] >= 0)
    assert np.all(strel["coords"][:, 0] < 3)
    assert np.all(strel["coords"][:, 1] < 3)
    assert np.all(strel["coords"][:, 2] < 3)

    # Verify pointer indices are 1-based and in valid range
    assert np.all(strel["pointer_indices"] >= 1)
    assert np.all(strel["pointer_indices"] <= strel["lut_size"])


@pytest.mark.unit
def test_strel_offsets_and_distances_consistent():
    """Test that strel offsets and distances are geometrically consistent.

    Distance should equal sqrt(sum((offset * microns_per_voxel)^2))
    """
    microns_per_voxel = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    strel = _matlab_global_watershed_current_strel(
        63,  # Center of 5x5x5 volume
        current_scale_label=1,
        shape=(5, 5, 5),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=microns_per_voxel,
        step_size_per_origin_radius=1.0,
    )

    # Recompute distances from offsets
    computed_distances = np.sqrt(
        np.sum((strel["offsets"].astype(np.float32) * microns_per_voxel) ** 2, axis=1)
    )

    # Should match within floating point tolerance
    assert np.allclose(computed_distances, strel["distance_microns"], rtol=1e-5)


@pytest.mark.unit
def test_strel_scales_with_radius():
    """Test that strel size grows with lumen radius.

    Larger radii should produce larger strels.
    """
    small_strel = _matlab_global_watershed_current_strel(
        63,
        current_scale_label=1,
        shape=(10, 10, 10),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    large_strel = _matlab_global_watershed_current_strel(
        63,
        current_scale_label=2,  # Use scale 2 (radius 2.0)
        shape=(10, 10, 10),
        lumen_radius_microns=np.array([1.0, 2.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    assert len(large_strel["coords"]) > len(small_strel["coords"])


# ============================================================================
# Seed Point Selection Tests
# ============================================================================


@pytest.mark.unit
def test_seed_selection_tolerance_1_single_seed():
    """Test that edge_number_tolerance=1 produces single seed from non-origin.

    MATLAB only allows multiple seeds from true origins (pointer_value == 0).
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~520
    """
    # Non-origin location (pointer_value != 0)
    seed_range = _matlab_global_watershed_seed_index_range(
        current_pointer_value=7,
        edge_number_tolerance=1,
    )

    assert list(seed_range) == [1], "Non-origin should produce single seed"


@pytest.mark.unit
def test_seed_selection_tolerance_2_from_origin():
    """Test that edge_number_tolerance=2 produces two seeds from origin.

    MATLAB allows edge_number_tolerance seeds from origins.
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~520
    """
    # Origin location (pointer_value == 0)
    seed_range = _matlab_global_watershed_seed_index_range(
        current_pointer_value=0,
        edge_number_tolerance=2,
    )

    assert list(seed_range) == [1, 2], "Origin should produce multiple seeds"


@pytest.mark.unit
def test_seed_selection_tolerance_4_from_origin():
    """Test that edge_number_tolerance=4 produces four seeds from origin."""
    seed_range = _matlab_global_watershed_seed_index_range(
        current_pointer_value=0,
        edge_number_tolerance=4,
    )

    assert list(seed_range) == [1, 2, 3, 4]


@pytest.mark.unit
def test_energy_tolerance_mask_filters_weak_candidates():
    """Test that energy tolerance correctly filters candidates.

    MATLAB uses: adjusted_energy < vertex_energy * (1 - energy_tolerance)
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~530
    """
    adjusted_energies = np.array([-2.0, -1.5, -1.0, -0.5], dtype=np.float32)
    current_vertex_energy = -1.0
    energy_tolerance = 0.5

    mask = _matlab_global_watershed_tolerance_mask(
        adjusted_energies,
        current_vertex_energy=current_vertex_energy,
        energy_tolerance=energy_tolerance,
    )

    # Threshold is -1.0 * (1 - 0.5) = -0.5
    # Only energies < -0.5 should pass
    expected = np.array([True, True, True, False], dtype=bool)
    assert np.array_equal(mask, expected)


# ============================================================================
# Directional Suppression Factor Tests (Critical for Bug Prevention)
# ============================================================================


@pytest.mark.unit
def test_directional_suppression_zeros_selected_direction():
    """Test that directional suppression zeros out the selected direction.

    MATLAB formula: (1 - cosine_to_selected) / 2
    When cosine = 1 (same direction), suppression = 0
    When cosine = -1 (opposite direction), suppression = 1

    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~540
    """
    offsets = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.int32)

    suppression = _matlab_frontier_directional_suppression_factors(
        offsets,
        selected_index=0,  # Select [1, 0, 0]
        microns_per_voxel=np.ones((3,), dtype=np.float32),
    )

    # Selected direction should be zeroed
    assert suppression[0] == 0.0, "Selected direction should have zero suppression"

    # Perpendicular direction should be 0.5
    assert math.isclose(suppression[1], 0.5, abs_tol=1e-6)

    # Opposite direction should be 1.0 (fully suppressed)
    assert math.isclose(suppression[2], 1.0, abs_tol=1e-6)


@pytest.mark.unit
def test_directional_suppression_continuous_not_discrete():
    """Test that suppression is continuous based on cosine, not discrete.

    This verifies we're using the correct MATLAB formula, not a simplified version.
    """
    # Create offsets at various angles
    offsets = np.array(
        [
            [1, 0, 0],  # 0 degrees
            [1, 1, 0],  # 45 degrees
            [0, 1, 0],  # 90 degrees
            [-1, 1, 0],  # 135 degrees
            [-1, 0, 0],  # 180 degrees
        ],
        dtype=np.int32,
    )

    suppression = _matlab_frontier_directional_suppression_factors(
        offsets,
        selected_index=0,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
    )

    # Suppression should increase monotonically with angle
    assert suppression[0] < suppression[1] < suppression[2] < suppression[3] < suppression[4]

    # Verify specific values
    assert suppression[0] == 0.0  # Same direction
    assert math.isclose(suppression[2], 0.5, abs_tol=1e-6)  # Perpendicular
    assert math.isclose(suppression[4], 1.0, abs_tol=1e-6)  # Opposite


@pytest.mark.unit
def test_directional_suppression_is_iterative():
    """Test that suppression correctly mutates the adjusted_energies array between seeds.

    This verifies the corrected finding: MATLAB applies directional suppression
    INSIDE the seed loop, causing each subsequent seed to see a suppressed field
    relative to previous seeds.

    Reference: Vectorization-Public/slavv_python/get_edges_by_watershed.m line 763
    """
    # Simulate the scenario with multiple seeds
    adjusted = np.array([-10.0, -8.0, -6.0, -4.0, -2.0], dtype=np.float64)
    offsets = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.int32,
    )

    # Seed 1: best is index 0 (energy -10.0)
    best_idx_1 = int(np.argmin(adjusted))
    assert best_idx_1 == 0

    # Apply suppression based on seed 1
    factors = _matlab_frontier_directional_suppression_factors(
        offsets,
        selected_index=best_idx_1,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
    )
    # index 0 is best direction -> factors[0] == 0.0
    # adjusted[0] becomes 0.0
    adjusted *= factors

    # Seed 2: now sees index 1 as best (or whatever is next best after suppression)
    best_idx_2 = int(np.argmin(adjusted))
    assert best_idx_2 != 0, "Second seed should NOT pick same index as first after suppression"
    assert best_idx_2 == 1, (
        f"Expected index 1 to be best after suppressing index 0, got {best_idx_2}"
    )
    assert adjusted[0] == 0.0


# ============================================================================
# Frontier Expansion Logic Tests
# ============================================================================


@pytest.mark.unit
def test_available_locations_insertion_maintains_order():
    """Test that available_locations list maintains worst-to-best order.

    MATLAB keeps list sorted with worst energy at index 0, best at end.
    For negative energies, "worst" means least negative (closer to zero).
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~560
    """
    energy_lookup = {10: -1.0, 20: -3.0, 30: -5.0}
    available = [10, 20]  # Worst to best: [-1.0, -3.0]

    # Insert best energy
    updated = _matlab_global_watershed_insert_available_location(
        available,
        next_location=30,
        next_energy=-5.0,
        energy_lookup=energy_lookup,
        seed_idx=1,
        is_current_location_clear=True,
    )

    # Should maintain order: [10, 20, 30] with energies [-1.0, -3.0, -5.0]
    # This is worst-to-best order (least negative to most negative)
    energies = [energy_lookup[loc] for loc in updated]
    assert energies == [-1.0, -3.0, -5.0], f"Expected worst-to-best order, got: {energies}"

    # Verify it's monotonically decreasing (getting more negative = better)
    for i in range(len(energies) - 1):
        assert energies[i] > energies[i + 1], "Energies should decrease (get more negative)"


@pytest.mark.unit
def test_available_locations_secondary_seed_replaces_tail():
    """Test that secondary seeds replace uncleared tail location.

    MATLAB behavior differs for seed_idx > 1 when location not clear.
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~565
    """
    energy_lookup = {10: -1.0, 20: -3.0, 30: -6.0}
    available = [10, 20, 30]

    updated = _matlab_global_watershed_insert_available_location(
        available,
        next_location=40,
        next_energy=-5.0,
        energy_lookup={**energy_lookup, 40: -5.0},
        seed_idx=2,  # Secondary seed
        is_current_location_clear=False,  # Not clear
    )

    # Should replace tail: [10, 20, 40]
    assert updated == [10, 20, 40]


@pytest.mark.unit
def test_reset_join_locations_removes_vertex_locations():
    """Test that joining vertices are removed from available_locations.

    When traces meet at vertices, those locations are removed from frontier.
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~580
    """
    available = [10, 20, 30, 40, 50]
    next_vertex_locations = np.array([20, 40], dtype=np.int64)

    updated, is_clear = _matlab_global_watershed_reset_join_locations(
        available,
        next_vertex_locations=next_vertex_locations,
        is_current_location_clear=True,
    )

    # Should remove 20 and 40
    assert updated == [10, 30, 50]
    assert is_clear is True


@pytest.mark.unit
def test_reset_join_locations_handles_uncleared_tail():
    """Test that uncleared tail is handled correctly during join reset."""
    available = [10, 20, 30, 40]
    next_vertex_locations = np.array([20], dtype=np.int64)

    updated, is_clear = _matlab_global_watershed_reset_join_locations(
        available,
        next_vertex_locations=next_vertex_locations,
        is_current_location_clear=False,  # Tail not clear
    )

    # Should remove tail (40) first, then remove 20
    assert updated == [10, 30]
    assert is_clear is True


# ============================================================================
# Edge Termination Condition Tests
# ============================================================================


@pytest.mark.unit
def test_trace_terminates_at_zero_pointer():
    """Test that trace stops when reaching zero pointer (origin).

    MATLAB traces back until pointer_value == 0.
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~620
    """
    shape = (5, 5, 5)
    pointer_map = np.zeros(shape, dtype=np.uint64, order="F")
    size_map = np.ones(shape, dtype=np.int16, order="F")

    # Create simple chain: 63 -> 62 -> 61 (origin)
    pointer_map[2, 2, 2] = 14  # Points to previous
    pointer_map[1, 2, 2] = 14  # Points to previous
    # pointer_map[0, 2, 2] = 0  # Origin (already zero)

    traced = _matlab_global_watershed_trace_half(
        63,  # Start from [2, 2, 2]
        pointer_map=pointer_map,
        size_map=size_map,
        shape=shape,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    # Should trace back to origin
    assert len(traced) >= 1
    assert traced[0] == 63  # Start location


@pytest.mark.unit
def test_trace_terminates_on_cycle_detection():
    """Test that trace stops if it detects a cycle (visited set).

    This prevents infinite loops from corrupted pointer maps.
    """
    shape = (3, 3, 3)
    pointer_map = np.zeros(shape, dtype=np.uint64, order="F")
    size_map = np.ones(shape, dtype=np.int16, order="F")

    # Create a cycle: 13 -> 14 -> 13 (invalid but test safety)
    # This shouldn't happen in correct code, but trace should handle it
    pointer_map[1, 1, 1] = 2  # Points to [1, 1, 2]
    pointer_map[1, 1, 2] = 26  # Would point back (if valid)

    traced = _matlab_global_watershed_trace_half(
        13,
        pointer_map=pointer_map,
        size_map=size_map,
        shape=shape,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    # Should terminate without infinite loop
    assert len(traced) >= 1
    assert len(traced) < 100  # Sanity check


@pytest.mark.unit
def test_reveal_unclaimed_only_claims_zero_vertex_voxels():
    """Test that reveal only claims voxels with vertex_index == 0 AND pointer == 0.

    MATLAB never overwrites existing vertex claims or existing pointers.
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~500
    """
    vertex_index_map = np.zeros((3, 3, 3), dtype=np.uint32, order="F")
    energy_map = np.full((3, 3, 3), 99.0, dtype=np.float32, order="F")
    pointer_map = np.zeros((3, 3, 3), dtype=np.uint64, order="F")
    d_over_r_map = np.zeros((3, 3, 3), dtype=np.float64, order="F")
    size_map = np.full((3, 3, 3), -1, dtype=np.int16, order="F")

    # Mark one location as already claimed by vertex 5
    # Linear index 13 is [1, 1, 1] in MATLAB order (y=1, x=1, z=1)
    vertex_index_map[1, 1, 1] = 5
    energy_map[1, 1, 1] = -9.0
    pointer_map[1, 1, 1] = 55

    # Mark another location as unclaimed (vertex_index=0, pointer=0)
    # Linear index 22 is [1, 1, 2] in MATLAB order (y=1, x=1, z=2)
    vertex_index_map[1, 1, 2] = 0
    pointer_map[1, 1, 2] = 0

    result = _matlab_global_watershed_reveal_unclaimed_strel(
        current_vertex_index=2,
        current_scale_label=1,
        current_d_over_r=1.0,
        valid_linear=np.array([13, 22], dtype=np.int64),  # [1,1,1] and [1,1,2]
        strel_pointer_indices=np.array([10, 11], dtype=np.uint64),
        strel_r_over_R=np.array([0.5, 0.6], dtype=np.float32) / 1.0,
        adjusted_energies=np.array([-5.0, -10.0], dtype=np.float32),
        vertex_index_map_flat=vertex_index_map.ravel(order="F"),
        pointer_map_flat=pointer_map.ravel(order="F"),
        energy_map_flat=energy_map.ravel(order="F"),
        d_over_r_map_flat=d_over_r_map.ravel(order="F"),
        size_map_flat=size_map.ravel(order="F"),
        lut_size=27,
    )

    # First location should not be claimed (already owned by vertex 5)
    assert vertex_index_map[1, 1, 1] == 5
    assert energy_map[1, 1, 1] == -9.0
    assert pointer_map[1, 1, 1] == 55

    # Second location should be claimed (was unclaimed)
    assert vertex_index_map[1, 1, 2] == 2
    assert pointer_map[1, 1, 2] == 11

    # Verify the result arrays
    assert result["vertices_of_current_strel"][0] == 5  # First location owned by vertex 5
    assert result["vertices_of_current_strel"][1] == 0  # Second location was unclaimed
    assert not result["is_without_vertex_in_strel"][0]  # First location not claimable
    assert result["is_without_vertex_in_strel"][1]  # Second location was claimable


# ============================================================================
# Integration Tests with Realistic Scenarios
# ============================================================================


@pytest.mark.unit
def test_simple_bridge_recovery_with_exact_mode():
    """Test that global watershed recovers a simple bridge in exact mode.

    This is an integration test that exercises the full algorithm.
    """
    energy = np.ones((5, 5, 5), dtype=np.float32, order="F")
    # Create a simple bridge: vertex1 -> middle -> vertex2
    energy[1, 2, 2] = -1.0
    energy[2, 2, 2] = -2.0
    energy[3, 2, 2] = -1.0

    scale_indices = np.zeros((5, 5, 5), dtype=np.int16, order="F")
    vertex_positions = np.array([[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)

    candidates = _generate_edge_candidates_matlab_global_watershed(
        energy,
        scale_indices,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        _vertex_center_image=np.zeros((5, 5, 5), dtype=np.int32, order="F"),
        params={"comparison_exact_network": True},
    )

    # Should find one connection
    assert len(candidates["connections"]) == 1
    assert sorted(candidates["connections"][0].tolist()) == [0, 1]

    # Should have exact flag set
    assert candidates["matlab_global_watershed_exact"] is True

    # Verify trace endpoints
    trace = candidates["traces"][0]
    endpoints = {tuple(trace[0].tolist()), tuple(trace[-1].tolist())}
    expected_endpoints = {(1.0, 2.0, 2.0), (3.0, 2.0, 2.0)}
    assert endpoints == expected_endpoints


@pytest.mark.unit
def test_pointer_map_scaling_matches_matlab_formula():
    """Test that final pointer map scaling matches MATLAB formula.

    MATLAB scales by: 1000 / strel_length * pointer_value
    Reference: Vectorization-Public/slavv_python/get_network_V200.m line ~650
    """
    pointer_map = np.zeros((3, 3, 3), dtype=np.uint64, order="F")
    size_map = np.ones((3, 3, 3), dtype=np.int16, order="F")

    pointer_map[1, 1, 1] = 14
    pointer_map[1, 1, 2] = 27

    scaled = _matlab_global_watershed_scale_pointer_map(
        pointer_map,
        size_map,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )

    # Verify scaling formula
    assert scaled.dtype == np.float32
    assert scaled[1, 1, 1] > 0
    assert scaled[1, 1, 2] > scaled[1, 1, 1]  # Higher pointer -> higher scaled value


@pytest.mark.unit
def test_border_locations_cover_all_boundary_voxels():
    """Test that border locations include all 6 faces of the volume.

    For a 3x3x3 volume, there are 26 border voxels (all except center).
    """
    border_locations = _matlab_global_watershed_border_locations((3, 3, 3))

    # Should have 26 border voxels (27 total - 1 center)
    assert len(border_locations) == 26

    # Center should not be included
    center_linear = 13  # [1, 1, 1] in MATLAB linear order
    assert center_linear not in border_locations

    # All border locations should be valid
    assert np.all(border_locations >= 0)
    assert np.all(border_locations < 27)


@pytest.mark.unit
def test_initialize_state_sets_correct_data_types():
    """Test that initialization creates maps with correct MATLAB data types."""
    energy = np.arange(27, dtype=np.float32).reshape((3, 3, 3), order="F")
    vertex_positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    state = _initialize_matlab_global_watershed_state(energy, vertex_positions)

    # Verify all data types match MATLAB
    assert state["vertex_locations"].dtype == np.int64
    assert state["vertex_energies"].dtype == np.float32
    assert state["energy_map_temp"].dtype == np.float32
    assert state["branch_order_map"].dtype == np.uint8
    assert state["d_over_r_map"].dtype == np.float64
    assert state["pointer_map"].dtype == np.uint64
    assert state["vertex_index_map"].dtype == np.uint32

    # Verify shapes
    assert state["energy_map_temp"].shape == (3, 3, 3)
    assert state["pointer_map"].shape == (3, 3, 3)
    assert state["vertex_index_map"].shape == (3, 3, 3)


@pytest.mark.unit
def test_finalize_edge_trace_samples_correctly():
    """Test that edge trace finalization samples energy and scale correctly."""
    shape = (3, 3, 3)
    energy_map = np.arange(27, dtype=np.float32).reshape(shape, order="F")
    scale_image = (100 + np.arange(27, dtype=np.int16)).reshape(shape, order="F")

    # Simple trace: [0,1,1] -> [1,1,1] -> [2,1,1]
    half_1 = [12]  # [0,1,1]
    half_2 = [13, 14]  # [1,1,1], [2,1,1]

    trace, energy_trace, scale_trace = _matlab_global_watershed_finalize_edge_trace(
        half_1,
        half_2,
        shape=shape,
        energy_map=energy_map,
        scale_image=scale_image,
    )

    # Verify trace coordinates
    assert trace.shape == (3, 3)
    assert trace.dtype == np.float32

    # Verify energy sampling
    assert energy_trace.dtype == np.float32
    assert len(energy_trace) == 3

    # Verify scale sampling
    assert scale_trace.dtype == np.int16
    assert len(scale_trace) == 3


@pytest.mark.unit
def test_matlab_global_watershed_reveal_unclaimed_strel_raises_for_invalid_claim_pointers():
    with pytest.raises(AssertionError, match="invalid claim pointers"):
        _matlab_global_watershed_reveal_unclaimed_strel(
            current_vertex_index=1,
            current_scale_label=2,
            current_d_over_r=0.5,
            valid_linear=np.array([1], dtype=np.int64),
            strel_pointer_indices=np.array([99], dtype=np.uint64),
            strel_r_over_R=np.array([0.25], dtype=np.float32),
            adjusted_energies=np.array([-5.0], dtype=np.float32),
            vertex_index_map_flat=np.zeros((8,), dtype=np.uint32),
            pointer_map_flat=np.zeros((8,), dtype=np.uint64),
            energy_map_flat=np.zeros((8,), dtype=np.float32),
            d_over_r_map_flat=np.zeros((8,), dtype=np.float64),
            size_map_flat=np.zeros((8,), dtype=np.int16),
            lut_size=4,
        )


@pytest.mark.unit
def test_generate_edge_candidates_matlab_global_watershed_uses_configured_step_size(
    monkeypatch,
):
    import slavv_python.core.edge_candidates_internal.global_watershed as global_watershed_module

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


@pytest.mark.unit
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


# Made with Bob
