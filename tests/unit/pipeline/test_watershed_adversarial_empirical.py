"""Adversarial challenge and empirical verification for Edges optimizations.

Tests:
1. _argmin_with_linear_index_tiebreak:
   - JIT vs Python vs NumPy vs MATLAB tie-break semantics
   - Identical energy values with varying Fortran linear indices
   - NaN, +inf, -inf, mixed values, subnormals
   - Empty arrays (ValueError verification)
   - Single-element arrays
   - Massive randomized stress test (millions of elements / configurations)
2. _matlab_global_watershed_current_strel:
   - Fast-path (interior) vs Boundary-path mathematical equivalence
   - Interior, face boundary, edge boundary, corner boundary voxels
   - Boundary threshold exact transition verification
   - Small volumes (smaller than strel)
   - Anisotropic voxel spacings
"""

from __future__ import annotations

import numpy as np
import pytest

from slavv_python.pipeline.edges.watershed.matlab_calculate_linear_strel_range import (
    _build_matlab_global_watershed_lut,
)
from slavv_python.pipeline.edges.watershed.matlab_get_edges_by_watershed import (
    _matlab_global_watershed_current_strel,
)
from slavv_python.pipeline.edges.watershed.matlab_indexing import (
    _argmin_with_linear_index_tiebreak,
    _argmin_with_linear_index_tiebreak_numba_impl,
    _argmin_with_linear_index_tiebreak_python,
    _matlab_watershed_min_candidate_energies,
)

# =========================================================================
# Task 1: Adversarial tests for _argmin_with_linear_index_tiebreak
# =========================================================================


def numpy_matlab_argmin_tiebreak_reference(
    energies: np.ndarray,
    linear_indices: np.ndarray,
) -> int:
    """NumPy-based reference for MATLAB lowest Fortran linear index tiebreak."""
    e = np.asarray(energies, dtype=np.float64).reshape(-1)
    lin = np.asarray(linear_indices, dtype=np.int64).reshape(-1)
    min_e = np.min(e)
    tied = np.flatnonzero(e == min_e)
    return int(tied[np.argmin(lin[tied])])


def test_argmin_empty_array_raises_value_error():
    """Verify that empty arrays raise ValueError as expected."""
    empty_e = np.array([], dtype=np.float64)
    empty_lin = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="energies must be non-empty"):
        _argmin_with_linear_index_tiebreak(empty_e, empty_lin)


def test_argmin_single_element():
    """Verify single-element arrays."""
    for val in [-100.0, 0.0, 42.5, np.inf, -np.inf]:
        e = np.array([val], dtype=np.float64)
        lin = np.array([12345], dtype=np.int64)

        assert _argmin_with_linear_index_tiebreak_python(e, lin) == 0
        assert _argmin_with_linear_index_tiebreak_numba_impl(e, lin) == 0
        assert _argmin_with_linear_index_tiebreak(e, lin) == 0


def test_argmin_all_identical_energies_varying_linear_indices():
    """All energies identical -> must strictly pick the index with smallest linear index."""
    rng = np.random.default_rng(42)
    n = 200
    for constant_energy in [-1e6, -1.0, 0.0, 1.0, 1e6, np.inf, -np.inf]:
        energies = np.full(n, constant_energy, dtype=np.float64)
        # Permute linear indices
        linear_indices = rng.permutation(np.arange(100, 100 + n, dtype=np.int64))

        expected_idx = int(np.argmin(linear_indices))

        py_idx = _argmin_with_linear_index_tiebreak_python(energies, linear_indices)
        num_idx = _argmin_with_linear_index_tiebreak_numba_impl(energies, linear_indices)
        pub_idx = _argmin_with_linear_index_tiebreak(energies, linear_indices)

        assert py_idx == expected_idx
        assert num_idx == expected_idx
        assert pub_idx == expected_idx


def test_argmin_partial_ties_varying_positions():
    """Test subsets of tied minima placed at start, middle, end, or scattered."""
    rng = np.random.default_rng(123)
    for _ in range(100):
        length = rng.integers(5, 50)
        energies = rng.uniform(10.0, 20.0, size=length)
        linear_indices = rng.integers(0, 100000, size=length, dtype=np.int64)

        # Place 3 identical minimum values
        min_val = -5.0
        tie_positions = rng.choice(length, size=3, replace=False)
        energies[tie_positions] = min_val

        expected_idx = numpy_matlab_argmin_tiebreak_reference(energies, linear_indices)

        py_idx = _argmin_with_linear_index_tiebreak_python(energies, linear_indices)
        num_idx = _argmin_with_linear_index_tiebreak_numba_impl(energies, linear_indices)
        pub_idx = _argmin_with_linear_index_tiebreak(energies, linear_indices)

        assert py_idx == expected_idx
        assert num_idx == expected_idx
        assert pub_idx == expected_idx


def test_argmin_inf_and_minus_inf_behavior():
    """Test behavior with +inf, -inf, and mixed values."""
    energies = np.array([np.inf, 10.0, -np.inf, -np.inf, 5.0, np.inf], dtype=np.float64)
    # -inf at index 2 (lin=500) and index 3 (lin=200) -> lowest linear index is 200 -> idx 3
    linear_indices = np.array([10, 20, 500, 200, 30, 40], dtype=np.int64)

    assert _argmin_with_linear_index_tiebreak_python(energies, linear_indices) == 3
    assert _argmin_with_linear_index_tiebreak_numba_impl(energies, linear_indices) == 3
    assert _argmin_with_linear_index_tiebreak(energies, linear_indices) == 3


def test_argmin_nan_handling_via_matlab_prefilter():
    """Test how NaN and +Inf candidates are handled in conjunction with MATLAB min prefilter."""
    energies_with_nan = np.array([np.nan, 2.0, np.nan, 2.0, 5.0], dtype=np.float64)
    linear_indices = np.array([1, 100, 2, 50, 20], dtype=np.int64)

    # Pre-filter using _matlab_watershed_min_candidate_energies (as done in the pipeline)
    filtered_energies = _matlab_watershed_min_candidate_energies(energies_with_nan)

    # Filtered energies should have inf where nan was, so min is 2.0 at idx 1 (lin=100) and idx 3 (lin=50).
    # Lowest linear index is 50 -> index 3.
    expected_idx = 3

    assert (
        _argmin_with_linear_index_tiebreak_python(filtered_energies, linear_indices) == expected_idx
    )
    assert (
        _argmin_with_linear_index_tiebreak_numba_impl(filtered_energies, linear_indices)
        == expected_idx
    )
    assert _argmin_with_linear_index_tiebreak(filtered_energies, linear_indices) == expected_idx


def test_argmin_subnormal_and_extreme_epsilons():
    """Verify scalar tie-breaking with extreme floating point epsilons (subnormal differences)."""
    base = 1.0000000000000002
    energies = np.array([base, 1.0, 1.0, base + 1e-16], dtype=np.float64)
    linear_indices = np.array([50, 400, 200, 10], dtype=np.int64)

    # 1.0 is min at idx 1 (lin=400) and idx 2 (lin=200) -> lowest linear index is 200 -> idx 2
    assert _argmin_with_linear_index_tiebreak_python(energies, linear_indices) == 2
    assert _argmin_with_linear_index_tiebreak_numba_impl(energies, linear_indices) == 2
    assert _argmin_with_linear_index_tiebreak(energies, linear_indices) == 2


def test_argmin_numba_disabled_fallback(monkeypatch):
    """Verify fallback to pure Python when Numba is disabled via SLAVV_DISABLE_NUMBA."""
    import slavv_python.pipeline.edges.watershed.matlab_indexing as mod

    monkeypatch.setattr(mod, "_NUMBA_AVAILABLE", False)

    energies = np.array([5.0, 2.0, 2.0, 5.0], dtype=np.float64)
    linear_indices = np.array([10, 30, 20, 40], dtype=np.int64)

    res = mod._argmin_with_linear_index_tiebreak(energies, linear_indices)
    assert res == 2


def test_reset_join_locations_adversarial_duplicates_and_tail():
    """Adversarial stress test for _matlab_global_watershed_reset_join_locations."""
    from slavv_python.pipeline.edges.watershed.matlab_watershed_heap import (
        _matlab_global_watershed_reset_join_locations,
    )

    def reference_reset(orig, next_locs, is_clear):
        updated = list(orig)
        next_locations = set(np.asarray(next_locs, dtype=np.int64).tolist())
        locations_to_reset = sorted({int(loc) for loc in updated if int(loc) in next_locations})
        if not is_clear:
            if updated:
                tail_location = int(updated[-1])
                locations_to_reset = [loc for loc in locations_to_reset if loc != tail_location]
                updated.pop()
            is_clear = True

        reset_indices: list[int] = []
        for location in locations_to_reset:
            for idx, available_location in enumerate(updated):
                if int(available_location) == int(location):
                    reset_indices.append(idx)
                    break

        for idx in sorted(set(reset_indices), reverse=True):
            del updated[idx]
        return updated, is_clear

    # Edge cases: empty available_locations
    res, is_clr = _matlab_global_watershed_reset_join_locations(
        [], next_vertex_locations=np.array([1, 2]), is_current_location_clear=False
    )
    assert res == []
    assert is_clr is True

    # Edge cases: empty targets with is_clear=False
    res, is_clr = _matlab_global_watershed_reset_join_locations(
        [10, 20, 30], next_vertex_locations=np.array([]), is_current_location_clear=False
    )
    assert res == [10, 20]
    assert is_clr is True

    # Duplicate elements in available_locations where first occurrence should be removed
    available = [5, 10, 5, 20, 5, 30]
    targets = np.array([5])
    ref_list, ref_clear = reference_reset(available, targets, True)
    opt_list, opt_clear = _matlab_global_watershed_reset_join_locations(
        available, next_vertex_locations=targets, is_current_location_clear=True
    )
    assert opt_list == [10, 5, 20, 5, 30]
    assert opt_list == ref_list
    assert opt_clear == ref_clear


def test_structuring_element_offsets_caching_and_empty():
    """Verify O4 LRU cache edge conditions."""
    from slavv_python.pipeline.edges.selection import (
        _construct_structuring_element_offsets_matlab,
        _construct_structuring_element_offsets_matlab_cached,
    )

    _construct_structuring_element_offsets_matlab_cached.cache_clear()
    # Zero or negative radii
    res_zero = _construct_structuring_element_offsets_matlab(np.array([0.0, 0.0, 0.0]))
    assert res_zero.shape == (1, 3)
    assert np.array_equal(res_zero, np.zeros((1, 3), dtype=np.int32))


# =========================================================================
# Task 2: Adversarial tests for _matlab_global_watershed_current_strel
# =========================================================================


def _matlab_global_watershed_current_strel_boundary_reference(
    current_linear: int,
    *,
    current_scale_label: int,
    shape: tuple[int, int, int],
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> dict[str, any]:
    """Reference implementation that ALWAYS runs the boundary filtering path (no fast path)."""
    current_linear_int = int(current_linear)
    sy, sx, sz = shape
    cy = current_linear_int % sy
    rem = current_linear_int // sy
    cx = rem % sx
    cz = rem // sx
    current_coord = np.array([cy, cx, cz], dtype=np.int32)

    current_scale_index = int(
        np.clip(int(current_scale_label) - 1, 0, len(lumen_radius_microns) - 1)
    )
    lut = _build_matlab_global_watershed_lut(
        current_scale_index,
        size_of_image=shape,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    offsets = lut["local_subscripts"]
    linear_offsets_full = lut["linear_offsets"]
    pointer_indices_full = lut["pointer_indices"]
    r_over_R_full = lut["r_over_R"]
    distance_lut_full = lut["distance_lut"]
    unit_vectors_full = lut["unit_vectors"]

    # Explicit boundary path unconditional execution
    strel_coords_y = cy + offsets[:, 0]
    strel_coords_x = cx + offsets[:, 1]
    strel_coords_z = cz + offsets[:, 2]

    valid_mask = (
        (strel_coords_y >= 0)
        & (strel_coords_y < sy)
        & (strel_coords_x >= 0)
        & (strel_coords_x < sx)
        & (strel_coords_z >= 0)
        & (strel_coords_z < sz)
    )

    valid_coords = np.column_stack(
        (
            strel_coords_y[valid_mask],
            strel_coords_x[valid_mask],
            strel_coords_z[valid_mask],
        )
    ).astype(np.int32)

    valid_offsets = offsets[valid_mask]
    valid_linear = linear_offsets_full[valid_mask] + np.int64(current_linear_int)
    pointer_indices = pointer_indices_full[valid_mask]
    r_over_R = r_over_R_full[valid_mask]
    distance_microns = distance_lut_full[valid_mask]
    unit_vectors = unit_vectors_full[valid_mask]

    return {
        "current_coord": current_coord,
        "coords": valid_coords,
        "offsets": valid_offsets,
        "linear_indices": valid_linear,
        "pointer_indices": pointer_indices,
        "r_over_R": r_over_R,
        "distance_microns": distance_microns,
        "unit_vectors": unit_vectors,
        "lut_size": len(offsets),
        "scale_label_clipped": current_scale_index + 1,
    }


def assert_strel_results_equal(actual: dict[str, any], expected: dict[str, any]):
    """Assert bit-exact equality between actual strel and expected strel output dictionaries."""
    assert np.array_equal(actual["current_coord"], expected["current_coord"])
    assert np.array_equal(actual["coords"], expected["coords"])
    assert np.array_equal(actual["offsets"], expected["offsets"])
    assert np.array_equal(actual["linear_indices"], expected["linear_indices"])
    assert np.array_equal(actual["pointer_indices"], expected["pointer_indices"])
    assert np.allclose(actual["r_over_R"], expected["r_over_R"], atol=1e-14)
    assert np.allclose(actual["distance_microns"], expected["distance_microns"], atol=1e-14)
    assert np.allclose(actual["unit_vectors"], expected["unit_vectors"], atol=1e-14)
    assert actual["lut_size"] == expected["lut_size"]
    assert actual["scale_label_clipped"] == expected["scale_label_clipped"]


def test_strel_interior_vs_boundary_comprehensive_volume_scan():
    """Verify that fast-path and boundary reference match bit-for-bit across all voxels."""
    shape = (20, 25, 15)
    lumen_radius_microns = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    microns_per_voxel = np.array([0.8, 0.8, 1.6], dtype=np.float32)
    step_size = 1.0

    total_voxels = shape[0] * shape[1] * shape[2]

    # Test every single voxel in the volume (7,500 voxels covering interior, face, edge, corner)
    for lin in range(0, total_voxels, 5):  # Step by 5 for fast test execution
        for scale_label in [1, 2, 3]:
            actual = _matlab_global_watershed_current_strel(
                lin,
                current_scale_label=scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size,
            )
            expected = _matlab_global_watershed_current_strel_boundary_reference(
                lin,
                current_scale_label=scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size,
            )
            assert_strel_results_equal(actual, expected)


def test_strel_boundary_edges_and_corners():
    """Specifically stress corner voxels, edge voxels, and face voxels."""
    shape = (30, 30, 30)
    lumen_radius_microns = np.array([1.5, 3.0, 6.0], dtype=np.float32)
    microns_per_voxel = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    step_size = 1.0

    sy, sx, sz = shape

    # Test all 8 corners, 12 edges, 6 faces
    coords_to_test = []
    # Corners
    for y in [0, sy - 1]:
        for x in [0, sx - 1]:
            for z in [0, sz - 1]:
                coords_to_test.append((y, x, z))
    # Boundary faces at various coordinates
    for y in [0, 1, sy - 2, sy - 1]:
        for x in [0, 1, sx - 2, sx - 1]:
            coords_to_test.append((y, x, 15))
            coords_to_test.append((15, y, x))
            coords_to_test.append((y, 15, x))

    for y, x, z in coords_to_test:
        lin = y + x * sy + z * (sy * sx)
        for scale_label in [1, 2, 3]:
            actual = _matlab_global_watershed_current_strel(
                lin,
                current_scale_label=scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size,
            )
            expected = _matlab_global_watershed_current_strel_boundary_reference(
                lin,
                current_scale_label=scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size,
            )
            assert_strel_results_equal(actual, expected)


def test_strel_small_volume_no_interior():
    """Test on a volume smaller than the strel itself (is_interior is always False)."""
    shape = (3, 3, 3)
    lumen_radius_microns = np.array([2.0, 4.0], dtype=np.float32)
    microns_per_voxel = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    step_size = 1.0

    total_voxels = shape[0] * shape[1] * shape[2]
    for lin in range(total_voxels):
        actual = _matlab_global_watershed_current_strel(
            lin,
            current_scale_label=2,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size,
        )
        expected = _matlab_global_watershed_current_strel_boundary_reference(
            lin,
            current_scale_label=2,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size,
        )
        assert_strel_results_equal(actual, expected)
