"""Unit tests verifying mathematical equivalence and fallback behavior of Watershed Numba kernels."""

from __future__ import annotations

import numpy as np

from slavv_python.pipeline.edges.watershed.matlab_get_edges_v300_geometry import (
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_frontier_adjusted_neighbor_energies_python,
    _matlab_frontier_directional_suppression_factors_numba_impl,
    _matlab_frontier_directional_suppression_factors_python,
)
from slavv_python.pipeline.edges.watershed.matlab_watershed_heap import (
    _claim_unowned_strel_arrays,
    _claim_unowned_strel_arrays_numba_impl,
)


def test_adjusted_neighbor_energies_python_vs_numba_equivalence():
    """Verify adjusted neighbor energies evaluate identically in Python and Numba."""
    np.random.seed(12345)
    n = 100
    raw_energies = np.random.uniform(-5.0, 5.0, size=n)
    raw_energies[0] = -np.inf
    raw_energies[1] = np.inf
    raw_energies[2] = np.nan

    neighbor_offsets = np.random.randint(-3, 4, size=(n, 3)).astype(np.int32)
    neighbor_unit_vectors = np.random.randn(n, 3)
    neighbor_unit_vectors /= np.linalg.norm(neighbor_unit_vectors, axis=1, keepdims=True)
    neighbor_r_over_R = np.random.uniform(0.0, 2.0, size=n)
    neighbor_scale_indices = np.random.randint(1, 6, size=n).astype(np.int16)
    propagated_scale_index = 2
    current_d_over_r = 0.75
    origin_radius_microns = 2.0
    current_forward_unit = np.array([0.57735, 0.57735, 0.57735])
    microns_per_voxel = np.array([1.0, 1.0, 2.0])
    lumen_radius_microns = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    py_res = _matlab_frontier_adjusted_neighbor_energies_python(
        raw_energies,
        neighbor_offsets=neighbor_offsets,
        neighbor_unit_vectors=neighbor_unit_vectors,
        neighbor_r_over_R=neighbor_r_over_R,
        neighbor_scale_indices=neighbor_scale_indices,
        propagated_scale_index=propagated_scale_index,
        current_d_over_r=current_d_over_r,
        origin_radius_microns=origin_radius_microns,
        current_forward_unit=current_forward_unit,
        microns_per_voxel=microns_per_voxel,
        lumen_radius_microns=lumen_radius_microns,
        radius_tolerance=0.5,
        distance_tolerance=3.0,
    )

    num_res = _matlab_frontier_adjusted_neighbor_energies(
        raw_energies,
        neighbor_offsets=neighbor_offsets,
        neighbor_unit_vectors=neighbor_unit_vectors,
        neighbor_r_over_R=neighbor_r_over_R,
        neighbor_scale_indices=neighbor_scale_indices,
        propagated_scale_index=propagated_scale_index,
        current_d_over_r=current_d_over_r,
        origin_radius_microns=origin_radius_microns,
        current_forward_unit=current_forward_unit,
        microns_per_voxel=microns_per_voxel,
        lumen_radius_microns=lumen_radius_microns,
        radius_tolerance=0.5,
        distance_tolerance=3.0,
    )

    assert np.allclose(py_res, num_res, atol=1e-12, equal_nan=True)


def test_directional_suppression_factors_python_vs_numba_equivalence():
    """Verify directional suppression factors evaluate identically in Python and Numba."""
    np.random.seed(54321)
    n = 50
    neighbor_offsets = np.random.randint(-2, 3, size=(n, 3)).astype(np.int32)
    selected_index = 5
    microns_per_voxel = np.array([0.8, 0.8, 1.6], dtype=np.float64)

    py_supp = _matlab_frontier_directional_suppression_factors_python(
        neighbor_offsets,
        selected_index=selected_index,
        microns_per_voxel=microns_per_voxel,
    )

    num_supp = _matlab_frontier_directional_suppression_factors_numba_impl(
        neighbor_offsets,
        int(selected_index),
        microns_per_voxel,
    )

    assert np.allclose(py_supp, num_supp, atol=1e-12, equal_nan=True)


def test_claim_unowned_strel_arrays_python_vs_numba_equivalence():
    """Verify atomic voxel claiming multi-map updates match between Python and Numba."""
    volume_size = 1000
    lut_size = 50

    # Initial state
    vert_map_py = np.zeros(volume_size, dtype=np.uint32)
    ptr_map_py = np.zeros(volume_size, dtype=np.uint64)
    nrg_map_py = np.full(volume_size, np.inf, dtype=np.float64)
    dor_map_py = np.zeros(volume_size, dtype=np.float64)
    siz_map_py = np.zeros(volume_size, dtype=np.int16)

    # Pre-claim some voxels
    vert_map_py[10:15] = 42
    vert_map_num = vert_map_py.copy()
    ptr_map_num = ptr_map_py.copy()
    nrg_map_num = nrg_map_py.copy()
    dor_map_num = dor_map_py.copy()
    siz_map_num = siz_map_py.copy()

    valid_linear = np.array([8, 9, 10, 11, 12, 15, 16, 17], dtype=np.int64)
    strel_ptrs = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)
    strel_r_over_R = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    adj_energies = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], dtype=np.float64)

    # Run Python claim
    v_py, empty_py = _claim_unowned_strel_arrays(
        current_vertex_index=99,
        current_scale_label=3,
        current_d_over_r=0.5,
        valid_linear=valid_linear,
        strel_pointer_indices=strel_ptrs,
        strel_r_over_R=strel_r_over_R,
        adjusted_energies=adj_energies,
        vertex_index_map_flat=vert_map_py,
        pointer_map_flat=ptr_map_py,
        energy_map_flat=nrg_map_py,
        d_over_r_map_flat=dor_map_py,
        size_map_flat=siz_map_py,
        lut_size=lut_size,
    )

    # Run Numba claim
    v_num, empty_num = _claim_unowned_strel_arrays_numba_impl(
        99,
        3,
        0.5,
        valid_linear,
        strel_ptrs,
        strel_r_over_R,
        adj_energies,
        vert_map_num,
        ptr_map_num,
        nrg_map_num,
        dor_map_num,
        siz_map_num,
        lut_size,
    )

    assert np.array_equal(v_py, v_num)
    assert np.array_equal(empty_py, empty_num)
    assert np.array_equal(vert_map_py, vert_map_num)
    assert np.array_equal(ptr_map_py, ptr_map_num)
    assert np.allclose(nrg_map_py, nrg_map_num, atol=1e-12, equal_nan=True)
    assert np.allclose(dor_map_py, dor_map_num, atol=1e-12, equal_nan=True)
    assert np.array_equal(siz_map_py, siz_map_num)


def test_insert_available_location_in_place_equivalence():
    """Verify in-place available locations insertion matches reference MATLAB slicing exactly."""
    from slavv_python.pipeline.edges.watershed.matlab_watershed_heap import (
        _matlab_global_watershed_insert_available_location,
    )

    np.random.seed(999)
    energy_lookup = np.random.uniform(0.0, 100.0, size=500)

    # Reference implementation using full list copying
    def reference_insert(orig, next_loc, next_nrg, s_idx, is_clear):
        original = list(orig)
        target_energy = float(next_nrg)
        if not original:
            return [int(next_loc)], True
        if s_idx == 1:
            if float(energy_lookup[int(original[0])]) <= target_energy:
                insert_at = 0
            else:
                insert_at = len(original)
                for idx in range(len(original) - 1, -1, -1):
                    if float(energy_lookup[int(original[idx])]) > target_energy:
                        insert_at = idx + 1
                        break
        elif float(energy_lookup[int(original[-1])]) >= target_energy:
            insert_at = len(original) if is_clear else len(original) - 1
        else:
            insert_at = len(original)
            for idx, loc in enumerate(original):
                if float(energy_lookup[int(loc)]) < target_energy:
                    insert_at = idx
                    break
        if not is_clear:
            updated = [*original[:insert_at], int(next_loc), *original[insert_at:-1]]
        else:
            updated = [*original[:insert_at], int(next_loc), *original[insert_at:]]
        return updated, True

    # Test random sequences of insertions
    for s_idx in [1, 2]:
        ref_list = []
        opt_list = []
        is_clear_ref = True
        is_clear_opt = True

        for step in range(50):
            loc = int(np.random.randint(0, 500))
            nrg = float(energy_lookup[loc])
            is_clear_flag = bool(step % 3 == 0)

            ref_list, is_clear_ref = reference_insert(ref_list, loc, nrg, s_idx, is_clear_flag)
            opt_list, is_clear_opt = _matlab_global_watershed_insert_available_location(
                opt_list, loc, nrg, energy_lookup, s_idx, is_clear_flag
            )

            assert ref_list == opt_list, f"Mismatch at step {step}, seed_idx {s_idx}"
            assert is_clear_ref == is_clear_opt


def test_argmin_with_linear_index_tiebreak_python_vs_numba_equivalence():
    """Verify Numba JIT scalar argmin tie-breaking matches Python fallback and NumPy."""
    from slavv_python.pipeline.edges.watershed.matlab_indexing import (
        _argmin_with_linear_index_tiebreak,
        _argmin_with_linear_index_tiebreak_numba_impl,
        _argmin_with_linear_index_tiebreak_python,
    )

    np.random.seed(42)
    # Test distinct values
    energies = np.array([5.0, 2.0, 3.0], dtype=np.float64)
    linear_indices = np.array([10, 20, 30], dtype=np.int64)
    assert _argmin_with_linear_index_tiebreak_python(energies, linear_indices) == 1
    assert _argmin_with_linear_index_tiebreak_numba_impl(energies, linear_indices) == 1
    assert _argmin_with_linear_index_tiebreak(energies, linear_indices) == 1

    # Test ties on energy with different linear indices
    energies_tied = np.array([5.0, 2.0, 2.0, 5.0], dtype=np.float64)
    linear_indices_tied = np.array([10, 30, 20, 40], dtype=np.int64)
    # Tie at idx 1 (lin=30) vs idx 2 (lin=20): lowest linear index is 20 -> idx 2
    assert _argmin_with_linear_index_tiebreak_python(energies_tied, linear_indices_tied) == 2
    assert _argmin_with_linear_index_tiebreak_numba_impl(energies_tied, linear_indices_tied) == 2
    assert _argmin_with_linear_index_tiebreak(energies_tied, linear_indices_tied) == 2

    # Test inf and -inf
    energies_inf = np.array([np.inf, np.inf, -np.inf, -np.inf], dtype=np.float64)
    linear_inf = np.array([10, 5, 200, 100], dtype=np.int64)
    assert _argmin_with_linear_index_tiebreak_python(energies_inf, linear_inf) == 3
    assert _argmin_with_linear_index_tiebreak_numba_impl(energies_inf, linear_inf) == 3
    assert _argmin_with_linear_index_tiebreak(energies_inf, linear_inf) == 3

    # Randomized stress test against NumPy reference
    for _ in range(50):
        n = np.random.randint(1, 100)
        e = np.random.choice([-10.0, 0.0, 1.5, 5.0, np.inf], size=n).astype(np.float64)
        lin = np.random.randint(0, 10000, size=n).astype(np.int64)

        # NumPy reference
        min_e = np.min(e)
        tied = np.flatnonzero(e == min_e)
        expected_idx = int(tied[np.argmin(lin[tied])])

        py_idx = _argmin_with_linear_index_tiebreak_python(e, lin)
        numba_idx = _argmin_with_linear_index_tiebreak_numba_impl(e, lin)
        pub_idx = _argmin_with_linear_index_tiebreak(e, lin)

        assert py_idx == expected_idx
        assert numba_idx == expected_idx
        assert pub_idx == expected_idx


def test_reset_join_locations_single_pass_equivalence():
    """Verify single-pass reset join locations matches reference quadratic scan exactly."""
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

    np.random.seed(777)
    for _ in range(100):
        length = np.random.randint(0, 50)
        available = np.random.randint(0, 100, size=length).tolist()
        num_targets = np.random.randint(0, 10)
        targets = np.random.randint(0, 100, size=num_targets).astype(np.int64)
        is_clear = bool(np.random.choice([True, False]))

        ref_updated, ref_clear = reference_reset(available, targets, is_clear)
        opt_updated, opt_clear = _matlab_global_watershed_reset_join_locations(
            available, next_vertex_locations=targets, is_current_location_clear=is_clear
        )

        assert ref_updated == opt_updated
        assert ref_clear == opt_clear


def test_structuring_element_offsets_memoization():
    """Verify structuring element offsets memoization returns identical arrays and hits cache."""
    from slavv_python.pipeline.edges.selection import (
        _construct_structuring_element_offsets_matlab,
        _construct_structuring_element_offsets_matlab_cached,
    )

    _construct_structuring_element_offsets_matlab_cached.cache_clear()

    radii1 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    offsets1 = _construct_structuring_element_offsets_matlab(radii1)
    info1 = _construct_structuring_element_offsets_matlab_cached.cache_info()
    assert info1.misses == 1
    assert info1.hits == 0

    # Second call with same radii should hit cache
    offsets2 = _construct_structuring_element_offsets_matlab(radii1)
    info2 = _construct_structuring_element_offsets_matlab_cached.cache_info()
    assert info2.hits == 1
    assert np.array_equal(offsets1, offsets2)

