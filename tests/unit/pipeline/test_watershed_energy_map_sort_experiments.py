"""Cheap experiments for the residual-hub energy-map / sort_edges hypothesis.

The full-volume 180709_E pair (26444, 38584) vs (34897, 38584) is slow to
re-emit. These experiments use a 3x3x3 claim map, a 3-edge toy hub, and the
crop raw dump (when present) to falsify the same claims:

1. MATLAB ``sort_edges`` ranks by ``max`` of the *claimed/penalized*
   ``energy_map`` (L445 write, L846 sample), not the original energy field.
2. Sampling the original field inverts that rank and degree-excess keeps the
   worse partner after a later resampled-max tie.
3. On crop, raw undirected pair sets already match — discovery is not the
   residual class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slavv_python.analytics.parity.experiments import (
    ArtifactClass,
    compare_same_class_pair_sets,
    load_edge_artifact,
)
from slavv_python.pipeline.edges.cleanup import remove_excess_vertex_degrees
from slavv_python.pipeline.edges.selection_payloads import (
    matlab_sort_edge_indices_by_raw_max,
    prepare_candidate_indices_for_cleanup,
)
from slavv_python.pipeline.edges.watershed.matlab_get_edges_by_watershed import (
    _matlab_global_watershed_assemble_results,
    _matlab_global_watershed_finalize_edge_trace,
)
from slavv_python.pipeline.edges.watershed.matlab_watershed_heap import VoxelClaimMap

_REPO = Path(__file__).resolve().parents[3]
_CROP_PY = _REPO / "workspace/runs/oracle_180709_E/crop_M_exact_v3/04_Edges/candidates.pkl"
_CROP_MAT = _REPO / "workspace/scratch/matlab_edge_dump/raw_watershed_candidates.mat"


@pytest.mark.unit
def test_experiment_traces_must_sample_claimed_energy_map_not_original() -> None:
    """Same voxels, two maps: original max stays deep; claimed map max is 0.

    Miniature of the residual traces: identical path, MATLAB L846 samples the
    map that received the L445 penalized write.
    """
    shape = (3, 3, 3)
    original = np.full(shape, -9.24, dtype=np.float64, order="F")
    original[1, 1, 1] = -42.0
    claim_map = VoxelClaimMap(
        shape,
        np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        original,
    )
    vertex_linear = int(claim_map.vertex_locations[0])
    claimed_linear = 22  # MATLAB [Y,X,Z] = (1, 1, 2)
    claim_map.energy_flat[claimed_linear] = 0.0

    _, from_claimed, _ = _matlab_global_watershed_finalize_edge_trace(
        [vertex_linear],
        [claimed_linear],
        shape=shape,
        energy_map=claim_map.energy_map,
        scale_image=None,
    )
    _, from_original, _ = _matlab_global_watershed_finalize_edge_trace(
        [vertex_linear],
        [claimed_linear],
        shape=shape,
        energy_map=original,
        scale_image=None,
    )
    assert float(np.nanmax(from_claimed)) == 0.0
    assert float(np.nanmax(from_original)) == pytest.approx(-9.24)
    assert claim_map.energy_flat[vertex_linear] == float("-inf")


@pytest.mark.unit
def test_experiment_raw_max_rank_not_resampled_tie_decides_degree_excess() -> None:
    """Toy hub: extra emitted first, worse raw max, equal resampled max.

    Degree-excess (max degree 2, three incident edges) must drop the extra
    after MATLAB ``sort_edges`` + ``clean_edge_pairs``, matching the full
    residual mechanism without a 84k-candidate volume.
    """
    extra, oracle, keeper = 1, 2, 4
    hub = 3
    connections = np.array(
        [[extra, hub], [oracle, hub], [keeper, hub]],
        dtype=np.int32,
    )
    raw_traces = [
        np.array([-np.inf, 0.0, -np.inf], dtype=np.float64),
        np.array([-np.inf, -0.24, -np.inf], dtype=np.float64),
        np.array([-np.inf, -1.0, -np.inf], dtype=np.float64),
    ]
    tied = np.array([-4.870152991855598, -4.870152991855598], dtype=np.float64)
    resampled_traces = [tied.copy(), tied.copy(), tied.copy()]

    ranked = matlab_sort_edge_indices_by_raw_max(raw_traces, [0, 1, 2])
    assert ranked == [2, 1, 0], "keeper, oracle, extra (best raw max first)"

    ordered = prepare_candidate_indices_for_cleanup(
        connections,
        np.array([-4.87, -4.87, -4.87], dtype=np.float64),
        resampled_traces,
        {},
        subset_indices=ranked,
        reject_nonnegative_energy_edges=False,
    )
    assert ordered == [2, 1, 0]

    keep = remove_excess_vertex_degrees(connections[ordered], np.zeros(3), max_degree=2)
    kept_pairs = {
        tuple(sorted((int(connections[i, 0]), int(connections[i, 1]))))
        for i, kept in zip(ordered, keep.tolist(), strict=True)
        if kept
    }
    assert tuple(sorted((oracle, hub))) in kept_pairs
    assert tuple(sorted((keeper, hub))) in kept_pairs
    assert tuple(sorted((extra, hub))) not in kept_pairs


@pytest.mark.unit
def test_experiment_original_field_sort_keeps_the_extra() -> None:
    """Control: ranking the original-field maxes inverts the survivor."""
    extra, oracle, hub = 1, 2, 3
    connections = np.array([[extra, hub], [oracle, hub]], dtype=np.int32)
    original_field_traces = [
        np.array([-19.2, -9.24, -15.3], dtype=np.float64),
        np.array([-16.4, -7.73, -15.3], dtype=np.float64),
    ]
    ranked = matlab_sort_edge_indices_by_raw_max(original_field_traces, [0, 1])
    assert ranked == [0, 1], "extra looks better on the original field"
    keep = remove_excess_vertex_degrees(connections[ranked], np.zeros(2), max_degree=1)
    kept = [
        tuple(sorted((int(connections[i, 0]), int(connections[i, 1]))))
        for i, flag in zip(ranked, keep.tolist(), strict=True)
        if flag
    ]
    assert kept == [(extra, hub)]


def _load_matlab_crop_pairs() -> np.ndarray:
    return load_edge_artifact(_CROP_MAT).connections


@pytest.mark.unit
@pytest.mark.skipif(
    not _CROP_PY.is_file() or not _CROP_MAT.is_file(), reason="crop artifacts absent"
)
def test_experiment_crop_raw_pair_sets_already_match() -> None:
    """Crop-scale check of the full-volume discovery claim: pair sets match."""
    report = compare_same_class_pair_sets(
        load_edge_artifact(_CROP_PY).connections,
        _load_matlab_crop_pairs(),
        left_class=ArtifactClass.RAW_CANDIDATE_SET,
        right_class=ArtifactClass.RAW_CANDIDATE_SET,
    )
    assert report.n_left == 19225
    assert report.n_right == 19225
    assert report.n_intersection == 19225
    assert report.n_only_left == 0
    assert report.n_only_right == 0


@pytest.mark.unit
def test_production_watershed_bakes_claimed_not_original_field_energy() -> None:
    """ADR 0013: assembled energy_traces must match claim_map, not original field.

    Paint a penalized claim write on an interior voxel, then ensure a production
    finalize sample of that path carries the claimed value (0.0) rather than the
    original-field deep negative.
    """
    shape = (3, 3, 3)
    original = np.full(shape, -9.24, dtype=np.float64, order="F")
    claim_map = VoxelClaimMap(
        shape,
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32),
        original,
    )
    start = int(claim_map.vertex_locations[0])
    end = int(claim_map.vertex_locations[1])
    claimed_linear = 13  # interior between seeds in 3x3x3 F-order
    claim_map.energy_flat[claimed_linear] = 0.0

    assembled = _matlab_global_watershed_assemble_results(
        edge_pairs=[(1, 2)],
        edge_halves=[([start], [claimed_linear, end])],
        shape=shape,
        energy_map_matlab=claim_map.energy_map,
        original_scale_image_matlab=None,
        vertex_positions=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32),
        vertex_index_map=claim_map.vertex_index_map,
        pointer_map=claim_map.pointer_map,
        size_map=np.ones(shape, dtype=np.int16, order="F"),
        d_over_r_map=claim_map.d_over_r_map,
        branch_order_map=claim_map.branch_order_map,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones(3, dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )
    energy_trace = np.asarray(assembled["energy_traces"][0], dtype=np.float64)
    assert float(np.nanmax(energy_trace)) == 0.0
    assert 0.0 in energy_trace.tolist()
    finite = energy_trace[np.isfinite(energy_trace)]
    assert -9.24 not in finite.tolist()


@pytest.mark.unit
def test_experiment_claimed_raw_max_ranks_oracle_ahead_of_extra() -> None:
    """E1: claimed maxes (MATLAB L846) put oracle before residual extra."""
    extra, oracle, hub = 1, 2, 3
    connections = np.array([[extra, hub], [oracle, hub]], dtype=np.int32)
    claimed_traces = [
        np.array([-np.inf, 0.0, -np.inf], dtype=np.float64),
        np.array([-np.inf, -0.239, -np.inf], dtype=np.float64),
    ]
    ranked = matlab_sort_edge_indices_by_raw_max(claimed_traces, [0, 1])
    assert ranked == [1, 0], "oracle (claimed -0.239) before extra (claimed 0.0)"
    keep = remove_excess_vertex_degrees(connections[ranked], np.zeros(2), max_degree=1)
    kept = [
        tuple(sorted((int(connections[i, 0]), int(connections[i, 1]))))
        for i, flag in zip(ranked, keep.tolist(), strict=True)
        if flag
    ]
    assert kept == [(oracle, hub)]


@pytest.mark.unit
@pytest.mark.skipif(
    not _CROP_PY.is_file() or not _CROP_MAT.is_file(), reason="crop artifacts absent"
)
def test_experiment_crop_stored_python_max_is_not_matlab_claimed_max() -> None:
    """Stored crop traces still look like the original field, not L846.

    After a crop Edges regen with ``claim_map.energy_map`` sampling this
    disagreement should shrink; until then it is the cheap crop reproduction
    of the full-volume energy-source bug.
    """
    import h5py

    from slavv_python.utils.safe_unpickle import safe_load

    python = safe_load(_CROP_PY)
    py_conn = np.asarray(python["connections"], dtype=np.int64)
    py_energy = python["energy_traces"]
    py_max_by_pair: dict[tuple[int, int], float] = {}
    for index, (a, b) in enumerate(py_conn.tolist()):
        key = (int(a), int(b)) if a < b else (int(b), int(a))
        py_max_by_pair[key] = float(np.nanmax(np.asarray(py_energy[index], dtype=np.float64)))

    with h5py.File(_CROP_MAT, "r") as handle:
        mat_pairs = _load_matlab_crop_pairs()
        refs = handle["edge_energies"][0]
        disagreed = 0
        compared = 0
        for index, (a, b) in enumerate(mat_pairs.tolist()):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            if key not in py_max_by_pair:
                continue
            mat_trace = np.asarray(handle[refs[index]], dtype=np.float64).reshape(-1)
            mat_max = float(np.nanmax(mat_trace))
            if not np.isfinite(mat_max) or not np.isfinite(py_max_by_pair[key]):
                continue
            compared += 1
            if abs(py_max_by_pair[key] - mat_max) > 1e-3:
                disagreed += 1
            if compared >= 200:
                break

    assert compared >= 50
    assert disagreed > compared // 2, (
        f"expected stored Python crop maxes to disagree with MATLAB claimed "
        f"maxes on most pairs; disagreed={disagreed}/{compared}"
    )
