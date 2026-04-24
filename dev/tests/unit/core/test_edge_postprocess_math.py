from __future__ import annotations

import numpy as np
from source.core._edges.postprocess import (
    _matlab_crop_edges_v200,
    _matlab_edge_endpoint_energy,
    finalize_edges_matlab_style,
    normalize_edges_matlab_style,
    prefilter_edge_indices_for_cleanup_matlab_style,
)


def test_matlab_edge_endpoint_energy_matches_geometric_mean_formula():
    trace = np.array([-4.0, -9.0], dtype=np.float32)

    endpoint_energy = _matlab_edge_endpoint_energy(trace)

    assert endpoint_energy == np.float32(-6.0)


def test_normalize_edges_matlab_style_matches_vectorize_v200_formulas():
    chosen_edges = {
        "energies": np.array([-3.0, -2.0], dtype=np.float32),
        "energy_traces": [
            np.array([-4.0, -9.0], dtype=np.float32),
            np.array([-1.0, -4.0], dtype=np.float32),
        ],
        "traces": [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ],
    }

    normalized = normalize_edges_matlab_style(chosen_edges)

    assert np.allclose(normalized["raw_energies"], np.array([-3.0, -2.0], dtype=np.float32))
    assert np.allclose(
        normalized["edge_endpoint_energies"],
        np.array([-6.0, -2.0], dtype=np.float32),
    )
    assert np.allclose(normalized["energies"], np.array([-0.5, -1.0], dtype=np.float32))
    assert np.allclose(
        normalized["energy_traces"][0],
        np.array([-2.0 / 3.0, -1.5], dtype=np.float32),
    )
    assert np.allclose(
        normalized["energy_traces"][1],
        np.array([-0.5, -2.0], dtype=np.float32),
    )
    assert np.allclose(normalized["raw_energy_traces"][0], np.array([-4.0, -9.0], dtype=np.float32))
    assert np.allclose(normalized["raw_energy_traces"][1], np.array([-1.0, -4.0], dtype=np.float32))


def test_normalize_edges_matlab_style_converts_nan_metric_to_negative_infinity():
    chosen_edges = {
        "energies": np.array([0.0], dtype=np.float32),
        "energy_traces": [np.array([0.0, 0.0], dtype=np.float32)],
    }

    normalized = normalize_edges_matlab_style(chosen_edges)

    assert normalized["edge_endpoint_energies"][0] == np.float32(-0.0)
    assert normalized["energies"][0] == np.float32(-np.inf)
    assert np.isnan(normalized["energy_traces"][0]).all()


def test_normalize_edges_matlab_style_leaves_empty_payload_unchanged():
    chosen_edges = {"energies": np.zeros((0,), dtype=np.float32), "energy_traces": []}

    normalized = normalize_edges_matlab_style(chosen_edges)

    assert normalized is chosen_edges
    assert normalized["energy_traces"] == []


def test_matlab_crop_edges_v200_excludes_edges_that_expand_past_image_bounds():
    excluded = _matlab_crop_edges_v200(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        ],
        [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ],
        [
            np.array([-2.0, -2.0], dtype=np.float32),
            np.array([-2.0, -2.0], dtype=np.float32),
        ],
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        size_of_image=(3, 3, 3),
    )

    assert excluded.tolist() == [True, False]


def test_prefilter_edge_indices_for_cleanup_matlab_style_crops_before_cleanup():
    kept_indices, cropped_count = prefilter_edge_indices_for_cleanup_matlab_style(
        [0, 1, 2],
        traces=[
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]], dtype=np.float32),
            np.array([[2.0, 3.0, 2.0], [3.0, 3.0, 2.0]], dtype=np.float32),
        ],
        scale_traces=[
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ],
        energy_traces=[
            np.array([-6.0, -6.0], dtype=np.float32),
            np.array([-5.0, -5.0], dtype=np.float32),
            np.array([-4.0, -4.0], dtype=np.float32),
        ],
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        size_of_image=(5, 5, 5),
    )

    assert kept_indices == [1, 2]
    assert cropped_count == 1


def test_finalize_edges_matlab_style_only_smooths_and_normalizes_post_cleanup():
    chosen_edges = {
        "traces": [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        ],
        "scale_traces": [
            np.array([0.0, 0.0], dtype=np.float32),
        ],
        "energy_traces": [
            np.array([-4.0, -4.0], dtype=np.float32),
        ],
        "energies": np.array([-4.0], dtype=np.float32),
        "connections": np.array([[1, 2]], dtype=np.int32),
        "connection_sources": ["frontier"],
        "chosen_candidate_indices": np.array([4], dtype=np.int32),
        "diagnostics": {"cropped_edge_count": 1},
    }

    finalized = finalize_edges_matlab_style(
        chosen_edges,
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        size_of_image=(5, 5, 5),
    )

    assert len(finalized["traces"]) == 1
    assert finalized["connections"].tolist() == [[1, 2]]
    assert finalized["chosen_candidate_indices"].tolist() == [4]
    assert finalized["diagnostics"]["cropped_edge_count"] == 1
    assert np.allclose(finalized["scale_traces"][0], np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(finalized["raw_energies"], np.array([-4.0], dtype=np.float32))
    assert np.allclose(finalized["energies"], np.array([-1.0], dtype=np.float32))


