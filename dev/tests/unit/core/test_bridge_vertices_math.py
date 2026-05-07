from __future__ import annotations

import numpy as np

from source.core.edges_internal.bridge_insertion import (
    _matlab_bridge_search_target,
    add_vertices_to_edges_matlab_style,
)
from source.core.network import construct_network
from source.core.vertices_internal.vertex_painting import (
    paint_vertex_center_image,
    paint_vertex_image,
)


def test_add_vertices_to_edges_matlab_style_inserts_bridge_vertex_and_splits_parent():
    energy = np.zeros((8, 8, 4), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)
    energy[1:6, 3, 1] = np.array([-9.0, -9.0, -10.0, -9.0, -9.0], dtype=np.float32)
    energy[3, 4:6, 1] = np.array([-8.0, -8.0], dtype=np.float32)

    vertices = {
        "positions": np.array(
            [[1.0, 3.0, 1.0], [5.0, 3.0, 1.0], [3.0, 5.0, 1.0]],
            dtype=np.float32,
        ),
        "scales": np.array([0, 0, 0], dtype=np.int16),
        "energies": np.array([-9.0, -9.0, -8.0], dtype=np.float32),
    }
    chosen_edges = {
        "traces": [
            np.array(
                [
                    [1.0, 3.0, 1.0],
                    [2.0, 3.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [4.0, 3.0, 1.0],
                    [5.0, 3.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array([[3.0, 5.0, 1.0], [3.0, 4.0, 1.0], [3.0, 3.0, 1.0]], dtype=np.float32),
        ],
        "scale_traces": [
            np.zeros((5,), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
        ],
        "energy_traces": [
            np.array([-9.0, -9.0, -10.0, -9.0, -9.0], dtype=np.float32),
            np.array([-8.0, -8.0, -10.0], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [2, 1]], dtype=np.int32),
        "connection_sources": ["frontier", "frontier"],
        "chosen_candidate_indices": np.array([5, 6], dtype=np.int32),
        "diagnostics": {},
    }

    bridged = add_vertices_to_edges_matlab_style(
        chosen_edges,
        vertices,
        energy=energy,
        scale_indices=scale_indices,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([0.49], dtype=np.float32),
        lumen_radius_pixels_axes=np.array([[0.49, 0.49, 0.49]], dtype=np.float32),
        size_of_image=energy.shape,
    )

    assert np.allclose(
        bridged["bridge_vertex_positions"], np.array([[2.0, 3.0, 1.0]], dtype=np.float32)
    )
    assert bridged["bridge_vertex_scales"].tolist() == [1]
    assert np.allclose(bridged["bridge_vertex_energies"], np.array([-9.0], dtype=np.float32))
    assert sorted(bridged["connections"].tolist()) == [[0, 3], [2, 3], [3, 1]]
    assert len(bridged["traces"]) == 3
    assert bridged["bridge_edges"]["connections"].tolist() == [[3, -1]]
    assert bridged["bridge_edges"]["edges2vertices"].tolist() == [[4, 0]]
    assert np.allclose(
        bridged["bridge_edges"]["traces"][0],
        np.array([[3.0, 3.0, 1.0], [2.0, 3.0, 1.0]], dtype=np.float32),
    )
    assert np.allclose(
        bridged["bridge_edges"]["mean_edge_energies"], np.array([-9.5], dtype=np.float32)
    )
    assert np.allclose(bridged["bridge_edges"]["energies"], np.array([-9.5], dtype=np.float32))
    assert bridged["diagnostics"]["bridge_vertex_count"] == 1
    assert bridged["diagnostics"]["bridge_edge_count"] == 1


def test_construct_network_uses_bridge_vertices_when_connections_reference_them():
    edges = {
        "traces": [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        ],
        "scale_traces": [np.zeros((3,), dtype=np.float32) for _ in range(3)],
        "energy_traces": [np.array([-5.0, -5.0, -5.0], dtype=np.float32) for _ in range(3)],
        "connections": np.array([[0, 3], [3, 1], [2, 3]], dtype=np.int32),
        "bridge_vertex_positions": np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        "lumen_radius_microns": np.array([0.49], dtype=np.float32),
    }
    vertices = {
        "positions": np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
            dtype=np.float32,
        ),
    }

    network = construct_network(
        edges,
        vertices,
        {"microns_per_voxel": [1.0, 1.0, 1.0], "remove_cycles": False},
    )

    assert len(network["adjacency_list"]) == 4
    assert set(network["adjacency_list"][3]) == {0, 1, 2}


def test_matlab_bridge_search_target_can_turn_through_branch_to_existing_vertex():
    energy = np.zeros((9, 9, 4), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)
    parent_trace = np.array(
        [[2.0, 4.0, 1.0], [3.0, 4.0, 1.0], [4.0, 4.0, 1.0], [5.0, 4.0, 1.0], [6.0, 4.0, 1.0]],
        dtype=np.float32,
    )
    branch_trace = np.array(
        [[5.0, 4.0, 1.0], [5.0, 5.0, 1.0], [5.0, 6.0, 1.0]],
        dtype=np.float32,
    )
    child_trace = np.array(
        [[4.0, 6.0, 1.0], [4.0, 5.0, 1.0], [4.0, 4.0, 1.0]],
        dtype=np.float32,
    )
    energy[
        parent_trace[:, 0].astype(np.int32),
        parent_trace[:, 1].astype(np.int32),
        parent_trace[:, 2].astype(np.int32),
    ] = np.array([-5.0, -1.0, -10.0, -8.0, -7.0], dtype=np.float32)
    energy[
        branch_trace[:, 0].astype(np.int32),
        branch_trace[:, 1].astype(np.int32),
        branch_trace[:, 2].astype(np.int32),
    ] = np.array([-8.0, -9.0, -11.0], dtype=np.float32)
    energy[
        child_trace[:, 0].astype(np.int32),
        child_trace[:, 1].astype(np.int32),
        child_trace[:, 2].astype(np.int32),
    ] = np.array([-4.0, -4.0, -10.0], dtype=np.float32)

    vertex_positions = np.array(
        [[2.0, 4.0, 1.0], [6.0, 4.0, 1.0], [4.0, 6.0, 1.0], [5.0, 6.0, 1.0]],
        dtype=np.float32,
    )
    vertex_scales = np.zeros((4,), dtype=np.int16)
    vertex_radii = np.array([[2.1, 2.1, 0.49]], dtype=np.float32)
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    vertex_volume_image = paint_vertex_image(
        vertex_positions,
        vertex_scales,
        vertex_radii,
        energy.shape,
    ).astype(bool, copy=False)

    target = _matlab_bridge_search_target(
        overlap_linear_index=int(4 + 4 * energy.shape[0] + 1 * energy.shape[0] * energy.shape[1]),
        traces=[parent_trace, branch_trace, child_trace],
        scale_traces=[
            np.zeros((5,), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
        ],
        child_edges=[(2, 2)],
        scale_indices=scale_indices,
        energy=energy,
        vertex_center_image=vertex_center_image,
        vertex_volume_image=vertex_volume_image,
        lumen_radius_pixels_axes=vertex_radii,
        lumen_radius_microns=np.array([0.49], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        image_shape=energy.shape,
        strel_apothem=1,
        max_edge_length_per_origin_radius=60.0,
    )

    assert target is not None
    assert target["kind"] == "existing_vertex"
    assert target["terminal_vertex"] == 3
    assert np.allclose(target["trace"][0], np.array([4.0, 4.0, 1.0], dtype=np.float32))
    assert np.allclose(target["trace"][-1], np.array([5.0, 6.0, 1.0], dtype=np.float32))


def test_add_vertices_to_edges_matlab_style_has_no_parent_half_fallback(monkeypatch):
    energy = np.zeros((6, 4, 2), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)
    energy[0:5, 0, 0] = np.array([-9.0, -9.0, -10.0, -9.0, -9.0], dtype=np.float32)
    energy[2, 1:3, 0] = np.array([-8.0, -8.0], dtype=np.float32)

    vertices = {
        "positions": np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
            dtype=np.float32,
        ),
        "scales": np.array([0, 0, 0], dtype=np.int16),
        "energies": np.array([-9.0, -9.0, -8.0], dtype=np.float32),
    }
    chosen_edges = {
        "traces": [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            np.array([[2.0, 2.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        ],
        "scale_traces": [
            np.zeros((5,), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
        ],
        "energy_traces": [
            np.array([-9.0, -9.0, -10.0, -9.0, -9.0], dtype=np.float32),
            np.array([-8.0, -8.0, -10.0], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [2, 1]], dtype=np.int32),
        "connection_sources": ["frontier", "frontier"],
        "chosen_candidate_indices": np.array([5, 6], dtype=np.int32),
        "diagnostics": {},
    }

    monkeypatch.setattr(
        "source.core.edges_internal.bridge_insertion._matlab_bridge_search_target",
        lambda *args, **kwargs: None,
    )

    bridged = add_vertices_to_edges_matlab_style(
        chosen_edges,
        vertices,
        energy=energy,
        scale_indices=scale_indices,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([0.49], dtype=np.float32),
        lumen_radius_pixels_axes=np.array([[0.49, 0.49, 0.49]], dtype=np.float32),
        size_of_image=energy.shape,
    )

    assert bridged["bridge_vertex_positions"].shape == (0, 3)
    assert bridged["bridge_edges"]["connections"].shape == (0, 2)
    assert bridged["bridge_edges"]["edges2vertices"].shape == (0, 2)
    assert bridged["bridge_edges"]["mean_edge_energies"].shape == (0,)
    assert sorted(bridged["connections"].tolist()) == [[0, 1], [2, 1]]


def test_add_vertices_to_edges_matlab_style_ignores_unrelated_edges_during_bridge_search():
    energy = np.zeros((8, 8, 4), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)
    energy[1:6, 3, 1] = np.array([-9.0, -9.0, -10.0, -9.0, -9.0], dtype=np.float32)
    energy[3, 4:6, 1] = np.array([-8.0, -8.0], dtype=np.float32)
    energy[1, 6, 1] = -20.0
    energy[2, 5, 1] = -20.0
    energy[3, 4, 1] = -20.0

    vertices = {
        "positions": np.array(
            [[1.0, 3.0, 1.0], [5.0, 3.0, 1.0], [3.0, 5.0, 1.0], [1.0, 6.0, 1.0], [4.0, 6.0, 1.0]],
            dtype=np.float32,
        ),
        "scales": np.array([0, 0, 0, 0, 0], dtype=np.int16),
        "energies": np.array([-9.0, -9.0, -8.0, -20.0, -20.0], dtype=np.float32),
    }
    chosen_edges = {
        "traces": [
            np.array(
                [
                    [1.0, 3.0, 1.0],
                    [2.0, 3.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [4.0, 3.0, 1.0],
                    [5.0, 3.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array([[3.0, 5.0, 1.0], [3.0, 4.0, 1.0], [3.0, 3.0, 1.0]], dtype=np.float32),
            np.array(
                [[1.0, 6.0, 1.0], [2.0, 5.0, 1.0], [3.0, 4.0, 1.0], [4.0, 6.0, 1.0]],
                dtype=np.float32,
            ),
        ],
        "scale_traces": [
            np.zeros((5,), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
        ],
        "energy_traces": [
            np.array([-9.0, -9.0, -10.0, -9.0, -9.0], dtype=np.float32),
            np.array([-8.0, -20.0, -10.0], dtype=np.float32),
            np.array([-20.0, -20.0, -20.0, -20.0], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [2, 1], [3, 4]], dtype=np.int32),
        "connection_sources": ["frontier", "frontier", "frontier"],
        "chosen_candidate_indices": np.array([5, 6, 7], dtype=np.int32),
        "diagnostics": {},
    }

    bridged = add_vertices_to_edges_matlab_style(
        chosen_edges,
        vertices,
        energy=energy,
        scale_indices=scale_indices,
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([0.49], dtype=np.float32),
        lumen_radius_pixels_axes=np.array([[0.49, 0.49, 0.49]], dtype=np.float32),
        size_of_image=energy.shape,
    )

    assert np.allclose(
        bridged["bridge_vertex_positions"], np.array([[2.0, 3.0, 1.0]], dtype=np.float32)
    )
    assert [3, 4] in bridged["connections"].tolist()
    assert all(
        not np.any(np.all(np.isclose(trace, np.array([3.0, 4.0, 1.0], dtype=np.float32)), axis=1))
        for trace in bridged["bridge_edges"]["traces"]
    )
