import numpy as np
from slavv_python.core.edge_primitives import trace_edge


def test_discrete_tracing_steps_snap_to_voxels():
    # Energy decreases along +x (axis 1) so tracing proceeds without energy rise.
    energy = np.tile(-np.arange(5)[None, :, None], (5, 1, 5)).astype(float)
    start_pos = np.array([2.0, 0.0, 2.0])  # (y, x, z)
    direction = np.array([0.0, 1.0, 0.0])
    trace = trace_edge(
        energy,
        start_pos,
        direction,
        step_size=1.0,
        max_edge_energy=0.0,
        vertex_positions=np.empty((0, 3)),
        vertex_scales=np.empty((0,), dtype=int),
        lumen_radius_pixels=np.array([1.0]),
        lumen_radius_microns=np.array([1.0]),
        max_steps=4,
        microns_per_voxel=np.ones(3),
        energy_sign=-1.0,
        discrete_steps=True,
    )
    coords = np.asarray(trace)
    assert np.allclose(coords, np.round(coords))


def test_continuous_tracing_can_recover_terminal_with_subvoxel_step_halving():
    energy = np.zeros((5, 5, 5), dtype=float)
    energy[:, 0, :] = -1.0
    energy[:, 2:, :] = -2.0

    vertex_positions = np.array(
        [
            [2.0, 0.75, 2.0],
            [2.0, 1.15, 2.0],
        ],
        dtype=float,
    )
    trace, metadata = trace_edge(
        energy,
        start_pos=vertex_positions[0],
        direction=np.array([0.0, 1.0, 0.0], dtype=float),
        step_size=1.0,
        max_edge_energy=0.0,
        vertex_positions=vertex_positions,
        vertex_scales=np.array([0, 0], dtype=int),
        lumen_radius_pixels=np.array([1.0], dtype=float),
        lumen_radius_microns=np.array([1.0], dtype=float),
        max_steps=4,
        microns_per_voxel=np.ones(3, dtype=float),
        energy_sign=-1.0,
        origin_vertex_idx=0,
        max_search_radius=5.0,
        return_metadata=True,
    )

    assert metadata["terminal_vertex"] == 1
    assert metadata["terminal_resolution"] in {"direct_hit", "reverse_near_hit"}
    assert len(trace) >= 3


def test_tracing_can_resolve_terminal_from_painted_vertex_volume():
    energy = np.tile(-np.arange(7, dtype=float)[None, :, None], (7, 1, 7))

    vertex_positions = np.array(
        [
            [3.0, 0.75, 3.0],
            [3.0, 4.6, 3.0],
        ],
        dtype=float,
    )
    vertex_center_image = np.zeros((7, 7, 7), dtype=np.uint16)
    vertex_center_image[3, 0, 3] = 1
    vertex_center_image[3, 4, 3] = 2
    vertex_image = np.zeros((7, 7, 7), dtype=np.uint16)
    vertex_image[3, 0:2, 3] = 1
    vertex_image[3, 2:4, 3] = 2

    trace, metadata = trace_edge(
        energy,
        start_pos=vertex_positions[0],
        direction=np.array([0.0, 1.0, 0.0], dtype=float),
        step_size=1.0,
        max_edge_energy=0.0,
        vertex_positions=vertex_positions,
        vertex_scales=np.array([0, 0], dtype=int),
        lumen_radius_pixels=np.array([1.0], dtype=float),
        lumen_radius_microns=np.array([1.0], dtype=float),
        max_steps=4,
        microns_per_voxel=np.ones(3, dtype=float),
        energy_sign=-1.0,
        vertex_center_image=vertex_center_image,
        vertex_image=vertex_image,
        origin_vertex_idx=0,
        max_search_radius=0.0,
        return_metadata=True,
    )

    assert metadata["terminal_vertex"] == 1
    assert metadata["terminal_resolution"] in {"direct_hit", "reverse_volume_hit"}
    assert len(trace) >= 3
