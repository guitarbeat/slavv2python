import numpy as np

from slavv_python.core.graph import (
    _matlab_edge_metrics,
    _matlab_get_strand_objects,
    _matlab_get_vessel_directions_v3,
    _matlab_network_topology,
    _matlab_smooth_edges_v2,
    _matlab_sort_network_v180,
)


def test_matlab_network_topology_matches_branching_strand_decomposition():
    connections = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [2, 4],
        ],
        dtype=np.int32,
    )
    edge_traces = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
    ]
    edge_scale_traces = [np.array([0.0, 0.0], dtype=np.float32) for _ in edge_traces]
    edge_energies = [np.array([-5.0, -5.0], dtype=np.float32) for _ in edge_traces]

    topology = _matlab_network_topology(
        connections,
        edge_traces,
        edge_scale_traces,
        edge_energies,
        n_vertices=5,
    )

    assert topology["bifurcations"].tolist() == [2]
    assert topology["strands"] == [[0, 1, 2], [2, 3], [2, 4]]
    assert [indices.tolist() for indices in topology["edge_indices_in_strands"]] == [
        [0, 1],
        [2],
        [3],
    ]
    assert [flags.tolist() for flags in topology["edge_backwards_in_strands"]] == [
        [False, False],
        [False],
        [False],
    ]
    assert np.allclose(
        topology["strand_traces"][0],
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_matlab_network_topology_breaks_cyclical_strand_by_worst_edge():
    connections = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
        ],
        dtype=np.int32,
    )
    edge_traces = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
    ]
    edge_scale_traces = [np.array([0.0, 0.0], dtype=np.float32) for _ in edge_traces]
    edge_energies = [np.array([-4.0, -4.0], dtype=np.float32) for _ in edge_traces]

    topology = _matlab_network_topology(
        connections,
        edge_traces,
        edge_scale_traces,
        edge_energies,
        n_vertices=3,
    )

    assert topology["bifurcations"].size == 0
    assert topology["strands"] == [[0, 1, 2]]
    assert [indices.tolist() for indices in topology["edge_indices_in_strands"]] == [[0, 1]]
    assert np.allclose(
        topology["strand_traces"][0],
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_matlab_sort_network_marks_backwards_edges_for_reversed_chain():
    connections = np.array(
        [
            [1, 0],
            [2, 1],
        ],
        dtype=np.int32,
    )

    vertex_indices, edge_indices, edge_backwards = _matlab_sort_network_v180(
        connections,
        [np.array([0, 2], dtype=np.int32)],
        [np.array([1, 0], dtype=np.int32)],
    )

    assert [vertices.tolist() for vertices in vertex_indices] == [[0, 1, 2]]
    assert [indices.tolist() for indices in edge_indices] == [[0, 1]]
    assert [flags.tolist() for flags in edge_backwards] == [[True, True]]


def test_matlab_get_strand_objects_preserves_scale_traces_and_stable_dedupe():
    edge_traces = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    ]
    edge_scale_traces = [
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
    ]
    edge_energies = [
        np.array([-4.0, -5.0], dtype=np.float32),
        np.array([-5.0, -6.0], dtype=np.float32),
    ]

    strand_spaces, strand_scales, strand_energies = _matlab_get_strand_objects(
        edge_traces,
        edge_scale_traces,
        edge_energies,
        [np.array([0, 1], dtype=np.int32)],
        [np.array([False, False])],
    )

    assert np.allclose(
        strand_spaces[0],
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert np.allclose(strand_scales[0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(strand_energies[0], np.array([-4.0, -5.0, -6.0], dtype=np.float32))


def test_matlab_smooth_edges_v2_and_vessel_directions_match_straight_strand_math():
    strand_space_traces = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    ]
    strand_scale_traces = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
    strand_energy_traces = [np.array([-2.0, -4.0, -2.0], dtype=np.float32)]

    smoothed_space, smoothed_scales, smoothed_energy = _matlab_smooth_edges_v2(
        strand_space_traces,
        strand_scale_traces,
        strand_energy_traces,
        np.sqrt(2.0) / 2.0,
        np.array([1.0, 2.0], dtype=np.float32),
        np.ones((3,), dtype=np.float32),
    )
    vessel_directions = _matlab_get_vessel_directions_v3(
        smoothed_space,
        np.ones((3,), dtype=np.float32),
    )

    assert np.allclose(smoothed_space[0][0], strand_space_traces[0][0])
    assert np.allclose(smoothed_space[0][-1], strand_space_traces[0][-1])
    assert np.allclose(smoothed_scales[0], np.ones((3,), dtype=np.float32))
    assert np.allclose(
        vessel_directions[0],
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert np.allclose(_matlab_edge_metrics(smoothed_energy), np.array([-2.0], dtype=np.float32))
