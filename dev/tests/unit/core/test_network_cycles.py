import numpy as np
from source.core import SLAVVProcessor
from source.core.graph import construct_network_resumable, trace_strand_sparse
from source.runtime import RunContext


def test_construct_network_prunes_cycles_and_detects_mismatched():
    processor = SLAVVProcessor()
    vertices = {
        "positions": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
    }
    edges = {
        "connections": np.array(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [1, 3],
            ],
            dtype=int,
        ),
        "traces": [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float),
        ],
    }

    network = processor.construct_network(edges, vertices, {"remove_cycles": True})

    cycle_pairs = [tuple(map(int, pair)) for pair in network["cycles"]]
    assert (0, 2) in cycle_pairs
    assert 2 not in network["adjacency_list"][0]
    assert len(network["strands"]) >= 1
    assert len(network["mismatched_strands"]) == 0


def test_construct_network_resumable_persists_standard_topology(tmp_path):
    vertices = {
        "positions": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
    }
    edges = {
        "connections": np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [2, 4],
            ],
            dtype=np.int32,
        ),
        "traces": [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
            np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=float),
        ],
    }
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="network")

    network = construct_network_resumable(
        edges,
        vertices,
        {"remove_cycles": True},
        run_context.stage("network"),
    )

    assert network["strands"]
    assert network["mismatched_strands"] == []
    assert run_context.stage("network").artifact_path("strands.pkl").exists()


def test_trace_strand_sparse_uses_sorted_neighbor_order():
    adjacency_list = {
        0: {3, 1},
        1: {0, 2},
        2: {1},
        3: {0},
    }
    visited = np.zeros(4, dtype=bool)

    strand = trace_strand_sparse(0, adjacency_list, visited)

    assert strand == [0, 1, 2]


