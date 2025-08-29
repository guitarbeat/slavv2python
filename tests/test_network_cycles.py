import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import SLAVVProcessor


def test_construct_network_prunes_cycles_and_detects_mismatched():
    processor = SLAVVProcessor()
    vertices = {
        'positions': np.array(
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
        'connections': np.array(
            [
                [0, 1],  # base edge
                [1, 2],  # continuation
                [2, 0],  # forms a cycle
                [1, 3],  # branch causing strand mismatch
            ],
            dtype=int,
        ),
        'traces': [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float),
        ],
    }
    network = processor.construct_network(edges, vertices, {})

    # The 2-0 edge should be pruned as a cycle
    cycle_pairs = [tuple(map(int, c)) for c in network['cycles']]
    assert (0, 2) in cycle_pairs
    assert not network['adjacency'][0, 2]

    # The branched component cannot be ordered linearly and is reported as mismatched
    assert len(network['mismatched_strands']) == 1
    assert set(map(int, network['mismatched_strands'][0])) == {0, 1, 2, 3}
