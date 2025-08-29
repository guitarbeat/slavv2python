import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import SLAVVProcessor


def test_generate_edge_directions_single_and_norms():
    processor = SLAVVProcessor()
    dirs1 = processor._generate_edge_directions(1)
    assert dirs1.shape == (1, 3)
    assert np.allclose(dirs1[0], [0, 0, 1])

    dirs5 = processor._generate_edge_directions(5)
    assert dirs5.shape == (5, 3)
    # all vectors should be unit length
    assert np.allclose(np.linalg.norm(dirs5, axis=1), 1.0, atol=1e-6)
    # ensure directions are unique
    assert len(np.unique(np.round(dirs5, 6), axis=0)) == 5
