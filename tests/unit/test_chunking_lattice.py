import sys
import pathlib
import numpy as np

# Add source path for imports
from src.slavv.vectorization_core import get_chunking_lattice


def test_chunking_lattice_reconstructs_volume():
    shape = (4, 4, 6)
    volume = np.random.rand(*shape)
    lattice = get_chunking_lattice(shape, max_voxels=64, margin=1)
    assert len(lattice) == 3

    out = np.zeros_like(volume)
    for chunk_slice, output_slice, inner_slice in lattice:
        chunk = volume[chunk_slice]
        out[output_slice] = chunk[inner_slice]
    assert np.allclose(out, volume)
