import numpy as np
from source.utils import get_chunking_lattice


def test_chunking_lattice_returns_single_chunk_when_volume_fits():
    shape = (10, 10, 10)
    lattice = get_chunking_lattice(shape, max_voxels=2000, margin=1)

    assert len(lattice) == 1
    chunk_slice, output_slice, inner_slice = lattice[0]
    assert chunk_slice == (slice(0, 10), slice(0, 10), slice(0, 10))
    assert output_slice == chunk_slice
    assert inner_slice == chunk_slice


def test_chunking_lattice_reconstructs_volume():
    shape = (4, 4, 6)
    volume = np.arange(np.prod(shape), dtype=float).reshape(shape)
    lattice = get_chunking_lattice(shape, max_voxels=64, margin=1)
    assert len(lattice) == 3

    out = np.zeros_like(volume)
    for chunk_slice, output_slice, inner_slice in lattice:
        chunk = volume[chunk_slice]
        out[output_slice] = chunk[inner_slice]
    assert np.allclose(out, volume)


def test_chunking_lattice_keeps_output_slices_contiguous_when_margin_is_clamped():
    shape = (6, 6, 8)
    lattice = get_chunking_lattice(shape, max_voxels=108, margin=5)

    processed_z = 0
    for _chunk_slice, output_slice, _inner_slice in lattice:
        assert output_slice[2].start == processed_z
        processed_z = output_slice[2].stop

    assert processed_z == shape[2]


