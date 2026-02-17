import io

import numpy as np
import pytest
import tifffile

from slavv.io import load_tiff_volume


def test_load_tiff_volume_valid():
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    buf.seek(0)
    vol = load_tiff_volume(buf)
    assert vol.shape == (2, 2, 2)


def test_load_tiff_volume_non_tiff():
    buf = io.BytesIO(b"not a tiff")
    with pytest.raises(ValueError):
        load_tiff_volume(buf)


def test_load_tiff_volume_wrong_dim():
    arr = np.zeros((4, 4), dtype=np.uint8)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    buf.seek(0)
    with pytest.raises(ValueError, match="3D volume"):
        load_tiff_volume(buf)


def test_load_tiff_volume_memmap(tmp_path):
    arr = np.zeros((2, 2, 2), dtype=np.uint8)
    path = tmp_path / 'vol.tif'
    tifffile.imwrite(path, arr)
    vol = load_tiff_volume(path, memory_map=True)
    assert vol.shape == (2, 2, 2)
    assert isinstance(vol, np.memmap)
