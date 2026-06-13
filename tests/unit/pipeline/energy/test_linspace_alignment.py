
import numpy as np
import pytest
from slavv_python.pipeline.energy.exact_mesh import _matlab_zero_based_linspace

def test_linspace_phase_alignment():
    """
    Verify that _matlab_zero_based_linspace correctly maps global indices
    to downsampled coordinates, accounting for the reading start phase.
    
    In the 180709_E volume (size 256, rf 9):
    - Reading start (1-based) is 1 + ((256-1) % 9) = 1 + 3 = 4.
    - Global index 3 (0-based) corresponds to the 4th physical voxel.
    - This voxel is the first element in the downsampled volume.
    - Its coordinate relative to the downsampled volume should be exactly 0.0.
    """
    size = 256
    rf = 9
    
    # MATLAB rules for single chunk
    writing_start = 1
    reading_start = 1 + ((size - 1) % rf) # 4
    
    # Offset: writing starts before reading
    offset = writing_start - reading_start # -3
    
    # Local start in coarse grid (floor(offset/rf))
    l_start = int(np.floor(float(offset) / float(rf))) # -1
    
    # Clamped local start (reading always covers writing in exact route)
    l_start_clamped = max(0, l_start) # 0
    
    mesh = _matlab_zero_based_linspace(offset, rf, size, l_start_clamped)
    
    # Global index 3 (j=3 in Python 0-based indexing) should be 0.0
    # because reading_start is 4 (1-based) which is index 3.
    assert np.isclose(mesh[3], 0.0, atol=1e-15)
    
    # Global index 12 (next downsampled point) should be 1.0
    assert np.isclose(mesh[12], 1.0, atol=1e-15)

def test_linspace_multi_chunk_alignment():
    """
    Verify alignment in a multi-chunk scenario (though single-chunk is preferred for parity).
    """
    # Case: Chunk 1 starts at global index 128 (writing_start=129)
    # Reading starts at global index 120 (reading_start=121)
    # rf = 9
    writing_start = 129
    reading_start = 121
    offset = writing_start - reading_start # 8
    rf = 9
    
    l_start = int(np.floor(float(offset) / float(rf))) # floor(8/9) = 0
    
    mesh = _matlab_zero_based_linspace(offset, rf, 128, l_start)
    
    # j=0 corresponds to global index 128.
    # Its coordinate relative to reading_start (121) is (129-121)/9 = 8/9.
    # Since l_start=0, coordinate relative to local volume is 8/9.
    assert np.isclose(mesh[0], 8.0/9.0, atol=1e-15)

    # Case: offset=15, rf=9
    offset = 15
    l_start = int(np.floor(float(offset) / float(rf))) # floor(15/9) = 1
    mesh = _matlab_zero_based_linspace(offset, rf, 128, l_start)
    
    # j=0 corresponds to offset 15.
    # Coord relative to reading_start is 15/9 = 1.666
    # Relative to local start (l_start=1), it should be 0.666
    assert np.isclose(mesh[0], 15.0/9.0 - 1.0, atol=1e-15)
