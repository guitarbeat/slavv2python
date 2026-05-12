
import numpy as np
import pytest
from slavv_python.core.edges.common import _build_matlab_local_strel_geometry

@pytest.mark.unit
def test_strel_geometry_anisotropic_microns():
    """Test that strel distances are correct with anisotropic microns.
    
    This catches misalignment bugs where offsets are applied to the wrong micron dimensions.
    """
    lumen_radius_microns = np.array([2.0], dtype=np.float32)
    # Anisotropic microns: Z is much larger than X, Y
    microns_per_voxel = np.array([2.0, 1.0, 0.5], dtype=np.float32) # dz, dx, dy (aligned order)
    step_size_per_origin_radius = 1.0
    
    geom = _build_matlab_local_strel_geometry(
        0, 
        lumen_radius_microns, 
        microns_per_voxel, 
        step_size_per_origin_radius=step_size_per_origin_radius
    )
    
    offsets = geom["local_subscripts"]
    distances = geom["distance_lut"]
    
    # [1, 0, 0] is a step in the first dimension (Z)
    # Expected distance: 1 * dz = 2.0
    idx_z = -1
    for i, off in enumerate(offsets):
        if np.array_equal(off, [1, 0, 0]):
            idx_z = i
            break
    assert idx_z != -1
    assert np.isclose(distances[idx_z], 2.0)
    
    # [0, 0, 1] is a step in the third dimension (Y)
    # Expected distance: 1 * dy = 0.5
    idx_y = -1
    for i, off in enumerate(offsets):
        if np.array_equal(off, [0, 0, 1]):
            idx_y = i
            break
    assert idx_y != -1
    assert np.isclose(distances[idx_y], 0.5)

@pytest.mark.unit
def test_strel_tie_breaking_order():
    """Test that the tie-breaking order matches MATLAB (small-Y fastest)."""
    lumen_radius_microns = np.array([5.0], dtype=np.float32) # Larger radius to ensure many valid offsets
    microns_per_voxel = np.ones((3,), dtype=np.float32)
    step_size_per_origin_radius = 1.0
    
    geom = _build_matlab_local_strel_geometry(
        0, 
        lumen_radius_microns, 
        microns_per_voxel, 
        step_size_per_origin_radius=step_size_per_origin_radius
    )
    
    offsets = geom["local_subscripts"]
    
    # Find two offsets that differ only in Y (index 2)
    found_y_step = False
    for i in range(len(offsets) - 1):
        if offsets[i][0] == offsets[i+1][0] and offsets[i][1] == offsets[i+1][1]:
            assert offsets[i][2] < offsets[i+1][2]
            found_y_step = True
            break
    assert found_y_step, "Could not find two adjacent offsets differing only in Y"

    # Also verify that X changes slower than Y but faster than Z
    found_x_step = False
    for i in range(len(offsets) - 1):
        if offsets[i][0] == offsets[i+1][0] and offsets[i][1] != offsets[i+1][1]:
            assert offsets[i][1] < offsets[i+1][1]
            found_x_step = True
            break
    assert found_x_step, "Could not find two adjacent offsets differing in X"
