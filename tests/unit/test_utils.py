
import pytest
import numpy as np
try:
    from slavv.utils import (
except ImportError:
    from src.slavv.utils import (
    calculate_path_length,
    validate_parameters,
    get_chunking_lattice,
    preprocess_image
)

def test_calculate_path_length():
    # Simple straight line
    path = np.array([[0, 0, 0], [0, 0, 10]], dtype=float)
    assert calculate_path_length(path) == 10.0

    # L-shape
    path = np.array([[0, 0, 0], [0, 3, 0], [4, 3, 0]], dtype=float)
    assert calculate_path_length(path) == 3.0 + 4.0

    # Empty/Single point
    assert calculate_path_length(np.array([], dtype=float)) == 0.0
    assert calculate_path_length(np.array([[1, 2, 3]], dtype=float)) == 0.0

def test_validate_parameters_defaults():
    params = {}
    validated = validate_parameters(params)
    assert validated['microns_per_voxel'] == [1.0, 1.0, 1.0]
    assert validated['radius_of_smallest_vessel_in_microns'] == 1.5
    assert validated['energy_sign'] == -1.0

def test_validate_parameters_invalid():
    with pytest.raises(ValueError, match="radius_of_smallest_vessel_in_microns must be positive"):
        validate_parameters({'radius_of_smallest_vessel_in_microns': -1})

    with pytest.raises(ValueError, match="microns_per_voxel must be a 3-element array"):
        validate_parameters({'microns_per_voxel': [1.0, 1.0]})

def test_get_chunking_lattice_small_volume():
    # Volume fits in one chunk
    shape = (10, 10, 10)
    max_voxels = 2000 # > 1000
    slices = get_chunking_lattice(shape, max_voxels, margin=1)
    
    assert len(slices) == 1
    chunk, output, inner = slices[0]
    # Full coverage
    assert chunk == (slice(0, 10), slice(0, 10), slice(0, 10))

def test_get_chunking_lattice_splitting():
    # Volume needs splitting
    # Plane size = 100 voxels. Max = 300. Max depth = 3.
    # Total depth = 6. Should split into chunks.
    shape = (10, 10, 6)
    max_voxels = 300
    margin = 1
    
    slices = get_chunking_lattice(shape, max_voxels, margin)
    assert len(slices) > 1
    
    # Verify total coverage
    processed_z = 0
    for chunk, output, inner in slices:
        z_start = output[2].start
        z_end = output[2].stop
        assert z_start == processed_z
        processed_z = z_end
    
    assert processed_z == 6

def test_preprocess_image():
    img = np.array([[[0, 100], [50, 100]]], dtype=float)
    params = {'bandpass_window': 0} # Disable bandpass for simple range check
    
    processed = preprocess_image(img, params)
    
    assert processed.min() == 0.0
    assert processed.max() == 1.0
    assert processed.shape == img.shape
