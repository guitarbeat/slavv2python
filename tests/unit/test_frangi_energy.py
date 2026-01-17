import pathlib
import sys
import numpy as np

from src.slavv.vectorization_core import SLAVVProcessor

def test_frangi_energy_method():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0
    proc = SLAVVProcessor()
    params = {
        'energy_method': 'frangi',
        'radius_of_smallest_vessel_in_microns': 1.0,
        'radius_of_largest_vessel_in_microns': 2.0,
        'scales_per_octave': 1.0,
    }
    result = proc.calculate_energy_field(image, params)
    assert result['energy'].shape == image.shape
    assert result['scale_indices'].shape == image.shape
    assert result['energy'][4,4,4] < 0
