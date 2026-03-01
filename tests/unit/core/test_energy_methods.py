"""Consolidated tests for alternative energy methods (Frangi/Sato)."""
import numpy as np
import pytest
from slavv.core import SLAVVProcessor


@pytest.mark.parametrize("method", ["frangi", "sato"])
def test_alternative_energy_methods(method):
    """Test that Frangi and Sato energy methods produce valid output."""
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0  # Create a tubular structure
    
    proc = SLAVVProcessor()
    params = {
        'energy_method': method,
        'radius_of_smallest_vessel_in_microns': 1.0,
        'radius_of_largest_vessel_in_microns': 2.0,
        'scales_per_octave': 1.0,
    }
    result = proc.calculate_energy_field(image, params)
    
    assert result['energy'].shape == image.shape
    assert result['scale_indices'].shape == image.shape
    assert result['energy'][4, 4, 4] < 0  # Tubular structure should have negative energy
