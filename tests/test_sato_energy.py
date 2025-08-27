import pathlib
import sys
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))
from vectorization_core import SLAVVProcessor


def test_sato_energy_method():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0
    proc = SLAVVProcessor()
    params = {
        'energy_method': 'sato',
        'radius_of_smallest_vessel_in_microns': 1.0,
        'radius_of_largest_vessel_in_microns': 2.0,
        'scales_per_octave': 1.0,
    }
    result = proc.calculate_energy_field(image, params)
    assert result['energy'].shape == image.shape
    assert result['scale_indices'].shape == image.shape
    assert result['energy'][4,4,4] < 0
