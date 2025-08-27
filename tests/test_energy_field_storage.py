import pathlib
import sys
import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import SLAVVProcessor, validate_parameters


def test_energy_field_no_full_storage():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters({})
    proc = SLAVVProcessor()
    result = proc.calculate_energy_field(img, params)
    assert 'energy_4d' not in result
    assert result['energy'].shape == img.shape


def test_energy_field_with_full_storage():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters({'return_all_scales': True})
    proc = SLAVVProcessor()
    result = proc.calculate_energy_field(img, params)
    assert 'energy_4d' in result
    energy_4d = result['energy_4d']
    assert energy_4d.shape[:3] == img.shape
    assert energy_4d.shape[3] == len(result['lumen_radius_pixels'])
