import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import SLAVVProcessor, validate_parameters


def test_validate_parameters_defaults():
    params = validate_parameters({})
    assert 'microns_per_voxel' in params
    assert len(params['microns_per_voxel']) == 3


def test_process_image_structure():
    processor = SLAVVProcessor()
    image = np.zeros((5, 5, 5), dtype=np.float32)
    result = processor.process_image(image, {})
    expected_keys = {'energy_data', 'vertices', 'edges', 'network', 'parameters'}
    assert expected_keys.issubset(result.keys())
    assert result['vertices']['positions'].shape[1] == 3
