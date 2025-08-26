import pathlib
import sys

import numpy as np
import pytest

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


def test_process_image_output_types():
    processor = SLAVVProcessor()
    image = np.zeros((5, 5, 5), dtype=np.float32)
    result = processor.process_image(image, {})

    vertices = result['vertices']
    edges = result['edges']
    network = result['network']

    assert vertices['positions'].dtype == np.float32
    assert vertices['scales'].dtype == np.int16
    assert vertices['radii_microns'].dtype == np.float32

    assert edges['connections'].dtype == np.int32
    assert len(edges['connections']) == len(edges['traces'])

    assert network['adjacency'].dtype == bool
    assert network['vertex_degrees'].dtype == np.int32


def test_validate_parameters_invalid_scales():
    with pytest.raises(ValueError):
        validate_parameters({'scales_per_octave': 0})


def test_validate_parameters_negative_bandpass():
    with pytest.raises(ValueError):
        validate_parameters({'bandpass_window': -1})
