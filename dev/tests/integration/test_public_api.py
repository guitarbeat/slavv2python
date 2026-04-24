import numpy as np
import pytest
import source.core as core
from source.core import SLAVVProcessor
from source.utils import validate_parameters


def test_process_image_structure():
    processor = SLAVVProcessor()
    image = np.zeros((5, 5, 5), dtype=np.float32)
    result = processor.process_image(image, {})
    expected_keys = {"energy_data", "vertices", "edges", "network", "parameters"}
    assert expected_keys.issubset(result.keys())
    assert result["vertices"]["positions"].shape[1] == 3


def test_process_image_output_types():
    processor = SLAVVProcessor()
    image = np.zeros((5, 5, 5), dtype=np.float32)
    result = processor.process_image(image, {})

    vertices = result["vertices"]
    edges = result["edges"]
    network = result["network"]

    assert vertices["positions"].dtype == np.float32
    assert vertices["scales"].dtype == np.int16
    assert vertices["radii_microns"].dtype == np.float32

    assert edges["connections"].dtype == np.int32
    assert len(edges["connections"]) == len(edges["traces"])

    assert isinstance(network["adjacency_list"], dict)
    assert network["vertex_degrees"].dtype == np.int32


def test_process_image_stop_after_energy_returns_plain_dict_payload():
    processor = SLAVVProcessor()
    image = np.zeros((5, 5, 5), dtype=np.float32)

    result = processor.process_image(image, {}, stop_after="energy")

    assert isinstance(result, dict)
    assert isinstance(result["parameters"], dict)
    assert isinstance(result["energy_data"], dict)
    assert "vertices" not in result


def test_validate_parameters_invalid_scales():
    with pytest.raises(ValueError, match="scales_per_octave must be positive"):
        validate_parameters({"scales_per_octave": 0})


def test_validate_parameters_negative_bandpass():
    with pytest.raises(ValueError, match="bandpass_window must be non-negative"):
        validate_parameters({"bandpass_window": -1})


def test_core_module_exports_minimal_public_api():
    assert core.__all__ == ["SLAVVProcessor"]
    assert hasattr(core, "SLAVVProcessor")
    for removed_name in (
        "compute_gradient",
        "estimate_vessel_directions",
        "extract_edges",
        "extract_edges_watershed",
        "extract_vertices",
        "find_terminal_vertex",
        "generate_edge_directions",
        "in_bounds",
        "near_vertex",
        "trace_edge",
    ):
        assert not hasattr(core, removed_name)


