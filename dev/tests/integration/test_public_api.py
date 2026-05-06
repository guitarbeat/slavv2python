from __future__ import annotations

import warnings

import numpy as np
import pytest

import source.core as core
from source.core import SlavvPipeline, SLAVVProcessor
from source.utils import validate_parameters


def test_run_structure():
    processor = SlavvPipeline()
    image = np.zeros((5, 5, 5), dtype=np.float32)
    result = processor.run(image, {})
    expected_keys = {"energy_data", "vertices", "edges", "network", "parameters"}
    assert expected_keys.issubset(result.keys())
    assert result["vertices"]["positions"].shape[1] == 3


def test_run_output_types():
    processor = SlavvPipeline()
    image = np.zeros((5, 5, 5), dtype=np.float32)
    result = processor.run(image, {})

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


def test_run_stop_after_energy_returns_plain_dict_payload():
    processor = SlavvPipeline()
    image = np.zeros((5, 5, 5), dtype=np.float32)

    result = processor.run(image, {}, stop_after="energy")

    assert isinstance(result, dict)
    assert isinstance(result["parameters"], dict)
    assert isinstance(result["energy_data"], dict)
    assert "vertices" not in result


def test_legacy_processor_alias_matches_preferred_pipeline():
    image = np.zeros((5, 5, 5), dtype=np.float32)
    preferred = SlavvPipeline().run(image, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = SLAVVProcessor().process_image(image, {})

    assert preferred.keys() == legacy.keys()
    np.testing.assert_array_equal(
        preferred["vertices"]["positions"], legacy["vertices"]["positions"]
    )
    np.testing.assert_array_equal(preferred["edges"]["connections"], legacy["edges"]["connections"])


def test_legacy_method_aliases_match_preferred_methods():
    processor = SlavvPipeline()
    image = np.zeros((5, 5, 5), dtype=np.float32)

    preferred_energy = processor.compute_energy(image, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_energy = processor.calculate_energy_field(image, {})

    np.testing.assert_array_equal(preferred_energy["energy"], legacy_energy["energy"])

    vertices = processor.extract_vertices(preferred_energy, {})
    edges = processor.extract_edges(preferred_energy, vertices, {})

    preferred_network = processor.build_network(edges, vertices, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_network = processor.construct_network(edges, vertices, {})

    np.testing.assert_array_equal(
        preferred_network["vertex_degrees"],
        legacy_network["vertex_degrees"],
    )


def test_validate_parameters_invalid_scales():
    with pytest.raises(ValueError, match="scales_per_octave must be positive"):
        validate_parameters({"scales_per_octave": 0})


def test_validate_parameters_negative_bandpass():
    with pytest.raises(ValueError, match="bandpass_window must be non-negative"):
        validate_parameters({"bandpass_window": -1})


def test_core_module_exports_preferred_and_legacy_pipeline_classes():
    assert set(core.__all__) == {"SlavvPipeline", "SLAVVProcessor"}
    assert hasattr(core, "SlavvPipeline")
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
