import numpy as np
import pytest

from slavv.utils import (
    calculate_path_length,
    get_chunking_lattice,
    preprocess_image,
    validate_parameters,
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
    assert validated["microns_per_voxel"] == [1.0, 1.0, 1.0]
    assert validated["radius_of_smallest_vessel_in_microns"] == 1.5
    assert validated["energy_sign"] == -1.0
    assert validated["comparison_exact_network"] is False
    assert validated["space_strel_apothem_edges"] == validated["space_strel_apothem"]
    assert validated["sigma_per_influence_vertices"] == 1.0
    assert validated["sigma_per_influence_edges"] == 0.5


def test_validate_parameters_preserves_edge_influence_overrides():
    validated = validate_parameters(
        {
            "sigma_per_influence_vertices": 2.0,
            "sigma_per_influence_edges": 2.0 / 3.0,
        }
    )

    assert validated["sigma_per_influence_vertices"] == 2.0
    assert validated["sigma_per_influence_edges"] == 2.0 / 3.0


def test_validate_parameters_preserves_parity_specific_overrides():
    validated = validate_parameters(
        {
            "parity_frontier_reachability_gate": False,
            "parity_require_mutual_frontier_participation": False,
            "parity_watershed_candidate_mode": "legacy_supplement",
            "parity_watershed_metric_threshold": -90.0,
            "parity_candidate_salvage_mode": "none",
            "parity_geodesic_salvage_k_nearest": 8,
            "parity_geodesic_salvage_box_margin_voxels": 5,
            "parity_geodesic_salvage_max_path_ratio": 3.0,
        }
    )

    assert validated["parity_frontier_reachability_gate"] is False
    assert validated["parity_require_mutual_frontier_participation"] is False
    assert validated["parity_watershed_candidate_mode"] == "legacy_supplement"
    assert validated["parity_watershed_metric_threshold"] == -90.0
    assert validated["parity_candidate_salvage_mode"] == "none"
    assert validated["parity_geodesic_salvage_k_nearest"] == 8
    assert validated["parity_geodesic_salvage_box_margin_voxels"] == 5
    assert validated["parity_geodesic_salvage_max_path_ratio"] == 3.0


def test_validate_parameters_coerces_integer_like_matlab_settings():
    validated = validate_parameters(
        {
            "max_voxels_per_node_energy": 100000.0,
            "space_strel_apothem": 1.0,
            "space_strel_apothem_edges": 2.0,
            "max_voxels_per_node": 6000.0,
            "number_of_edges_per_vertex": 4.0,
        }
    )

    assert validated["max_voxels_per_node_energy"] == 100000
    assert validated["space_strel_apothem"] == 1
    assert validated["space_strel_apothem_edges"] == 2
    assert validated["max_voxels_per_node"] == 6000
    assert validated["number_of_edges_per_vertex"] == 4
    assert isinstance(validated["number_of_edges_per_vertex"], int)


def test_validate_parameters_invalid():
    with pytest.raises(ValueError, match="radius_of_smallest_vessel_in_microns must be positive"):
        validate_parameters({"radius_of_smallest_vessel_in_microns": -1})

    with pytest.raises(ValueError, match="microns_per_voxel must be a 3-element array"):
        validate_parameters({"microns_per_voxel": [1.0, 1.0]})

    with pytest.raises(ValueError, match="number_of_edges_per_vertex must be an integer value"):
        validate_parameters({"number_of_edges_per_vertex": 4.5})


def test_get_chunking_lattice_small_volume():
    # Volume fits in one chunk
    shape = (10, 10, 10)
    max_voxels = 2000  # > 1000
    slices = get_chunking_lattice(shape, max_voxels, margin=1)

    assert len(slices) == 1
    chunk, _output, _inner = slices[0]
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
    for _chunk, output, _inner in slices:
        z_start = output[2].start
        z_end = output[2].stop
        assert z_start == processed_z
        processed_z = z_end

    assert processed_z == 6


def test_preprocess_image():
    img = np.array([[[0, 100], [50, 100]]], dtype=float)
    params = {"bandpass_window": 0}  # Disable bandpass for simple range check

    processed = preprocess_image(img, params)

    assert processed.min() == 0.0
    assert processed.max() == 1.0
    assert processed.shape == img.shape
