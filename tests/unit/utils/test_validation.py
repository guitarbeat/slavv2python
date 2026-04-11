import pytest

from slavv.utils import validate_parameters


def test_validate_parameters_defaults():
    validated = validate_parameters({})
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


def test_validate_parameters_warns_for_unusual_excitation_wavelength():
    with pytest.warns(UserWarning, match="Excitation wavelength outside typical range"):
        validated = validate_parameters({"excitation_wavelength_in_microns": 3.5})

    assert validated["excitation_wavelength_in_microns"] == 3.5


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"radius_of_smallest_vessel_in_microns": -1},
            "radius_of_smallest_vessel_in_microns must be positive",
        ),
        (
            {
                "radius_of_smallest_vessel_in_microns": 3.0,
                "radius_of_largest_vessel_in_microns": 3.0,
            },
            "radius_of_largest_vessel_in_microns must be larger than smallest",
        ),
        (
            {"microns_per_voxel": [1.0, 1.0]},
            "microns_per_voxel must be a 3-element array",
        ),
        (
            {"microns_per_voxel": [0.0, 1.0, 1.0]},
            "microns_per_voxel values must be positive",
        ),
        (
            {"number_of_edges_per_vertex": 4.5},
            "number_of_edges_per_vertex must be an integer value",
        ),
        (
            {"parity_watershed_candidate_mode": "unknown_mode"},
            "parity_watershed_candidate_mode must be one of",
        ),
        (
            {"parity_candidate_salvage_mode": "unknown_mode"},
            "parity_candidate_salvage_mode must be one of",
        ),
    ],
)
def test_validate_parameters_rejects_invalid_values(params, message):
    with pytest.raises(ValueError, match=message):
        validate_parameters(params)
