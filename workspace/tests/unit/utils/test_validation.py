import pytest
from slavv_python.utils import validate_parameters


def test_validate_parameters_defaults():
    validated = validate_parameters({})
    assert validated["microns_per_voxel"] == [1.0, 1.0, 1.0]
    assert validated["radius_of_smallest_vessel_in_microns"] == 1.5
    assert validated["energy_projection_mode"] == "matlab"
    assert validated["energy_sign"] == -1.0
    assert validated["space_strel_apothem_edges"] == validated["space_strel_apothem"]
    assert validated["sigma_per_influence_vertices"] == 1.0
    assert validated["sigma_per_influence_edges"] == 0.5
    # Default for matlab_compat (which is the default profile if none specified)
    assert validated["comparison_exact_network_use_conflict_painting"] is False


def test_validate_parameters_paper_profile_enables_conflict_painting():
    validated = validate_parameters({"pipeline_profile": "paper"})
    assert validated["comparison_exact_network_use_conflict_painting"] is True


def test_validate_parameters_preserves_edge_influence_overrides():
    validated = validate_parameters(
        {
            "sigma_per_influence_vertices": 2.0,
            "sigma_per_influence_edges": 2.0 / 3.0,
        }
    )

    assert validated["sigma_per_influence_vertices"] == 2.0
    assert validated["sigma_per_influence_edges"] == 2.0 / 3.0


def test_validate_parameters_accepts_simpleitk_objectness_energy_method():
    validated = validate_parameters({"energy_method": "simpleitk_objectness"})

    assert validated["energy_method"] == "simpleitk_objectness"


def test_validate_parameters_accepts_cupy_hessian_energy_method():
    validated = validate_parameters({"energy_method": "cupy_hessian"})

    assert validated["energy_method"] == "cupy_hessian"


def test_validate_parameters_accepts_zarr_energy_storage_format():
    validated = validate_parameters({"energy_storage_format": "zarr"})

    assert validated["energy_storage_format"] == "zarr"


def test_validate_parameters_accepts_paper_energy_projection_mode():
    validated = validate_parameters({"energy_projection_mode": "paper"})

    assert validated["energy_projection_mode"] == "paper"


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


def test_validate_parameters_rejects_legacy_parity_controls():
    with pytest.raises(ValueError, match="legacy parity parameters are no longer supported"):
        validate_parameters(
            {
                "comparison_exact_network": True,
                "parity_watershed_candidate_mode": "all_contacts",
                "parity_geodesic_salvage_k_nearest": 10,
            }
        )


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
    ],
)
def test_validate_parameters_rejects_invalid_values(params, message):
    with pytest.raises(ValueError, match=message):
        validate_parameters(params)
