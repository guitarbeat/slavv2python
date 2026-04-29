"""Parameter validation functions for source."""

from __future__ import annotations

import warnings
from typing import Any


def _coerce_integral_parameter(name: str, value: Any) -> Any:
    """Accept MATLAB-style numeric scalars for integer pipeline settings."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer value")
        return int(value)
    return value


def validate_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Validate parameters and populate defaults for the maintained pipeline."""
    legacy_parity_keys = sorted(str(key) for key in params if str(key).startswith("parity_"))
    if legacy_parity_keys:
        joined = ", ".join(legacy_parity_keys)
        raise ValueError(f"legacy parity parameters are no longer supported: {joined}")

    validated: dict[str, Any] = {}

    # Voxel size parameters
    try:
        validated["microns_per_voxel"] = [
            float(value) for value in params.get("microns_per_voxel", [1.0, 1.0, 1.0])
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("microns_per_voxel must be a 3-element array") from exc
    if len(validated["microns_per_voxel"]) != 3:
        raise ValueError("microns_per_voxel must be a 3-element array")
    if any(value <= 0 for value in validated["microns_per_voxel"]):
        raise ValueError("microns_per_voxel values must be positive")

    # Vessel size parameters
    validated["radius_of_smallest_vessel_in_microns"] = params.get(
        "radius_of_smallest_vessel_in_microns", 1.5
    )
    validated["radius_of_largest_vessel_in_microns"] = params.get(
        "radius_of_largest_vessel_in_microns", 50.0
    )

    if validated["radius_of_smallest_vessel_in_microns"] <= 0:
        raise ValueError("radius_of_smallest_vessel_in_microns must be positive")
    if (
        validated["radius_of_largest_vessel_in_microns"]
        <= validated["radius_of_smallest_vessel_in_microns"]
    ):
        raise ValueError("radius_of_largest_vessel_in_microns must be larger than smallest")

    # PSF parameters
    validated["approximating_PSF"] = params.get("approximating_PSF", True)
    if validated["approximating_PSF"]:
        validated["numerical_aperture"] = params.get("numerical_aperture", 0.95)
        validated["excitation_wavelength_in_microns"] = params.get(
            "excitation_wavelength_in_microns", 1.3
        )
        validated["sample_index_of_refraction"] = params.get("sample_index_of_refraction", 1.33)

        # Validate excitation wavelength (common range for two-photon microscopy)
        if not (0.7 <= validated["excitation_wavelength_in_microns"] <= 3.0):
            warnings.warn(
                "Excitation wavelength outside typical range (0.7-3.0 Î¼m). "
                "This may indicate an error or unusual experimental setup.",
                stacklevel=2,
            )

    # Scale parameters
    validated["scales_per_octave"] = params.get("scales_per_octave", 1.5)
    validated["gaussian_to_ideal_ratio"] = params.get("gaussian_to_ideal_ratio", 1.0)
    validated["spherical_to_annular_ratio"] = params.get("spherical_to_annular_ratio", 1.0)
    validated["energy_projection_mode"] = (
        str(params.get("energy_projection_mode", "matlab")).strip().lower()
    )
    validated["energy_sign"] = params.get("energy_sign", -1.0)
    if validated["energy_sign"] not in (-1, 1):
        raise ValueError("energy_sign must be -1 or 1")
    if validated["scales_per_octave"] <= 0:
        raise ValueError("scales_per_octave must be positive (e.g., 1.5)")
    if validated["gaussian_to_ideal_ratio"] < 0:
        raise ValueError(
            "gaussian_to_ideal_ratio must be non-negative; use 0.0 for an ideal-only kernel"
        )
    if validated["gaussian_to_ideal_ratio"] > 1:
        raise ValueError("gaussian_to_ideal_ratio must be between 0 and 1 inclusive")
    if validated["spherical_to_annular_ratio"] < 0:
        raise ValueError(
            "spherical_to_annular_ratio must be non-negative; use 0.0 for annular-only weighting"
        )
    if validated["spherical_to_annular_ratio"] > 1:
        raise ValueError("spherical_to_annular_ratio must be between 0 and 1 inclusive")
    if validated["energy_projection_mode"] not in ("matlab", "paper"):
        raise ValueError("energy_projection_mode must be 'matlab' or 'paper'")

    # Processing parameters
    validated["max_voxels_per_node_energy"] = params.get("max_voxels_per_node_energy", 1e5)
    validated["energy_upper_bound"] = params.get("energy_upper_bound", 0.0)
    validated["space_strel_apothem"] = params.get("space_strel_apothem", 1)
    validated["space_strel_apothem_edges"] = params.get(
        "space_strel_apothem_edges",
        validated["space_strel_apothem"],
    )
    validated["max_voxels_per_node"] = params.get("max_voxels_per_node", 6000)
    validated["length_dilation_ratio"] = params.get("length_dilation_ratio", 1.0)
    validated["number_of_edges_per_vertex"] = params.get("number_of_edges_per_vertex", 4)
    validated["step_size_per_origin_radius"] = params.get("step_size_per_origin_radius", 1.0)
    validated["max_edge_length_per_origin_radius"] = params.get(
        "max_edge_length_per_origin_radius", 60.0
    )
    validated["sigma_per_influence_vertices"] = params.get("sigma_per_influence_vertices", 1.0)
    validated["sigma_per_influence_edges"] = params.get("sigma_per_influence_edges", 0.5)
    validated["max_edge_energy"] = params.get("max_edge_energy", 0.0)
    validated["min_hair_length_in_microns"] = params.get("min_hair_length_in_microns", 0.0)
    validated["bandpass_window"] = params.get("bandpass_window", 0.0)
    validated["edge_method"] = params.get("edge_method", "tracing")
    if validated["edge_method"] not in ("tracing", "watershed"):
        raise ValueError("edge_method must be 'tracing' or 'watershed'")
    validated["energy_method"] = params.get("energy_method", "hessian")
    if validated["energy_method"] not in (
        "hessian",
        "frangi",
        "sato",
        "simpleitk_objectness",
        "cupy_hessian",
    ):
        raise ValueError(
            "energy_method must be 'hessian', 'frangi', 'sato', "
            "'simpleitk_objectness', or 'cupy_hessian'"
        )
    validated["direction_method"] = params.get("direction_method", "hessian")
    if validated["direction_method"] not in ("hessian", "uniform"):
        raise ValueError("direction_method must be 'hessian' or 'uniform'")
    validated["return_all_scales"] = params.get("return_all_scales", False)
    validated["energy_storage_format"] = str(params.get("energy_storage_format", "auto")).strip()
    if validated["energy_storage_format"] not in ("auto", "npy", "zarr"):
        raise ValueError("energy_storage_format must be 'auto', 'npy', or 'zarr'")
    if validated["max_voxels_per_node_energy"] <= 0:
        raise ValueError(
            "max_voxels_per_node_energy must be positive; increase to process larger volumes"
        )
    if validated["length_dilation_ratio"] <= 0:
        raise ValueError("length_dilation_ratio must be positive")
    if validated["max_voxels_per_node"] <= 0:
        raise ValueError("max_voxels_per_node must be positive")
    if validated["number_of_edges_per_vertex"] < 1:
        raise ValueError("number_of_edges_per_vertex must be at least 1")
    if validated["step_size_per_origin_radius"] <= 0:
        raise ValueError("step_size_per_origin_radius must be positive; try 0.5 for finer tracing")
    if validated["max_edge_length_per_origin_radius"] <= 0:
        raise ValueError("max_edge_length_per_origin_radius must be positive")
    if validated["sigma_per_influence_vertices"] <= 0:
        raise ValueError("sigma_per_influence_vertices must be positive")
    if validated["sigma_per_influence_edges"] <= 0:
        raise ValueError("sigma_per_influence_edges must be positive")
    if validated["min_hair_length_in_microns"] < 0:
        raise ValueError("min_hair_length_in_microns cannot be negative")
    if validated["bandpass_window"] < 0:
        raise ValueError("bandpass_window must be non-negative; set 0 to disable")
    validated["discrete_tracing"] = params.get("discrete_tracing", False)

    for key in (
        "max_voxels_per_node_energy",
        "space_strel_apothem",
        "space_strel_apothem_edges",
        "max_voxels_per_node",
        "number_of_edges_per_vertex",
    ):
        validated[key] = _coerce_integral_parameter(key, validated[key])

    # Preserve workflow-specific extension keys outside the retired parity surface.
    for key, value in params.items():
        validated.setdefault(key, value)

    return validated
