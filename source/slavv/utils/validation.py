"""
Parameter validation functions for SLAVV.
"""

from __future__ import annotations

import warnings
from typing import Any


def _coerce_integral_parameter(name: str, value: Any) -> int:
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
    """
    Validate and set default parameters based on MATLAB implementation
    """
    validated = {}

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
                "Excitation wavelength outside typical range (0.7-3.0 μm). "
                "This may indicate an error or unusual experimental setup.",
                stacklevel=2,
            )

    # Scale parameters
    validated["scales_per_octave"] = params.get("scales_per_octave", 1.5)
    validated["gaussian_to_ideal_ratio"] = params.get("gaussian_to_ideal_ratio", 1.0)
    validated["spherical_to_annular_ratio"] = params.get("spherical_to_annular_ratio", 1.0)
    validated["energy_sign"] = params.get("energy_sign", -1.0)
    if validated["energy_sign"] not in (-1, 1):
        raise ValueError("energy_sign must be -1 or 1")
    if validated["scales_per_octave"] <= 0:
        raise ValueError("scales_per_octave must be positive (e.g., 1.5)")
    if validated["gaussian_to_ideal_ratio"] <= 0:
        raise ValueError(
            "gaussian_to_ideal_ratio must be positive; try 1.0 to disable prefiltering"
        )
    if validated["spherical_to_annular_ratio"] <= 0:
        raise ValueError(
            "spherical_to_annular_ratio must be positive; use 1.0 to skip annular subtraction"
        )

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
    ):
        raise ValueError(
            "energy_method must be 'hessian', 'frangi', 'sato', "
            "or 'simpleitk_objectness'"
        )
    validated["direction_method"] = params.get("direction_method", "hessian")
    if validated["direction_method"] not in ("hessian", "uniform"):
        raise ValueError("direction_method must be 'hessian' or 'uniform'")
    validated["return_all_scales"] = params.get("return_all_scales", False)
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
    validated["comparison_exact_network"] = bool(params.get("comparison_exact_network", False))
    validated["parity_frontier_reachability_gate"] = bool(
        params.get("parity_frontier_reachability_gate", True)
    )
    validated["parity_require_mutual_frontier_participation"] = bool(
        params.get("parity_require_mutual_frontier_participation", True)
    )
    parity_watershed_candidate_mode = str(
        params.get("parity_watershed_candidate_mode", "all_contacts")
    ).strip()
    allowed_parity_watershed_candidate_modes = {
        "all_contacts",
        "remaining_origin_contacts",
        "legacy_supplement",
    }
    if parity_watershed_candidate_mode not in allowed_parity_watershed_candidate_modes:
        raise ValueError(
            "parity_watershed_candidate_mode must be one of "
            "'all_contacts', 'remaining_origin_contacts', or 'legacy_supplement'"
        )
    validated["parity_watershed_candidate_mode"] = parity_watershed_candidate_mode
    parity_watershed_metric_threshold = params.get("parity_watershed_metric_threshold")
    validated["parity_watershed_metric_threshold"] = (
        None
        if parity_watershed_metric_threshold in (None, "")
        else float(parity_watershed_metric_threshold)
    )
    parity_candidate_salvage_mode = str(params.get("parity_candidate_salvage_mode", "auto")).strip()
    allowed_parity_candidate_salvage_modes = {
        "auto",
        "none",
        "frontier_deficit_geodesic",
        "all_origins_geodesic",
    }
    if parity_candidate_salvage_mode not in allowed_parity_candidate_salvage_modes:
        raise ValueError(
            "parity_candidate_salvage_mode must be one of "
            "'auto', 'none', 'frontier_deficit_geodesic', or 'all_origins_geodesic'"
        )
    validated["parity_candidate_salvage_mode"] = parity_candidate_salvage_mode
    validated["parity_geodesic_salvage_k_nearest"] = int(
        params.get("parity_geodesic_salvage_k_nearest", 10)
    )
    validated["parity_geodesic_salvage_box_margin_voxels"] = int(
        params.get("parity_geodesic_salvage_box_margin_voxels", 4)
    )
    validated["parity_geodesic_salvage_max_path_ratio"] = float(
        params.get("parity_geodesic_salvage_max_path_ratio", 2.5)
    )
    if validated["parity_geodesic_salvage_k_nearest"] < 1:
        raise ValueError("parity_geodesic_salvage_k_nearest must be at least 1")
    if validated["parity_geodesic_salvage_box_margin_voxels"] < 0:
        raise ValueError("parity_geodesic_salvage_box_margin_voxels cannot be negative")
    if validated["parity_geodesic_salvage_max_path_ratio"] < 1.0:
        raise ValueError("parity_geodesic_salvage_max_path_ratio must be at least 1.0")

    for key in (
        "max_voxels_per_node_energy",
        "space_strel_apothem",
        "space_strel_apothem_edges",
        "max_voxels_per_node",
        "number_of_edges_per_vertex",
    ):
        validated[key] = _coerce_integral_parameter(key, validated[key])

    return validated
