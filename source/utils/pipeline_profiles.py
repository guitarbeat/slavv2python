"""Shared parameter presets for public SLAVV pipeline profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

PIPELINE_PROFILE_CHOICES: tuple[str, ...] = ("paper", "matlab_compat")

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "paper": {
        "pipeline_profile": "paper",
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "radius_of_smallest_vessel_in_microns": 1.5,
        "radius_of_largest_vessel_in_microns": 50.0,
        "approximating_PSF": True,
        "numerical_aperture": 0.95,
        "excitation_wavelength_in_microns": 1.3,
        "sample_index_of_refraction": 1.33,
        "scales_per_octave": 1.5,
        "gaussian_to_ideal_ratio": 1.0,
        "spherical_to_annular_ratio": 1.0,
        "energy_projection_mode": "paper",
        "energy_method": "hessian",
        "energy_storage_format": "auto",
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "space_strel_apothem_edges": 1,
        "max_voxels_per_node_energy": 100000,
        "max_voxels_per_node": 6000,
        "length_dilation_ratio": 1.0,
        "number_of_edges_per_vertex": 4,
        "step_size_per_origin_radius": 1.0,
        "max_edge_length_per_origin_radius": 60.0,
        "sigma_per_influence_vertices": 1.0,
        "sigma_per_influence_edges": 0.5,
        "max_edge_energy": 0.0,
        "min_hair_length_in_microns": 0.0,
        "bandpass_window": 0.0,
        "edge_method": "tracing",
        "direction_method": "hessian",
        "return_all_scales": False,
        "discrete_tracing": False,
        "comparison_exact_network_use_conflict_painting": True,
    },
    "matlab_compat": {
        "pipeline_profile": "matlab_compat",
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "radius_of_smallest_vessel_in_microns": 1.5,
        "radius_of_largest_vessel_in_microns": 50.0,
        "approximating_PSF": True,
        "numerical_aperture": 0.95,
        "excitation_wavelength_in_microns": 1.3,
        "sample_index_of_refraction": 1.33,
        "scales_per_octave": 1.5,
        "gaussian_to_ideal_ratio": 1.0,
        "spherical_to_annular_ratio": 1.0,
        "energy_projection_mode": "matlab",
        "energy_method": "hessian",
        "energy_storage_format": "auto",
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "space_strel_apothem_edges": 1,
        "max_voxels_per_node_energy": 100000,
        "max_voxels_per_node": 6000,
        "length_dilation_ratio": 1.0,
        "number_of_edges_per_vertex": 4,
        "step_size_per_origin_radius": 1.0,
        "max_edge_length_per_origin_radius": 60.0,
        "sigma_per_influence_vertices": 1.0,
        "sigma_per_influence_edges": 0.5,
        "max_edge_energy": 0.0,
        "min_hair_length_in_microns": 0.0,
        "bandpass_window": 0.0,
        "edge_method": "tracing",
        "direction_method": "hessian",
        "return_all_scales": False,
        "discrete_tracing": False,
        "comparison_exact_network_use_conflict_painting": False,
    },
}


def normalize_pipeline_profile_name(profile: str) -> str:
    """Return a normalized pipeline profile name or raise for unsupported values."""
    normalized = str(profile).strip().lower()
    if normalized not in _PROFILE_DEFAULTS:
        valid = ", ".join(PIPELINE_PROFILE_CHOICES)
        raise ValueError(f"pipeline_profile must be one of: {valid}")
    return normalized


def get_pipeline_profile_defaults(profile: str) -> dict[str, Any]:
    """Return a copy of the default parameters for a named pipeline profile."""
    normalized = normalize_pipeline_profile_name(profile)
    return deepcopy(_PROFILE_DEFAULTS[normalized])


def apply_pipeline_profile(
    parameters: Mapping[str, Any],
    *,
    default_profile: str | None = None,
) -> dict[str, Any]:
    """Seed profile defaults, then overlay explicit user parameters."""
    raw_parameters = dict(parameters)
    requested_profile = raw_parameters.get("pipeline_profile", default_profile)
    if requested_profile is None:
        return raw_parameters

    normalized_profile = normalize_pipeline_profile_name(str(requested_profile))
    resolved_parameters = get_pipeline_profile_defaults(normalized_profile)
    resolved_parameters.update(
        {key: value for key, value in raw_parameters.items() if value is not None}
    )
    resolved_parameters["pipeline_profile"] = normalized_profile
    return resolved_parameters


__all__ = [
    "PIPELINE_PROFILE_CHOICES",
    "apply_pipeline_profile",
    "get_pipeline_profile_defaults",
    "normalize_pipeline_profile_name",
]
