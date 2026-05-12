"""Shared internal helpers for the SLAVV CLI."""

from __future__ import annotations

import os
import sys
from typing import Any

from ...workflows import apply_pipeline_profile
from .parser import _EXPORT_FILE_NAMES


def _prepare_run_parameters(args) -> dict[str, Any]:
    """Combine profile defaults with explicit user CLI overrides."""
    requested_parameters = {
        "energy_method": args.energy_method,
        "energy_projection_mode": args.energy_projection_mode,
        "energy_storage_format": args.energy_storage_format,
        "edge_method": args.edge_method,
        "radius_of_smallest_vessel_in_microns": args.vessel_radius,
        "radius_of_largest_vessel_in_microns": args.largest_vessel_radius,
        "microns_per_voxel": args.microns_per_voxel,
        "scales_per_octave": args.scales_per_octave,
        "gaussian_to_ideal_ratio": args.gaussian_to_ideal_ratio,
        "spherical_to_annular_ratio": args.spherical_to_annular_ratio,
        "energy_upper_bound": args.energy_upper_bound,
        "space_strel_apothem": args.space_strel_apothem,
        "space_strel_apothem_edges": args.space_strel_apothem_edges,
        "length_dilation_ratio": args.length_dilation_ratio,
        "number_of_edges_per_vertex": args.number_of_edges_per_vertex,
        "step_size_per_origin_radius": args.step_size_per_origin_radius,
        "max_edge_length_per_origin_radius": args.max_edge_length_per_origin_radius,
        "max_edge_energy": args.max_edge_energy,
        "min_hair_length_in_microns": args.min_hair_length_in_microns,
        "n_jobs": args.n_jobs,
    }
    return apply_pipeline_profile(
        requested_parameters,
        default_profile=args.pipeline_profile,
    )


def _require_existing_file(path: str) -> str:
    """Ensure a file exists or exit the CLI with an error."""
    if not os.path.isfile(path):
        print(f"Error: Required file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def _resolve_export_artifact_paths(output_dir: str, export_formats: list[str]) -> dict[str, str]:
    """Build run-state artifact paths for requested exports."""
    return {
        fmt: os.path.join(output_dir, _EXPORT_FILE_NAMES[fmt])
        for fmt in export_formats
        if fmt != "csv"
    }


__all__ = [
    "_prepare_run_parameters",
    "_require_existing_file",
    "_resolve_export_artifact_paths",
]
