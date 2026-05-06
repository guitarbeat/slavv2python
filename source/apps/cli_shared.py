"""Shared helpers for the SLAVV CLI."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from source.utils import apply_pipeline_profile

if TYPE_CHECKING:
    import argparse

from .cli_parser import _EXPORT_FILE_NAMES

_DETAILED_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_SIMPLE_LOG_FORMAT = "%(asctime)s %(message)s"


def _build_pipeline_parameters(args: argparse.Namespace) -> dict:
    """Convert parsed CLI arguments to a SLAVV parameters dict."""
    requested_parameters = {
        "pipeline_profile": args.pipeline_profile,
        "energy_method": args.energy_method,
        "energy_projection_mode": args.energy_projection_mode,
        "energy_storage_format": args.energy_storage_format,
        "edge_method": args.edge_method,
        "radius_of_smallest_vessel_in_microns": args.vessel_radius,
        "radius_of_largest_vessel_in_microns": args.largest_vessel_radius,
        "microns_per_voxel": list(args.microns_per_voxel) if args.microns_per_voxel else None,
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


def _configure_logging(verbose: bool, *, format_string: str) -> None:
    """Configure command logging with the requested verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=format_string)


def _require_existing_file(path: str, *, label: str = "file") -> None:
    """Exit with a consistent error if a required file path is missing."""
    if not os.path.isfile(path):
        print(f"Error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)


def _expand_export_formats(export_formats: list[str]) -> list[str]:
    """Normalize CLI export selections into concrete formats."""
    return ["csv", "json", "casx", "vmv", "mat"] if "all" in export_formats else export_formats


def _build_export_artifacts(output_dir: str, export_formats: list[str]) -> dict[str, str]:
    """Build run-state artifact paths for requested exports."""
    return {
        fmt: os.path.join(output_dir, _EXPORT_FILE_NAMES[fmt])
        for fmt in export_formats
        if fmt != "csv"
    }
