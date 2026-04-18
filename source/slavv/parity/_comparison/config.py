"""Parameter and artifact-discovery helpers for parity comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_parameters(params_file: str | None = None) -> dict[str, Any]:
    """Load parameters from JSON file or use defaults."""
    if params_file and os.path.exists(params_file):
        with open(params_file, encoding="utf-8") as handle:
            params = json.load(handle)
    else:
        params = {
            "microns_per_voxel": [1.0, 1.0, 1.0],
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 50.0,
            "approximating_PSF": True,
            "excitation_wavelength_in_microns": 1.3,
            "numerical_aperture": 0.95,
            "sample_index_of_refraction": 1.33,
            "scales_per_octave": 1.5,
            "gaussian_to_ideal_ratio": 1.0,
            "spherical_to_annular_ratio": 1.0,
            "max_voxels_per_node_energy": 1e5,
        }

    if "microns_per_voxel" in params:
        params["microns_per_voxel"] = np.array(params["microns_per_voxel"])

    return params


def discover_matlab_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Discover the newest MATLAB batch folder and key output artifacts."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return {}

    batch_folders = sorted(
        path for path in output_path.iterdir() if path.is_dir() and path.name.startswith("batch_")
    )
    if not batch_folders:
        return {}

    batch_folder = batch_folders[-1]
    artifacts: dict[str, Any] = {"batch_folder": str(batch_folder)}

    vectors_dir = batch_folder / "vectors"
    if vectors_dir.exists():
        artifacts["vectors_dir"] = str(vectors_dir)
        if network_files := sorted(vectors_dir.glob("network_*.mat")):
            artifacts["network_mat"] = str(network_files[-1])

    return artifacts
