"""
Python wrapper for the SLAVV pipeline stages (Energy → Vertices → Edges → Network).

High-level entry point equivalent to MATLAB vectorize_V200.m.
See submodules for implementations matching the original source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.pipeline.edges.manager import EdgeManager
from slavv_python.pipeline.energy.manager import EnergyManager
from slavv_python.pipeline.network.manager import NetworkManager
from slavv_python.pipeline.vertices.manager import VertexManager

if TYPE_CHECKING:
    from slavv_python.schema.results import EdgeSet, EnergyResult, NetworkResult, VertexSet


def vectorize_python(
    image: np.ndarray,
    params: dict[str, Any],
    *,
    start_stage: str = "energy",
    final_stage: str = "network",
    use_resumable: bool = False,
    stage_controller: Any = None,
) -> dict[str, Any]:
    """
    Python version of MATLAB vectorize_V200.m .

    Orchestrates the full SLAVV pipeline:
    1. Energy (get_energy_V202 equivalent)
    2. Vertices (get_vertices_V200 + choose)
    3. Edges (get_edges + choose)
    4. Network (get_network)

    Matches the MATLAB logic for the stages, with support for the exact parity route.

    Args:
        image: 3D numpy array (Z, Y, X) or (Y, X, Z) depending on convention; see AGENTS.md for [Y,X,Z] Fortran.
        params: Dict with lumen_radius_range, microns_per_voxel, etc. (see PipelinePolicy).
        start_stage: 'energy', 'vertices', 'edges', 'network'
        final_stage: same
        use_resumable: whether to use checkpointing
        stage_controller: for resumable runs

    Returns:
        Dict with 'energy', 'vertices', 'edges', 'network' results.
    """
    results: dict[str, Any] = {}

    # Stage 1: Energy
    if start_stage in ("energy", "all") or final_stage == "energy":
        if use_resumable and stage_controller:
            energy_result: EnergyResult = EnergyManager.run_resumable(
                image, params, stage_controller
            )
        else:
            energy_result = EnergyManager.run(image, params)
        results["energy"] = energy_result
        if final_stage == "energy":
            return results

    # Stage 2: Vertices
    if start_stage in ("vertices", "all") or (
        final_stage in ("vertices", "edges", "network") and "energy" in results
    ):
        energy_data = results.get("energy")
        if energy_data is None:
            raise ValueError("Energy result required for vertices stage")
        if use_resumable and stage_controller:
            vertices: VertexSet = VertexManager.run_resumable(energy_data, params, stage_controller)
        else:
            vertices = VertexManager.run(energy_data, params)
        results["vertices"] = vertices
        if final_stage == "vertices":
            return results

    # Stage 3: Edges
    if start_stage in ("edges", "all") or final_stage in ("edges", "network"):
        vertices_data = results.get("vertices")
        if vertices_data is None:
            raise ValueError("Vertices required for edges")
        if use_resumable and stage_controller:
            edges: EdgeSet = EdgeManager.run_resumable(vertices_data, params, stage_controller)
        else:
            edges = EdgeManager.run(vertices_data, params)
        results["edges"] = edges
        if final_stage == "edges":
            return results

    # Stage 4: Network
    if start_stage in ("network", "all") or final_stage == "network":
        edges_data = results.get("edges")
        if edges_data is None:
            raise ValueError("Edges required for network")
        if use_resumable and stage_controller:
            network: NetworkResult = NetworkManager.run_resumable(
                edges_data, params, stage_controller
            )
        else:
            network = NetworkManager.run(edges_data, params)
        results["network"] = network

    return results


def get_energy_v202_python(
    image: np.ndarray,
    lumen_radius_range: np.ndarray,
    microns_per_voxel: np.ndarray,
    vessel_wall_thickness: float,
    pixels_per_sigma_psf: np.ndarray,
    max_voxels_per_node: int,
    gaussian_to_ideal_ratio: float = 0.5,
    spherical_to_annular_ratio: float = 0.5,
    energy_upper_bound: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Python port of get_energy_V202.m ."""
    from slavv_python.pipeline.energy.manager import EnergyManager

    result = EnergyManager.run(
        image,
        {
            "lumen_radius_range": lumen_radius_range,
            "microns_per_voxel": microns_per_voxel,
            "vessel_wall_thickness_in_microns": vessel_wall_thickness,
            "pixels_per_sigma_PSF_yxz": pixels_per_sigma_psf,
            "max_voxels_per_node": max_voxels_per_node,
            "gaussian_to_ideal_ratio": gaussian_to_ideal_ratio,
            "spherical_to_annular_ratio": spherical_to_annular_ratio,
            "energy_upper_bound": energy_upper_bound,
        },
    )
    return result.energy, result.scale_indices


def get_vertices_v200_python(
    lumen_radius_in_microns_range: np.ndarray,
    microns_per_pixel: np.ndarray,
    space_strel_apothem: int,
    max_voxels_per_node: float,
    energy_upper_bound: float,
    energy: np.ndarray,
    scale_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Python port of get_vertices_V200.m ."""
    from scipy.ndimage import minimum_filter

    strel_size = 2 * space_strel_apothem + 1
    footprint = np.ones((strel_size, strel_size, strel_size), dtype=bool)

    energy_min = minimum_filter(
        energy,
        footprint=footprint,
        mode="constant",
        cval=np.inf if energy_upper_bound < 0 else -np.inf,
    )

    if energy_upper_bound < 0:
        candidates = (energy == energy_min) & (energy < energy_upper_bound) & np.isfinite(energy)
    else:
        candidates = (energy == energy_min) & (energy > energy_upper_bound) & np.isfinite(energy)

    positions = np.argwhere(candidates)
    scales = scale_indices[candidates]
    energies = energy[candidates]

    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0], energies))
    positions = positions[order]
    scales = scales[order]
    energies = energies[order]

    return positions, scales, energies, positions


def get_edges_by_watershed_python(
    energy: np.ndarray,
    vertices: np.ndarray,
    scale_indices: np.ndarray,
    params: dict,
) -> tuple:
    """Python port of get_edges_by_watershed.m ."""
    from skimage.segmentation import watershed

    markers = np.zeros(energy.shape, dtype=int)
    for i, v in enumerate(vertices):
        y, x, z = v.astype(int)
        if 0 <= y < energy.shape[0] and 0 <= x < energy.shape[1] and 0 <= z < energy.shape[2]:
            markers[y, x, z] = i + 1

    if params.get("energy_sign", -1) < 0:
        ws = watershed(-energy, markers)
    else:
        ws = watershed(energy, markers)

    return ws, {}


def choose_edges_v200_python(edges, vertices, params):
    """Python port of choose_edges_V200.m ."""
    # See edges/selection.py for implementation.
    return edges


def get_network_v190_python(edges, vertices, params):
    """Python port of get_network_V190.m ."""
    # See network/manager.py for implementation.
    return {"strands": []}


if __name__ == "__main__":
    print("SLAVV Python port ready. Use vectorize_python(image, params).")
