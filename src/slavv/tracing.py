
"""
Vertex and Edge tracing logic for SLAVV.
Handles vertex extraction (local maxima/minima) and edge tracing through the energy field.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import feature  # Needed for Hessian
from skimage.segmentation import watershed

# Imports from sibling modules
from .energy import compute_gradient_impl, spherical_structuring_element

logger = logging.getLogger(__name__)

def in_bounds(pos: np.ndarray, shape: Tuple[int, ...]) -> bool:
    """Check if the floored position lies within array bounds."""
    # Optimization for 3D case which is the bottleneck in tracing
    if len(shape) == 3:
        return (0 <= pos[0] < shape[0] and
                0 <= pos[1] < shape[1] and
                0 <= pos[2] < shape[2])

    pos_int = np.floor(pos).astype(int)
    return np.all((pos_int >= 0) & (pos_int < np.array(shape)))

def compute_gradient(energy: np.ndarray, pos: np.ndarray, microns_per_voxel: np.ndarray) -> np.ndarray:
    """Compute gradient at ``pos`` using central differences (wrapper for implementation)."""
    pos_int = np.round(pos).astype(np.int64)
    # Ensure proper dtypes for Numba compatibility (if enabled in impl)
    energy_arr = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_arr = np.asarray(microns_per_voxel, dtype=np.float64)
    return compute_gradient_impl(energy_arr, pos_int, mpv_arr)

def generate_edge_directions(n_directions: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate uniformly distributed unit vectors on the sphere.
    
    Parameters
    ----------
    n_directions : int
        Number of direction vectors to generate.
    seed : int, optional
        Random seed for reproducibility. If None, uses unseeded RNG.
    """
    if n_directions <= 0:
        return np.empty((0, 3))
    if n_directions == 1:
        return np.array([[0, 0, 1]], dtype=float)

    # Generate random points from a 3D standard normal distribution
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n_directions, 3))
    # Normalize to unit vectors
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def extract_vertices(energy_data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract vertices as local extrema in the energy field.
    MATLAB Equivalent: `get_vertices_V200.m`
    """
    logger.info("Extracting vertices")
    
    energy = energy_data['energy']
    scale_indices = energy_data['scale_indices']
    lumen_radius_pixels = energy_data['lumen_radius_pixels']
    energy_sign = energy_data.get('energy_sign', -1.0)
    lumen_radius_microns = energy_data['lumen_radius_microns']

    # Parameters
    energy_upper_bound = params.get('energy_upper_bound', 0.0)
    space_strel_apothem = params.get('space_strel_apothem', 1)
    length_dilation_ratio = params.get('length_dilation_ratio', 1.0)
    voxel_size = np.array(params.get('microns_per_voxel', [1.0, 1.0, 1.0]), dtype=float)

    # Find local extrema using a spacing-aware structuring element
    strel = spherical_structuring_element(space_strel_apothem, voxel_size)
    if energy_sign < 0:
        filt = ndi.minimum_filter(energy, footprint=strel, mode="nearest")
        extrema = (energy <= filt) & (energy < energy_upper_bound)
    else:
        filt = ndi.maximum_filter(energy, footprint=strel, mode="nearest")
        extrema = (energy >= filt) & (energy > energy_upper_bound)

    # Get coordinates of extrema
    coords = np.where(extrema)
    vertex_positions = np.column_stack(coords)
    vertex_scales = scale_indices[coords]
    vertex_energies = energy[coords]

    # Sort by energy (best first depending on sign)
    if energy_sign < 0:
        sort_indices = np.argsort(vertex_energies)
    else:
        sort_indices = np.argsort(-vertex_energies)
    vertex_positions = vertex_positions[sort_indices]
    vertex_scales = vertex_scales[sort_indices]
    vertex_energies = vertex_energies[sort_indices]

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 vertices")
        return {
            'positions': np.empty((0, 3), dtype=np.float32),
            'scales': np.empty((0,), dtype=np.int16),
            'energies': np.empty((0,), dtype=np.float32),
            'radii_pixels': np.empty((0,), dtype=np.float32),
            'radii_microns': np.empty((0,), dtype=np.float32),
            'radii': np.empty((0,), dtype=np.float32),
        }

    # Volume exclusion: remove overlapping vertices using energy-ordered cKDTree
    vertex_positions_microns = vertex_positions * voxel_size
    max_radius = np.max(lumen_radius_microns) * length_dilation_ratio
    tree = cKDTree(vertex_positions_microns)
    keep_mask = np.ones(len(vertex_positions), dtype=bool)
    
    for i, pos in enumerate(vertex_positions_microns):
        if not keep_mask[i]:
            continue
        radius_i = lumen_radius_microns[vertex_scales[i]] * length_dilation_ratio
        neighbors = tree.query_ball_point(pos, radius_i + max_radius)
        for j in neighbors:
            if j <= i or not keep_mask[j]:
                continue
            radius_j = lumen_radius_microns[vertex_scales[j]] * length_dilation_ratio
            dist = np.linalg.norm(vertex_positions_microns[j] - pos)
            if dist < (radius_i + radius_j):
                keep_mask[j] = False

    vertex_positions = vertex_positions[keep_mask]
    vertex_scales = vertex_scales[keep_mask]
    vertex_energies = vertex_energies[keep_mask]

    logger.info(f"Extracted {len(vertex_positions)} vertices")

    # Standardize output dtypes
    vertex_positions = vertex_positions.astype(np.float32)
    vertex_scales = vertex_scales.astype(np.int16)
    vertex_energies = vertex_energies.astype(np.float32)
    radii_pixels = lumen_radius_pixels[vertex_scales].astype(np.float32)
    radii_microns = lumen_radius_microns[vertex_scales].astype(np.float32)

    return {
        'positions': vertex_positions,
        'scales': vertex_scales,
        'energies': vertex_energies,
        'radii_pixels': radii_pixels,
        'radii_microns': radii_microns,
        'radii': radii_microns,
    }

def near_vertex(pos: np.ndarray, vertex_positions: np.ndarray,
                vertex_scales: np.ndarray, lumen_radius_microns: np.ndarray,
                microns_per_voxel: np.ndarray,
                tree: Optional[cKDTree] = None,
                max_search_radius: float = 0.0) -> Optional[int]:
    """Return the index of a nearby vertex if within its physical radius; otherwise None"""
    if tree is not None:
        # Optimized spatial query
        pos_microns = pos * microns_per_voxel
        # Query candidates within max possible radius
        candidates = tree.query_ball_point(pos_microns, max_search_radius)
        for i in candidates:
            # Check specific radius for this candidate
            vertex_pos = vertex_positions[i]
            vertex_scale = vertex_scales[i]
            radius = lumen_radius_microns[vertex_scale]
            diff = pos_microns - (vertex_pos * microns_per_voxel)
            if np.linalg.norm(diff) < radius:
                return i
        return None
    else:
        # Fallback linear scan
        for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
            radius = lumen_radius_microns[vertex_scale]
            diff = (pos - vertex_pos) * microns_per_voxel
            if np.linalg.norm(diff) < radius:
                return i
        return None

def find_terminal_vertex(pos: np.ndarray, vertex_positions: np.ndarray,
                         vertex_scales: np.ndarray, lumen_radius_microns: np.ndarray,
                         microns_per_voxel: np.ndarray,
                         tree: Optional[cKDTree] = None,
                         max_search_radius: float = 0.0) -> Optional[int]:
    """Find the index of a terminal vertex near a given position, if any."""
    return near_vertex(pos, vertex_positions, vertex_scales,
                       lumen_radius_microns, microns_per_voxel,
                       tree=tree, max_search_radius=max_search_radius)



def estimate_vessel_directions(energy: np.ndarray, pos: np.ndarray, radius: float,
                               microns_per_voxel: np.ndarray) -> np.ndarray:
    """Estimate vessel directions at a vertex via local Hessian analysis."""
    # Determine a small neighborhood around the vertex
    sigma = max(radius / 2.0, 1.0)
    center = np.round(pos).astype(int)
    r = int(max(1, np.ceil(sigma)))
    slices = tuple(
        slice(max(c - r, 0), min(c + r + 1, s))
        for c, s in zip(center, energy.shape)
    )
    patch = energy[slices]
    # Fallback to uniform directions if patch is too small
    if patch.ndim != 3 or min(patch.shape) < 3:
        return generate_edge_directions(2)

    # Rescale patch to account for anisotropic voxel spacing
    scale = microns_per_voxel / microns_per_voxel.min()
    if not np.allclose(scale, 1):
        patch = ndi.zoom(patch, scale, order=1, mode="nearest")

    # --- EXPLANATION FOR JUNIOR DEVS ---
    # WHY: We need to find the direction of the vessel at this point.
    # HOW: We calculate the Hessian matrix (second-order partial derivatives) of intensity.
    #      The eigenvalues of the Hessian describe the local curvature:
    #      - Small eigenvector -> Direction of least curvature (along the vessel).
    #      - Large eigenvectors -> Direction of high curvature (across the vessel wall).
    #      We pick the eigenvector corresponding to the smallest absolute eigenvalue
    #      as the vessel direction.
    # -----------------------------------

    # Compute Hessian in the local patch and extract center values
    hessian_elems = [
        h * (radius ** 2)
        for h in feature.hessian_matrix(
            patch, sigma=sigma, mode='nearest', order='rc'
        )
    ]
    patch_center = tuple(np.array(patch.shape) // 2)
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h[patch_center] for h in hessian_elems]
    H = np.array([
        [Hxx, Hxy, Hxz],
        [Hxy, Hyy, Hyz],
        [Hxz, Hyz, Hzz],
    ])
    # Eigen decomposition to find principal axis
    try:
        w, v = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return generate_edge_directions(2)
    if not np.all(np.isfinite(w)):
        return generate_edge_directions(2)

    # Fallback if eigenvalues are nearly isotropic or all zero
    w_abs = np.sort(np.abs(w))
    max_eig = w_abs[-1]
    if max_eig == 0 or (w_abs[1] - w_abs[0]) < 1e-6 * max_eig:
        return generate_edge_directions(2)

    direction = v[:, np.argmin(np.abs(w))]
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return generate_edge_directions(2)
    direction = direction / norm
    return np.stack((direction, -direction))

def trace_edge(
    energy: np.ndarray,
    start_pos: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    max_edge_energy: float,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
    max_steps: int,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
    discrete_steps: bool = False,
    tree: Optional[cKDTree] = None,
    max_search_radius: float = 0.0,
) -> List[np.ndarray]:
    """Trace an edge through the energy field with adaptive step sizing."""
    trace = [start_pos.copy()]
    current_pos = start_pos.copy()
    current_dir = direction.copy()
    prev_energy = energy[tuple(np.floor(current_pos).astype(int))]

    for _ in range(max_steps):
        attempt = 0
        while attempt < 10:
            next_pos = current_pos + current_dir * step_size
            if discrete_steps:
                next_pos = np.round(next_pos)
                if np.array_equal(next_pos, current_pos):
                    return trace
            if not in_bounds(next_pos, energy.shape):
                return trace
            pos_int = np.floor(next_pos).astype(int)
            current_energy = energy[pos_int[0], pos_int[1], pos_int[2]]
            if (energy_sign < 0 and current_energy > max_edge_energy) or (
                energy_sign > 0 and current_energy < max_edge_energy
            ):
                return trace
            if (energy_sign < 0 and current_energy > prev_energy) or (
                energy_sign > 0 and current_energy < prev_energy
            ):
                step_size *= 0.5
                if step_size < 0.5:
                    return trace
                attempt += 1
                continue
            break

        trace.append(next_pos.copy())
        current_pos = next_pos.copy()
        prev_energy = current_energy

        # Optimized gradient computation:
        # Avoids wrapper overhead by calling implementation directly.
        pos_int = np.round(current_pos).astype(np.int64)
        gradient = compute_gradient_impl(energy, pos_int, microns_per_voxel)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-12:
            # Project gradient onto plane perpendicular to current direction
            perp_grad = gradient - current_dir * np.dot(gradient, current_dir)
            # Steer along ridge by opposing gradient direction
            current_dir = current_dir - np.sign(energy_sign) * perp_grad
            norm = np.linalg.norm(current_dir)
            if norm > 1e-12:
                current_dir = (current_dir / norm).astype(float)

        terminal_vertex_idx = near_vertex(
            current_pos, vertex_positions, vertex_scales,
            lumen_radius_microns, microns_per_voxel,
            tree=tree, max_search_radius=max_search_radius
        )
        if terminal_vertex_idx is not None:
            trace.append(vertex_positions[terminal_vertex_idx].copy())
            break

    return trace

def extract_edges(energy_data: Dict[str, Any], vertices: Dict[str, Any], 
                  params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract edges by tracing from vertices through energy field.
    MATLAB Equivalent: `get_edges_V300.m`
    """
    logger.info("Extracting edges")
    
    energy = energy_data["energy"]
    vertex_positions = vertices["positions"]
    vertex_scales = vertices["scales"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    energy_sign = energy_data.get("energy_sign", -1.0)

    # Parameters
    max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    length_ratio = params.get("length_dilation_ratio", 1.0)
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    discrete_tracing = params.get("discrete_tracing", False)
    direction_method = params.get("direction_method", "hessian")

    edges = []
    edge_connections = []
    edge_energies: List[float] = []
    edges_per_vertex = np.zeros(len(vertex_positions), dtype=int)
    existing_pairs = set()

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 edges")
        return {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "energies": np.zeros((0,), dtype=np.float32),
            "vertex_positions": vertex_positions.astype(np.float32),
        }

    # Build cKDTree for optimized spatial queries
    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0

    # Prepare arrays once for performance (avoiding overhead in trace_edge)
    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        if edges_per_vertex[vertex_idx] >= max_edges_per_vertex:
            continue
        start_radius = lumen_radius_pixels[start_scale]
        step_size = start_radius * step_size_ratio
        max_length = start_radius * length_ratio
        max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))

        if direction_method == "hessian":
            directions = estimate_vessel_directions(
                energy, start_pos, start_radius, microns_per_voxel
            )
            if directions.shape[0] < max_edges_per_vertex:
                extra = generate_edge_directions(
                    max_edges_per_vertex - directions.shape[0]
                )
                directions = np.vstack([directions, extra])
            else:
                directions = directions[:max_edges_per_vertex]
        else:
            directions = generate_edge_directions(max_edges_per_vertex)
        
        for direction in directions:
            if edges_per_vertex[vertex_idx] >= max_edges_per_vertex:
                break
            edge_trace = trace_edge(
                energy_prepared,
                start_pos,
                direction,
                step_size,
                max_edge_energy,
                vertex_positions,
                vertex_scales,
                lumen_radius_pixels,
                lumen_radius_microns,
                max_steps,
                mpv_prepared,
                energy_sign,
                discrete_steps=discrete_tracing,
                tree=tree,
                max_search_radius=max_vertex_radius,
            )
            if len(edge_trace) > 1:  # Valid edge found
                terminal_vertex = find_terminal_vertex(
                    edge_trace[-1], vertex_positions, vertex_scales,
                    lumen_radius_microns, microns_per_voxel,
                    tree=tree, max_search_radius=max_vertex_radius
                )
                if terminal_vertex == vertex_idx:
                    continue
                if terminal_vertex is not None:
                    if edges_per_vertex[terminal_vertex] >= max_edges_per_vertex:
                        continue
                    pair = tuple(sorted((vertex_idx, terminal_vertex)))
                    if pair in existing_pairs:
                        continue
                edge_arr = np.asarray(edge_trace, dtype=np.float32)
                edges.append(edge_arr)
                idx = np.floor(edge_arr).astype(int)
                energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
                edge_energies.append(float(np.mean(energies)))
                edge_connections.append([
                    vertex_idx,
                    terminal_vertex if terminal_vertex is not None else -1,
                ])
                edges_per_vertex[vertex_idx] += 1
                if terminal_vertex is not None:
                    edges_per_vertex[terminal_vertex] += 1
                    existing_pairs.add(pair)
    
    logger.info(f"Extracted {len(edges)} edges")

    edge_connections = np.asarray(edge_connections, dtype=np.int32).reshape(-1, 2)

    return {
        "traces": edges,
        "connections": edge_connections,
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32)
    }

def extract_edges_watershed(energy_data: Dict[str, Any],
                            vertices: Dict[str, Any],
                            params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract edges using watershed segmentation seeded at vertices."""
    logger.info("Extracting edges via watershed")

    energy = energy_data["energy"]
    energy_sign = float(energy_data.get("energy_sign", -1.0))
    vertex_positions = vertices["positions"]

    markers = np.zeros_like(energy, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(energy.shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

    labels = watershed(-energy_sign * energy, markers)
    structure = ndi.generate_binary_structure(3, 1)

    edges = []
    connections = []
    edge_energies: List[float] = []
    seen = set()

    for label in range(1, len(vertex_positions) + 1):
        region = labels == label
        dilated = ndi.binary_dilation(region, structure)
        neighbors = np.unique(labels[dilated & (labels != label)])
        for neighbor in neighbors:
            if neighbor <= label or neighbor == 0:
                continue
            pair = (label - 1, neighbor - 1)
            if pair in seen:
                continue
            boundary = (
                ndi.binary_dilation(labels == neighbor, structure) & region
            ) | (
                ndi.binary_dilation(region, structure) & (labels == neighbor)
            )
            coords = np.argwhere(boundary)
            if coords.size == 0:
                continue
            coords = coords.astype(np.float32)
            edges.append(coords)
            idx = np.floor(coords).astype(int)
            energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
            edge_energies.append(float(np.mean(energies)))
            connections.append([label - 1, neighbor - 1])
            seen.add(pair)

    logger.info("Extracted %d watershed edges", len(edges))

    return {
        "traces": edges,
        "connections": np.asarray(connections, dtype=np.int32),
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }

__all__ = [
    "in_bounds",
    "compute_gradient",
    "generate_edge_directions",
    "extract_vertices",
    "near_vertex",
    "find_terminal_vertex",
    "estimate_vessel_directions",
    "trace_edge",
    "extract_edges",
    "extract_edges_watershed",
]
