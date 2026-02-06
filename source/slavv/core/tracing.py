
"""
Vertex and Edge tracing logic for SLAVV.
Handles vertex extraction (local maxima/minima) and edge tracing through the energy field.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, Set

import math
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import feature  # Needed for Hessian
from skimage.segmentation import watershed
from skimage.draw import ellipsoid

# Imports from sibling modules
from .energy import compute_gradient_impl, spherical_structuring_element

logger = logging.getLogger(__name__)

def in_bounds(pos: np.ndarray, shape: Tuple[int, ...]) -> bool:
    """Check if the floored position lies within array bounds."""
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


def paint_vertex_image(
    vertex_positions: np.ndarray,  # Shape: (N, 3) - [y, x, z] positions
    vertex_scales: np.ndarray,     # Shape: (N,) - scale indices  
    lumen_radius_pixels: np.ndarray,  # Shape: (M, 3) - [ry, rx, rz] per scale
    image_shape: Tuple[int, int, int]  # (height, width, depth)
) -> np.ndarray:
    """
    Create a volume where each voxel is labeled with its vertex membership (1-indexed, 0=background).
    
    This matches MATLAB's approach for fast O(1) vertex detection during edge tracing.
    Paints ellipsoidal regions around each vertex with the vertex index.
    
    Parameters
    ----------
    vertex_positions : np.ndarray
        Vertex positions as (y, x, z) coordinates
    vertex_scales : np.ndarray
        Scale index for each vertex
    lumen_radius_pixels : np.ndarray
        Radii for each scale in pixels [ry, rx, rz]
    image_shape : tuple
        Shape of the output volume (height, width, depth)
        
    Returns
    -------
    vertex_image : np.ndarray
        Volume where each voxel contains vertex index (1-indexed) or 0 for background
    """
    vertex_image = np.zeros(image_shape, dtype=np.uint16)  # Supports up to 65k vertices
    
    for i, (pos, scale) in enumerate(zip(vertex_positions, vertex_scales)):
        # Get ellipsoid radii for this vertex's scale
        radii = lumen_radius_pixels[scale]  # [ry, rx, rz]
        
        # Generate ellipsoid mask using skimage
        try:
            # ellipsoid returns a 3D boolean array
            ellipsoid_mask = ellipsoid(radii[0], radii[1], radii[2], spacing=(1.0, 1.0, 1.0))
            # Get coordinates of True voxels (centered at origin of mask array)
            coords = np.where(ellipsoid_mask)
            # Center the ellipsoid coordinates (they're currently offset from origin)
            center = np.array(ellipsoid_mask.shape) // 2
            rr = coords[0] - center[0]
            cc = coords[1] - center[1]
            dd = coords[2] - center[2]
            
            # Offset to vertex position (convert to int)
            y_coords = rr + int(np.round(pos[0]))
            x_coords = cc + int(np.round(pos[1]))
            z_coords = dd + int(np.round(pos[2]))
            
            # Clip to image bounds
            valid_mask = (
                (y_coords >= 0) & (y_coords < image_shape[0]) &
                (x_coords >= 0) & (x_coords < image_shape[1]) &
                (z_coords >= 0) & (z_coords < image_shape[2])
            )
            
            y_coords = y_coords[valid_mask]
            x_coords = x_coords[valid_mask]
            z_coords = z_coords[valid_mask]
            
            # Paint vertex index (1-indexed, so i+1)
            vertex_image[y_coords, x_coords, z_coords] = i + 1
            
        except Exception as e:
            logger.warning(f"Failed to paint vertex {i} at {pos} with scale {scale}: {e}")
            continue
    
    logger.info(f"Painted {len(vertex_positions)} vertices into volume image")
    return vertex_image


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

def vertex_at_position(pos: np.ndarray, vertex_image: np.ndarray) -> Optional[int]:
    """
    Fast O(1) vertex lookup using pre-computed vertex volume image.
    
    Parameters
    ----------
    pos : np.ndarray
        Position in voxel coordinates [y, x, z]
    vertex_image : np.ndarray
        Volume where each voxel contains vertex index (1-indexed) or 0
        
    Returns
    -------
    vertex_idx : Optional[int]
        Vertex index (0-indexed) if position is within a vertex region, None otherwise
    """
    pos_int = np.floor(pos).astype(int)
    
    # Check bounds
    if not np.all((pos_int >= 0) & (pos_int < np.array(vertex_image.shape))):
        return None
    
    vertex_id = vertex_image[pos_int[0], pos_int[1], pos_int[2]]
    
    if vertex_id > 0:
        return int(vertex_id - 1)  # Convert from 1-indexed to 0-indexed
    return None


def near_vertex(pos: np.ndarray, vertex_positions: np.ndarray,
                vertex_scales: np.ndarray, lumen_radius_microns: np.ndarray,
                microns_per_voxel: np.ndarray,
                tree: Optional[cKDTree] = None,
                max_search_radius: float = 0.0) -> Optional[int]:
    """Return the index of a nearby vertex if within its physical radius; otherwise None
    
    Uses a tolerance of 0.5 voxels to account for traces ending near but not exactly at vertices.
    """
    # Tolerance: 0.5 voxels in physical units (use average voxel size)
    tolerance_microns = 0.5 * np.mean(microns_per_voxel)
    
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
            if np.linalg.norm(diff) <= radius + tolerance_microns:
                return i
        return None
    else:
        # Fallback linear scan
        for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
            radius = lumen_radius_microns[vertex_scale]
            diff = (pos - vertex_pos) * microns_per_voxel
            if np.linalg.norm(diff) <= radius + tolerance_microns:
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
    vertex_image: Optional[np.ndarray] = None,
    tree: Optional[cKDTree] = None,
    max_search_radius: float = 0.0,
) -> List[np.ndarray]:
    """Trace an edge through the energy field with adaptive step sizing."""
    trace = [start_pos.copy()]

    # Initialize scalar position and direction to avoid array overhead in loop
    cx, cy, cz = float(start_pos[0]), float(start_pos[1]), float(start_pos[2])
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])

    # Precompute for optimized gradient calc (scalarized)
    inv_mpv_2x_0 = 1.0 / (2.0 * microns_per_voxel[0])
    inv_mpv_2x_1 = 1.0 / (2.0 * microns_per_voxel[1])
    inv_mpv_2x_2 = 1.0 / (2.0 * microns_per_voxel[2])

    shape_0, shape_1, shape_2 = energy.shape

    p0 = int(math.floor(cx))
    p1 = int(math.floor(cy))
    p2 = int(math.floor(cz))
    prev_energy = energy[p0, p1, p2]

    for _ in range(max_steps):
        attempt = 0
        while attempt < 10:
            nx = cx + dx * step_size
            ny = cy + dy * step_size
            nz = cz + dz * step_size

            if discrete_steps:
                nx = round(nx)
                ny = round(ny)
                nz = round(nz)
                if nx == cx and ny == cy and nz == cz:
                    return trace

            # Inline bounds check for speed using scalars
            if (nx < 0 or nx >= shape_0 or
                ny < 0 or ny >= shape_1 or
                nz < 0 or nz >= shape_2):
                return trace

            p0 = int(math.floor(nx))
            p1 = int(math.floor(ny))
            p2 = int(math.floor(nz))
            current_energy = energy[p0, p1, p2]

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

        # Update current position
        cx, cy, cz = nx, ny, nz
        # Create array only when storing to trace
        current_pos_arr = np.array([cx, cy, cz], dtype=np.float64)
        trace.append(current_pos_arr)

        prev_energy = current_energy

        # Optimized gradient computation:
        # Avoids wrapper overhead by calling implementation directly.
        # Use scalar args to avoid allocating arrays
        p0 = int(round(cx))
        p1 = int(round(cy))
        p2 = int(round(cz))

        # Inline gradient computation to avoid function call and allocation
        # Manual clamping
        gp0 = p0
        if gp0 < 1: gp0 = 1
        elif gp0 > shape_0 - 2: gp0 = shape_0 - 2

        gp1 = p1
        if gp1 < 1: gp1 = 1
        elif gp1 > shape_1 - 2: gp1 = shape_1 - 2

        gp2 = p2
        if gp2 < 1: gp2 = 1
        elif gp2 > shape_2 - 2: gp2 = shape_2 - 2

        # Compute gradient components using scalar inv_mpv_2x
        g0 = (energy[gp0+1, gp1, gp2] - energy[gp0-1, gp1, gp2]) * inv_mpv_2x_0
        g1 = (energy[gp0, gp1+1, gp2] - energy[gp0, gp1-1, gp2]) * inv_mpv_2x_1
        g2 = (energy[gp0, gp1, gp2+1] - energy[gp0, gp1, gp2-1]) * inv_mpv_2x_2

        # Manual norm
        grad_norm = math.sqrt(g0**2 + g1**2 + g2**2)

        if grad_norm > 1e-12:
            # Project gradient onto plane perpendicular to current direction
            dot_prod = g0*dx + g1*dy + g2*dz

            perp_grad0 = g0 - dx * dot_prod
            perp_grad1 = g1 - dy * dot_prod
            perp_grad2 = g2 - dz * dot_prod

            # Steer along ridge by opposing gradient direction
            sign = 1.0 if energy_sign >= 0 else -1.0
            dx = dx - sign * perp_grad0
            dy = dy - sign * perp_grad1
            dz = dz - sign * perp_grad2

            norm = math.sqrt(dx**2 + dy**2 + dz**2)
            if norm > 1e-12:
                inv_norm = 1.0 / norm
                dx *= inv_norm
                dy *= inv_norm
                dz *= inv_norm

        # Check if we've reached a vertex (use vertex_image for O(1) lookup if available)
        if vertex_image is not None:
            terminal_vertex_idx = vertex_at_position(current_pos_arr, vertex_image)
        else:
            terminal_vertex_idx = near_vertex(
                current_pos_arr, vertex_positions, vertex_scales,
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

    # Build vertex volume image for O(1) vertex detection (matching MATLAB approach)
    logger.info("Creating vertex volume image...")
    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    vertex_image = paint_vertex_image(
        vertex_positions, vertex_scales, lumen_radius_pixels_axes, energy.shape
    )
    logger.info("Vertex volume image created")
    
    # Also build cKDTree as fallback for out-of-volume queries
    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0

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
                vertex_image=vertex_image,
                tree=tree,
                max_search_radius=max_search_radius,
            )
            if len(edge_trace) > 1:  # Valid edge found
                # Use vertex image for fast O(1) lookup at endpoint
                terminal_vertex = vertex_at_position(edge_trace[-1], vertex_image)
                
                # If endpoint check failed and trace is short, check earlier points
                if terminal_vertex is None and len(edge_trace) <= 5:
                    for point in reversed(edge_trace[-len(edge_trace):-1]):
                        terminal_vertex = vertex_at_position(point, vertex_image)
                        if terminal_vertex is not None and terminal_vertex != vertex_idx:
                            break
                        elif terminal_vertex == vertex_idx:
                            terminal_vertex = None
                            
                # Skip self-connections
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

    logger.info("Running watershed on volume (this may take several minutes)...")
    labels = watershed(-energy_sign * energy, markers)
    logger.info("Watershed complete, extracting edges between regions...")
    structure = ndi.generate_binary_structure(3, 1)

    edges = []
    connections = []
    edge_energies: List[float] = []
    seen = set()
    n_vertices = len(vertex_positions)
    log_interval = max(1, n_vertices // 20)  # Log ~20 times over the loop

    for label in range(1, n_vertices + 1):
        if label % log_interval == 0 or label == n_vertices:
            logger.info("Watershed progress: vertex %d / %d, edges so far: %d",
                        label, n_vertices, len(edges))
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
