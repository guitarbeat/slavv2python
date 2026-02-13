
"""
Geometric operations and network statistics for SLAVV.
Handles registration, spatial metrics, and statistical analysis of the vascular network.
"""
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def calculate_branching_angles(
    strands: List[List[int]],
    vertex_positions: np.ndarray,
    microns_per_voxel: List[float],
    bifurcations: np.ndarray,
) -> List[float]:
    """Compute pairwise angles at each bifurcation."""
    vertex_positions = np.asarray(vertex_positions, dtype=float)
    scale = np.asarray(microns_per_voxel, dtype=float)
    bifurcations = set(int(b) for b in np.asarray(bifurcations, dtype=int))
    directions: Dict[int, List[np.ndarray]] = {}

    for strand in strands:
        if len(strand) < 2:
            continue
        start, next_idx = strand[0], strand[1]
        end, prev_idx = strand[-1], strand[-2]

        vec_start = (vertex_positions[next_idx] - vertex_positions[start]) * scale
        vec_end = (vertex_positions[prev_idx] - vertex_positions[end]) * scale

        if start in bifurcations:
            directions.setdefault(start, []).append(vec_start)
        if end in bifurcations:
            directions.setdefault(end, []).append(vec_end)

    angles: List[float] = []
    for v in directions:
        vecs = [vec for vec in directions[v] if np.linalg.norm(vec) > 0]
        if len(vecs) < 2:
            continue
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                direction_vec_a = vecs[i]
                direction_vec_b = vecs[j]
                norm_a = np.linalg.norm(direction_vec_a)
                norm_b = np.linalg.norm(direction_vec_b)
                if norm_a == 0 or norm_b == 0:
                    continue
                cosang = np.clip(np.dot(direction_vec_a, direction_vec_b) / (norm_a * norm_b), -1.0, 1.0)
                angles.append(float(np.degrees(np.arccos(cosang))))

    return angles

def calculate_surface_area(strands: List[List[int]], vertex_positions: np.ndarray,
                           radii: np.ndarray, microns_per_voxel: List[float]) -> float:
    """Compute total vessel surface area."""
    vertex_positions = np.asarray(vertex_positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    scale = np.asarray(microns_per_voxel, dtype=float)
    total_area = 0.0

    for strand in strands:
        if len(strand) < 2:
            continue
        for i in range(len(strand) - 1):
            v1 = strand[i]
            v2 = strand[i + 1]
            pos1 = vertex_positions[v1] * scale
            pos2 = vertex_positions[v2] * scale
            length = np.linalg.norm(pos2 - pos1)
            radius = 0.5 * (radii[v1] + radii[v2])
            total_area += 2 * np.pi * radius * length

    return float(total_area)

def calculate_vessel_volume(strands: List[List[int]], vertex_positions: np.ndarray,
                            radii: np.ndarray, microns_per_voxel: List[float]) -> float:
    """Compute total vessel volume."""
    vertex_positions = np.asarray(vertex_positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    scale = np.asarray(microns_per_voxel, dtype=float)
    total_volume = 0.0

    for strand in strands:
        if len(strand) < 2:
            continue
        for i in range(len(strand) - 1):
            v1 = strand[i]
            v2 = strand[i + 1]
            pos1 = vertex_positions[v1] * scale
            pos2 = vertex_positions[v2] * scale
            length = np.linalg.norm(pos2 - pos1)
            radius = 0.5 * (radii[v1] + radii[v2])
            total_volume += np.pi * radius**2 * length

    return float(total_volume)

def get_edges_for_vertex(connections: np.ndarray, vertex_index: int) -> np.ndarray:
    """Return indices of edges incident to a given vertex."""
    connections = np.asarray(connections)
    if connections.size == 0:
        return np.empty((0,), dtype=int)
    mask = (connections[:, 0] == vertex_index) | (connections[:, 1] == vertex_index)
    return np.flatnonzero(mask)

def get_edge_metric(
    trace: np.ndarray,
    energy: Optional[np.ndarray] = None,
    method: str = "mean_energy",
) -> float:
    """Compute a simple metric for a single edge trace."""
    arr = np.asarray(trace)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    if method == "length" or energy is None:
        diffs = np.diff(arr, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    # Energy-based metrics
    coords = np.floor(arr).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, energy.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, energy.shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, energy.shape[2] - 1)
    samples = energy[coords[:, 0], coords[:, 1], coords[:, 2]]
    if method == "mean_energy":
        return float(np.mean(samples))
    if method == "min_energy":
        return float(np.min(samples))
    if method == "max_energy":
        return float(np.max(samples))
    if method == "median_energy":
        return float(np.median(samples))
    # Fallback
    return float(np.mean(samples))

def resample_vectors(trace: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline trace at approximately uniform arc-length spacing."""
    pts = np.asarray(trace, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or step <= 0:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    total = arclen[-1]
    if total == 0:
        return pts[[0]].copy()
    num = max(2, int(np.floor(total / step)) + 1)
    targets = np.linspace(0.0, total, num)
    out = np.empty((len(targets), pts.shape[1]), dtype=float)
    for d in range(pts.shape[1]):
        out[:, d] = np.interp(targets, arclen, pts[:, d])
    return out

def smooth_edge_traces(traces: List[np.ndarray], sigma: float = 1.0) -> List[np.ndarray]:
    """Smooth each polyline trace with a 1D Gaussian along its path."""
    smoothed: List[np.ndarray] = []
    for t in traces:
        arr = np.asarray(t, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3:
            smoothed.append(arr.copy())
            continue
        out = np.empty_like(arr)
        for d in range(arr.shape[1]):
            out[:, d] = gaussian_filter1d(arr[:, d], sigma=sigma, mode="nearest")
        smoothed.append(out)
    return smoothed

def transform_vector_set(
    positions: np.ndarray,
    *,
    matrix: Optional[np.ndarray] = None,
    scale: Optional[List[float]] = None,
    rotation: Optional[np.ndarray] = None,
    translate: Optional[List[float]] = None,
) -> np.ndarray:
    """Apply geometric transforms to a set of positions."""
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if matrix is not None:
        M = np.asarray(matrix, dtype=float)
        if M.shape != (4, 4):
            raise ValueError("matrix must be 4x4")
        homo = np.c_[pts, np.ones((pts.shape[0], 1))]
        out = homo @ M.T
        return out[:, :3]
    out = pts.copy()
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        if s.shape != (3,):
            raise ValueError("scale must be length-3")
        out = out * s
    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        out = out @ R.T
    if translate is not None:
        t = np.asarray(translate, dtype=float)
        if t.shape != (3,):
            raise ValueError("translate must be length-3")
        out = out + t
    return out

def subsample_vectors(trace: np.ndarray, step: int) -> np.ndarray:
    """Subsample a polyline by keeping every ``step``-th point."""
    arr = np.asarray(trace)
    if arr.ndim != 2 or step <= 1:
        return arr.copy()
    idx = np.arange(0, arr.shape[0], step, dtype=int)
    if idx[-1] != arr.shape[0] - 1:
        idx = np.r_[idx, arr.shape[0] - 1]
    return arr[idx]

def icp_register_rigid(
    source: np.ndarray,
    target: np.ndarray,
    *,
    with_scale: bool = False,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Iterative closest point (rigid) registration using Kabsch per iteration."""
    X = np.asarray(source, dtype=float)
    Y = np.asarray(target, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != 3 or Y.shape[1] != 3:
        raise ValueError("source and target must be (N,3) arrays")
    if len(X) == 0 or len(Y) == 0:
        return np.eye(4), 0.0

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    s = 1.0
    R = np.eye(3)
    t = muY - muX

    tree = cKDTree(Y)
    prev_err = np.inf

    for _ in range(max_iters):
        Xp = (X @ R.T) * s + t
        dists, idx = tree.query(Xp, k=1)
        Ymatch = Y[idx]

        Xm = X
        Ym = Ymatch
        Xc = Xm - Xm.mean(axis=0)
        Yc = Ym - Ym.mean(axis=0)
        H = Xc.T @ Yc
        U, Svals, Vt = np.linalg.svd(H)
        Rk = Vt.T @ U.T
        if np.linalg.det(Rk) < 0:
            Vt[-1, :] *= -1
            Rk = Vt.T @ U.T
        sk = 1.0
        if with_scale:
            denom = np.sum(Xc ** 2)
            if denom > 0:
                sk = float(np.sum(Svals) / denom)
        tk = Ym.mean(axis=0) - sk * (Rk @ Xm.mean(axis=0))

        R = Rk @ R
        s = sk * s
        t = Rk @ t + tk

        Xp = (X @ R.T) * s + t
        err = float(np.sqrt(np.mean(np.sum((Xp - Ymatch) ** 2, axis=1))))
        if abs(prev_err - err) < tol:
            prev_err = err
            break
        prev_err = err

    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T, prev_err

def register_vector_sets(
    source: np.ndarray,
    target: np.ndarray,
    *,
    method: str = "rigid",
    with_scale: bool = False,
    return_error: bool = False,
) -> Any:
    """Register source points to target points and return a 4x4 transform."""
    X = np.asarray(source, dtype=float)
    Y = np.asarray(target, dtype=float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("source and target must have shape (N, 3)")
    num_points = X.shape[0]
    if method not in {"rigid", "affine"}:
        raise ValueError("method must be 'rigid' or 'affine'")

    if method == "affine":
        if num_points < 4:
            raise ValueError("affine registration requires at least 4 points")
        Xh = np.c_[X, np.ones((num_points, 1))]
        A, *_ = np.linalg.lstsq(Xh, Y, rcond=None)
        T = np.eye(4)
        T[:3, :3] = A[:3, :].T
        T[:3, 3] = A[3, :]
        Xp = (Xh @ A)
        err = float(np.sqrt(np.mean(np.sum((Xp - Y) ** 2, axis=1))))
        return (T, err) if return_error else T

    if num_points < 3:
        raise ValueError("rigid registration requires at least 3 points")
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    s = 1.0
    if with_scale:
        denom = np.sum(Xc ** 2)
        if denom > 0:
            s = float(np.sum(S) / denom)
    t = muY - s * (R @ muX)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    Xp = (np.c_[X, np.ones((num_points, 1))] @ T.T)[:, :3]
    err = float(np.sqrt(np.mean(np.sum((Xp - Y) ** 2, axis=1))))
    return (T, err) if return_error else T

def register_strands(
    vertices_a: np.ndarray,
    edges_a: np.ndarray,
    vertices_b: np.ndarray,
    edges_b: np.ndarray,
    *,
    method: str = "rigid",
    with_scale: bool = False,
    match_threshold: float = 2.0,
    max_iters: int = 50,
) -> Dict[str, Any]:
    """Register and merge two networks (A onto B) and return the merged result."""
    Va = np.asarray(vertices_a, dtype=float)
    Vb = np.asarray(vertices_b, dtype=float)
    Ea = np.atleast_2d(np.asarray(edges_a, dtype=int))
    Eb = np.atleast_2d(np.asarray(edges_b, dtype=int))

    if Va.size == 0:
        return {"vertices": Vb.copy(), "edges": Eb.copy(), "transform": np.eye(4), "rms": 0.0}
    if Vb.size == 0:
        return {"vertices": Va.copy(), "edges": Ea.copy(), "transform": np.eye(4), "rms": 0.0}

    if method == "rigid":
        T, rms = icp_register_rigid(Va, Vb, with_scale=with_scale, max_iters=max_iters)
    elif method == "affine":
        tree = cKDTree(Vb)
        _, idx = tree.query(Va, k=1)
        Ta, rms = register_vector_sets(Va, Vb[idx], method="affine", return_error=True)
        T = Ta
    else:
        raise ValueError("method must be 'rigid' or 'affine'")

    Va_h = np.c_[Va, np.ones((Va.shape[0], 1))]
    Va_t = (Va_h @ T.T)[:, :3]

    tree = cKDTree(Vb)
    dists, idx = tree.query(Va_t, k=1)
    merged_vertices = Vb.tolist()
    a_to_merged = np.empty((Va.shape[0],), dtype=int)
    for i, (d, j) in enumerate(zip(dists, idx)):
        if np.isfinite(d) and d <= match_threshold:
            a_to_merged[i] = int(j)
        else:
            a_to_merged[i] = len(merged_vertices)
            merged_vertices.append(Va_t[i].tolist())

    merged_vertices_arr = np.asarray(merged_vertices, dtype=float)

    edges_a_mapped = np.vstack([
        np.minimum(a_to_merged[Ea[:, 0]], a_to_merged[Ea[:, 1]]),
        np.maximum(a_to_merged[Ea[:, 0]], a_to_merged[Ea[:, 1]]),
    ]).T

    edges_b_norm = np.vstack([
        np.minimum(Eb[:, 0], Eb[:, 1]),
        np.maximum(Eb[:, 0], Eb[:, 1]),
    ]).T

    all_edges = np.vstack([edges_b_norm, edges_a_mapped])
    mask = all_edges[:, 0] != all_edges[:, 1]
    all_edges = all_edges[mask]
    if all_edges.size:
        view = np.ascontiguousarray(all_edges).view([("a", all_edges.dtype), ("b", all_edges.dtype)])
        _, idx_unique = np.unique(view, return_index=True)
        merged_edges = all_edges[np.sort(idx_unique)]
    else:
        merged_edges = all_edges

    return {"vertices": merged_vertices_arr, "edges": merged_edges.astype(int), "transform": T, "rms": rms}

def calculate_network_statistics(
    strands: List[List[int]],
    bifurcations: np.ndarray,
    vertex_positions: np.ndarray,
    radii: np.ndarray,
    microns_per_voxel: List[float],
    image_shape: Tuple[int, ...],
    edge_energies: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate aggregate metrics for a traced vascular network."""
    stats = {}
    
    stats['num_strands'] = len(strands)
    stats['num_bifurcations'] = len(bifurcations)
    stats['num_vertices'] = len(vertex_positions)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(vertex_positions)))
    strand_lengths: List[float] = []
    edge_lengths: List[float] = []
    edge_radii: List[float] = []
    tortuosities: List[float] = []
    for strand in strands:
        if len(strand) > 1:
            length = 0.0
            for i in range(len(strand) - 1):
                pos1 = vertex_positions[strand[i]] * microns_per_voxel
                pos2 = vertex_positions[strand[i + 1]] * microns_per_voxel
                seg_length = np.linalg.norm(pos2 - pos1)
                length += seg_length
                edge_lengths.append(seg_length)
                edge_radii.append((radii[strand[i]] + radii[strand[i + 1]]) / 2.0)
                G.add_edge(strand[i], strand[i + 1], weight=seg_length)
            strand_lengths.append(length)

            start = vertex_positions[strand[0]] * microns_per_voxel
            end = vertex_positions[strand[-1]] * microns_per_voxel
            euclidean = np.linalg.norm(end - start)
            if euclidean > 0:
                tortuosities.append(length / euclidean)

    if strand_lengths:
        stats['mean_strand_length'] = np.mean(strand_lengths)
        stats['total_length'] = np.sum(strand_lengths)
        stats['strand_length_std'] = np.std(strand_lengths)
    else:
        stats['mean_strand_length'] = 0
        stats['total_length'] = 0
        stats['strand_length_std'] = 0

    if tortuosities:
        stats['mean_tortuosity'] = np.mean(tortuosities)
        stats['tortuosity_std'] = np.std(tortuosities)
    else:
        stats['mean_tortuosity'] = 0
        stats['tortuosity_std'] = 0

    stats['num_edges'] = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    if degrees:
        stats['mean_degree'] = float(np.mean(degrees))
        stats['degree_std'] = float(np.std(degrees))
    else:
        stats['mean_degree'] = 0.0
        stats['degree_std'] = 0.0

    stats['num_connected_components'] = nx.number_connected_components(G)
    stats['num_endpoints'] = sum(1 for _, d in G.degree() if d == 1)
    pairwise_lengths: List[float] = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_nodes() < 2:
            continue
        for u, dist_map in nx.all_pairs_dijkstra_path_length(sub, weight='weight'):
            for v, dist in dist_map.items():
                if u < v:
                    pairwise_lengths.append(dist)
    if pairwise_lengths:
        stats['avg_path_length'] = float(np.mean(pairwise_lengths))
        stats['network_diameter'] = float(np.max(pairwise_lengths))
    else:
        stats['avg_path_length'] = 0.0
        stats['network_diameter'] = 0.0
    stats['clustering_coefficient'] = (
        float(nx.average_clustering(G, weight='weight'))
        if G.number_of_nodes() > 1
        else 0.0
    )

    betweenness = nx.betweenness_centrality(G, weight='weight')
    if betweenness:
        vals = np.fromiter(betweenness.values(), dtype=float)
        stats['betweenness_mean'] = float(np.mean(vals))
        stats['betweenness_std'] = float(np.std(vals))
    else:
        stats['betweenness_mean'] = 0.0
        stats['betweenness_std'] = 0.0

    closeness = nx.closeness_centrality(G, distance='weight')
    if closeness:
        cvals = np.fromiter(closeness.values(), dtype=float)
        stats['closeness_mean'] = float(np.mean(cvals))
        stats['closeness_std'] = float(np.std(cvals))
    else:
        stats['closeness_mean'] = 0.0
        stats['closeness_std'] = 0.0

    eigen = (
        nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        if G.number_of_nodes() > 0
        else {}
    )
    if eigen:
        evals = np.fromiter(eigen.values(), dtype=float)
        stats['eigenvector_mean'] = float(np.mean(evals))
        stats['eigenvector_std'] = float(np.std(evals))
    else:
        stats['eigenvector_mean'] = 0.0
        stats['eigenvector_std'] = 0.0

    stats['graph_density'] = float(nx.density(G))

    if edge_lengths:
        stats['mean_edge_length'] = float(np.mean(edge_lengths))
        stats['edge_length_std'] = float(np.std(edge_lengths))
    else:
        stats['mean_edge_length'] = 0.0
        stats['edge_length_std'] = 0.0

    if edge_radii:
        stats['mean_edge_radius'] = float(np.mean(edge_radii))
        stats['edge_radius_std'] = float(np.std(edge_radii))
    else:
        stats['mean_edge_radius'] = 0.0
        stats['edge_radius_std'] = 0.0

    angles = calculate_branching_angles(strands, vertex_positions, microns_per_voxel, bifurcations)
    if angles:
        stats['mean_branch_angle'] = float(np.mean(angles))
        stats['branch_angle_std'] = float(np.std(angles))
    else:
        stats['mean_branch_angle'] = 0.0
        stats['branch_angle_std'] = 0.0
    
    if len(radii) > 0:
        stats['mean_radius'] = np.mean(radii)
        stats['radius_std'] = np.std(radii)
        stats['min_radius'] = np.min(radii)
        stats['max_radius'] = np.max(radii)

    if edge_energies is not None and len(edge_energies) > 0:
        stats['mean_edge_energy'] = float(np.mean(edge_energies))
        stats['edge_energy_std'] = float(np.std(edge_energies))
    else:
        stats['mean_edge_energy'] = 0.0
        stats['edge_energy_std'] = 0.0

    total_volume = calculate_vessel_volume(strands, vertex_positions, radii, microns_per_voxel)
    image_volume = np.prod(image_shape) * np.prod(microns_per_voxel)
    stats['total_volume'] = total_volume
    stats['volume_fraction'] = total_volume / image_volume if image_volume > 0 else 0

    total_surface_area = calculate_surface_area(strands, vertex_positions, radii, microns_per_voxel)
    stats['total_surface_area'] = total_surface_area
    stats['surface_area_density'] = total_surface_area / image_volume if image_volume > 0 else 0
    
    stats['length_density'] = stats.get('total_length', 0) / image_volume if image_volume > 0 else 0
    stats['bifurcation_density'] = len(bifurcations) / image_volume if image_volume > 0 else 0
    stats['vertex_density'] = stats['num_vertices'] / image_volume if image_volume > 0 else 0
    stats['edge_density'] = stats['num_edges'] / image_volume if image_volume > 0 else 0

    return stats

def crop_vertices(vertex_positions: np.ndarray,
                  bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Crop vertices to an axis-aligned bounding box."""
    vertex_positions = np.asarray(vertex_positions)
    bounds = np.asarray(bounds, dtype=float)

    mask = (
        (vertex_positions[:, 0] >= bounds[0, 0]) & (vertex_positions[:, 0] <= bounds[0, 1]) &
        (vertex_positions[:, 1] >= bounds[1, 0]) & (vertex_positions[:, 1] <= bounds[1, 1]) &
        (vertex_positions[:, 2] >= bounds[2, 0]) & (vertex_positions[:, 2] <= bounds[2, 1])
    )
    return vertex_positions[mask], mask

def crop_edges(edge_indices: np.ndarray, vertex_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove edges whose endpoints are not both retained in ``vertex_mask``."""
    vertex_mask = np.asarray(vertex_mask, dtype=bool)
    keep = vertex_mask[edge_indices[:, 0]] & vertex_mask[edge_indices[:, 1]]
    return edge_indices[keep], keep

def crop_vertices_by_mask(vertex_positions: np.ndarray, mask_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop vertices by a 3D binary mask."""
    vertex_positions = np.asarray(vertex_positions)
    mask_volume = np.asarray(mask_volume, dtype=bool)

    coords = np.floor(vertex_positions).astype(int)
    in_bounds = (
        (coords[:, 0] >= 0) & (coords[:, 0] < mask_volume.shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < mask_volume.shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < mask_volume.shape[2])
    )

    mask = np.zeros(len(vertex_positions), dtype=bool)
    valid_indices = np.where(in_bounds)[0]
    valid_coords = coords[in_bounds]
    mask[valid_indices] = mask_volume[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]
    return vertex_positions[mask], mask

__all__ = [
    "calculate_branching_angles",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "get_edges_for_vertex",
    "get_edge_metric",
    "resample_vectors",
    "smooth_edge_traces",
    "transform_vector_set",
    "subsample_vectors",
    "icp_register_rigid",
    "register_vector_sets",
    "register_strands",
    "calculate_network_statistics",
    "crop_vertices",
    "crop_edges",
    "crop_vertices_by_mask",
]
