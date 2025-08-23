"""
Core vectorization functions for SLAVV (Segmentation-Less, Automated, Vascular Vectorization)

This module implements the 4-step SLAVV algorithm:
1. Energy image formation using Hessian-based filtering
2. Vertex extraction via local minima detection  
3. Edge extraction through gradient following
4. Network construction and connectivity analysis

Based on the MATLAB implementation by Samuel Alexander Mihelic
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import feature
from skimage.segmentation import watershed
import warnings
from typing import Tuple, List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "SLAVVProcessor",
    "preprocess_image",
    "validate_parameters",
    "get_chunking_lattice",
    "calculate_network_statistics",
]

class SLAVVProcessor:
    """Main class for SLAVV vectorization processing"""
    
    def __init__(self):
        self.energy_data = None
        self.vertices = None
        self.edges = None
        self.network = None
        
    def process_image(self, image: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete SLAVV processing pipeline
        
        Args:
            image: 3D input image array (y, x, z)
            parameters: Dictionary of processing parameters
            
        Returns:
            Dictionary containing all processing results
        """
        logger.info("Starting SLAVV processing pipeline")

        # Validate and populate default parameters
        parameters = validate_parameters(parameters)

        # Step 0: Image preprocessing
        image = preprocess_image(image, parameters)

        # Step 1: Energy image formation
        energy_data = self.calculate_energy_field(image, parameters)
        
        # Step 2: Vertex extraction
        vertices = self.extract_vertices(energy_data, parameters)
        
        # Step 3: Edge extraction
        edge_method = parameters.get('edge_method', 'tracing')
        if edge_method == 'watershed':
            edges = self.extract_edges_watershed(energy_data, vertices, parameters)
        else:
            edges = self.extract_edges(energy_data, vertices, parameters)
        
        # Step 4: Network construction
        network = self.construct_network(edges, vertices, parameters)
        
        results = {
            'energy_data': energy_data,
            'vertices': vertices,
            'edges': edges,
            'network': network,
            'parameters': parameters
        }
        
        logger.info("SLAVV processing pipeline completed")
        return results

    def calculate_energy_field(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate multi-scale energy field using Hessian-based filtering

        This implements the energy calculation from get_energy_V202 in MATLAB,
        including PSF prefiltering and configurable Gaussian/annular ratios.
        """
        logger.info("Calculating energy field")
        
        # Extract parameters with defaults from MATLAB
        microns_per_voxel = params.get('microns_per_voxel', [1.0, 1.0, 1.0])
        radius_smallest = params.get('radius_of_smallest_vessel_in_microns', 1.5)
        radius_largest = params.get('radius_of_largest_vessel_in_microns', 50.0)
        scales_per_octave = params.get('scales_per_octave', 1.5)
        gaussian_to_ideal_ratio = params.get('gaussian_to_ideal_ratio', 1.0)
        spherical_to_annular_ratio = params.get('spherical_to_annular_ratio', 1.0)
        approximating_PSF = params.get('approximating_PSF', True)
        energy_sign = params.get('energy_sign', -1.0)  # -1 for bright vessels

        # Cache voxel spacing for anisotropy handling
        voxel_size = np.array(microns_per_voxel, dtype=float)
        
        # PSF calculation (from MATLAB implementation)
        if approximating_PSF:
            numerical_aperture = params.get('numerical_aperture', 0.95)
            excitation_wavelength = params.get('excitation_wavelength_in_microns', 1.3)
            sample_index_of_refraction = params.get('sample_index_of_refraction', 1.33)
            
            # PSF calculation based on Zipfel et al.
            if numerical_aperture <= 0.7:
                coefficient, exponent = 0.320, 1.0
            else:
                coefficient, exponent = 0.325, 0.91
                
            microns_per_sigma_PSF = [
                excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
                excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
                excitation_wavelength / (2**0.5) * 0.532 / (
                    sample_index_of_refraction - 
                    (sample_index_of_refraction**2 - numerical_aperture**2)**0.5
                )
            ]
        else:
            microns_per_sigma_PSF = [0.0, 0.0, 0.0]
            
        microns_per_sigma_PSF = np.array(microns_per_sigma_PSF, dtype=float)
        pixels_per_sigma_PSF = microns_per_sigma_PSF / voxel_size
        
        # Calculate scale range following MATLAB ordination
        largest_per_smallest_ratio = radius_largest / radius_smallest
        final_scale = int(
            np.floor(np.log2(largest_per_smallest_ratio) * scales_per_octave)
        )
        scale_ordinates = np.arange(0, final_scale + 1)
        scale_factors = 2 ** (scale_ordinates / scales_per_octave)
        lumen_radius_microns = radius_smallest * scale_factors

        # Convert radii to pixels per axis then average for scalar pixel radii
        lumen_radius_pixels_axes = lumen_radius_microns[:, None] / voxel_size[None, :]
        lumen_radius_pixels = lumen_radius_pixels_axes.mean(axis=1)

        # Use float32 image once for all scales
        image = image.astype(np.float32)

        total_voxels = int(np.prod(image.shape))
        max_voxels = int(params.get("max_voxels_per_node_energy", 1e5))
        if total_voxels > max_voxels:
            max_sigma = (
                (lumen_radius_microns[-1] / voxel_size)
                / max(gaussian_to_ideal_ratio, 1e-12)
            )
            if approximating_PSF:
                max_sigma = np.sqrt(max_sigma**2 + pixels_per_sigma_PSF**2)
            margin = int(np.ceil(np.max(max_sigma)))
            lattice = get_chunking_lattice(image.shape, max_voxels, margin)
            energy_4d = np.zeros((*image.shape, len(scale_factors)), dtype=np.float32)
            for chunk_slice, out_slice, inner_slice in lattice:
                chunk_img = image[chunk_slice]
                sub_params = params.copy()
                sub_params["max_voxels_per_node_energy"] = chunk_img.size + 1
                chunk_data = self.calculate_energy_field(chunk_img, sub_params)
                energy_4d[out_slice + (slice(None),)] = chunk_data["energy_4d"][
                    inner_slice + (slice(None),)
                ]
            if energy_sign < 0:
                energy_3d = np.min(energy_4d, axis=3)
                scale_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
            else:
                energy_3d = np.max(energy_4d, axis=3)
                scale_indices = np.argmax(energy_4d, axis=3).astype(np.int16)
            return {
                "energy": energy_3d,
                "scale_indices": scale_indices,
                "lumen_radius_microns": lumen_radius_microns,
                "lumen_radius_pixels": lumen_radius_pixels,
                "lumen_radius_pixels_axes": lumen_radius_pixels_axes,
                "pixels_per_sigma_PSF": pixels_per_sigma_PSF,
                "microns_per_sigma_PSF": microns_per_sigma_PSF,
                "energy_sign": energy_sign,
                "energy_4d": energy_4d,
                "image_shape": image.shape,
            }

        # Multi-scale energy calculation with per-scale PSF weighting
        energy_4d = np.zeros((*image.shape, len(scale_factors)), dtype=np.float32)

        for scale_idx, _ in enumerate(lumen_radius_pixels):
            # Calculate Gaussian sigmas at this scale using physical voxel spacing
            radius_microns = lumen_radius_microns[scale_idx]
            sigma_scale = (
                (radius_microns / voxel_size) / max(gaussian_to_ideal_ratio, 1e-12)
            )
            sigma_scale = np.asarray(sigma_scale, dtype=float)

            if approximating_PSF:
                sigma_object = np.sqrt(sigma_scale**2 + pixels_per_sigma_PSF**2)
            else:
                sigma_object = sigma_scale

            smoothed_object = gaussian_filter(image, sigma=tuple(sigma_object))
            if spherical_to_annular_ratio > 0:
                annular_scale = sigma_scale * spherical_to_annular_ratio
                if approximating_PSF:
                    sigma_background = np.sqrt(
                        annular_scale**2 + pixels_per_sigma_PSF**2
                    )
                else:
                    sigma_background = annular_scale
                smoothed_background = gaussian_filter(
                    image, sigma=tuple(sigma_background)
                )
                smoothed = smoothed_object - smoothed_background
            else:
                smoothed = smoothed_object

            # Calculate Hessian eigenvalues with PSF-weighted sigma
            hessian = feature.hessian_matrix(smoothed, sigma=tuple(sigma_object))
            eigenvals = feature.hessian_matrix_eigvals(hessian)
            
            # Energy function: enhance tubular structures
            # Negative eigenvalues indicate bright ridges
            lambda1, lambda2, lambda3 = eigenvals
            
            # Frangi-like vesselness measure
            vesselness = np.zeros_like(lambda1)
            
            # Only consider voxels where lambda2 and lambda3 < 0 (bright tubular structures in 3D)
            mask = (lambda2 < 0) & (lambda3 < 0)
            
            if np.any(mask):
                # Ratios for tubular structure detection
                Ra = np.abs(lambda2[mask]) / (np.abs(lambda3[mask]) + 1e-12)
                Rb = np.abs(lambda1[mask]) / (np.sqrt(np.abs(lambda2[mask] * lambda3[mask])) + 1e-12)
                S = np.sqrt(lambda1[mask]**2 + lambda2[mask]**2 + lambda3[mask]**2)
                
                # Vesselness response (Frangi-inspired)
                alpha, beta, c = 0.5, 0.5, np.max(S) + 1e-12
                vesselness[mask] = (
                    (1.0 - np.exp(-(Ra**2) / (2 * (alpha**2)))) *
                    np.exp(-(Rb**2) / (2 * (beta**2))) *
                    (1.0 - np.exp(-(S**2) / (2 * (c**2))))
                )
             
            energy_4d[:, :, :, scale_idx] = energy_sign * vesselness
        
        # Projection across scales uses sign-aware extremum
        if energy_sign < 0:
            energy_3d = np.min(energy_4d, axis=3)
            scale_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
        else:
            energy_3d = np.max(energy_4d, axis=3)
            scale_indices = np.argmax(energy_4d, axis=3).astype(np.int16)
        
        return {
            'energy': energy_3d,
            'scale_indices': scale_indices,
            'lumen_radius_microns': lumen_radius_microns,
            'lumen_radius_pixels': lumen_radius_pixels,
            'lumen_radius_pixels_axes': lumen_radius_pixels_axes,
            'pixels_per_sigma_PSF': pixels_per_sigma_PSF,
            'microns_per_sigma_PSF': microns_per_sigma_PSF,
            'energy_sign': energy_sign,
            'energy_4d': energy_4d,
            'image_shape': image.shape
        }

    @staticmethod
    def _spherical_structuring_element(radius: int) -> np.ndarray:
        """Create a 3D spherical structuring element with the given radius."""
        radius = int(radius)
        ax = np.arange(-radius, radius + 1)
        xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
        return (xx**2 + yy**2 + zz**2) <= radius**2

    def extract_vertices(self, energy_data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract vertices as local extrema in the energy field

        Minima correspond to bright vessels (default energy_sign < 0) and
        maxima to dark vessels when ``energy_sign`` is positive. This mirrors
        ``get_vertices_V200`` in MATLAB.

        Returns radii in both pixel and micron units for downstream
        processing and physical measurements.
        """
        logger.info("Extracting vertices")
        
        energy = energy_data['energy']
        scale_indices = energy_data['scale_indices']
        lumen_radius_pixels = energy_data['lumen_radius_pixels']
        energy_sign = energy_data.get('energy_sign', -1.0)

        # Parameters
        energy_upper_bound = params.get('energy_upper_bound', 0.0)
        space_strel_apothem = params.get('space_strel_apothem', 1)
        length_dilation_ratio = params.get('length_dilation_ratio', 1.0)

        # Find local extrema using a spherical structuring element
        strel = self._spherical_structuring_element(space_strel_apothem)
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

        # Volume exclusion: remove overlapping vertices using energy-ordered cKDTree
        lumen_radius_microns = energy_data['lumen_radius_microns']
        voxel_size = np.array(params.get('microns_per_voxel', [1.0, 1.0, 1.0]), dtype=float)
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

    def extract_edges(self, energy_data: Dict[str, Any], vertices: Dict[str, Any], 
                     params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract edges by tracing from vertices through energy field
        
        This implements edge extraction from get_edges_V300 in MATLAB
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
        
        edges = []
        edge_connections = []
        edges_per_vertex = np.zeros(len(vertex_positions), dtype=int)
        existing_pairs = set()

        for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
            if edges_per_vertex[vertex_idx] >= max_edges_per_vertex:
                continue
            start_radius = lumen_radius_pixels[start_scale]
            step_size = start_radius * step_size_ratio
            max_length = start_radius * length_ratio
            max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))

            # Estimate likely vessel directions from local Hessian analysis
            directions = self._estimate_vessel_directions(energy, start_pos, start_radius)
            if directions.shape[0] < max_edges_per_vertex:
                extra = self._generate_edge_directions(max_edges_per_vertex - directions.shape[0])
                directions = np.vstack([directions, extra])
            else:
                directions = directions[:max_edges_per_vertex]
            
            for direction in directions:
                if edges_per_vertex[vertex_idx] >= max_edges_per_vertex:
                    break
                edge_trace = self._trace_edge(
                    energy, start_pos, direction, step_size,
                    max_edge_energy, vertex_positions, vertex_scales,
                    lumen_radius_pixels, lumen_radius_microns,
                    max_steps, microns_per_voxel, energy_sign
                )
                if len(edge_trace) > 1:  # Valid edge found
                    terminal_vertex = self._find_terminal_vertex(
                        edge_trace[-1], vertex_positions, vertex_scales,
                        lumen_radius_microns, microns_per_voxel
                    )
                    if terminal_vertex == vertex_idx:
                        continue
                    if terminal_vertex is not None:
                        if edges_per_vertex[terminal_vertex] >= max_edges_per_vertex:
                            continue
                        pair = tuple(sorted((vertex_idx, terminal_vertex)))
                        if pair in existing_pairs:
                            continue
                    edges.append(np.asarray(edge_trace, dtype=np.float32))
                    edge_connections.append([
                        vertex_idx,
                        terminal_vertex if terminal_vertex is not None else -1,
                    ])
                    edges_per_vertex[vertex_idx] += 1
                    if terminal_vertex is not None:
                        edges_per_vertex[terminal_vertex] += 1
                        existing_pairs.add(pair)
        
        logger.info(f"Extracted {len(edges)} edges")

        edge_connections = np.asarray(edge_connections, dtype=np.int32)

        return {
            "traces": edges,
            "connections": edge_connections,
            "vertex_positions": vertex_positions.astype(np.float32)
        }

    def extract_edges_watershed(self, energy_data: Dict[str, Any],
                                vertices: Dict[str, Any],
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract edges using watershed segmentation seeded at vertices.

        This provides an alternative to gradient-based tracing and approximates
        the MATLAB ``get_edges_by_watershed.m`` behavior by growing regions from
        vertices and capturing boundaries where regions touch.
        """
        logger.info("Extracting edges via watershed")

        energy = energy_data["energy"]
        vertex_positions = vertices["positions"]

        markers = np.zeros_like(energy, dtype=np.int32)
        idxs = np.floor(vertex_positions).astype(int)
        idxs = np.clip(idxs, 0, np.array(energy.shape) - 1)
        markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

        labels = watershed(-energy, markers)
        structure = ndi.generate_binary_structure(3, 1)

        edges = []
        connections = []
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
                edges.append(coords.astype(np.float32))
                connections.append([label - 1, neighbor - 1])
                seen.add(pair)

        logger.info("Extracted %d watershed edges", len(edges))

        return {
            "traces": edges,
            "connections": np.asarray(connections, dtype=np.int32),
            "vertex_positions": vertex_positions.astype(np.float32),
        }

    def construct_network(self, edges: Dict[str, Any], vertices: Dict[str, Any],
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Construct network from traced edges and detected vertices.

        Deduplicates edges, preserves their traces, and tracks dangling edges
        lacking terminal vertices. Optionally removes short hair-like edges and
        cyclic connections, reporting orphan vertices. This approximates network
        construction in ``get_network_V190.m``.
        """
        logger.info("Constructing network")

        edge_traces = edges["traces"]
        edge_connections = edges["connections"]
        vertex_positions = vertices["positions"]

        # Parameter for hair removal and physical scaling
        microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
        min_hair_length = params.get("min_hair_length_in_microns", 0.0)
        remove_cycles = params.get("remove_cycles", True)

        # Build adjacency matrix and edge list for graph
        n_vertices = len(vertex_positions)
        adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)

        # Store actual edges in a dictionary keyed by sorted vertex index pairs
        graph_edges: Dict[Tuple[int, int], np.ndarray] = {}
        dangling_edges: List[Dict[str, Any]] = []

        for trace, (start_vertex, end_vertex) in zip(edge_traces, edge_connections):
            if start_vertex < 0 or end_vertex < 0:
                dangling_edges.append({
                    "start": int(start_vertex) if start_vertex >= 0 else None,
                    "end": int(end_vertex) if end_vertex >= 0 else None,
                    "trace": trace,
                })
                continue

            adjacency[start_vertex, end_vertex] = True
            adjacency[end_vertex, start_vertex] = True

            key = tuple(sorted((start_vertex, end_vertex)))
            if key not in graph_edges:
                graph_edges[key] = trace

        # Optionally remove short hairs connected to degree-1 vertices
        if min_hair_length > 0:
            to_remove = []
            for (v0, v1), trace in graph_edges.items():
                length = np.sum(
                    np.linalg.norm(np.diff(trace, axis=0) * microns_per_voxel, axis=1)
                )
                if length < min_hair_length and (
                    np.sum(adjacency[v0]) == 1 or np.sum(adjacency[v1]) == 1
                ):
                    adjacency[v0, v1] = adjacency[v1, v0] = False
                    to_remove.append((v0, v1))
            for key in to_remove:
                del graph_edges[key]

        # Optionally remove cycles by building a spanning forest
        cycles: List[Tuple[int, int]] = []
        if remove_cycles and graph_edges:
            parent = np.arange(n_vertices)

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for (v0, v1) in list(graph_edges.keys()):
                r0, r1 = find(v0), find(v1)
                if r0 == r1:
                    cycles.append((v0, v1))
                    adjacency[v0, v1] = adjacency[v1, v0] = False
                    del graph_edges[(v0, v1)]
                else:
                    parent[r1] = r0

        # Find connected components (strands)
        strands = []
        visited = np.zeros(n_vertices, dtype=bool)

        for vertex_idx in range(n_vertices):
            if not visited[vertex_idx]:
                strand = self._trace_strand(vertex_idx, adjacency, visited)
                if len(strand) > 1:
                    strands.append(strand)

        # Sort strands and flag ordering mismatches
        strands, mismatched = self._sort_and_validate_strands(strands, adjacency)

        # Find bifurcation vertices (degree > 2) and orphans (degree == 0)
        vertex_degrees = np.sum(adjacency, axis=1).astype(np.int32)
        bifurcations = np.where(vertex_degrees > 2)[0].astype(np.int32)
        orphans = np.where(vertex_degrees == 0)[0].astype(np.int32)

        logger.info(
            "Constructed network with %d strands, %d bifurcations, %d orphans, removed %d cycles, and %d mismatched strands",
            len(strands),
            len(bifurcations),
            len(orphans),
            len(cycles),
            len(mismatched),
        )

        return {
            "strands": strands,
            "bifurcations": bifurcations,
            "orphans": orphans,
            "cycles": cycles,
            "mismatched_strands": mismatched,
            "adjacency": adjacency.astype(bool),
            "vertex_degrees": vertex_degrees,
            "graph_edges": graph_edges,
            "dangling_edges": dangling_edges,
        }

    def _estimate_vessel_directions(self, energy: np.ndarray, pos: np.ndarray, radius: float) -> np.ndarray:
        """Estimate vessel directions at a vertex via local Hessian analysis.

        Parameters
        ----------
        energy : np.ndarray
            3D energy field.
        pos : np.ndarray
            Vertex position in pixel coordinates.
        radius : float
            Estimated vessel radius in pixels.

        Returns
        -------
        np.ndarray
            Opposing unit direction vectors of shape ``(2, 3)``. Falls back to
            uniformly distributed directions if the neighborhood is ill-conditioned
            or undersized.
        """
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
            return self._generate_edge_directions(2)

        # Compute Hessian in the local patch and extract center values
        hessian_elems = [h * (radius ** 2) for h in feature.hessian_matrix(patch, sigma=sigma)]
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
            return self._generate_edge_directions(2)
        if not np.all(np.isfinite(w)):
            return self._generate_edge_directions(2)
        direction = v[:, np.argmin(np.abs(w))]
        norm = np.linalg.norm(direction)
        if norm == 0 or not np.isfinite(norm):
            return self._generate_edge_directions(2)
        direction = direction / norm
        return np.stack((direction, -direction))

    def _generate_edge_directions(self, n_directions: int) -> np.ndarray:
        """Generate uniformly distributed directions for edge tracing using spherical Fibonacci spiral"""
        if n_directions == 1:
            return np.array([[0, 0, 1]])  # Single direction
        
        # Generate directions on unit sphere using spherical Fibonacci spiral
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        
        for i in range(n_directions):
            y = 1 - (i / float(n_directions - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)

    def _trace_edge(self, energy: np.ndarray, start_pos: np.ndarray, direction: np.ndarray,
                    step_size: float, max_energy: float, vertex_positions: np.ndarray,
                    vertex_scales: np.ndarray, lumen_radius_pixels: np.ndarray,
                    lumen_radius_microns: np.ndarray, max_steps: int,
                    microns_per_voxel: np.ndarray, energy_sign: float) -> List[np.ndarray]:
        """Trace an edge through the energy field with adaptive step sizing"""
        trace = [start_pos.copy()]
        current_pos = start_pos.copy()
        current_dir = direction.copy()
        prev_energy = energy[tuple(np.floor(current_pos).astype(int))]

        for _ in range(max_steps):
            attempt = 0
            while attempt < 10:
                next_pos = current_pos + current_dir * step_size
                if not self._in_bounds(next_pos, energy.shape):
                    return trace
                pos_int = np.floor(next_pos).astype(int)
                current_energy = energy[pos_int[0], pos_int[1], pos_int[2]]
                if (energy_sign < 0 and current_energy > max_energy) or (
                    energy_sign > 0 and current_energy < max_energy
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

            gradient = self._compute_gradient(energy, current_pos, microns_per_voxel)
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-12:
                # Project gradient onto plane perpendicular to current direction
                perp_grad = gradient - current_dir * np.dot(gradient, current_dir)
                # Steer along ridge by opposing gradient direction
                current_dir = current_dir - np.sign(energy_sign) * perp_grad
                norm = np.linalg.norm(current_dir)
                if norm > 1e-12:
                    current_dir = (current_dir / norm).astype(float)

            terminal_vertex_idx = self._near_vertex(
                current_pos, vertex_positions, vertex_scales,
                lumen_radius_microns, microns_per_voxel
            )
            if terminal_vertex_idx is not None:
                trace.append(vertex_positions[terminal_vertex_idx].copy())
                break

        return trace

    def _trace_strand(self, start_vertex_idx: int, adjacency: np.ndarray, visited: np.ndarray) -> List[int]:
        """Recursively trace a strand (connected component) in the network"""
        strand = []
        stack = [start_vertex_idx]
        visited[start_vertex_idx] = True
        
        while stack:
            current_vertex = stack.pop()
            strand.append(current_vertex)
            
            neighbors = np.where(adjacency[current_vertex])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return strand

    def _sort_and_validate_strands(
        self, strands: List[List[int]], adjacency: np.ndarray
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Sort vertices within each strand and flag ordering mismatches.

        Parameters
        ----------
        strands : List[List[int]]
            Connected components of the network.
        adjacency : np.ndarray
            Symmetric adjacency matrix for the network.

        Returns
        -------
        Tuple[List[List[int]], List[List[int]]]
            Sorted strands and any strands containing ordering mismatches.
        """
        sorted_strands: List[List[int]] = []
        mismatched: List[List[int]] = []
        for strand in strands:
            if len(strand) < 2:
                sorted_strands.append(strand)
                continue
            sub_adj = adjacency[strand][:, strand]
            degrees = np.sum(sub_adj, axis=1)
            endpoints = np.where(degrees == 1)[0]
            start_idx = int(endpoints[0]) if len(endpoints) else 0
            ordered = [strand[start_idx]]
            visited = {strand[start_idx]}
            current = strand[start_idx]
            while len(ordered) < len(strand):
                neighbors = np.where(adjacency[current])[0]
                next_candidates = [n for n in neighbors if n in strand and n not in visited]
                if not next_candidates:
                    mismatched.append(strand)
                    break
                nxt = next_candidates[0]
                ordered.append(nxt)
                visited.add(nxt)
                current = nxt
            if len(ordered) == len(strand):
                sorted_strands.append(ordered)
        return sorted_strands, mismatched

    def _in_bounds(self, pos: np.ndarray, shape: Tuple[int, ...]) -> bool:
        """Check if the floored position lies within array bounds."""
        pos_int = np.floor(pos).astype(int)
        return np.all((pos_int >= 0) & (pos_int < np.array(shape)))

    def _near_vertex(self, pos: np.ndarray, vertex_positions: np.ndarray,
                    vertex_scales: np.ndarray, lumen_radius_microns: np.ndarray,
                    microns_per_voxel: np.ndarray) -> Optional[int]:
        """Return the index of a nearby vertex if within its physical radius; otherwise None"""
        for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
            radius = lumen_radius_microns[vertex_scale]
            diff = (pos - vertex_pos) * microns_per_voxel
            if np.linalg.norm(diff) < radius:
                return i
        return None

    def _find_terminal_vertex(self, pos: np.ndarray, vertex_positions: np.ndarray,
                              vertex_scales: np.ndarray, lumen_radius_microns: np.ndarray,
                              microns_per_voxel: np.ndarray) -> Optional[int]:
        """Find the index of a terminal vertex near a given position, if any."""
        return self._near_vertex(pos, vertex_positions, vertex_scales,
                                 lumen_radius_microns, microns_per_voxel)

    def _compute_gradient(self, energy: np.ndarray, pos: np.ndarray,
                          microns_per_voxel: np.ndarray) -> np.ndarray:
        """Compute gradient at position using central differences with voxel size scaling"""
        pos_int = np.round(pos).astype(int)
        gradient = np.zeros(3, dtype=float)

        for i in range(3):
            if 0 < pos_int[i] < energy.shape[i] - 1:
                pos_plus = pos_int.copy()
                pos_minus = pos_int.copy()
                pos_plus[i] += 1
                pos_minus[i] -= 1
                diff = energy[tuple(pos_plus)] - energy[tuple(pos_minus)]
                gradient[i] = diff / (2.0 * microns_per_voxel[i])

        return gradient


def preprocess_image(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Normalize intensities and optionally correct axial banding.

    Parameters
    ----------
    image:
        Raw input image array (y, x, z).
    params:
        Processing parameters. Uses ``bandpass_window`` to control the
        axial Gaussian smoothing window for band removal.

    Returns
    -------
    np.ndarray
        Preprocessed image with intensities scaled to ``[0, 1]``.
    """

    img = image.astype(np.float32)

    # Scale intensities to [0, 1]
    img -= img.min()
    max_val = img.max()
    if max_val > 0:
        img /= max_val

    # Simple axial band correction inspired by `fix_intensity_bands.m`
    band_window = params.get("bandpass_window", 0.0)
    if band_window > 0:
        background = ndi.gaussian_filter(img, sigma=(0, 0, band_window))
        img = np.clip(img - background, 0, 1)

    return img

def validate_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set default parameters based on MATLAB implementation
    """
    validated = {}
    
    # Voxel size parameters
    validated['microns_per_voxel'] = params.get('microns_per_voxel', [1.0, 1.0, 1.0])
    if len(validated['microns_per_voxel']) != 3:
        raise ValueError("microns_per_voxel must be a 3-element array")
    
    # Vessel size parameters
    validated['radius_of_smallest_vessel_in_microns'] = params.get(
        'radius_of_smallest_vessel_in_microns', 1.5)
    validated['radius_of_largest_vessel_in_microns'] = params.get(
        'radius_of_largest_vessel_in_microns', 50.0)
    
    if validated['radius_of_smallest_vessel_in_microns'] <= 0:
        raise ValueError("radius_of_smallest_vessel_in_microns must be positive")
    if validated['radius_of_largest_vessel_in_microns'] <= validated['radius_of_smallest_vessel_in_microns']:
        raise ValueError("radius_of_largest_vessel_in_microns must be larger than smallest")
    
    # PSF parameters
    validated['approximating_PSF'] = params.get('approximating_PSF', True)
    if validated['approximating_PSF']:
        validated['numerical_aperture'] = params.get('numerical_aperture', 0.95)
        validated['excitation_wavelength_in_microns'] = params.get(
            'excitation_wavelength_in_microns', 1.3)
        validated['sample_index_of_refraction'] = params.get(
            'sample_index_of_refraction', 1.33)
        
        # Validate excitation wavelength (common range for two-photon microscopy)
        if not (0.7 <= validated['excitation_wavelength_in_microns'] <= 3.0):
            warnings.warn(
                "Excitation wavelength outside typical range (0.7-3.0 μm). "
                "This may indicate an error or unusual experimental setup."
            )
    
    # Scale parameters
    validated['scales_per_octave'] = params.get('scales_per_octave', 1.5)
    validated['gaussian_to_ideal_ratio'] = params.get('gaussian_to_ideal_ratio', 1.0)
    validated['spherical_to_annular_ratio'] = params.get('spherical_to_annular_ratio', 1.0)
    validated['energy_sign'] = params.get('energy_sign', -1.0)
    if validated['energy_sign'] not in (-1, 1):
        raise ValueError('energy_sign must be -1 or 1')
    
    # Processing parameters
    validated['max_voxels_per_node_energy'] = params.get('max_voxels_per_node_energy', 1e5)
    validated['energy_upper_bound'] = params.get('energy_upper_bound', 0.0)
    validated['space_strel_apothem'] = params.get('space_strel_apothem', 1)
    validated['length_dilation_ratio'] = params.get('length_dilation_ratio', 1.0)
    validated['number_of_edges_per_vertex'] = params.get('number_of_edges_per_vertex', 4)
    validated['step_size_per_origin_radius'] = params.get('step_size_per_origin_radius', 1.0)
    validated['max_edge_energy'] = params.get('max_edge_energy', 0.0)
    validated['min_hair_length_in_microns'] = params.get('min_hair_length_in_microns', 0.0)
    validated['bandpass_window'] = params.get('bandpass_window', 0.0)
    validated['edge_method'] = params.get('edge_method', 'tracing')
    if validated['edge_method'] not in ('tracing', 'watershed'):
        raise ValueError("edge_method must be 'tracing' or 'watershed'")

    return validated


def get_chunking_lattice(
    shape: Tuple[int, int, int], max_voxels: int, margin: int
) -> List[Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice], Tuple[slice, slice, slice]]]:
    """Generate overlapping z-axis chunks to limit voxel processing.

    Parameters
    ----------
    shape:
        Image shape as ``(y, x, z)``.
    max_voxels:
        Maximum voxels allowed per chunk including margins.
    margin:
        Overlap in voxels applied on both sides of each chunk along ``z``.

    Returns
    -------
    list of tuples
        ``(chunk_slice, output_slice, inner_slice)`` where ``chunk_slice``
        indexes the padded region in the source image, ``output_slice``
        corresponds to the destination region, and ``inner_slice`` selects the
        interior region of the chunk to copy into ``output_slice``.
    """

    y, x, z = shape
    plane_voxels = y * x
    max_depth = max_voxels // plane_voxels
    if max_depth <= 0 or max_depth >= z:
        return [
            (
                (slice(0, y), slice(0, x), slice(0, z)),
                (slice(0, y), slice(0, x), slice(0, z)),
                (slice(0, y), slice(0, x), slice(0, z)),
            )
        ]

    margin = min(margin, max_depth // 2)
    core_depth = max_depth - 2 * margin
    if core_depth <= 0:
        core_depth = 1

    slices = []
    start = 0
    while start < z:
        end = min(start + core_depth, z)
        pad_before = margin if start > 0 else 0
        pad_after = margin if end < z else 0
        chunk_slice = (
            slice(0, y),
            slice(0, x),
            slice(start - pad_before, end + pad_after),
        )
        output_slice = (slice(0, y), slice(0, x), slice(start, end))
        inner_slice = (
            slice(0, y),
            slice(0, x),
            slice(pad_before, pad_before + (end - start)),
        )
        slices.append((chunk_slice, output_slice, inner_slice))
        start = end

    return slices

def calculate_network_statistics(strands: List[List[int]], bifurcations: np.ndarray,
                               vertex_positions: np.ndarray, radii: np.ndarray,
                               microns_per_voxel: List[float], image_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Calculate comprehensive network statistics
    """
    stats = {}
    
    # Basic counts
    stats['num_strands'] = len(strands)
    stats['num_bifurcations'] = len(bifurcations)
    stats['num_vertices'] = len(vertex_positions)
    
    # Strand lengths
    strand_lengths = []
    for strand in strands:
        if len(strand) > 1:
            length = 0
            for i in range(len(strand) - 1):
                pos1 = vertex_positions[strand[i]] * microns_per_voxel
                pos2 = vertex_positions[strand[i+1]] * microns_per_voxel
                length += np.linalg.norm(pos2 - pos1)
            strand_lengths.append(length)
    
    if strand_lengths:
        stats['mean_strand_length'] = np.mean(strand_lengths)
        stats['total_length'] = np.sum(strand_lengths)
        stats['strand_length_std'] = np.std(strand_lengths)
    else:
        stats['mean_strand_length'] = 0
        stats['total_length'] = 0
        stats['strand_length_std'] = 0
    
    # Vessel radii statistics
    if len(radii) > 0:
        stats['mean_radius'] = np.mean(radii)
        stats['radius_std'] = np.std(radii)
        stats['min_radius'] = np.min(radii)
        stats['max_radius'] = np.max(radii)
    
    # Volume fraction
    image_volume = np.prod(image_shape) * np.prod(microns_per_voxel)
    vessel_volume = np.sum(np.pi * radii**2) * stats.get('total_length', 0)
    stats['volume_fraction'] = vessel_volume / image_volume if image_volume > 0 else 0
    
    # Density measures
    stats['length_density'] = stats.get('total_length', 0) / image_volume if image_volume > 0 else 0
    stats['bifurcation_density'] = len(bifurcations) / image_volume if image_volume > 0 else 0
    
    return stats

