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
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters, feature, morphology
from skimage.measure import label, regionprops
import h5py
import warnings
from typing import Tuple, List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Step 1: Energy image formation
        energy_data = self.calculate_energy_field(image, parameters)
        
        # Step 2: Vertex extraction
        vertices = self.extract_vertices(energy_data, parameters)
        
        # Step 3: Edge extraction
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
        
        This implements the energy calculation from get_energy_V202 in MATLAB
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
            
        pixels_per_sigma_PSF = np.array(microns_per_sigma_PSF) / np.array(microns_per_voxel)
        
        # Calculate scale range
        largest_per_smallest_ratio = (radius_largest / radius_smallest) ** 3
        final_scale = round(np.log(largest_per_smallest_ratio) / np.log(2) * scales_per_octave)
        
        scale_ordinates = np.arange(-1, final_scale + 2)
        scale_factors = 2 ** (scale_ordinates / scales_per_octave / 3)
        lumen_radius_microns = radius_smallest * scale_factors
        lumen_radius_pixels = lumen_radius_microns / np.array(microns_per_voxel)
        
        # Multi-scale energy calculation
        energy_4d = np.zeros((*image.shape, len(scale_factors)))
        scale_indices = np.zeros(image.shape, dtype=np.int16)
        
        for scale_idx, radius_pixels in enumerate(lumen_radius_pixels):
            # Calculate Hessian at this scale
            sigma = radius_pixels / 2  # Approximate relationship
            
            # Apply Gaussian smoothing
            smoothed = gaussian_filter(image.astype(np.float32), sigma)
            
            # Calculate Hessian eigenvalues
            hessian = feature.hessian_matrix(smoothed, sigma=sigma[0])
            eigenvals = feature.hessian_matrix_eigvals(hessian)
            
            # Energy function: enhance tubular structures
            # Negative eigenvalues indicate bright ridges
            lambda1, lambda2, lambda3 = eigenvals
            
            # Frangi-like vesselness measure
            vesselness = np.zeros_like(lambda1)
            
            # Only consider voxels where lambda3 < 0 (bright structures)
            mask = lambda3 < 0
            
            if np.any(mask):
                # Ratios for tubular structure detection
                Ra = np.abs(lambda2[mask]) / np.abs(lambda3[mask])
                Rb = np.abs(lambda1[mask]) / np.sqrt(np.abs(lambda2[mask] * lambda3[mask]))
                S = np.sqrt(lambda1[mask]**2 + lambda2[mask]**2 + lambda3[mask]**2)
                
                # Vesselness response
                vesselness[mask] = (
                    (1 - np.exp(-Ra**2 / (2 * 0.5**2))) *
                    np.exp(-Rb**2 / (2 * 0.5**2)) *
                    (1 - np.exp(-S**2 / (2 * np.max(S)**2)))
                )
            
            energy_4d[:, :, :, scale_idx] = -vesselness  # Negative for minima detection
        
        # Min projection across scales
        energy_3d = np.min(energy_4d, axis=3)
        scale_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
        
        return {
            'energy': energy_3d,
            'scale_indices': scale_indices,
            'lumen_radius_microns': lumen_radius_microns,
            'lumen_radius_pixels': lumen_radius_pixels,
            'pixels_per_sigma_PSF': pixels_per_sigma_PSF,
            'energy_4d': energy_4d
        }

    def extract_vertices(self, energy_data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract vertices as local minima in the energy field
        
        This implements vertex extraction from get_vertices_V200 in MATLAB
        """
        logger.info("Extracting vertices")
        
        energy = energy_data['energy']
        scale_indices = energy_data['scale_indices']
        lumen_radius_pixels = energy_data['lumen_radius_pixels']
        
        # Parameters
        energy_upper_bound = params.get('energy_upper_bound', 0.0)
        space_strel_apothem = params.get('space_strel_apothem', 1)
        length_dilation_ratio = params.get('length_dilation_ratio', 1.0)
        
        # Find local minima
        min_filter = ndi.minimum_filter(energy, size=2*space_strel_apothem+1)
        local_minima = (energy == min_filter) & (energy < energy_upper_bound)
        
        # Get coordinates of minima
        coords = np.where(local_minima)
        vertex_positions = np.column_stack(coords)
        vertex_scales = scale_indices[coords]
        vertex_energies = energy[coords]
        
        # Sort by energy (best first)
        sort_indices = np.argsort(vertex_energies)
        vertex_positions = vertex_positions[sort_indices]
        vertex_scales = vertex_scales[sort_indices]
        vertex_energies = vertex_energies[sort_indices]
        
        # Volume exclusion: remove overlapping vertices
        # This part needs to be optimized for performance for large datasets
        kept_indices = []
        for i in range(len(vertex_positions)):
            current_pos = vertex_positions[i]
            current_scale = vertex_scales[i]
            current_radius = lumen_radius_pixels[current_scale] * length_dilation_ratio
            
            is_overlapping = False
            for j in kept_indices:
                prev_pos = vertex_positions[j]
                prev_scale = vertex_scales[j]
                prev_radius = lumen_radius_pixels[prev_scale] * length_dilation_ratio
                
                distance = np.linalg.norm(current_pos - prev_pos)
                if distance < (current_radius + prev_radius):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                kept_indices.append(i)
        
        # Keep only non-overlapping vertices
        vertex_positions = vertex_positions[kept_indices]
        vertex_scales = vertex_scales[kept_indices]
        vertex_energies = vertex_energies[kept_indices]
        
        logger.info(f"Extracted {len(vertex_positions)} vertices")
        
        return {
            'positions': vertex_positions,
            'scales': vertex_scales,
            'energies': vertex_energies,
            'radii': lumen_radius_pixels[vertex_scales]
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
        
        # Parameters
        max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
        step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
        max_edge_energy = params.get("max_edge_energy", 0.0)
        
        edges = []
        edge_connections = []
        
        for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
            start_radius = lumen_radius_pixels[start_scale]
            step_size = start_radius * step_size_ratio
            
            # Try multiple directions from this vertex
            directions = self._generate_edge_directions(max_edges_per_vertex)
            
            for direction in directions:
                edge_trace = self._trace_edge(
                    energy, start_pos, direction, step_size, 
                    max_edge_energy, vertex_positions, lumen_radius_pixels
                )
                if len(edge_trace) > 1:  # Valid edge found
                    edges.append(edge_trace)
                    
                    # Find terminal vertex if any
                    terminal_vertex = self._find_terminal_vertex(
                        edge_trace[-1], vertex_positions, vertex_scales, lumen_radius_pixels
                    )                    
                    edge_connections.append((vertex_idx, terminal_vertex))
        
        logger.info(f"Extracted {len(edges)} edges")
        
        return {
            "traces": edges,
            "connections": edge_connections,
            "vertex_positions": vertex_positions
        }

    def construct_network(self, edges: Dict[str, Any], vertices: Dict[str, Any], 
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct network by connecting edges into strands
        
        This implements network construction from get_network_V190 in MATLAB
        """
        logger.info("Constructing network")
        
        edge_traces = edges["traces"]
        edge_connections = edges["connections"]
        vertex_positions = vertices["positions"]
        
        # Build adjacency matrix and edge list for graph
        n_vertices = len(vertex_positions)
        adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)
        
        # Store actual edges (traces) in a dictionary for easy lookup
        # Key: tuple (start_vertex_idx, end_vertex_idx), Value: edge_trace
        graph_edges = {}
        
        for i, (start_vertex, end_vertex) in enumerate(edge_connections):
            if start_vertex is not None and end_vertex is not None:
                adjacency[start_vertex, end_vertex] = True
                adjacency[end_vertex, start_vertex] = True  # Assuming undirected graph for now
                
                # Store the edge trace. Ensure consistent key order.
                key = tuple(sorted((start_vertex, end_vertex)))
                graph_edges[key] = edge_traces[i]

        # Find connected components (strands)
        strands = []
        visited = np.zeros(n_vertices, dtype=bool)
        
        for vertex_idx in range(n_vertices):
            if not visited[vertex_idx]:
                strand = self._trace_strand(vertex_idx, adjacency, visited)
                if len(strand) > 1:
                    strands.append(strand)
        
        # Find bifurcation vertices (degree > 2)
        vertex_degrees = np.sum(adjacency, axis=1)
        bifurcations = np.where(vertex_degrees > 2)[0]
        
        logger.info(f"Constructed network with {len(strands)} strands and {len(bifurcations)} bifurcations")
        
        return {
            "strands": strands,
            "bifurcations": bifurcations,
            "adjacency": adjacency,
            "vertex_degrees": vertex_degrees,
            "graph_edges": graph_edges # Add the actual edge traces to the network output
        }

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
                   lumen_radius_pixels: np.ndarray) -> List[np.ndarray]:
        """Trace an edge through the energy field"""
        trace = [start_pos.copy()]
        current_pos = start_pos.copy()
        current_dir = direction.copy()
        
        max_steps = 500  # Prevent infinite loops, increased from 100
        
        for step in range(max_steps):
            # Take step in current direction
            next_pos = current_pos + current_dir * step_size
            
            # Check bounds
            if not self._in_bounds(next_pos, energy.shape):
                break
            
            # Check energy threshold
            pos_int = np.round(next_pos).astype(int)
            
            # Ensure pos_int is within bounds before accessing energy array
            if not (0 <= pos_int[0] < energy.shape[0] and
                    0 <= pos_int[1] < energy.shape[1] and
                    0 <= pos_int[2] < energy.shape[2]):
                break

            current_energy = energy[pos_int[0], pos_int[1], pos_int[2]]
            if current_energy > max_energy:
                break
            
            trace.append(next_pos.copy())
            current_pos = next_pos.copy()
            
            # Update direction based on energy gradient (simplified for now)
            # In MATLAB, this involves more complex gradient descent
            # For now, we'll just continue in the same direction or a slightly perturbed one
            # A more accurate implementation would calculate the gradient of the energy field
            # and adjust the direction to move towards lower energy.
            
            # Check if near another vertex (terminal vertex)
            terminal_vertex_idx = self._near_vertex(current_pos, vertex_positions, lumen_radius_pixels)
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

    def _in_bounds(self, pos: np.ndarray, shape: Tuple[int, ...]) -> bool:
        """Check if a position is within image bounds"""
        for i in range(len(pos)):
            if not (0 <= pos[i] < shape[i]):
                return False
        return True
                break
            
            # Update position and direction based on local gradient
            gradient = self._compute_gradient(energy, current_pos) # Use current_pos for gradient calculation
            
            # Normalize gradient and update direction
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-10:
                current_dir = -gradient / grad_norm
            else:
                # If gradient is zero, try to continue in the same direction or stop
                # For now, just break to avoid infinite loop
                break
            
            current_pos = next_pos
            
            trace.append(current_pos.copy())
            
            # Check if we've reached another vertex
            if self._near_vertex(current_pos, vertex_positions, vertex_scales, lumen_radius_pixels):
                break       
        return trace

    def _find_terminal_vertex(self, end_pos: np.ndarray, vertex_positions: np.ndarray,
                             vertex_scales: np.ndarray, lumen_radius_pixels: np.ndarray) -> Optional[int]:
        """Find if edge terminates at a vertex"""
        # Iterate through existing vertices to check for proximity
        for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
            # Calculate the radius of the vertex based on its scale
            radius = lumen_radius_pixels[vertex_scale]
            
            # Calculate Euclidean distance between the end of the trace and the vertex
            distance = np.linalg.norm(end_pos - vertex_pos)
            
            # If the distance is less than the vertex's radius, consider it a terminal vertex
            if distance < radius:
                return i
        return None

    def _trace_strand(self, start_vertex: int, adjacency: np.ndarray, 
                     visited: np.ndarray) -> List[int]:
        """Trace a connected strand of vertices"""
        strand = []
        queue = [start_vertex]
        
        while queue:
            vertex = queue.pop(0)
            if visited[vertex]:
                continue
                
            visited[vertex] = True
            strand.append(vertex)
            
            # Add connected vertices to queue
            neighbors = np.where(adjacency[vertex])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    queue.append(neighbor)
        
        return strand

    def _in_bounds(self, pos: np.ndarray, shape: Tuple[int, ...]) -> bool:
        """Check if position is within image bounds"""
        return all(0 <= p < s for p, s in zip(pos, shape))

    def _near_vertex(self, pos: np.ndarray, vertex_positions: np.ndarray, 
                    vertex_scales: np.ndarray, lumen_radius_pixels: np.ndarray) -> bool:
        """Check if position is near any vertex"""
        for vertex_pos, vertex_scale in zip(vertex_positions, vertex_scales):
            radius = lumen_radius_pixels[vertex_scale]
            if np.linalg.norm(pos - vertex_pos) < radius:
                return True
        return False

    def _compute_gradient(self, energy: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Compute gradient at given position using finite differences"""
        pos_int = np.round(pos).astype(int)
        gradient = np.zeros(3)
        
        for i in range(3):
            if pos_int[i] > 0 and pos_int[i] < energy.shape[i] - 1:
                pos_plus = pos_int.copy()
                pos_minus = pos_int.copy()
                pos_plus[i] += 1
                pos_minus[i] -= 1
                
                gradient[i] = (energy[tuple(pos_plus)] - energy[tuple(pos_minus)]) / 2
        
        return gradient

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
    
    # Processing parameters
    validated['max_voxels_per_node_energy'] = params.get('max_voxels_per_node_energy', 1e5)
    validated['energy_upper_bound'] = params.get('energy_upper_bound', 0.0)
    validated['space_strel_apothem'] = params.get('space_strel_apothem', 1)
    validated['length_dilation_ratio'] = params.get('length_dilation_ratio', 1.0)
    validated['number_of_edges_per_vertex'] = params.get('number_of_edges_per_vertex', 4)
    validated['step_size_per_origin_radius'] = params.get('step_size_per_origin_radius', 1.0)
    validated['max_edge_energy'] = params.get('max_edge_energy', 0.0)
    
    return validated

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

