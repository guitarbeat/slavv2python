
"""
Main pipeline orchestration for SLAVV.
Coordinates the energy, tracing, and graph construction steps.
"""
import os
import joblib
import numpy as np
import logging
from typing import Dict, Any, Callable, Optional

from . import energy, tracing, graph
from .. import utils

logger = logging.getLogger(__name__)

class SLAVVProcessor:
    """Main class for SLAVV vectorization processing"""
    
    def __init__(self):
        self.energy_data = None
        self.vertices = None
        self.edges = None
        self.network = None
        
    def process_image(
        self,
        image: np.ndarray,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete SLAVV processing pipeline.

        MATLAB Equivalent: `vectorize_V200.m` (with resume capability)

        Args:
            image: 3D input image array (y, x, z)
            parameters: Dictionary of processing parameters
            progress_callback: Optional callable receiving ``(fraction, stage)``
                updates as the pipeline advances from 0.0 to 1.0.
            checkpoint_dir: Optional directory path. If provided, intermediate steps
                (Energy, Vertices, Edges, Network) will be saved/loaded from this directory.
                Enables resuming crashed runs or inspecting intermediate results.

        Returns:
            Dictionary containing all processing results
        """
        if image.ndim != 3 or 0 in image.shape:
            raise ValueError("Input image must be a non-empty 3D array")

        logger.info("Starting SLAVV processing pipeline")
        
        # Imports for checkpointing
        paths = {}
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            paths = {
                'energy': os.path.join(checkpoint_dir, 'checkpoint_energy.pkl'),
                'vertices': os.path.join(checkpoint_dir, 'checkpoint_vertices.pkl'),
                'edges': os.path.join(checkpoint_dir, 'checkpoint_edges.pkl'),
                'network': os.path.join(checkpoint_dir, 'checkpoint_network.pkl'),
            }
        
        if progress_callback:
            progress_callback(0.0, "start")

        # Validate and populate default parameters
        parameters = utils.validate_parameters(parameters)

        # Step 0: Image preprocessing (fast, typically not cached)
        image = utils.preprocess_image(image, parameters)
        if progress_callback:
            progress_callback(0.2, "preprocess")

        # Step 1: Energy image formation
        if checkpoint_dir and paths and os.path.exists(paths['energy']):
            logger.info(f"Loading cached Energy Field from {paths['energy']}")
            energy_data = joblib.load(paths['energy'])
        else:
            energy_data = self.calculate_energy_field(image, parameters)
            if checkpoint_dir and paths:
                logger.info(f"Saving Energy Field to {paths['energy']}")
                joblib.dump(energy_data, paths['energy'])
                
        if progress_callback:
            progress_callback(0.4, "energy")

        # Step 2: Vertex extraction
        if checkpoint_dir and paths and os.path.exists(paths['vertices']):
            logger.info(f"Loading cached Vertices from {paths['vertices']}")
            vertices = joblib.load(paths['vertices'])
        else:
            vertices = self.extract_vertices(energy_data, parameters)
            if checkpoint_dir and paths:
                logger.info(f"Saving Vertices to {paths['vertices']}")
                joblib.dump(vertices, paths['vertices'])
                
        if progress_callback:
            progress_callback(0.6, "vertices")

        # Step 3: Edge extraction
        if checkpoint_dir and paths and os.path.exists(paths['edges']):
            logger.info(f"Loading cached Edges from {paths['edges']}")
            edges = joblib.load(paths['edges'])
        else:
            edge_method = parameters.get('edge_method', 'tracing')
            if edge_method == 'watershed':
                edges = self.extract_edges_watershed(energy_data, vertices, parameters)
            else:
                edges = self.extract_edges(energy_data, vertices, parameters)
            if checkpoint_dir and paths:
                logger.info(f"Saving Edges to {paths['edges']}")
                joblib.dump(edges, paths['edges'])
                
        if progress_callback:
            progress_callback(0.8, "edges")

        # Step 4: Network construction
        if checkpoint_dir and paths and os.path.exists(paths['network']):
            logger.info(f"Loading cached Network from {paths['network']}")
            network = joblib.load(paths['network'])
        else:
            network = self.construct_network(edges, vertices, parameters)
            if checkpoint_dir and paths:
                logger.info(f"Saving Network to {paths['network']}")
                joblib.dump(network, paths['network'])
                
        if progress_callback:
            progress_callback(1.0, "network")
        
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
        """Calculate multi-scale energy field using Hessian. Delegates to ``energy`` module."""
        from .. import utils as utils_module
        return energy.calculate_energy_field(image, params, utils_module.get_chunking_lattice)

    def extract_vertices(self, energy_data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract vertices as local extrema. Delegates to ``tracing`` module."""
        return tracing.extract_vertices(energy_data, params)

    def extract_edges(self, energy_data: Dict[str, Any], vertices: Dict[str, Any], 
                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract edges by tracing. Delegates to ``tracing`` module."""
        return tracing.extract_edges(energy_data, vertices, params)
    
    def extract_edges_watershed(self, energy_data: Dict[str, Any], vertices: Dict[str, Any], 
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract edges by watershed. Delegates to ``tracing`` module."""
        return tracing.extract_edges_watershed(energy_data, vertices, params)

    def construct_network(self, edges: Dict[str, Any], vertices: Dict[str, Any],
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Construct network from traces. Delegates to ``graph`` module."""
        return graph.construct_network(edges, vertices, params)

    # Legacy private methods exposed for compatibility/internal use
    # These static methods can be attached to the class if needed, or kept as module calls
    # Since original code used `self._method`, we can map them.
    
    @staticmethod
    def _spherical_structuring_element(radius, mpv):
        return energy.spherical_structuring_element(radius, mpv)
    
    @staticmethod
    def _trace_edge(*args, **kwargs):
        return tracing.trace_edge(*args, **kwargs)
    
    @staticmethod
    def _generate_edge_directions(*args, **kwargs):
        return tracing.generate_edge_directions(*args, **kwargs)

    @staticmethod
    def _estimate_vessel_directions(*args, **kwargs):
        return tracing.estimate_vessel_directions(*args, **kwargs)
        
    @staticmethod
    def _near_vertex(*args, **kwargs):
        return tracing.near_vertex(*args, **kwargs)

    @staticmethod
    def _find_terminal_vertex(*args, **kwargs):
        return tracing.find_terminal_vertex(*args, **kwargs)
        
    @staticmethod
    def _compute_gradient(*args, **kwargs):
        return tracing.compute_gradient(*args, **kwargs)
        
    @staticmethod
    def _in_bounds(*args, **kwargs):
        return tracing.in_bounds(*args, **kwargs)
    
    @staticmethod
    def _trace_strand(*args, **kwargs):
        """Legacy method - delegates to sparse implementation.
        
        Note: Signature differs from original dense implementation.
        Use graph.trace_strand_sparse directly for new code.
        """
        return graph.trace_strand_sparse(*args, **kwargs)

    @staticmethod
    def _trace_strand_sparse(*args, **kwargs):
        return graph.trace_strand_sparse(*args, **kwargs)

    @staticmethod
    def _sort_and_validate_strands_sparse(*args, **kwargs):
        return graph.sort_and_validate_strands_sparse(*args, **kwargs)
