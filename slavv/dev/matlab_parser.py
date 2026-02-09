#!/usr/bin/env python3
"""
MATLAB Output Parser for SLAVV Vectorization Results

This module loads and extracts data from MATLAB .mat files produced by vectorize_V200.
It handles the structure of MATLAB batch output folders and provides utilities to
extract vertices, edges, network statistics, and timing information.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


class MATLABParseError(Exception):
    """Exception raised when MATLAB output parsing fails."""
    pass


def find_batch_folder(output_dir: Union[str, Path]) -> Optional[Path]:
    """Find the most recent MATLAB batch folder in the output directory.
    
    Parameters
    ----------
    output_dir : str | Path
        Directory to search for batch folders
        
    Returns
    -------
    Optional[Path]
        Path to the most recent batch folder, or None if not found
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return None
    
    # Find all batch folders (format: batch_YYMMDD-HHmmss)
    batch_folders = [
        d for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith('batch_')
    ]
    
    if not batch_folders:
        logger.warning(f"No batch folders found in {output_dir}")
        return None
    
    # Sort by name (chronological due to timestamp format) and return most recent
    batch_folders.sort()
    return batch_folders[-1]


def load_mat_file_safe(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely load a MATLAB .mat file with error handling.
    
    Parameters
    ----------
    file_path : Path
        Path to the .mat file
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing MATLAB data, or None if loading fails
    """
    try:
        # Load with squeeze_me=True to remove singleton dimensions
        # struct_as_record=False to access struct fields as attributes
        data = loadmat(
            str(file_path),
            squeeze_me=True,
            struct_as_record=False,
            mat_dtype=True
        )
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def extract_vertices(mat_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract vertex information from MATLAB network data.
    
    Parameters
    ----------
    mat_data : Dict[str, Any]
        Dictionary loaded from MATLAB .mat file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'positions': Nx4 array (x, y, z, scale_index)
        - 'radii': Nx1 array of vessel radii
        - 'count': number of vertices
    """
    vertices_info = {
        'positions': np.array([]),
        'radii': np.array([]),
        'count': 0
    }
    
    # Try to find vertex structure in various possible locations
    vertex_struct = None
    
    # Common locations in MATLAB output
    if 'vertex' in mat_data:
        vertex_struct = mat_data['vertex']
    elif 'vertices' in mat_data:
        vertex_struct = mat_data['vertices']
    
    if vertex_struct is None:
        logger.warning("No vertex structure found in MATLAB data")
        return vertices_info
    
    # Extract positions
    if hasattr(vertex_struct, 'space_subscripts'):
        positions = np.array(vertex_struct.space_subscripts)
        if positions.ndim == 1 and positions.size > 0:
            positions = positions.reshape(-1, 1)
        vertices_info['positions'] = positions
        vertices_info['count'] = positions.shape[0] if positions.size > 0 else 0
    elif hasattr(vertex_struct, 'positions'):
        positions = np.array(vertex_struct.positions)
        vertices_info['positions'] = positions
        vertices_info['count'] = positions.shape[0] if positions.size > 0 else 0
    
    # Extract radii (scale values converted to microns)
    if hasattr(vertex_struct, 'scale_subscripts'):
        # scale_subscripts are indices into a scale array
        # We may need to convert these to actual radii
        scale_indices = np.array(vertex_struct.scale_subscripts)
        vertices_info['scale_indices'] = scale_indices
        
        # Try to find the scale array to convert to actual radii
        if 'lumen_radius_in_microns_range' in mat_data:
            radii_range = np.array(mat_data['lumen_radius_in_microns_range'])
            if scale_indices.size > 0:
                # Map scale indices to radii
                vertices_info['radii'] = radii_range[scale_indices.astype(int)]
        elif hasattr(vertex_struct, 'radii'):
            vertices_info['radii'] = np.array(vertex_struct.radii)
    elif hasattr(vertex_struct, 'radii'):
        vertices_info['radii'] = np.array(vertex_struct.radii)
    
    logger.info(f"Extracted {vertices_info['count']} vertices from MATLAB data")
    return vertices_info


def extract_edges(mat_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract edge information from MATLAB network data.
    
    Parameters
    ----------
    mat_data : Dict[str, Any]
        Dictionary loaded from MATLAB .mat file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'indices': Mx2 array of vertex connectivity
        - 'traces': List of edge trajectories (if available)
        - 'count': number of edges
        - 'total_length': total network length in microns (if available)
    """
    edges_info = {
        'indices': np.array([]),
        'traces': [],
        'count': 0,
        'total_length': 0.0
    }
    
    # Try to find edge structure
    edge_struct = None
    if 'edge' in mat_data:
        edge_struct = mat_data['edge']
    elif 'edges' in mat_data:
        edge_struct = mat_data['edges']
    elif 'edge_indices' in mat_data:
        edges_info['indices'] = np.array(mat_data['edge_indices'])
        edges_info['count'] = edges_info['indices'].shape[0] if edges_info['indices'].size > 0 else 0
    
    if edge_struct is not None:
        # Extract connectivity
        if hasattr(edge_struct, 'vertices'):
            edges_info['indices'] = np.array(edge_struct.vertices)
            edges_info['count'] = edges_info['indices'].shape[0] if edges_info['indices'].size > 0 else 0
        
        # Extract edge traces (space subscripts)
        if hasattr(edge_struct, 'space_subscripts'):
            space_subs = edge_struct.space_subscripts
            # This is typically a cell array in MATLAB
            if isinstance(space_subs, np.ndarray):
                if space_subs.dtype == object:
                    # Array of arrays
                    edges_info['traces'] = [np.array(trace) for trace in space_subs if trace is not None]
                else:
                    edges_info['traces'] = [space_subs]
        
        # Extract edge lengths
        if hasattr(edge_struct, 'lengths'):
            lengths = np.array(edge_struct.lengths)
            edges_info['lengths'] = lengths
            edges_info['total_length'] = np.sum(lengths)
    
    logger.info(f"Extracted {edges_info['count']} edges from MATLAB data")
    return edges_info


def extract_network_stats(mat_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract network statistics from MATLAB data.
    
    Parameters
    ----------
    mat_data : Dict[str, Any]
        Dictionary loaded from MATLAB .mat file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing network statistics
    """
    stats = {
        'strand_count': 0,
        'total_length_microns': 0.0,
        'mean_radius_microns': 0.0,
        'network_volume_microns3': 0.0
    }
    
    # Look for network or strand information
    if 'network' in mat_data:
        network = mat_data['network']
        if hasattr(network, 'strand'):
            strands = network.strand
            if isinstance(strands, np.ndarray):
                stats['strand_count'] = len(strands)
    
    if 'strand' in mat_data:
        strands = mat_data['strand']
        if isinstance(strands, np.ndarray):
            stats['strand_count'] = len(strands)
    
    # Extract aggregate statistics if available
    vertices_info = extract_vertices(mat_data)
    if vertices_info['radii'].size > 0:
        stats['mean_radius_microns'] = float(np.mean(vertices_info['radii']))
    
    edges_info = extract_edges(mat_data)
    if edges_info['total_length'] > 0:
        stats['total_length_microns'] = float(edges_info['total_length'])
    
    logger.info(f"Extracted network statistics: {stats['strand_count']} strands")
    return stats


def extract_stage_timings(batch_folder: Path) -> Dict[str, float]:
    """Extract pipeline stage timing information from MATLAB log or settings.
    
    Parameters
    ----------
    batch_folder : Path
        Path to the MATLAB batch output folder
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping stage names to elapsed time in seconds
    """
    timings = {
        'energy': 0.0,
        'vertices': 0.0,
        'edges': 0.0,
        'network': 0.0,
        'total': 0.0
    }
    
    # Try to find timing information in settings folder
    settings_dir = batch_folder / 'settings'
    if settings_dir.exists():
        workflow_files = list(settings_dir.glob('workflow_*.mat'))
        if workflow_files:
            # Load the most recent workflow file
            workflow_data = load_mat_file_safe(workflow_files[-1])
            if workflow_data:
                # Look for timing fields
                if 'time_stamps' in workflow_data:
                    time_stamps = workflow_data['time_stamps']
                    # Parse timing structure if available
                    # This is highly dependent on MATLAB output format
                    pass
    
    # Try to parse log file if available
    log_file = batch_folder.parent / 'matlab_run.log'
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
            # Look for timing patterns in log
            # Pattern: "Elapsed time: XX.XX seconds"
            elapsed_match = re.search(r'Elapsed time:\s*([\d.]+)\s*seconds', log_content)
            if elapsed_match:
                timings['total'] = float(elapsed_match.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse log file: {e}")
    
    return timings


def load_matlab_batch_results(batch_folder: Union[str, Path]) -> Dict[str, Any]:
    """Load all results from a MATLAB batch output folder.
    
    This is the main entry point for loading MATLAB results.
    
    Parameters
    ----------
    batch_folder : str | Path
        Path to the MATLAB batch output folder (batch_YYMMDD-HHmmss)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'vertices': vertex information
        - 'edges': edge information
        - 'network_stats': network statistics
        - 'timings': stage timing information
        - 'batch_folder': path to batch folder
        - 'files': list of loaded file paths
        
    Raises
    ------
    MATLABParseError
        If the batch folder structure is invalid or files cannot be loaded
    """
    batch_path = Path(batch_folder)
    
    if not batch_path.exists():
        raise MATLABParseError(f"Batch folder does not exist: {batch_folder}")
    
    if not batch_path.is_dir():
        raise MATLABParseError(f"Batch folder is not a directory: {batch_folder}")
    
    logger.info(f"Loading MATLAB results from: {batch_path}")
    
    results = {
        'vertices': {'count': 0, 'positions': np.array([]), 'radii': np.array([])},
        'edges': {'count': 0, 'indices': np.array([]), 'traces': [], 'total_length': 0.0},
        'network_stats': {'strand_count': 0},
        'timings': {},
        'batch_folder': str(batch_path),
        'files': []
    }
    
    # Look for network file in vectors directory
    vectors_dir = batch_path / 'vectors'
    if vectors_dir.exists():
        # Find network .mat file
        network_files = list(vectors_dir.glob('network_*.mat'))
        if network_files:
            network_file = network_files[-1]  # Most recent
            logger.info(f"Loading network file: {network_file}")
            
            mat_data = load_mat_file_safe(network_file)
            if mat_data:
                results['vertices'] = extract_vertices(mat_data)
                results['edges'] = extract_edges(mat_data)
                results['network_stats'] = extract_network_stats(mat_data)
                results['files'].append(str(network_file))
            else:
                logger.warning(f"Failed to load network file: {network_file}")
        else:
            logger.warning(f"No network files found in {vectors_dir}")
            
            # Try to load vertices and edges separately
            vertices_files = list(vectors_dir.glob('vertices_*.mat'))
            edges_files = list(vectors_dir.glob('edges_*.mat'))
            
            if vertices_files:
                vertices_file = vertices_files[-1]
                logger.info(f"Loading vertices file: {vertices_file}")
                mat_data = load_mat_file_safe(vertices_file)
                if mat_data:
                    results['vertices'] = extract_vertices(mat_data)
                    results['files'].append(str(vertices_file))
            
            if edges_files:
                edges_file = edges_files[-1]
                logger.info(f"Loading edges file: {edges_file}")
                mat_data = load_mat_file_safe(edges_file)
                if mat_data:
                    results['edges'] = extract_edges(mat_data)
                    results['files'].append(str(edges_file))
    else:
        logger.warning(f"Vectors directory not found: {vectors_dir}")
    
    # Extract timing information
    results['timings'] = extract_stage_timings(batch_path)
    
    # Validate that we loaded something useful
    if results['vertices'].get('count', 0) == 0 and results['edges'].get('count', 0) == 0:
        logger.warning("No vertices or edges were extracted from MATLAB data")
    
    return results


def load_matlab_results_from_output_dir(output_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Find and load MATLAB results from an output directory.
    
    Convenience function that finds the most recent batch folder and loads results.
    
    Parameters
    ----------
    output_dir : str | Path
        Directory containing MATLAB batch output folders
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Results dictionary from load_matlab_batch_results, or None if no batch folder found
    """
    batch_folder = find_batch_folder(output_dir)
    if batch_folder is None:
        return None
    
    try:
        return load_matlab_batch_results(batch_folder)
    except MATLABParseError as e:
        logger.error(f"Failed to load MATLAB results: {e}")
        return None


if __name__ == '__main__':
    # Test the parser with command-line arguments
    import sys
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Parse MATLAB vectorization output')
    parser.add_argument('batch_folder', help='Path to MATLAB batch folder')
    parser.add_argument('--output', help='Output JSON file for parsed results')
    
    args = parser.parse_args()
    
    try:
        results = load_matlab_batch_results(args.batch_folder)
        
        print("\n" + "="*60)
        print("MATLAB Results Summary")
        print("="*60)
        print(f"Vertices: {results['vertices'].get('count', 0)}")
        print(f"Edges: {results['edges'].get('count', 0)}")
        print(f"Strands: {results['network_stats'].get('strand_count', 0)}")
        print(f"Total length: {results['network_stats'].get('total_length_microns', 0):.2f} microns")
        print(f"Mean radius: {results['network_stats'].get('mean_radius_microns', 0):.2f} microns")
        print(f"\nTiming: {results['timings'].get('total', 0):.2f} seconds")
        print(f"\nFiles loaded: {len(results['files'])}")
        for file_path in results['files']:
            print(f"  - {file_path}")
        print("="*60)
        
        if args.output:
            # Save to JSON (with numpy array conversion)
            output_data = {
                'vertices_count': results['vertices'].get('count', 0),
                'edges_count': results['edges'].get('count', 0),
                'network_stats': results['network_stats'],
                'timings': results['timings'],
                'batch_folder': results['batch_folder']
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    except MATLABParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
