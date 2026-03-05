#!/usr/bin/env python3
"""
MATLAB Output Parser for SLAVV Vectorization Results

This module loads and extracts data from MATLAB .mat files produced by vectorize_V200.
It handles the structure of MATLAB batch output folders and provides utilities to
extract vertices, edges, network statistics, and timing information.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

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
    """Extract vertex information from MATLAB network data."""
    vertices_info = {
        'positions': np.array([]),
        'radii': np.array([]),
        'count': 0
    }
    
    # Check for root-level arrays (common in this dataset)
    if 'vertex_space_subscripts' in mat_data:
        positions = np.array(mat_data['vertex_space_subscripts'])
        if positions.ndim == 1 and positions.size > 0:
            positions = positions.reshape(-1, 1)
        vertices_info['positions'] = positions
        vertices_info['count'] = positions.shape[0] if positions.size > 0 else 0
        
        # Handle scales/radii
        if 'vertex_scale_subscripts' in mat_data:
            scale_indices = np.array(mat_data['vertex_scale_subscripts'])
            vertices_info['scale_indices'] = scale_indices
            
            # Try to map to radii if range is available
            if 'lumen_radius_in_microns_range' in mat_data:
                 radii_range = np.array(mat_data['lumen_radius_in_microns_range'])
                 if scale_indices.size > 0:
                     # Matlab indices are 1-based usually, check min
                     # If they are from Matlab, they might be 1-based.
                     # But scipy.io might load them as is. 
                     # Let's assume 1-based if coming from Matlab.
                     # Actually, let's check min value.
                     if np.min(scale_indices) >= 1:
                         vertices_info['radii'] = radii_range[scale_indices.astype(int) - 1]
                     else:
                         vertices_info['radii'] = radii_range[scale_indices.astype(int)]
    
    # Fallback to struct-based extraction
    elif 'vertex' in mat_data or 'vertices' in mat_data:
        vertex_struct = mat_data.get('vertex', mat_data.get('vertices'))
        if hasattr(vertex_struct, 'space_subscripts'):
            positions = np.array(vertex_struct.space_subscripts)
            vertices_info['positions'] = positions
            vertices_info['count'] = positions.shape[0] if positions.size > 0 else 0
        elif hasattr(vertex_struct, 'positions'):
            positions = np.array(vertex_struct.positions)
            vertices_info['positions'] = positions
            vertices_info['count'] = positions.shape[0] if positions.size > 0 else 0
            
        if hasattr(vertex_struct, 'radii'):
             vertices_info['radii'] = np.array(vertex_struct.radii)

    logger.info(f"Extracted {vertices_info['count']} vertices")
    return vertices_info



def extract_edges(mat_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract edge information from MATLAB network data."""
    edges_info = {
        'connections': np.array([]),
        'traces': [],
        'count': 0,
        'total_length': 0.0
    }
    
    # Check for root-level arrays
    if 'edges2vertices' in mat_data:
        indices = np.array(mat_data['edges2vertices'])
        # Matlab indices are 1-based, convert to 0-based
        if indices.size > 0 and np.min(indices) >= 1:
            indices = indices - 1
        edges_info['connections'] = indices
        edges_info['count'] = indices.shape[0] if indices.size > 0 else 0
    
    if 'edge_space_subscripts' in mat_data:
        space_subs = mat_data['edge_space_subscripts']
        if isinstance(space_subs, np.ndarray):
            if space_subs.dtype == object:
                 edges_info['traces'] = [np.array(t) if t is not None else np.array([]) for t in space_subs]
            else:
                 edges_info['traces'] = [space_subs] # Single edge?
    
    # Fallback to struct-based
    if edges_info['count'] == 0:
        edge_struct = mat_data.get('edge', mat_data.get('edges'))
        if edge_struct is not None:
             if hasattr(edge_struct, 'vertices'):
                indices = np.array(edge_struct.vertices)
                # Check for 1-based indexing heuristic
                if indices.size > 0 and np.min(indices) >= 1:
                     indices = indices - 1
                edges_info['connections'] = indices
                edges_info['count'] = indices.shape[0]
             
             if hasattr(edge_struct, 'space_subscripts'):
                 space_subs = edge_struct.space_subscripts
                 if isinstance(space_subs, np.ndarray) and space_subs.dtype == object:
                      edges_info['traces'] = [np.array(t) for t in space_subs if t is not None]

    logger.info(f"Extracted {edges_info['count']} edges")
    return edges_info



def extract_network_data(mat_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract network topology and statistics."""
    network_data = {
        'strands': [],
        'stats': {}
    }
    
    # Extract strands (topology)
    if 'strand_subscripts' in mat_data:
        strands = mat_data['strand_subscripts']
        if isinstance(strands, np.ndarray):
            # Handle object array (cell array)
            if strands.dtype == object:
                # Filter out None and empty arrays
                # Do NOT flatten coordinate arrays (N, 4) -> keep them as is
                network_data['strands'] = [np.array(s) if s is not None and s.size > 0 else np.array([]) for s in strands]
                # Filter out empty entries
                network_data['strands'] = [s for s in network_data['strands'] if s.size > 0]
            else:
                # Single array or different format
                network_data['strands'] = [strands]
    elif 'strand' in mat_data:
        # Alternative key used by some MATLAB outputs
        s = mat_data['strand']
        if isinstance(s, np.ndarray) and s.size > 0:
            network_data['strands'] = [np.array(x) for x in s] if s.dtype == object else [s]
                  
    # Extract statistics
    if 'network_statistics' in mat_data:
        ns = mat_data['network_statistics']
        # Helper to extract scalar or array
        def get_val(obj, name):
            if hasattr(obj, name):
                val = getattr(obj, name)
                if isinstance(val, np.ndarray) and val.size == 1:
                    return val.item()
                # Handle 0-d arrays
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    return val.item()
                return val
            return None

        network_data['stats']['strand_count'] = get_val(ns, 'num_strands') or 0
        network_data['stats']['total_length_microns'] = get_val(ns, 'length') or 0.0
        network_data['stats']['mean_radius_microns'] = get_val(ns, 'strand_ave_radii')
        
        # Handle mean radius being an array
        mr = network_data['stats']['mean_radius_microns']
        if isinstance(mr, np.ndarray):
             network_data['stats']['mean_radius_microns'] = float(np.mean(mr)) if mr.size > 0 else 0.0
             
    return network_data


def extract_network_stats(mat_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract network statistics from MATLAB data. Returns a flat stats dict."""
    net = extract_network_data(mat_data)
    stats = net.get('stats', {})
    # Ensure expected keys with defaults
    return {
        'strand_count': stats.get('strand_count', len(net.get('strands', []))),
        'total_length_microns': stats.get('total_length_microns', 0.0),
        'mean_radius_microns': stats.get('mean_radius_microns', 0.0),
    }


def load_matlab_batch_results(batch_folder: Union[str, Path]) -> Dict[str, Any]:
    """Load and aggregate results from a MATLAB batch output folder."""
    batch_path = Path(batch_folder)
    if not batch_path.exists():
        raise MATLABParseError(f"Batch folder not found: {batch_folder}")
        
    logger.info(f"Loading MATLAB results from: {batch_path}")
    
    results = {
        'vertices': {'count': 0, 'positions': np.array([]), 'radii': np.array([])},
        'edges': {'count': 0, 'indices': np.array([]), 'traces': [], 'total_length': 0.0},
        'network': {'strands': []},
        'network_stats': {},
        'timings': {},
        'batch_folder': str(batch_path),
        'files': []
    }
    
    vectors_dir = batch_path / 'vectors'
    if not vectors_dir.exists():
        logger.warning(f"Vectors directory not found: {vectors_dir}")
        return results

    # Helper to merge dicts
    def merge_info(target, source):
        for k, v in source.items():
            if isinstance(v, np.ndarray):
                if v.size > 0:
                    target[k] = v
            elif isinstance(v, list):
                if v:
                    target[k] = v
            elif v:
                target[k] = v

    # 1. Load Vertices
    v_files = list(vectors_dir.glob('vertices_*.mat'))
    if v_files:
        f = v_files[-1]
        logger.info(f"Loading vertices: {f.name}")
        data = load_mat_file_safe(f)
        if data:
            v_info = extract_vertices(data)
            merge_info(results['vertices'], v_info)
            results['files'].append(str(f))

    # 2. Load Edges
    e_files = list(vectors_dir.glob('edges_*.mat'))
    if e_files:
        f = e_files[-1]
        logger.info(f"Loading edges: {f.name}")
        data = load_mat_file_safe(f)
        if data:
            e_info = extract_edges(data)
            merge_info(results['edges'], e_info)
            results['files'].append(str(f))
            
    # 3. Load Network (for stats and topology)
    n_files = list(vectors_dir.glob('network_*.mat'))
    if n_files:
        f = n_files[-1]
        logger.info(f"Loading network: {f.name}")
        data = load_mat_file_safe(f)
        if data:
            net_data = extract_network_data(data)
            if net_data.get('strands'):
                results['network']['strands'] = net_data['strands']
            results['network_stats'].update(net_data.get('stats', {}))
            results['files'].append(str(f))
            
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
