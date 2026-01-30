#!/usr/bin/env python3
"""
Compare MATLAB and Python implementations of SLAVV vectorization.

This script runs both implementations with identical parameters and compares:
- Runtime performance
- Vertex counts and statistics
- Edge counts and statistics
- Network statistics

Usage:
    python scripts/compare_matlab_python.py \
        --input "data/slavv_test_volume.tif" \
        --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
        --output-dir "comparison_output"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.slavv.pipeline import SLAVVProcessor
from src.slavv.io_utils import load_tiff_volume, export_pipeline_results

# Import MATLAB parser
sys.path.insert(0, str(Path(__file__).parent))
from matlab_output_parser import load_matlab_results_from_output_dir, load_matlab_batch_results


def load_parameters(params_file: Optional[str] = None) -> Dict[str, Any]:
    """Load parameters from JSON file or use defaults."""
    if params_file and os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        # Use Python defaults
        params = {
            'microns_per_voxel': [1.0, 1.0, 1.0],
            'radius_of_smallest_vessel_in_microns': 1.5,
            'radius_of_largest_vessel_in_microns': 50.0,
            'approximating_PSF': True,
            'excitation_wavelength_in_microns': 1.3,
            'numerical_aperture': 0.95,
            'sample_index_of_refraction': 1.33,
            'scales_per_octave': 1.5,
            'gaussian_to_ideal_ratio': 1.0,
            'spherical_to_annular_ratio': 1.0,
            'max_voxels_per_node_energy': 1e5,
        }
    
    # Convert lists to numpy arrays where needed
    if 'microns_per_voxel' in params:
        params['microns_per_voxel'] = np.array(params['microns_per_voxel'])
    
    return params


def run_matlab_vectorization(
    input_file: str,
    output_dir: str,
    matlab_path: str,
    batch_script: str = None
) -> Dict[str, Any]:
    """Run MATLAB vectorization via CLI."""
    print("\n" + "="*60)
    print("Running MATLAB Implementation")
    print("="*60)
    
    if batch_script is None:
        batch_script = os.path.join(
            os.path.dirname(__file__),
            'run_matlab_cli.bat'
        )
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run MATLAB via batch script
    cmd = [
        batch_script,
        input_file,
        output_dir,
        matlab_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root
        )
        elapsed_time = time.time() - start_time
        
        print(f"\nMATLAB execution completed in {elapsed_time:.2f} seconds")
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Try to find MATLAB output files
        matlab_results = {
            'success': True,
            'elapsed_time': elapsed_time,
            'output_dir': output_dir,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # Look for batch folder in output directory
        batch_folders = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('batch_')
        ]
        
        if batch_folders:
            # Use most recent batch folder
            batch_folder = os.path.join(output_dir, sorted(batch_folders)[-1])
            matlab_results['batch_folder'] = batch_folder
            
            # Look for network results
            vectors_dir = os.path.join(batch_folder, 'vectors')
            if os.path.exists(vectors_dir):
                matlab_results['vectors_dir'] = vectors_dir
                
                # Try to find network .mat file
                mat_files = [
                    f for f in os.listdir(vectors_dir)
                    if f.startswith('network_') and f.endswith('.mat')
                ]
                if mat_files:
                    matlab_results['network_mat'] = os.path.join(vectors_dir, mat_files[0])
        
        return matlab_results
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\nMATLAB execution failed after {elapsed_time:.2f} seconds")
        print(f"Exit code: {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        
        return {
            'success': False,
            'elapsed_time': elapsed_time,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def run_python_vectorization(
    input_file: str,
    output_dir: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Run Python vectorization."""
    print("\n" + "="*60)
    print("Running Python Implementation")
    print("="*60)
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Load image
    print("Loading image...")
    image = load_tiff_volume(input_file)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    
    # Initialize processor
    processor = SLAVVProcessor()
    
    # Run pipeline
    print("Running pipeline...")
    start_time = time.time()
    
    def progress_callback(frac, stage):
        print(f"  Progress: {frac*100:.1f}% - {stage}")
    
    try:
        results = processor.process_image(
            image,
            params,
            progress_callback=progress_callback,
            checkpoint_dir=os.path.join(output_dir, 'checkpoints')
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\nPython execution completed in {elapsed_time:.2f} seconds")
        
        # Extract statistics
        python_results = {
            'success': True,
            'elapsed_time': elapsed_time,
            'output_dir': output_dir,
            'vertices_count': len(results['vertices']['positions']) if 'vertices' in results else 0,
            'edges_count': len(results['edges']['traces']) if 'edges' in results else 0,
            'network_strands_count': len(results['network']['strands']) if 'network' in results else 0,
            'results': results
        }
        
        # Export results
        print("Exporting results...")
        export_pipeline_results(results, output_dir, base_name="python_comparison")
        
        return python_results
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nPython execution failed after {elapsed_time:.2f} seconds")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'elapsed_time': elapsed_time,
            'error': str(e)
        }


def match_vertices(
    matlab_positions: np.ndarray,
    python_positions: np.ndarray,
    distance_threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match vertices between MATLAB and Python using nearest neighbors.
    
    Parameters
    ----------
    matlab_positions : np.ndarray
        MATLAB vertex positions (Nx3 or Nx4)
    python_positions : np.ndarray
        Python vertex positions (Mx3 or Mx4)
    distance_threshold : float
        Maximum distance for a match (in voxels)
        
    Returns
    -------
    tuple
        (matched_matlab_indices, matched_python_indices, distances)
    """
    if matlab_positions.size == 0 or python_positions.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Use only spatial coordinates (first 3 columns)
    matlab_xyz = matlab_positions[:, :3] if matlab_positions.ndim == 2 else matlab_positions.reshape(-1, 3)
    python_xyz = python_positions[:, :3] if python_positions.ndim == 2 else python_positions.reshape(-1, 3)
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(python_xyz)
    
    # Find nearest Python vertex for each MATLAB vertex
    distances, python_indices = tree.query(matlab_xyz)
    
    # Filter by distance threshold
    valid_matches = distances < distance_threshold
    
    matlab_indices = np.arange(len(matlab_xyz))[valid_matches]
    matched_python_indices = python_indices[valid_matches]
    matched_distances = distances[valid_matches]
    
    return matlab_indices, matched_python_indices, matched_distances


def compare_vertices(matlab_verts: Dict[str, Any], python_verts: Dict[str, Any]) -> Dict[str, Any]:
    """Compare vertex information between MATLAB and Python.
    
    Parameters
    ----------
    matlab_verts : Dict[str, Any]
        MATLAB vertex data from parser
    python_verts : Dict[str, Any]
        Python vertex data
        
    Returns
    -------
    Dict[str, Any]
        Comparison metrics
    """
    comparison = {
        'matlab_count': matlab_verts.get('count', 0),
        'python_count': python_verts.get('count', 0),
        'count_difference': 0,
        'count_percent_difference': 0.0,
        'position_rmse': None,
        'matched_vertices': 0,
        'unmatched_matlab': 0,
        'unmatched_python': 0,
        'radius_correlation': None,
        'radius_stats': {}
    }
    
    matlab_count = comparison['matlab_count']
    python_count = comparison['python_count']
    
    if matlab_count > 0 or python_count > 0:
        comparison['count_difference'] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison['count_percent_difference'] = (comparison['count_difference'] / avg_count) * 100.0
    
    # Match vertices if both have data
    matlab_positions = matlab_verts.get('positions', np.array([]))
    python_positions = python_verts.get('positions', np.array([]))
    
    if matlab_positions.size > 0 and python_positions.size > 0:
        matlab_idx, python_idx, distances = match_vertices(matlab_positions, python_positions)
        
        comparison['matched_vertices'] = len(matlab_idx)
        comparison['unmatched_matlab'] = matlab_count - len(matlab_idx)
        comparison['unmatched_python'] = python_count - len(python_idx)
        
        if len(distances) > 0:
            comparison['position_rmse'] = float(np.sqrt(np.mean(distances**2)))
            comparison['position_mean_distance'] = float(np.mean(distances))
            comparison['position_median_distance'] = float(np.median(distances))
            comparison['position_95th_percentile'] = float(np.percentile(distances, 95))
        
        # Compare radii for matched vertices
        matlab_radii = matlab_verts.get('radii', np.array([]))
        python_radii = python_verts.get('radii', np.array([]))
        
        if len(matlab_idx) > 0 and matlab_radii.size > 0 and python_radii.size > 0:
            matched_matlab_radii = matlab_radii[matlab_idx]
            matched_python_radii = python_radii[python_idx]
            
            # Compute correlation
            if len(matched_matlab_radii) > 1:
                pearson_r, pearson_p = stats.pearsonr(matched_matlab_radii, matched_python_radii)
                spearman_r, spearman_p = stats.spearmanr(matched_matlab_radii, matched_python_radii)
                
                comparison['radius_correlation'] = {
                    'pearson_r': float(pearson_r),
                    'pearson_p': float(pearson_p),
                    'spearman_r': float(spearman_r),
                    'spearman_p': float(spearman_p)
                }
            
            # Radius statistics
            comparison['radius_stats'] = {
                'matlab_mean': float(np.mean(matched_matlab_radii)),
                'matlab_std': float(np.std(matched_matlab_radii)),
                'python_mean': float(np.mean(matched_python_radii)),
                'python_std': float(np.std(matched_python_radii)),
                'mean_difference': float(np.mean(matched_matlab_radii - matched_python_radii)),
                'rmse': float(np.sqrt(np.mean((matched_matlab_radii - matched_python_radii)**2)))
            }
    
    return comparison


def compare_edges(matlab_edges: Dict[str, Any], python_edges: Dict[str, Any]) -> Dict[str, Any]:
    """Compare edge information between MATLAB and Python.
    
    Parameters
    ----------
    matlab_edges : Dict[str, Any]
        MATLAB edge data from parser
    python_edges : Dict[str, Any]
        Python edge data
        
    Returns
    -------
    Dict[str, Any]
        Comparison metrics
    """
    comparison = {
        'matlab_count': matlab_edges.get('count', 0),
        'python_count': python_edges.get('count', 0),
        'count_difference': 0,
        'count_percent_difference': 0.0,
        'total_length': {}
    }
    
    matlab_count = comparison['matlab_count']
    python_count = comparison['python_count']
    
    if matlab_count > 0 or python_count > 0:
        comparison['count_difference'] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison['count_percent_difference'] = (comparison['count_difference'] / avg_count) * 100.0
    
    # Compare total lengths if available
    matlab_total_length = matlab_edges.get('total_length', 0.0)
    if matlab_total_length > 0:
        comparison['total_length']['matlab'] = float(matlab_total_length)
    
    # Calculate Python edge lengths
    python_traces = python_edges.get('traces', [])
    if python_traces:
        python_total_length = 0.0
        for trace in python_traces:
            if isinstance(trace, np.ndarray) and trace.size > 0:
                # Calculate path length
                if trace.ndim == 2 and trace.shape[0] > 1:
                    diffs = np.diff(trace[:, :3], axis=0)
                    lengths = np.sqrt(np.sum(diffs**2, axis=1))
                    python_total_length += np.sum(lengths)
        
        comparison['total_length']['python'] = float(python_total_length)
        
        if matlab_total_length > 0 and python_total_length > 0:
            comparison['total_length']['difference'] = float(abs(matlab_total_length - python_total_length))
            comparison['total_length']['percent_difference'] = float(
                (comparison['total_length']['difference'] / ((matlab_total_length + python_total_length) / 2.0)) * 100.0
            )
    
    return comparison


def compare_networks(matlab_stats: Dict[str, Any], python_network: Dict[str, Any]) -> Dict[str, Any]:
    """Compare network-level statistics.
    
    Parameters
    ----------
    matlab_stats : Dict[str, Any]
        MATLAB network statistics
    python_network : Dict[str, Any]
        Python network data
        
    Returns
    -------
    Dict[str, Any]
        Comparison metrics
    """
    comparison = {
        'matlab_strand_count': matlab_stats.get('strand_count', 0),
        'python_strand_count': 0
    }
    
    # Extract Python strand count
    if 'strands' in python_network:
        comparison['python_strand_count'] = len(python_network['strands'])
    
    matlab_count = comparison['matlab_strand_count']
    python_count = comparison['python_strand_count']
    
    if matlab_count > 0 or python_count > 0:
        comparison['strand_count_difference'] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison['strand_count_percent_difference'] = (comparison['strand_count_difference'] / avg_count) * 100.0
    
    return comparison


def compare_results(
    matlab_results: Dict[str, Any],
    python_results: Dict[str, Any],
    matlab_parsed: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compare MATLAB and Python results with detailed analysis.
    
    Parameters
    ----------
    matlab_results : Dict[str, Any]
        MATLAB execution results
    python_results : Dict[str, Any]
        Python execution results
    matlab_parsed : Optional[Dict[str, Any]]
        Parsed MATLAB output data (from matlab_output_parser)
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive comparison metrics
    """
    print("\n" + "="*70)
    print(" "*20 + "COMPARISON SUMMARY")
    print("="*70)
    
    comparison = {
        'matlab': {
            'success': matlab_results.get('success', False),
            'elapsed_time': matlab_results.get('elapsed_time', 0),
        },
        'python': {
            'success': python_results.get('success', False),
            'elapsed_time': python_results.get('elapsed_time', 0),
            'vertices_count': python_results.get('vertices_count', 0),
            'edges_count': python_results.get('edges_count', 0),
            'network_strands_count': python_results.get('network_strands_count', 0),
        },
        'performance': {},
        'vertices': {},
        'edges': {},
        'network': {}
    }
    
    # Performance comparison
    if comparison['matlab']['success'] and comparison['python']['success']:
        matlab_time = comparison['matlab']['elapsed_time']
        python_time = comparison['python']['elapsed_time']
        speedup = matlab_time / python_time if python_time > 0 else 0
        
        comparison['performance'] = {
            'matlab_time_seconds': matlab_time,
            'python_time_seconds': python_time,
            'speedup': speedup,
            'faster': 'MATLAB' if matlab_time < python_time else 'Python'
        }
        
        # Format times nicely
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{int(seconds//60)}m {int(seconds%60)}s"
            else:
                return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
        
        print(f"\nPerformance:")
        print(f"  MATLAB:  {format_time(matlab_time):>12}  ({matlab_time:.1f}s)")
        print(f"  Python:  {format_time(python_time):>12}  ({python_time:.1f}s)")
        print(f"  Speedup: {speedup:>12.1f}x  ({comparison['performance']['faster']} is faster)")
    
    # Detailed comparison if MATLAB data was parsed
    if matlab_parsed:
        print("\nDetailed Result Comparison:")
        
        # Add MATLAB counts to comparison
        comparison['matlab']['vertices_count'] = matlab_parsed['vertices'].get('count', 0)
        comparison['matlab']['edges_count'] = matlab_parsed['edges'].get('count', 0)
        comparison['matlab']['strand_count'] = matlab_parsed['network_stats'].get('strand_count', 0)
        
        # Compare vertices
        python_verts = {
            'count': comparison['python']['vertices_count'],
            'positions': python_results.get('results', {}).get('vertices', {}).get('positions', np.array([])),
            'radii': python_results.get('results', {}).get('vertices', {}).get('radii', np.array([]))
        }
        
        comparison['vertices'] = compare_vertices(matlab_parsed['vertices'], python_verts)
        
        # Print comparison table
        print(f"\nResults Comparison:")
        print("-" * 70)
        print(f"{'Component':<15} {'MATLAB':>15} {'Python':>15} {'Difference':>15}")
        print("-" * 70)
        
        # Vertices
        matlab_verts = comparison['vertices']['matlab_count']
        python_verts = comparison['vertices']['python_count']
        diff_verts = python_verts - matlab_verts
        print(f"{'Vertices':<15} {matlab_verts:>15,} {python_verts:>15,} {diff_verts:>+15,}")
        
        if comparison['vertices'].get('matched_vertices', 0) > 0:
            matched = comparison['vertices']['matched_vertices']
            print(f"{'  Matched':<15} {matched:>15,}")
            if comparison['vertices'].get('position_rmse'):
                rmse = comparison['vertices']['position_rmse']
                print(f"{'  Pos RMSE':<15} {rmse:>14.3f} voxels")
        
        # Compare edges
        python_edge_data = {
            'count': comparison['python']['edges_count'],
            'traces': python_results.get('results', {}).get('edges', {}).get('traces', [])
        }
        
        comparison['edges'] = compare_edges(matlab_parsed['edges'], python_edge_data)
        
        # Edges
        matlab_edges = comparison['edges']['matlab_count']
        python_edges = comparison['edges']['python_count']
        diff_edges = python_edges - matlab_edges
        print(f"{'Edges':<15} {matlab_edges:>15,} {python_edges:>15,} {diff_edges:>+15,}")
        
        if 'total_length' in comparison['edges'] and comparison['edges']['total_length']:
            tl = comparison['edges']['total_length']
            if 'matlab' in tl and 'python' in tl:
                print(f"{'  Total length':<15} {tl['matlab']:>14.1f} {tl['python']:>14.1f} microns")
        
        # Compare networks
        python_network = python_results.get('results', {}).get('network', {})
        comparison['network'] = compare_networks(matlab_parsed['network_stats'], python_network)
        
        # Strands
        matlab_strands = comparison['network']['matlab_strand_count']
        python_strands = comparison['network']['python_strand_count']
        diff_strands = python_strands - matlab_strands
        print(f"{'Strands':<15} {matlab_strands:>15,} {python_strands:>15,} {diff_strands:>+15,}")
        
        print("-" * 70)
    else:
        # Basic comparison without parsed MATLAB data
        print(f"\nPython Results:")
        print(f"  Vertices: {comparison['python']['vertices_count']}")
        print(f"  Edges: {comparison['python']['edges_count']}")
        print(f"  Network strands: {comparison['python']['network_strands_count']}")
        print("\nNote: MATLAB results not parsed. Install scipy to enable detailed comparison.")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Compare MATLAB and Python SLAVV implementations'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input TIFF file path'
    )
    parser.add_argument(
        '--matlab-path',
        required=True,
        help='Path to MATLAB executable (e.g., C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for results (default: comparisons/YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
        '--params',
        help='JSON file with parameters (default: scripts/comparison_params.json)'
    )
    parser.add_argument(
        '--skip-matlab',
        action='store_true',
        help='Skip MATLAB execution (for testing Python only)'
    )
    parser.add_argument(
        '--skip-python',
        action='store_true',
        help='Skip Python execution (for testing MATLAB only)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    # Load parameters
    params_file = args.params or os.path.join(
        os.path.dirname(__file__),
        'comparison_params.json'
    )
    params = load_parameters(params_file)
    
    # Create output directories with timestamp if not specified
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('comparisons') / f'{timestamp}_comparison'
    else:
        output_dir = Path(args.output_dir)
    matlab_output = output_dir / 'matlab_results'
    python_output = output_dir / 'python_results'
    
    os.makedirs(matlab_output, exist_ok=True)
    os.makedirs(python_output, exist_ok=True)
    
    # Run MATLAB
    matlab_results = None
    if not args.skip_matlab:
        matlab_results = run_matlab_vectorization(
            args.input,
            str(matlab_output),
            args.matlab_path
        )
    else:
        print("\nSkipping MATLAB execution (--skip-matlab)")
    
    # Run Python
    python_results = None
    if not args.skip_python:
        python_results = run_python_vectorization(
            args.input,
            str(python_output),
            params
        )
    else:
        print("\nSkipping Python execution (--skip-python)")
    
    # Compare results
    if matlab_results and python_results:
        # Try to load parsed MATLAB data
        matlab_parsed = None
        if matlab_results.get('success') and matlab_results.get('batch_folder'):
            print("\nLoading MATLAB output data...")
            try:
                matlab_parsed = load_matlab_batch_results(matlab_results['batch_folder'])
                print(f"Successfully loaded MATLAB data from {matlab_results['batch_folder']}")
            except Exception as e:
                print(f"Warning: Could not load MATLAB output data: {e}")
                print("Comparison will proceed with basic metrics only.")
        
        comparison = compare_results(matlab_results, python_results, matlab_parsed)
        
        # Save comparison report
        report_file = output_dir / 'comparison_report.json'
        with open(report_file, 'w') as f:
            # Remove non-serializable items for JSON
            report = {
                'matlab': comparison['matlab'].copy(),
                'python': comparison['python'].copy(),
                'performance': comparison['performance'],
                'vertices': comparison.get('vertices', {}),
                'edges': comparison.get('edges', {}),
                'network': comparison.get('network', {})
            }
            json.dump(report, f, indent=2)
        
        print(f"\nComparison report saved to: {report_file}")
        
        # Generate summary.txt automatically
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from generate_summary import generate_summary
            summary_file = output_dir / 'summary.txt'
            generate_summary(output_dir, summary_file)
        except Exception as e:
            print(f"Note: Could not auto-generate summary: {e}")
    
    print("\n" + "="*70)
    print(" "*25 + "COMPARISON COMPLETE")
    print("="*70)
    
    # Print final summary
    if matlab_results and python_results:
        if matlab_results.get('success') and python_results.get('success'):
            print("\nOutput files:")
            if report_file.exists():
                print(f"  - Report: {report_file}")
            if (output_dir / 'python_results').exists():
                print(f"  - Python results: {output_dir / 'python_results'}")
            if (output_dir / 'matlab_results').exists():
                print(f"  - MATLAB results: {output_dir / 'matlab_results'}")
            print(f"\nNext steps:")
            print(f"  1. View summary: cat {output_dir}/summary.txt")
            print(f"  2. Generate plots: python scripts/visualize_comparison.py --comparison-report {report_file}")
            print(f"  3. See all runs: python scripts/list_comparisons.py")
        else:
            print("\nWARNING: One or both implementations failed. Check logs for details.")
    
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
