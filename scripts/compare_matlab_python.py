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
from src.slavv.visualization import NetworkVisualizer

from src.slavv.visualization import NetworkVisualizer

# Import dev tools
from src.slavv.dev.matlab_parser import load_matlab_results_from_output_dir, load_matlab_batch_results
from src.slavv.dev.metrics import compare_vertices, compare_edges, compare_networks, compare_results


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
    
    # Define the path to the MATLAB repository
    # This assumes the 'legacy' folder is at the project root level
    MATLAB_REPO_PATH = project_root / 'legacy' / 'Vectorization-Public'
    
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
        
        # Export VMV and CASX formats for visualization
        print("Exporting VMV and CASX formats...")
        try:
            visualizer = NetworkVisualizer()
            vmv_path = os.path.join(output_dir, "network.vmv")
            casx_path = os.path.join(output_dir, "network.casx")
            csv_base = os.path.join(output_dir, "network")
            json_path = os.path.join(output_dir, "network.json")
            
            visualizer.export_network_data(results, vmv_path, format='vmv')
            print(f"  VMV export: {vmv_path}")
            
            visualizer.export_network_data(results, casx_path, format='casx')
            print(f"  CASX export: {casx_path}")
            
            visualizer.export_network_data(results, csv_base, format='csv')
            print(f"  CSV export: {csv_base}_vertices.csv, {csv_base}_edges.csv")
            
            visualizer.export_network_data(results, json_path, format='json')
            print(f"  JSON export: {json_path}")
            
            python_results['exports'] = {
                'vmv': vmv_path,
                'casx': casx_path,
                'csv': csv_base,
                'json': json_path
            }
        except Exception as e:
            print(f"  Warning: Export failed: {e}")
            import traceback
            traceback.print_exc()
        
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
        from src.slavv.dev.reporting import generate_summary
        summary_file = output_dir / 'summary.txt'
        generate_summary(output_dir, summary_file)
    except Exception as e:
        print(f"Note: Could not auto-generate summary: {e}")
    
    # Generate manifest automatically
    try:
        from generate_comparison_manifest import generate_manifest
        manifest_file = output_dir / 'MANIFEST.md'
        generate_manifest(output_dir, manifest_file)
        print(f"Manifest generated: {manifest_file}")
    except Exception as e:
        print(f"Note: Could not auto-generate manifest: {e}")
    
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
