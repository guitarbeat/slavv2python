
"""
Comparison execution module.

This module handles the execution of MATLAB and Python vectorization pipelines
for comparison purposes.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from source.slavv.core import SLAVVProcessor
from source.slavv.io import load_tiff_volume, export_pipeline_results
from source.slavv.visualization import NetworkVisualizer
from source.slavv.dev.matlab_parser import load_matlab_batch_results
from source.slavv.dev.metrics import compare_results
from source.slavv.dev.reporting import generate_summary
from source.slavv.dev.management import generate_manifest
from source.slavv.utils import get_system_info, get_matlab_info

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
    project_root: Path,
    batch_script: str = None
) -> Dict[str, Any]:
    """Run MATLAB vectorization via CLI."""
    print("\n" + "="*60)
    print("Running MATLAB Implementation")
    print("="*60)
    
    # Define the path to the MATLAB repository
    MATLAB_REPO_PATH = project_root / 'external' / 'Vectorization-Public'
    
    if batch_script is None:
        batch_script = str(project_root / 'scripts' / 'run_matlab_cli.bat')
    
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
    
    # Capture system info
    system_info = get_system_info()
    matlab_info = get_matlab_info(matlab_path)
    system_info['matlab'] = matlab_info
    
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
            'stderr': result.stderr,
            'system_info': system_info
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
            'stderr': e.stderr,
            'system_info': system_info
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
    
    # Capture system info
    system_info = get_system_info()
    
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
            'results': results,
            'system_info': system_info
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
            'error': str(e),
            'system_info': system_info
        }

def orchestrate_comparison(
    input_file: str,
    output_dir: Path,
    matlab_path: str,
    project_root: Path,
    params: Dict[str, Any],
    skip_matlab: bool = False,
    skip_python: bool = False
) -> int:
    """Run full comparison workflow."""
    
    matlab_output = output_dir / 'matlab_results'
    python_output = output_dir / 'python_results'
    
    os.makedirs(matlab_output, exist_ok=True)
    os.makedirs(python_output, exist_ok=True)
    
    # Run MATLAB
    matlab_results = None
    if not skip_matlab:
        matlab_results = run_matlab_vectorization(
            input_file,
            str(matlab_output),
            matlab_path,
            project_root
        )
    else:
        print("\nSkipping MATLAB execution (--skip-matlab)")
    
    # Run Python
    python_results = None
    if not skip_python:
        python_results = run_python_vectorization(
            input_file,
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
        summary_file = output_dir / 'summary.txt'
        generate_summary(output_dir, summary_file)
    except Exception as e:
        print(f"Note: Could not auto-generate summary: {e}")
    
    # Generate manifest automatically
    try:
        manifest_file = output_dir / 'MANIFEST.md'
        generate_manifest(output_dir, manifest_file)
        print(f"Manifest generated: {manifest_file}")
    except Exception as e:
        print(f"Note: Could not auto-generate manifest: {e}")
    
    # Print final summary status
    # Print final summary status
    if matlab_results and python_results:
        success = matlab_results.get('success') and python_results.get('success')
        return 0 if success else 1
    return 0


def run_standalone_comparison(
    matlab_dir: Path,
    python_dir: Path,
    output_dir: Path,
    project_root: Path
) -> int:
    """
    Run comparison on existing result directories.
    
    Args:
        matlab_dir: Directory containing MATLAB results (e.g., .../TIMESTAMP_matlab_run)
        python_dir: Directory containing Python results (e.g., .../TIMESTAMP_python_run)
        output_dir: Directory to save comparison results
        project_root: Root of the repository
    
    Returns:
        0 if successful, 1 otherwise
    """
    print("\n" + "="*60)
    print("Running Standalone Comparison")
    print("="*60)
    print(f"MATLAB Dir: {matlab_dir}")
    print(f"Python Dir: {python_dir}")
    print(f"Output Dir: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Reconstruct MATLAB results dict
    matlab_results = {
        'success': True, # Assume success if we are selecting it
        'output_dir': str(matlab_dir),
        'elapsed_time': 0.0 # Unknown
    }
    
    # Find batch folder
    batch_folders = [
        d for d in os.listdir(matlab_dir)
        if os.path.isdir(os.path.join(matlab_dir, d)) and d.startswith('batch_')
    ]
    
    if batch_folders:
        batch_folder = os.path.join(matlab_dir, sorted(batch_folders)[-1])
        matlab_results['batch_folder'] = batch_folder
        print(f"Found MATLAB batch folder: {batch_folder}")
    else:
        # Check inside matlab_results subdir if it exists (for combined runs)
        sub_matlab = matlab_dir / 'matlab_results'
        if sub_matlab.exists():
            batch_folders_sub = [
                d for d in os.listdir(sub_matlab)
                if os.path.isdir(os.path.join(sub_matlab, d)) and d.startswith('batch_')
            ]
            if batch_folders_sub:
                batch_folder = os.path.join(sub_matlab, sorted(batch_folders_sub)[-1])
                matlab_results['batch_folder'] = batch_folder
                print(f"Found MATLAB batch folder (in subdir): {batch_folder}")
            else:
                 print("Warning: No MATLAB batch_* folder found.")
        else:
            print("Warning: No MATLAB batch_* folder found.")

    # 2. Reconstruct Python results dict
    python_results = {
        'success': True,
        'output_dir': str(python_dir),
        'elapsed_time': 0.0
    }
    
    # Load python results
    # Try finding python_comparison_*.json
    import glob
    # Check root of python dir
    json_files = glob.glob(str(python_dir / "python_comparison_*.json"))
    # Check python_results subdir
    if not json_files:
        json_files = glob.glob(str(python_dir / "python_results" / "python_comparison_*.json"))
    
    if json_files:
        # Load the latest one
        latest_json = sorted(json_files)[-1]
        print(f"Loading Python results from: {latest_json}")
        try:
            with open(latest_json, 'r') as f:
                loaded_data = json.load(f)
            
            # Need to convert lists back to numpy arrays for metric comparison
            # This is a simplified reconstruction. For full fidelity we'd need convert_lists_to_arrays
            # But compare_results should handle basic lists or we can do a quick pass
            
            def recursive_restore(d):
                if isinstance(d, dict):
                    return {k: recursive_restore(v) for k, v in d.items()}
                elif isinstance(d, list):
                    # Heuristic: if list of numbers, make array?
                    # Actually compare_vertices expects arrays for 'positions'
                    # Let's hope numpy array conversion happens or is tolerant
                    return np.array(d)
                return d

            # Specifically restore vertices/positions and edges/traces
            if 'vertices' in loaded_data and 'positions' in loaded_data['vertices']:
                loaded_data['vertices']['positions'] = np.array(loaded_data['vertices']['positions'])
                if 'radii' in loaded_data['vertices']:
                    loaded_data['vertices']['radii'] = np.array(loaded_data['vertices']['radii'])
            
            # Edges traces are list of arrays
            if 'edges' in loaded_data and 'traces' in loaded_data['edges']:
                loaded_data['edges']['traces'] = [np.array(t) for t in loaded_data['edges']['traces']]

            python_results['results'] = loaded_data
            python_results['vertices_count'] = len(loaded_data.get('vertices', {}).get('positions', []))
            python_results['edges_count'] = len(loaded_data.get('edges', {}).get('traces', []))
            
        except Exception as e:
            print(f"Error loading Python JSON: {e}")
            python_results['success'] = False
    else:
        print("Warning: No python_comparison_*.json found.")
        python_results['success'] = False

    # 3. Compare
    # Try to load parsed MATLAB data
    matlab_parsed = None
    if matlab_results.get('batch_folder'):
        print("\nLoading MATLAB output data...")
        try:
            matlab_parsed = load_matlab_batch_results(matlab_results['batch_folder'])
            print(f"Successfully loaded MATLAB data")
        except Exception as e:
            print(f"Warning: Could not load MATLAB output data: {e}")
    
    comparison = compare_results(matlab_results, python_results, matlab_parsed)
    
    # 4. Save
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
    
    # Generate summary
    try:
        summary_file = output_dir / 'summary.txt'
        generate_summary(output_dir, summary_file)
    except Exception as e:
        print(f"Note: Could not auto-generate summary: {e}")
        
    return 0
