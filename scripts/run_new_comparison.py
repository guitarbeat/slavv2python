
import os
import sys
import time
import json
import shutil
import glob
import logging
from datetime import datetime
from pathlib import Path
import subprocess

# Ensure slavv is in path
sys.path.append(os.getcwd())

from slavv.dev.comparison import run_python_vectorization, load_parameters, run_matlab_vectorization
from slavv.dev.matlab_parser import load_matlab_batch_results
from slavv.dev.metrics import compare_results
from slavv.visualization import NetworkVisualizer
from slavv.dev.reporting import generate_summary
from slavv.dev.management import generate_manifest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_matlab():
    """Find the latest MATLAB installation on Windows."""
    base_path = Path(r"C:\Program Files\MATLAB")
    if not base_path.exists():
        return None
    
    # Look for R20* folders
    versions = sorted([d for d in base_path.iterdir() if d.name.startswith("R20")], reverse=True)
    
    for v in versions:
        exe = v / "bin" / "matlab.exe"
        if exe.exists():
            return str(exe)
    
    return None

def main():
    # 1. Setup
    project_root = Path(".")
    input_file = "tests/data/slavv_test_volume.tif"
    
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    # Find MATLAB
    matlab_path = find_matlab()
    if not matlab_path:
        logging.error("MATLAB executable not found in C:\\Program Files\\MATLAB\\R20*\\bin\\matlab.exe")
        logging.error("Please ensure MATLAB is installed or manually provide the path in the script.")
        # Can't proceed with full comparison if MATLAB is missing
        # But user might want to run Python only? The request was "run both".
        return
    logging.info(f"Found MATLAB at: {matlab_path}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"comparisons/{timestamp}_full_run")
    
    logging.info(f"Starting full comparison run in: {base_dir}")
    
    # Define structure
    input_dir = base_dir / "01_Input"
    output_dir = base_dir / "02_Output"
    analysis_dir = base_dir / "03_Analysis"
    
    for d in [input_dir, output_dir, analysis_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Load parameters
    params = load_parameters()
    
    # 2. Run MATLAB
    logging.info("\n" + "="*50)
    logging.info("Step 1: Running MATLAB Vectorization")
    logging.info("="*50)
    
    matlab_results_dir = input_dir / "matlab_results"
    os.makedirs(matlab_results_dir, exist_ok=True)
    
    matlab_run_data = run_matlab_vectorization(
        input_file=input_file,
        output_dir=str(matlab_results_dir),
        matlab_path=matlab_path,
        project_root=project_root
    )
    
    if not matlab_run_data.get('success'):
        logging.error("MATLAB execution failed!")
        return
    
    # Load parsed MATLAB data for VMV generation
    logging.info("Parsing MATLAB results for visualization...")
    try:
        matlab_parsed = load_matlab_batch_results(str(matlab_run_data['batch_folder']))
        matlab_parsed['parameters'] = params
        
        # Generate MATLAB VMV/CASX
        viz = NetworkVisualizer()
        viz.export_network_data(matlab_parsed, output_dir / "matlab_network.vmv", format='vmv')
        viz.export_network_data(matlab_parsed, output_dir / "matlab_network.casx", format='casx')
        logging.info("MATLAB visualization files generated.")
        
    except Exception as e:
        logging.error(f"Failed to process MATLAB results: {e}")
        import traceback
        traceback.print_exc()
        # Continue anyway? Maybe.
    
    # 3. Run Python
    logging.info("\n" + "="*50)
    logging.info("Step 2: Running Python Vectorization")
    logging.info("="*50)
    
    python_results_dir = output_dir / "python_results"
    os.makedirs(python_results_dir, exist_ok=True)
    
    python_run_data = run_python_vectorization(
        input_file=input_file,
        output_dir=str(python_results_dir),
        params=params
    )
    
    if not python_run_data.get('success'):
        logging.error("Python execution failed!")
        return

    # 4. Compare
    logging.info("\n" + "="*50)
    logging.info("Step 3: Comparing Results")
    logging.info("="*50)
    
    try:
        # Construct metadata for comparison
        matlab_meta = {
            'success': True,
            'output_dir': str(matlab_results_dir),
            'batch_folder': str(matlab_run_data['batch_folder']),
            'elapsed_time': matlab_run_data.get('elapsed_time', 0)
        }
        
        comparison = compare_results(matlab_meta, python_run_data, matlab_parsed)
        
        # Save Report
        report_file = analysis_dir / 'comparison_report.json'
        with open(report_file, 'w') as f:
            # Helper to convert numpy to list
            def default(o):
                if isinstance(o, (np.integer, np.int32, np.int64)): return int(o)
                if isinstance(o, (np.floating, np.float32, np.float64)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return str(o)
                
            report = {
                'matlab': comparison['matlab'],
                'python': comparison['python'],
                'performance': comparison['performance'],
                'vertices': comparison.get('vertices', {}),
                'edges': comparison.get('edges', {}),
                'network': comparison.get('network', {})
            }
            json.dump(report, f, indent=2, default=default)
            
        logging.info(f"Comparison report saved to: {report_file}")
        
    except Exception as e:
        logging.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

    logging.info("\nFull Comparison Pipeline Complete!")
    logging.info(f"Results are in: {base_dir}")

if __name__ == "__main__":
    main()
