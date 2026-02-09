import os
import sys
from pathlib import Path
import json
import numpy as np
import copy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure slavv is in path
sys.path.append(os.getcwd())

from slavv.dev.comparison import run_python_vectorization, load_parameters
from slavv.dev.matlab_parser import load_matlab_batch_results
from slavv.dev.metrics import compare_results
from slavv.visualization import NetworkVisualizer

def main():
    # Paths
    project_root = Path(".")
    base_dir = Path("comparisons/20260206_173559_matlab_run")
    matlab_batch = base_dir / "01_Input/matlab_results/batch_260206-173729"
    
    # Output Directories
    output_vis_dir = base_dir / "02_Output"
    output_analysis_dir = base_dir / "03_Analysis"
    input_file = "tests/data/slavv_test_volume.tif"
    
    # Ensure params exist
    os.makedirs(output_vis_dir, exist_ok=True)
    os.makedirs(output_analysis_dir, exist_ok=True)
    
    # Check if paths exist
    if not matlab_batch.exists():
        logging.error(f"MATLAB batch folder not found: {matlab_batch}")
        return
    
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    # 1. Load MATLAB Output & Generate VMV
    logging.info("Loading MATLAB data...")
    try:
        matlab_data = load_matlab_batch_results(str(matlab_batch))
        
        # Inject parameters for visualization
        params = load_parameters()
        matlab_data['parameters'] = params
        
        logging.info("MATLAB data loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load MATLAB data: {e}")
        return

    logging.info("Generating MATLAB VMV...")
    try:
        viz = NetworkVisualizer()
        vmv_path = output_vis_dir / "matlab_network.vmv"
        casx_path = output_vis_dir / "matlab_network.casx"
        
        viz.export_network_data(matlab_data, vmv_path, format='vmv')
        logging.info(f"MATLAB VMV exported to: {vmv_path}")
        
        viz.export_network_data(matlab_data, casx_path, format='casx')
        logging.info(f"MATLAB CASX exported to: {casx_path}")
    except Exception as e:
        logging.error(f"Failed to export VMV/CASX: {e}")
        # Continue anyway to run python comparison?
        # Yes, let's continue.

    # 2. Run Python
    logging.info("Running Python vectorization...")
    python_output_dir = output_vis_dir / "python_results"
    os.makedirs(python_output_dir, exist_ok=True)
    
    try:
        params = load_parameters() # Defaults
        python_results = run_python_vectorization(input_file, str(python_output_dir), params)
    except Exception as e:
        logging.error(f"Python vectorization failed: {e}")
        return

    # 3. Compare
    logging.info("Comparing results...")
    
    # Construct metadata wrapper for MATLAB results expected by compare_results
    matlab_results_meta = {
        'success': True,
        'output_dir': str(matlab_batch.parent),
        'batch_folder': str(matlab_batch),
        'elapsed_time': matlab_data['timings'].get('total', 0)
    }

    try:
        comparison = compare_results(matlab_results_meta, python_results, matlab_data)
        
        # 4. Save Report
        report_file = output_analysis_dir / 'comparison_report.json'
        with open(report_file, 'w') as f:
            report = {
                'matlab': copy.deepcopy(comparison['matlab']),
                'python': copy.deepcopy(comparison['python']),
                'performance': comparison['performance'],
                'vertices': comparison.get('vertices', {}),
                'edges': comparison.get('edges', {}),
                'network': comparison.get('network', {})
            }
            json.dump(report, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
            
        logging.info(f"Comparison report saved to: {report_file}")
        
    except Exception as e:
        logging.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

