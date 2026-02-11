
import sys
import os
import json
import logging
import time
import glob
from pathlib import Path
import copy
import joblib
import numpy as np

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.absolute()
sys.path.append(str(project_root))

from slavv.dev.matlab_parser import load_matlab_batch_results
from slavv.dev.metrics import compare_results
from slavv.dev.reporting import generate_summary
from slavv.dev.management import generate_manifest

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def load_python_results_from_checkpoints(output_dir):
    """Reconstruct Python results from checkpoints."""
    checkpoint_dir = output_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    results = {}
    
    # Load Vertices
    v_path = checkpoint_dir / 'checkpoint_vertices.pkl'
    if v_path.exists():
        logger.info(f"Loading vertices from {v_path}")
        results['vertices'] = joblib.load(v_path)
    else:
        logger.warning("Vertices checkpoint not found")
        results['vertices'] = {'positions': []}

    # Load Edges
    e_path = checkpoint_dir / 'checkpoint_edges.pkl'
    if e_path.exists():
        logger.info(f"Loading edges from {e_path}")
        results['edges'] = joblib.load(e_path)
    else:
        logger.warning("Edges checkpoint not found")
        results['edges'] = {'traces': []}

    # Load Network
    n_path = checkpoint_dir / 'checkpoint_network.pkl'
    if n_path.exists():
        logger.info(f"Loading network from {n_path}")
        results['network'] = joblib.load(n_path)
    else:
        logger.warning("Network checkpoint not found")
        results['network'] = {'strands': []}

    return results

def main():
    # Define paths based on the manual run timestamp
    # We could parse this from args, but for this task we know the specific run
    # To be generic, let's find the latest manual run folder
    comparisons_dir = project_root / 'comparisons'
    
    # Find most recent run folder
    run_folders = sorted([d for d in comparisons_dir.iterdir() if d.is_dir() and 'manual_run' in d.name])
    if not run_folders:
        logger.error("No manual run folder found in comparisons directory")
        return 1
    
    run_dir = run_folders[-1]
    logger.info(f"Using run directory: {run_dir}")

    matlab_dir = run_dir / '01_Input' / 'matlab_results'
    python_dir = run_dir / '02_Output' / 'python_results'
    analysis_dir = run_dir / '03_Analysis'
    
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. Load MATLAB Results
    matlab_results = {
        'success': False,
        'output_dir': str(matlab_dir),
        'elapsed_time': 0.0
    }
    
    try:
        # Find batch folder
        batch_folders = [d for d in matlab_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
        if batch_folders:
            batch_folder = sorted(batch_folders)[-1]
            logger.info(f"Found MATLAB batch folder: {batch_folder}")
            
            matlab_parsed = load_matlab_batch_results(batch_folder)
            matlab_results['batch_folder'] = str(batch_folder)
            matlab_results['success'] = True
            
            # Try to get timing from parsed data
            if 'timings' in matlab_parsed:
                matlab_results['elapsed_time'] = matlab_parsed['timings'].get('total', 0.0)
        else:
            logger.error("No MATLAB batch folder found!")
            matlab_parsed = None
            
    except Exception as e:
        logger.error(f"Failed to load MATLAB results: {e}")
        matlab_parsed = None


    # 2. Load Python Results
    python_results = {
        'success': False,
        'output_dir': str(python_dir),
        'elapsed_time': 0.0 # Unknown without logs/metadata
    }
    
    # Try checkpoints first as they are most reliable for this specific run state
    try:
        results_dict = load_python_results_from_checkpoints(python_dir)
        if results_dict:
            python_results['results'] = results_dict
            python_results['vertices_count'] = len(results_dict.get('vertices', {}).get('positions', []))
            python_results['edges_count'] = len(results_dict.get('edges', {}).get('traces', []))
            python_results['network_strands_count'] = len(results_dict.get('network', {}).get('strands', []))
            python_results['success'] = True
            logger.info("Successfully loaded Python results from checkpoints")
        else:
             logger.error("Failed to load Python results from checkpoints")
             
    except Exception as e:
        logger.error(f"Error loading Python checkpoints: {e}")

    # 3. Run Comparison
    if matlab_results['success'] and python_results['success']:
        logger.info("Running comparison metrics...")
        comparison = compare_results(matlab_results, python_results, matlab_parsed)
        
        # 4. Save Report
        report_file = analysis_dir / 'comparison_report.json'
        
        # Serialize helper
        def default_serializer(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            return str(o)

        with open(report_file, 'w') as f:
            report = {
                'matlab': copy.deepcopy(comparison['matlab']),
                'python': copy.deepcopy(comparison['python']),
                'performance': comparison['performance'],
                'vertices': comparison.get('vertices', {}),
                'edges': comparison.get('edges', {}),
                'network': comparison.get('network', {})
            }
            json.dump(report, f, indent=2, default=default_serializer)
        
        logger.info(f"Comparison report saved to: {report_file}")
        
        # 5. Generate Summary
        generate_summary(run_dir, analysis_dir / 'summary.txt')
        generate_manifest(run_dir, analysis_dir / 'MANIFEST.md')
        
        return 0
    else:
        logger.error("Cannot run comparison: Missing MATLAB or Python results.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
