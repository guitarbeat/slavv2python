#!/usr/bin/env python3
"""
Compare MATLAB and Python implementations of SLAVV vectorization.

This script runs both implementations with identical parameters and compares:
- Runtime performance
- Vertex counts and statistics
- Edge counts and statistics
- Network statistics

Usage:
    python workspace/scripts/cli/compare_matlab_python.py \
        --input "data/slavv_test_volume.tif" \
        --matlab-path "C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe" \
        --output-dir "comparison_output"
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'source'))

from slavv.evaluation.comparison import load_parameters, orchestrate_comparison

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
        help='JSON file with parameters (default: workspace/scripts/cli/comparison_params.json)'
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

    if not os.path.exists(params_file):
        print(f"ERROR: Parameters file not found: {params_file}")
        return 1

    params = load_parameters(params_file)

    # Create output directories with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('comparisons') / f'{timestamp}_comparison'
    else:
        output_dir = Path(args.output_dir)

    print(f"Output directory: {output_dir}")

    # Run orchestration
    return orchestrate_comparison(
        input_file=args.input,
        output_dir=output_dir,
        matlab_path=args.matlab_path,
        project_root=project_root,
        params=params,
        skip_matlab=args.skip_matlab,
        skip_python=args.skip_python
    )

if __name__ == '__main__':
    sys.exit(main())
