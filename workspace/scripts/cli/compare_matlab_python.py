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
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "source"))

from slavv.evaluation.comparison import load_parameters, orchestrate_comparison  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MATLAB and Python SLAVV implementations")
    parser.add_argument("--input", required=True, help="Input TIFF file path")
    parser.add_argument(
        "--matlab-path",
        required=False,
        help="Path to MATLAB executable (e.g., C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: comparisons/YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--params",
        help="JSON file with parameters (default: workspace/scripts/cli/comparison_params.json)",
    )
    parser.add_argument(
        "--skip-matlab", action="store_true", help="Skip MATLAB execution (for testing Python only)"
    )
    parser.add_argument(
        "--skip-python", action="store_true", help="Skip Python execution (for testing MATLAB only)"
    )
    return parser


def _validate_cli_args(args: argparse.Namespace):
    if args.skip_matlab and args.skip_python:
        print("ERROR: --skip-matlab and --skip-python cannot be used together.")
        return 2

    if not args.skip_matlab and not args.matlab_path:
        print("ERROR: --matlab-path is required unless --skip-matlab is set.")
        return 2

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    return None


def _resolve_params_file(user_params):
    if user_params:
        return Path(user_params)
    return Path(__file__).resolve().with_name("comparison_params.json")


def _resolve_output_dir(user_output_dir):
    if user_output_dir:
        return Path(user_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("comparisons") / f"{timestamp}_comparison"


def main():
    parser = _build_parser()

    args = parser.parse_args()

    validation_error = _validate_cli_args(args)
    if validation_error is not None:
        return validation_error

    params_file = _resolve_params_file(args.params)
    if not params_file.is_file():
        print(f"ERROR: Parameters file not found: {params_file}")
        return 1

    try:
        params = load_parameters(str(params_file))
    except (OSError, ValueError) as exc:
        print(f"ERROR: Failed to load parameters from {params_file}: {exc}")
        return 1

    output_dir = _resolve_output_dir(args.output_dir)

    print(f"Output directory: {output_dir}")

    # Run orchestration
    return orchestrate_comparison(
        input_file=args.input,
        output_dir=output_dir,
        matlab_path=args.matlab_path or "",
        project_root=project_root,
        params=params,
        skip_matlab=args.skip_matlab,
        skip_python=args.skip_python,
    )


if __name__ == "__main__":
    sys.exit(main())
