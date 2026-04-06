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
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "source"))

from slavv.evaluation.comparison import (  # noqa: E402
    load_parameters,
    orchestrate_comparison,
    run_standalone_comparison,
)
from slavv.evaluation.management import list_runs  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MATLAB and Python SLAVV implementations")
    parser.add_argument("--input", required=False, help="Input TIFF file path")
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
        "--standalone-matlab-dir",
        help="Existing MATLAB results directory for standalone comparison reuse",
    )
    parser.add_argument(
        "--standalone-python-dir",
        help="Existing Python results directory for standalone comparison reuse",
    )
    parser.add_argument(
        "--skip-matlab", action="store_true", help="Skip MATLAB execution (for testing Python only)"
    )
    parser.add_argument(
        "--skip-python", action="store_true", help="Skip Python execution (for testing MATLAB only)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run output-root preflight checks and exit without executing MATLAB or Python",
    )
    parser.add_argument(
        "--minimal-exports",
        action="store_true",
        help="Reduce Python export workload for comparison runs (skip VMV/CASX/CSV extras)",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Reuse the latest existing comparison run root instead of creating a fresh timestamped one",
    )
    parser.add_argument(
        "--comparison-depth",
        choices=("shallow", "deep"),
        default="deep",
        help="Choose whether comparison should parse full MATLAB batch data or only compute shallow summary metrics",
    )
    parser.add_argument(
        "--python-result-source",
        choices=("auto", "checkpoints-only", "export-json-only", "network-json-only"),
        default="auto",
        help="Choose which Python result surface standalone comparison should trust",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace):
    standalone_mode = bool(args.standalone_matlab_dir or args.standalone_python_dir)

    if standalone_mode and not (args.standalone_matlab_dir and args.standalone_python_dir):
        print(
            "ERROR: --standalone-matlab-dir and --standalone-python-dir must be provided together."
        )
        return 2

    if args.skip_matlab and args.skip_python:
        print("ERROR: --skip-matlab and --skip-python cannot be used together.")
        return 2

    if standalone_mode and (args.skip_matlab or args.skip_python or args.validate_only):
        print(
            "ERROR: standalone comparison cannot be combined with --skip-matlab, --skip-python, or --validate-only."
        )
        return 2

    if not standalone_mode and not args.input:
        print("ERROR: --input is required unless standalone comparison directories are provided.")
        return 2

    if (
        not standalone_mode
        and not args.validate_only
        and not args.skip_matlab
        and not args.matlab_path
    ):
        print("ERROR: --matlab-path is required unless --skip-matlab is set.")
        return 2

    if standalone_mode:
        matlab_dir = Path(args.standalone_matlab_dir)
        python_dir = Path(args.standalone_python_dir)
        if not matlab_dir.exists():
            print(f"ERROR: Standalone MATLAB directory not found: {matlab_dir}")
            return 1
        if not python_dir.exists():
            print(f"ERROR: Standalone Python directory not found: {python_dir}")
            return 1
    else:
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"ERROR: Input file not found: {args.input}")
            return 1
    return None


def _resolve_params_file(user_params):
    if user_params:
        return Path(user_params)
    return Path(__file__).resolve().with_name("comparison_params.json")


def _is_run_root(path: Path) -> bool:
    return any(
        (path / child).exists() for child in ("01_Input", "02_Output", "03_Analysis", "99_Metadata")
    )


def _find_latest_run_root(base_dir: Path) -> Optional[Path]:
    runs = list_runs(base_dir)
    if not runs:
        return None
    latest = runs[0].get("path")
    return latest if isinstance(latest, Path) else None


def _resolve_output_dir(user_output_dir, *, resume_latest: bool = False):
    if user_output_dir:
        requested = Path(user_output_dir)
        if not resume_latest:
            return requested
        if _is_run_root(requested):
            return requested
        latest = _find_latest_run_root(requested)
        if latest is not None:
            return latest
        return requested
    if resume_latest:
        latest = _find_latest_run_root(Path("comparisons"))
        if latest is not None:
            return latest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("comparisons") / f"{timestamp}_comparison"


def main():
    parser = _build_parser()

    args = parser.parse_args()

    validation_error = _validate_cli_args(args)
    if validation_error is not None:
        return validation_error

    output_dir = _resolve_output_dir(args.output_dir, resume_latest=args.resume_latest)
    print(f"Output directory: {output_dir}")

    if args.standalone_matlab_dir and args.standalone_python_dir:
        return run_standalone_comparison(
            matlab_dir=Path(args.standalone_matlab_dir),
            python_dir=Path(args.standalone_python_dir),
            output_dir=output_dir,
            project_root=project_root,
            python_result_source=args.python_result_source,
            comparison_depth=args.comparison_depth,
        )

    params_file = _resolve_params_file(args.params)
    if not params_file.is_file():
        print(f"ERROR: Parameters file not found: {params_file}")
        return 1

    try:
        params = load_parameters(str(params_file))
    except (OSError, ValueError) as exc:
        print(f"ERROR: Failed to load parameters from {params_file}: {exc}")
        return 1

    # Run orchestration
    return orchestrate_comparison(
        input_file=args.input,
        output_dir=output_dir,
        matlab_path=args.matlab_path or "",
        project_root=project_root,
        params=params,
        skip_matlab=args.skip_matlab,
        skip_python=args.skip_python,
        validate_only=args.validate_only,
        minimal_exports=args.minimal_exports,
        comparison_depth=args.comparison_depth,
    )


if __name__ == "__main__":
    sys.exit(main())
