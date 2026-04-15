"""Packaged CLI for MATLAB/Python parity and comparison workflows."""

from __future__ import annotations

# ruff: noqa: UP045
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from slavv.parity.comparison import (
    load_parameters,
    orchestrate_comparison,
    run_matlab_health_check_workflow,
    run_standalone_comparison,
)
from slavv.parity.run_layout import list_runs
from slavv.parity.workflow_assessment import assess_loop_request, determine_loop_kind
from slavv.runtime import load_run_snapshot

warnings.filterwarnings("ignore", category=DeprecationWarning)
project_root = Path(__file__).resolve().parents[3]
DEFAULT_COMPARISON_PARAMS = project_root / "dev" / "scripts" / "cli" / "comparison_params.json"


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
        help="JSON file with parameters (default: dev/scripts/cli/comparison_params.json)",
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
        "--matlab-health-check",
        action="store_true",
        help="Run a lightweight MATLAB launch probe after output-root preflight and exit",
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
    parser.add_argument(
        "--python-parity-rerun-from",
        choices=("edges", "network"),
        default="edges",
        help=(
            "Choose the Python stage to rerun after importing reusable MATLAB checkpoints. "
            "'edges' is the default imported-MATLAB parity loop; 'network' is the stage-isolated "
            "MATLAB-edges-to-Python-network probe."
        ),
    )
    return parser


def _validate_cli_args(args: argparse.Namespace):
    standalone_mode = bool(args.standalone_matlab_dir or args.standalone_python_dir)

    if args.matlab_health_check:
        disallowed = [
            standalone_mode,
            bool(args.input),
            args.skip_matlab,
            args.skip_python,
            args.validate_only,
        ]
        if any(disallowed):
            print(
                "ERROR: --matlab-health-check cannot be combined with run, standalone, skip, or validate-only flags."
            )
            return 2
        if not args.output_dir:
            print("ERROR: --output-dir is required when --matlab-health-check is used.")
            return 2
        if not args.matlab_path:
            print("ERROR: --matlab-path is required when --matlab-health-check is used.")
            return 2
        return None

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

    if args.skip_python and args.python_parity_rerun_from != "edges":
        print(
            "ERROR: --python-parity-rerun-from is only meaningful when Python execution is enabled."
        )
        return 2

    if standalone_mode and args.python_parity_rerun_from != "edges":
        print("ERROR: --python-parity-rerun-from cannot be used with standalone comparison mode.")
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
    return DEFAULT_COMPARISON_PARAMS


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


def _normalize_resume_path(path: Union[Path, str]) -> str:
    normalized = Path(path)
    try:
        normalized = normalized.resolve()
    except OSError:
        normalized = normalized.absolute()
    return str(normalized).replace("\\", "/").lower()


def _normalize_resume_value(value):
    if isinstance(value, dict):
        return {str(key): _normalize_resume_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_resume_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except TypeError:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _normalize_comparison_request_params(
    params: dict | None, python_parity_rerun_from: str
) -> dict | None:
    if params is None:
        return None
    normalized = dict(params)
    normalized["comparison_exact_network"] = True
    normalized.setdefault("python_parity_rerun_from", python_parity_rerun_from)
    return normalized


def _load_recorded_comparison_params(run_root: Path):
    params_path = run_root / "99_Metadata" / "comparison_params.normalized.json"
    if not params_path.exists():
        return None
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["comparison_exact_network"] = True
    params.setdefault("python_parity_rerun_from", "edges")
    return params


def _has_reusable_matlab_artifacts(run_root: Path) -> bool:
    matlab_dir = run_root / "01_Input" / "matlab_results"
    if not matlab_dir.exists():
        return False
    return any(child.name.startswith("batch_") for child in matlab_dir.iterdir())


def _has_reusable_python_checkpoints(run_root: Path) -> bool:
    checkpoint_dir = run_root / "02_Output" / "python_results" / "checkpoints"
    if not checkpoint_dir.exists():
        return False
    return any(checkpoint_dir.glob("checkpoint_*.pkl"))


def _has_required_resume_artifacts(
    run_root: Path,
    *,
    standalone_mode: bool,
    validate_only: bool,
    skip_matlab: bool,
    skip_python: bool,
) -> tuple[bool, str]:
    if standalone_mode or validate_only:
        return True, "no staged artifact requirement for this mode"
    if skip_matlab and not skip_python:
        if _has_reusable_python_checkpoints(run_root):
            return True, "python checkpoint surface is reusable"
        return False, "missing reusable Python checkpoints for a skip-matlab parity rerun"
    if skip_python and not skip_matlab:
        if _has_reusable_matlab_artifacts(run_root):
            return True, "matlab batch surface is reusable"
        return False, "missing reusable MATLAB batch artifacts for a skip-python rerun"
    if _has_reusable_matlab_artifacts(run_root):
        return True, "matlab batch surface is reusable"
    return False, "missing reusable MATLAB batch artifacts for a full comparison rerun"


def _is_resume_candidate_compatible(
    run_root: Path,
    input_path: Optional[Path],
    params: Optional[dict] = None,
) -> tuple[bool, str]:
    if input_path is None:
        return True, "no input compatibility check required"
    snapshot = load_run_snapshot(run_root)
    if snapshot is None:
        return False, "missing run snapshot"
    recorded_input = snapshot.provenance.get("input_file")
    if not recorded_input:
        return False, "missing recorded input provenance"
    if _normalize_resume_path(recorded_input) != _normalize_resume_path(input_path):
        return False, f"recorded input '{recorded_input}' does not match '{input_path}'"
    if params is not None:
        recorded_params = _load_recorded_comparison_params(run_root)
        if recorded_params is None:
            return False, "missing normalized comparison params"
        if _normalize_resume_value(recorded_params) != _normalize_resume_value(params):
            return False, "recorded comparison parameters do not match the current request"
    return True, "input provenance matches"


def _build_fresh_output_dir(base_dir: Optional[Path] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = base_dir or Path("comparisons")
    return parent / f"{timestamp}_comparison"


def _loop_kind_for_resolution(
    *,
    standalone_mode: bool,
    validate_only: bool,
    skip_matlab: bool,
    skip_python: bool,
    python_parity_rerun_from: str = "edges",
) -> str:
    return determine_loop_kind(
        standalone_mode=standalone_mode,
        validate_only=validate_only,
        skip_matlab=skip_matlab,
        skip_python=skip_python,
        python_parity_rerun_from=python_parity_rerun_from,
    )


def _resolve_output_dir(
    user_output_dir,
    *,
    resume_latest: bool = False,
    input_path: Optional[Path] = None,
    params: Optional[dict] = None,
    standalone_mode: bool = False,
    validate_only: bool = False,
    skip_matlab: bool = False,
    skip_python: bool = False,
    python_parity_rerun_from: str = "edges",
):
    loop_kind = _loop_kind_for_resolution(
        standalone_mode=standalone_mode,
        validate_only=validate_only,
        skip_matlab=skip_matlab,
        skip_python=skip_python,
        python_parity_rerun_from=python_parity_rerun_from,
    )

    def select_compatible_run(base_dir: Path) -> Path | None:
        latest_blocking_reason: str | None = None
        if loop_kind in {"skip_matlab_edges", "skip_matlab_network"}:
            reusable_verdicts = {"reuse_ready"}
        elif loop_kind in {"standalone_analysis", "validate_only"}:
            reusable_verdicts = {"analysis_ready"}
        else:
            reusable_verdicts = {"analysis_ready", "reuse_ready", "fresh_matlab_required"}
        for index, entry in enumerate(list_runs(base_dir)):
            candidate = entry.get("path")
            if not isinstance(candidate, Path):
                continue
            assessment = assess_loop_request(
                candidate,
                loop_kind=loop_kind,
                input_path=input_path,
                params=params,
            )
            if assessment.verdict in reusable_verdicts:
                print(f"Reusing latest compatible run root: {candidate}")

                # Display reuse eligibility summary after resume-latest compatibility check
                from slavv.parity.cli_summaries import (
                    format_reuse_eligibility_summary,
                    generate_reuse_commands,
                )

                if input_path:
                    assessment.reuse_commands = generate_reuse_commands(
                        assessment,
                        run_root=candidate,
                        input_file=input_path,
                    )

                summary = format_reuse_eligibility_summary(
                    assessment,
                    run_root=candidate,
                    input_file=input_path if input_path else Path(""),
                )
                print("\n" + summary)

                return candidate
            if index == 0:
                latest_blocking_reason = (
                    assessment.reasons[0]
                    if assessment.reasons
                    else assessment.artifact_reason or assessment.compatibility_reason
                )
        if latest_blocking_reason:
            print(
                "Latest discovered run root is not compatible with the current input; "
                "starting a fresh run root instead "
                f"({latest_blocking_reason})."
            )
        return None

    if user_output_dir:
        requested = Path(user_output_dir)
        if not resume_latest:
            return requested
        if _is_run_root(requested):
            return requested
        latest = select_compatible_run(requested)
        if latest is not None:
            return latest
        return _build_fresh_output_dir(requested)
    if resume_latest:
        latest = select_compatible_run(Path("comparisons"))
        if latest is not None:
            return latest
    return _build_fresh_output_dir(Path(user_output_dir) if user_output_dir else None)


def main():
    parser = _build_parser()

    args = parser.parse_args()

    validation_error = _validate_cli_args(args)
    if validation_error is not None:
        return validation_error

    if args.standalone_matlab_dir and args.standalone_python_dir:
        output_dir = _resolve_output_dir(
            args.output_dir,
            resume_latest=args.resume_latest,
            input_path=Path(args.input) if args.input else None,
            standalone_mode=True,
            validate_only=args.validate_only,
            skip_matlab=args.skip_matlab,
            skip_python=args.skip_python,
            python_parity_rerun_from=args.python_parity_rerun_from,
        )
        print(f"Output directory: {output_dir}")
        return run_standalone_comparison(
            matlab_dir=Path(args.standalone_matlab_dir),
            python_dir=Path(args.standalone_python_dir),
            output_dir=output_dir,
            project_root=project_root,
            python_result_source=args.python_result_source,
            comparison_depth=args.comparison_depth,
        )

    if args.matlab_health_check:
        output_dir = Path(args.output_dir)
        print(f"Output directory: {output_dir}")
        return run_matlab_health_check_workflow(
            output_dir=output_dir,
            matlab_path=args.matlab_path,
            project_root=project_root,
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

    compatibility_params = _normalize_comparison_request_params(
        params, args.python_parity_rerun_from
    )

    output_dir = _resolve_output_dir(
        args.output_dir,
        resume_latest=args.resume_latest,
        input_path=Path(args.input) if args.input else None,
        params=compatibility_params,
        standalone_mode=False,
        validate_only=args.validate_only,
        skip_matlab=args.skip_matlab,
        skip_python=args.skip_python,
        python_parity_rerun_from=args.python_parity_rerun_from,
    )
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
        validate_only=args.validate_only,
        minimal_exports=args.minimal_exports,
        comparison_depth=args.comparison_depth,
        python_result_source=args.python_result_source,
        python_parity_rerun_from=args.python_parity_rerun_from,
    )


if __name__ == "__main__":
    raise SystemExit(main())
