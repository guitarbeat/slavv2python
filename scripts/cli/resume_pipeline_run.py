"""Resume a SLAVV pipeline run in an existing run directory.

Deprecated: prefer `parity_experiment.py resume-exact-run` for init-exact-run directories.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from slavv_python.analytics.parity.constants import EXPERIMENT_PROVENANCE_PATH
from slavv_python.analytics.parity.resume import resume_exact_run
from slavv_python.engine import SlavvPipeline
from slavv_python.engine.state import load_json_dict
from slavv_python.storage import load_tiff_volume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume SlavvPipeline in an existing run dir.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument(
        "--stop-after",
        default="network",
        help="Pipeline stop stage (default: network).",
    )
    parser.add_argument("--oracle-root")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()

    if (run_dir / EXPERIMENT_PROVENANCE_PATH).is_file():
        resume_exact_run(
            run_dir,
            dataset_root=None,
            oracle_root=Path(args.oracle_root).expanduser().resolve()
            if args.oracle_root
            else None,
            stop_after=args.stop_after,
            skip_preflight=bool(args.skip_preflight),
        )
        print(f"resumed exact-route pipeline in {run_dir}")
        return 0

    dataset_file = args.dataset_file.expanduser().resolve()
    params_path = run_dir / "99_Metadata" / "validated_params.json"
    params = load_json_dict(params_path)
    if params is None:
        raise FileNotFoundError(f"missing validated params: {params_path}")

    image = load_tiff_volume(str(dataset_file))
    SlavvPipeline().run(
        image,
        params,
        run_dir=str(run_dir),
        stop_after=args.stop_after,
    )
    print(f"resumed pipeline in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
