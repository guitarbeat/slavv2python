"""Resume a SLAVV pipeline run in an existing run directory.

Used when init-exact-run refuses because run_snapshot.json still shows status=running
but no process is active. Energy and other resumable stages continue from checkpoints.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
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
