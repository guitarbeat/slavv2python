"""Batch one-voxel probes for crop Energy mismatch groups or regression fixtures.

Thin CLI wrapper around :mod:`tests.support.parity_harness`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slavv_python.analytics.parity.proof.energy_proof_evidence import require_energy_proof_evidence
from tests.support.parity_harness import (
    REPO_ROOT,
    export_regression_probe_jsonl,
    run_mismatch_group_batch,
    run_voxel_regression_fixture,
    select_mismatch_group_requests,
)

DEFAULT_PROBE_REQUESTS = (
    REPO_ROOT / "workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/energy_probe_requests.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/batch_voxel_probe.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch Energy voxel parity probes")
    parser.add_argument(
        "--mode",
        choices=("mismatch-groups", "regression", "export-jsonl", "export-requests"),
        default="mismatch-groups",
        help="mismatch-groups: Python probes; export-requests: MATLAB batch input; regression: fixture",
    )
    parser.add_argument("--probe-requests", type=Path, default=DEFAULT_PROBE_REQUESTS)
    parser.add_argument("--top-groups", type=int, default=15)
    parser.add_argument("--coordinates-per-group", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.mode == "export-jsonl":
        export_regression_probe_jsonl(args.output)
        print(f"exported probe JSONL: {args.output}")
        return 0

    if args.mode == "export-requests":
        if not args.probe_requests.is_file():
            raise FileNotFoundError(f"probe requests not found: {args.probe_requests}")
        require_energy_proof_evidence(args.probe_requests.parent.parent)
        summary = select_mismatch_group_requests(
            args.probe_requests,
            top_groups=args.top_groups,
            coordinates_per_group=args.coordinates_per_group,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json

        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"exported {len(summary['requests'])} batch requests: {args.output}")
        return 0

    if args.mode == "regression":
        summary = run_voxel_regression_fixture()
    else:
        if not args.probe_requests.is_file():
            raise FileNotFoundError(f"probe requests not found: {args.probe_requests}")
        require_energy_proof_evidence(args.probe_requests.parent.parent)
        summary = run_mismatch_group_batch(
            args.probe_requests,
            top_groups=args.top_groups,
            coordinates_per_group=args.coordinates_per_group,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json

    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{args.mode}: {summary['passed']}/{summary['probed']} passed")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
