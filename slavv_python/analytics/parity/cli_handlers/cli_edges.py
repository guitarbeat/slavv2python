"""CLI handlers for edge capture/replay and trace utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from slavv_python.analytics.parity.cli_handlers.cli_proofs import (
    handle_prove_exact,
    handle_prove_luts,
)
from slavv_python.analytics.parity.cli_handlers.cli_runs import handle_preflight_exact
from slavv_python.analytics.parity.cli_handlers.cli_support import _build_exact_proof_source_surface
from slavv_python.analytics.parity.oracle.models import ExactProofSourceSurface
from slavv_python.analytics.parity.oracle.surfaces import load_oracle_surface
from slavv_python.analytics.parity.probes.crop_export import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_SOURCE,
)
from slavv_python.analytics.parity.probes.crop_export import main as export_crop_main
from slavv_python.analytics.parity.probes.trace_comparator import main as compare_traces_main
from slavv_python.analytics.parity.proof.coordinator import ExactProofCoordinator
from slavv_python.analytics.parity.proof.index import (
    deduplicate_index_records,
    resolve_experiment_root,
)

if TYPE_CHECKING:
    import argparse


def handle_capture_candidates(args: argparse.Namespace) -> None:
    """Capture candidate pairs from a Python run for parity comparison."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source_surface = _build_exact_proof_source_surface(run_root, oracle_root)
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    report, _, _ = ExactProofCoordinator(source_surface).capture_candidates(
        dest_run_root,
        include_debug_maps=bool(getattr(args, "debug_maps", False)),
    )
    if not report.get("passed"):
        sys.exit(1)


def handle_replay_edges(args: argparse.Namespace) -> None:
    """Replay edge discovery from candidates for parity verification."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None

    oracle_surface = load_oracle_surface(oracle_root)
    source_surface = ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=run_root / "02_Output" / "python_results" / "checkpoints",
        validated_params_path=run_root / "99_Metadata" / "validated_params.json",
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
    )

    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    report, _, _ = ExactProofCoordinator.run_edge_replay(source_surface, dest_run_root)
    if not report.get("passed"):
        sys.exit(1)


def handle_fail_fast(args: argparse.Namespace) -> None:
    """Run cheap gates first and stop at the first failing gate."""
    handle_preflight_exact(args)
    handle_prove_luts(args)
    handle_capture_candidates(args)
    handle_replay_edges(args)
    handle_prove_exact(args)


def handle_dedupe(args: argparse.Namespace) -> None:
    """Clean up and deduplicate index.jsonl in the experiment workspace root."""
    repo_root = Path.cwd()
    exp_root = resolve_experiment_root(repo_root / "workspace") or resolve_experiment_root(
        repo_root
    )
    if not exp_root:
        raise RuntimeError("Could not find experiment root directory from CWD.")

    print(f"Auditing and deduplicating SLAVV index under {exp_root}...")
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        print("[Dry Run Mode] No files will be modified on disk.")

    removed = deduplicate_index_records(exp_root, dry_run=dry_run)

    if removed:
        print(f"\nFound {len(removed)} stale or duplicate records to remove:")
        for r in removed:
            print(f"  - ID: {r.get('id') or r.get('run_id')} (Kind: {r.get('kind')})")
            print(f"    Path: {r.get('path') or r.get('run_root')}")
        if not dry_run:
            print("\nSuccessfully cleaned up and updated index.jsonl!")
        else:
            print("\nDry run completed. Run without --dry-run to apply these changes.")
    else:
        print("\nWorkspace index is already perfectly canonical and clean!")


def handle_compare_traces(args: argparse.Namespace) -> None:
    """Compare two SLAVV JSONL execution traces for divergences."""
    result = compare_traces_main(
        [str(args.trace1), str(args.trace2), "--energy-tol", str(args.energy_tol)]
    )
    if result:
        sys.exit(result)


def handle_export_crop(args: argparse.Namespace) -> None:
    """Export the 180709_E tier-M center crop TIFF for parity pre-gate."""
    source = args.source or DEFAULT_SOURCE
    output = args.output or (Path("workspace/scratch/180709_E_crop_M") / DEFAULT_OUTPUT_NAME)
    argv = ["--source", str(source), "--output", str(output)]
    if getattr(args, "write_metadata", False):
        argv.append("--write-metadata")
    result = export_crop_main(argv)
    if result:
        sys.exit(result)
