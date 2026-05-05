"""Developer CLI wrapper for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from source.runtime import find_repo_root

REPO_ROOT = find_repo_root(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.analysis.parity.constants import (
    EXACT_STAGE_ORDER,
    DEFAULT_MEMORY_SAFETY_FRACTION,
)
from source.analysis.parity.execution import ensure_dest_run_layout, write_run_manifest, validate_exact_proof_source_surface
from source.analysis.parity.proofs import CHECKPOINTS_DIR
from source.analysis.parity.promotion import materialize_dataset_record as _materialize_dataset_record
from source.analysis.parity.utils import fingerprint_file
from source.io.tiff import load_tiff_volume
from source.core.pipeline import SLAVVProcessor
from source.core.edges import (
    choose_edges_for_workflow,
    add_vertices_to_edges_matlab_style,
    finalize_edges_matlab_style,
)
from source.analysis.parity.cli import (
    handle_rerun_python,
    handle_summarize,
    handle_normalize_recordings,
    handle_diagnose_gaps,
    handle_prove_exact,
    handle_preflight_exact,
    handle_prove_luts,
    handle_capture_candidates,
    handle_replay_edges,
    handle_fail_fast,
    handle_promote_oracle,
    handle_promote_dataset,
    handle_promote_report,
    handle_init_exact_run,
)

# Internal aliases for test monkeypatching
_handle_rerun_python = handle_rerun_python
_handle_summarize = handle_summarize
_handle_normalize_recordings = handle_normalize_recordings
_handle_diagnose_gaps = handle_diagnose_gaps
_handle_prove_exact = handle_prove_exact
_handle_preflight_exact = handle_preflight_exact
_handle_prove_luts = handle_prove_luts
_handle_capture_candidates = handle_capture_candidates
_handle_replay_edges = handle_replay_edges
_handle_fail_fast = handle_fail_fast
from source.analysis.parity.proofs import (
    run_exact_parity_proof,
    run_exact_preflight,
    run_candidate_capture,
    run_edge_replay,
    run_lut_proof,
    run_exact_preflight as _run_preflight_exact,
    run_lut_proof as _run_prove_luts,
    run_candidate_capture as _run_capture_candidates,
    run_edge_replay as _run_replay_edges,
)
from source.io.matlab_exact_proof import render_exact_proof_report
from source.io.matlab_fail_fast import (
    render_lut_proof_report,
    render_candidate_coverage_report,
)
from source.analysis.parity.reports import render_exact_preflight_report

def build_parser() -> argparse.ArgumentParser:
    """Build the developer parity experiment parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Developer helpers for rerunning and proving native-first exact-route parity "
            "against an existing staged comparison run."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # Rerun Python
    rerun = subparsers.add_parser("rerun-python")
    rerun.add_argument("--source-run-root", required=True)
    rerun.add_argument("--dest-run-root", required=True)
    rerun.add_argument("--input")
    rerun.add_argument("--rerun-from", choices=("edges", "network"), default="edges")
    rerun.add_argument("--params-file")
    rerun.set_defaults(handler=handle_rerun_python)

    # Summarize
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-root", required=True)
    summarize.set_defaults(handler=handle_summarize)

    # Normalize Recordings
    normalize = subparsers.add_parser("normalize-recordings")
    normalize.add_argument("--run-root", required=True)
    normalize.set_defaults(handler=handle_normalize_recordings)

    # Diagnose Gaps
    diagnose = subparsers.add_parser("diagnose-gaps")
    diagnose.add_argument("--run-root", required=True)
    diagnose.add_argument("--limit", type=int, default=10)
    diagnose.set_defaults(handler=handle_diagnose_gaps)

    # Prove Exact
    prove = subparsers.add_parser("prove-exact")
    prove.add_argument("--source-run-root", required=True)
    prove.add_argument("--oracle-root")
    prove.add_argument("--dest-run-root", required=True)
    prove.add_argument("--stage", choices=(*EXACT_STAGE_ORDER, "all"), default="all")
    prove.add_argument("--report-path")
    prove.set_defaults(handler=handle_prove_exact)

    # Preflight Exact
    preflight = subparsers.add_parser("preflight-exact")
    preflight.add_argument("--source-run-root", required=True)
    preflight.add_argument("--oracle-root")
    preflight.add_argument("--dest-run-root", required=True)
    preflight.add_argument("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION)
    preflight.add_argument("--force", action="store_true")
    preflight.set_defaults(handler=handle_preflight_exact)

    # Prove LUTs
    luts = subparsers.add_parser("prove-luts")
    luts.add_argument("--source-run-root", required=True)
    luts.add_argument("--oracle-root")
    luts.add_argument("--dest-run-root", required=True)
    luts.set_defaults(handler=handle_prove_luts)

    # Capture Candidates
    capture = subparsers.add_parser("capture-candidates")
    capture.add_argument("--source-run-root", required=True)
    capture.add_argument("--oracle-root")
    capture.add_argument("--dest-run-root", required=True)
    capture.add_argument("--include-debug-maps", action="store_true")
    capture.set_defaults(handler=handle_capture_candidates)

    # Replay Edges
    replay = subparsers.add_parser("replay-edges")
    replay.add_argument("--source-run-root", required=True)
    replay.add_argument("--oracle-root")
    replay.add_argument("--dest-run-root", required=True)
    replay.set_defaults(handler=handle_replay_edges)

    # Fail Fast
    fail_fast = subparsers.add_parser("fail-fast")
    fail_fast.add_argument("--source-run-root", required=True)
    fail_fast.add_argument("--oracle-root")
    fail_fast.add_argument("--dest-run-root", required=True)
    fail_fast.add_argument("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION)
    fail_fast.add_argument("--force", action="store_true")
    fail_fast.add_argument("--debug-maps", action="store_true")
    fail_fast.set_defaults(handler=handle_fail_fast)

    # Promote Oracle
    promote_oracle = subparsers.add_parser("promote-oracle")
    promote_oracle.add_argument("--matlab-batch-dir", required=True)
    promote_oracle.add_argument("--oracle-root", required=True)
    promote_oracle.add_argument("--dataset-file")
    promote_oracle.add_argument("--dataset-hash")
    promote_oracle.add_argument("--oracle-id")
    promote_oracle.add_argument("--matlab-source-version")
    promote_oracle.set_defaults(handler=handle_promote_oracle)

    # Promote Report
    promote_report = subparsers.add_parser("promote-report")
    promote_report.add_argument("--run-root", required=True)
    promote_report.add_argument("--report-root")
    promote_report.set_defaults(handler=handle_promote_report)

    # Promote Dataset
    promote_dataset = subparsers.add_parser("promote-dataset")
    promote_dataset.add_argument("--dataset-file", required=True)
    promote_dataset.add_argument("--experiment-root", required=True)
    promote_dataset.set_defaults(handler=handle_promote_dataset)

    # Init Exact Run
    init_run = subparsers.add_parser("init-exact-run")
    init_run.add_argument("--dataset-root", required=True)
    init_run.add_argument("--oracle-root", required=True)
    init_run.add_argument("--dest-run-root", required=True)
    init_run.add_argument("--stop-after", default="vertices")
    init_run.add_argument("--energy-storage-format", default="npy")
    init_run.set_defaults(handler=handle_init_exact_run)

    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
