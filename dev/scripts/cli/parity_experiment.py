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
    handle_prove_exact as _handle_prove_exact,
)
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

    # ... (I'll add the rest of the subparsers in the final version)

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
