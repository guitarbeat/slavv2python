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
    DEV_RUNS_ROOT,
    EXACT_STAGE_ORDER,
    DEFAULT_MEMORY_SAFETY_FRACTION,
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
    handle_init_exact_run,
    handle_promote_report,
)

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

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
