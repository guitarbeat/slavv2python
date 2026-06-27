"""CLI entry points for native-first MATLAB-oracle parity experiments.

Thin facade: re-exports stage handlers from focused submodules and hosts the
sequence proof handler plus the patchable proof-surface helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import (
    ANALYSIS_DIR,
    EXACT_PROOF_JSON_PATH,
    EXACT_STAGE_ORDER,
)
from .coordinator import ExactProofCoordinator
from .utils import (
    write_json_with_hash,
)

if TYPE_CHECKING:
    import argparse

from .cli_diagnostics import (
    handle_compare_energy_probes,
    handle_diagnose_energy,
    handle_diagnose_gaps,
    handle_inspect_energy_evidence,
    handle_normalize_recordings,
    handle_record_parity_hypothesis,
    handle_summarize,
    handle_trace_vertex,
)
from .cli_edges import (
    handle_capture_candidates,
    handle_compare_traces,
    handle_dedupe,
    handle_export_crop,
    handle_fail_fast,
    handle_replay_edges,
)
from .cli_proofs import (
    handle_prove_energy_ulp,
    handle_prove_exact,
    handle_prove_luts,
)
from .cli_runs import (
    handle_ensure_oracle_artifacts,
    handle_init_exact_run,
    handle_launch_exact_run,
    handle_preflight_exact,
    handle_promote_dataset,
    handle_promote_oracle,
    handle_promote_report,
    handle_rerun_python,
    handle_resume_exact_run,
    handle_status_exact_run,
)
from .cli_support import _build_exact_proof_source_surface


def handle_prove_exact_sequence(args: argparse.Namespace) -> None:
    """Run prove-exact for each stage in order; stop at the first failure."""
    from shutil import copy2

    from slavv_python.analytics.parity.proof_report import render_exact_proof_report

    run_root = Path(args.source_run_root).expanduser().resolve()
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source_surface = _build_exact_proof_source_surface(run_root, oracle_root)
    coordinator = ExactProofCoordinator(source_surface)

    stage_results: list[dict[str, Any]] = []
    for stage in EXACT_STAGE_ORDER:
        report, json_path, _text_path = coordinator.prove(
            dest_run_root,
            stage_arg=stage,
            strict_floats=bool(getattr(args, "strict_floats", False)),
            max_ulps=getattr(args, "max_ulps", None),
        )
        if json_path is not None and json_path.is_file():
            stage_json = dest_run_root / ANALYSIS_DIR / f"exact_proof_{stage}.json"
            stage_json.parent.mkdir(parents=True, exist_ok=True)
            copy2(json_path, stage_json)
        passed = bool(report.get("passed"))
        stage_results.append({"stage": stage, "passed": passed})
        print(f"prove-exact --stage {stage}: {'PASS' if passed else 'FAIL'}")
        if report.get("stage_summaries"):
            print(render_exact_proof_report(report))
        if not passed:
            import sys

            sys.exit(1)

    summary = {
        "passed": True,
        "stages": stage_results,
        "source_run_root": str(run_root),
        "dest_run_root": str(dest_run_root),
    }
    from .utils import write_text_with_hash

    summary_json = dest_run_root / EXACT_PROOF_JSON_PATH
    summary_text = dest_run_root / ANALYSIS_DIR / "exact_proof_sequence.txt"
    write_json_with_hash(summary_json, summary)
    write_text_with_hash(
        summary_text,
        "\n".join(
            [
                "Exact proof sequence (all stages passed)",
                *(f"  {row['stage']}: PASS" for row in stage_results),
            ]
        ),
    )
    print(str(summary_json))


__all__ = [
    "_build_exact_proof_source_surface",
    "handle_capture_candidates",
    "handle_compare_energy_probes",
    "handle_compare_traces",
    "handle_dedupe",
    "handle_diagnose_energy",
    "handle_diagnose_gaps",
    "handle_ensure_oracle_artifacts",
    "handle_export_crop",
    "handle_fail_fast",
    "handle_init_exact_run",
    "handle_inspect_energy_evidence",
    "handle_launch_exact_run",
    "handle_normalize_recordings",
    "handle_preflight_exact",
    "handle_promote_dataset",
    "handle_promote_oracle",
    "handle_promote_report",
    "handle_prove_energy_ulp",
    "handle_prove_exact",
    "handle_prove_exact_sequence",
    "handle_prove_luts",
    "handle_record_parity_hypothesis",
    "handle_replay_edges",
    "handle_rerun_python",
    "handle_resume_exact_run",
    "handle_status_exact_run",
    "handle_summarize",
    "handle_trace_vertex",
]
