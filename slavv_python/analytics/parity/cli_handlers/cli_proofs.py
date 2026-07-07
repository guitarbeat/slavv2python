"""CLI handlers for exact-route parity proofs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.analytics.parity.cli_handlers.cli_support import (
    _build_exact_proof_source_surface,
    _copy_stage_proof_json,
)
from slavv_python.analytics.parity.constants import (
    ANALYSIS_DIR,
    EXACT_PROOF_JSON_PATH,
    EXACT_STAGE_ORDER,
)
from slavv_python.analytics.parity.oracle.matlab_vector_loader import (
    load_normalized_matlab_vectors,
)
from slavv_python.analytics.parity.oracle.python_checkpoint_loader import (
    load_normalized_python_checkpoints,
)
from slavv_python.analytics.parity.proof.coordinator import ExactProofCoordinator
from slavv_python.analytics.parity.proof.energy_proof_evidence import (
    require_energy_proof_evidence,
)
from slavv_python.analytics.parity.proof.energy_ulp_proof import (
    build_energy_ulp_proof_report,
    persist_energy_ulp_proof_report,
)
from slavv_python.analytics.parity.proof.proof_report import render_exact_proof_report
from slavv_python.analytics.parity.utils import (
    payload_hash,
    write_json_with_hash,
    write_text_with_hash,
)
from slavv_python.engine.state import load_json_dict

if TYPE_CHECKING:
    import argparse


def handle_prove_energy_ulp(args: argparse.Namespace) -> None:
    """Run advisory Energy ULP proof (strict scales, bounded float ULP)."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    require_energy_proof_evidence(dest_run_root)
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source = _build_exact_proof_source_surface(run_root, oracle_root)
    if source.matlab_batch_dir is None:
        raise RuntimeError("Exact proof source is missing MATLAB batch directory")
    matlab = load_normalized_matlab_vectors(source.matlab_batch_dir, ("energy",))["energy"]
    python = load_normalized_python_checkpoints(source.checkpoints_dir, ("energy",))["energy"]
    params = load_json_dict(source.validated_params_path) or {}
    report = build_energy_ulp_proof_report(
        np.asarray(matlab["energy"]),
        np.asarray(python["energy"]),
        np.asarray(matlab["scale_indices"]),
        np.asarray(python["scale_indices"]),
        max_ulps=max(0, int(args.max_ulps)),
        provenance={
            "source_run_root": str(run_root),
            "dest_run_root": str(dest_run_root),
            "oracle_id": source.oracle_surface.oracle_id,
            "matlab_batch_dir": str(source.matlab_batch_dir),
            "params_fingerprint": payload_hash(params),
        },
    )
    path = persist_energy_ulp_proof_report(dest_run_root, report)
    print(path)
    if not report["passed"]:
        sys.exit(1)


def handle_prove_exact(args: argparse.Namespace) -> None:
    """Orchestrate a full-artifact exact proof."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source_surface = _build_exact_proof_source_surface(run_root, oracle_root)
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    stage_arg = getattr(args, "stage", "all")
    report, json_path, _text_path = ExactProofCoordinator(source_surface).prove(
        dest_run_root,
        stage_arg=stage_arg,
        report_path_arg=getattr(args, "report_path", None),
        strict_floats=bool(getattr(args, "strict_floats", False)),
        max_ulps=getattr(args, "max_ulps", None),
    )
    _copy_stage_proof_json(dest_run_root, json_path, stage_arg)

    if report.get("stage_summaries"):
        print(render_exact_proof_report(report))

    if not report.get("passed"):
        sys.exit(1)


def handle_prove_exact_sequence(args: argparse.Namespace) -> None:
    """Run prove-exact for each stage in order; stop at the first failure."""
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
        _copy_stage_proof_json(dest_run_root, json_path, stage)
        passed = bool(report.get("passed"))
        stage_results.append({"stage": stage, "passed": passed})
        print(f"prove-exact --stage {stage}: {'PASS' if passed else 'FAIL'}")
        if report.get("stage_summaries"):
            print(render_exact_proof_report(report))
        if not passed:
            sys.exit(1)

    summary = {
        "passed": True,
        "stages": stage_results,
        "source_run_root": str(run_root),
        "dest_run_root": str(dest_run_root),
    }
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


def handle_prove_luts(args: argparse.Namespace) -> None:
    """Verify exact parity for lookup tables."""
    report, _, _ = ExactProofCoordinator.run_lut_proof(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
    )
    if not report.get("passed"):
        sys.exit(1)
