"""CLI handlers for exact-route parity proofs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from slavv_python.analytics.parity.proof.coordinator import ExactProofCoordinator
from slavv_python.analytics.parity.proof.proofs import (
    run_lut_proof,
)
from slavv_python.engine.state import load_json_dict

if TYPE_CHECKING:
    import argparse

from slavv_python.analytics.parity.cli_handlers.cli_support import _build_exact_proof_source_surface


def handle_prove_energy_ulp(args: argparse.Namespace) -> None:
    """Run advisory Energy ULP proof (strict scales, bounded float ULP)."""
    from slavv_python.analytics.parity.oracle.matlab_vector_loader import (
        load_normalized_matlab_vectors,
    )
    from slavv_python.analytics.parity.oracle.python_checkpoint_loader import (
        load_normalized_python_checkpoints,
    )
    from slavv_python.analytics.parity.proof.energy_proof_evidence import (
        require_energy_proof_evidence,
    )
    from slavv_python.analytics.parity.proof.energy_ulp_proof import (
        build_energy_ulp_proof_report,
        persist_energy_ulp_proof_report,
    )
    from slavv_python.analytics.parity.utils import payload_hash

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
        import sys

        sys.exit(1)


def handle_prove_exact(args: argparse.Namespace) -> None:
    """Orchestrate a full-artifact exact proof."""
    from shutil import copy2

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
    # When proving a single named stage, also write a per-stage JSON file
    # (e.g., exact_proof_edges.json) so callers can track per-stage evidence.
    if stage_arg not in (None, "all") and json_path is not None and json_path.is_file():
        from slavv_python.analytics.parity.constants import ANALYSIS_DIR

        stage_json = dest_run_root / ANALYSIS_DIR / f"exact_proof_{stage_arg}.json"
        stage_json.parent.mkdir(parents=True, exist_ok=True)
        if json_path.resolve() != stage_json.resolve():
            copy2(json_path, stage_json)

    if not report.get("passed"):
        import sys

        sys.exit(1)


def handle_prove_luts(args: argparse.Namespace) -> None:
    """Verify exact parity for lookup tables."""
    report, _, _ = run_lut_proof(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
    )
    if not report.get("passed"):
        import sys

        sys.exit(1)
