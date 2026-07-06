"""Shared helpers for native-first MATLAB-oracle parity CLI handlers."""

from __future__ import annotations

from shutil import copy2
from typing import TYPE_CHECKING

from slavv_python.analytics.parity.constants import ANALYSIS_DIR
from slavv_python.analytics.parity.oracle.models import ExactProofSourceSurface
from slavv_python.analytics.parity.oracle.surfaces import load_oracle_surface

if TYPE_CHECKING:
    from pathlib import Path


def _copy_stage_proof_json(
    dest_run_root: Path,
    json_path: Path | None,
    stage_arg: str | None,
) -> None:
    """Write a per-stage proof JSON when proving a single named stage."""
    if stage_arg in (None, "all") or json_path is None or not json_path.is_file():
        return
    stage_json = dest_run_root / ANALYSIS_DIR / f"exact_proof_{stage_arg}.json"
    stage_json.parent.mkdir(parents=True, exist_ok=True)
    if json_path.resolve() != stage_json.resolve():
        copy2(json_path, stage_json)


def _build_exact_proof_source_surface(
    run_root: Path,
    oracle_root: Path | None,
) -> ExactProofSourceSurface:
    """Resolve oracle paths and return the exact-proof source surface."""
    if oracle_root is None and (run_root / "01_Input" / "matlab_results").is_dir():
        oracle_root = run_root
    oracle_surface = load_oracle_surface(oracle_root)
    return ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=run_root / "02_Output" / "python_results" / "checkpoints",
        validated_params_path=run_root / "99_Metadata" / "validated_params.json",
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
    )
