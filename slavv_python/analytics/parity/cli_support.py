"""Shared helpers for native-first MATLAB-oracle parity CLI handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import ExactProofSourceSurface
from .surfaces import (
    load_oracle_surface,
)

if TYPE_CHECKING:
    from pathlib import Path


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
