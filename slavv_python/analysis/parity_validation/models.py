"""Data models for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class RunCounts:
    """Vertex, edge, and strand counts for a run."""

    vertices: int
    edges: int
    strands: int


@dataclass(frozen=True)
class OracleSurface:
    """Authority surface for a preserved MATLAB truth package."""

    oracle_root: Path
    manifest_path: Path | None
    matlab_batch_dir: Path
    matlab_vector_paths: dict[str, Path]
    oracle_id: str | None
    matlab_source_version: str | None
    dataset_hash: str | None


@dataclass(frozen=True)
class DatasetSurface:
    """Authority surface for a preserved dataset package."""

    dataset_root: Path
    manifest_path: Path
    input_file: Path
    dataset_hash: str


@dataclass(frozen=True)
class SourceRunSurface:
    """Surface for a slavv_python run root used as a comparison baseline."""

    run_root: Path
    checkpoints_dir: Path
    comparison_report_path: Path
    validated_params_path: Path
    run_snapshot_path: Path | None


@dataclass(frozen=True)
class ExactProofSourceSurface:
    """Surface for an exact-route proof against a MATLAB oracle."""

    run_root: Path
    checkpoints_dir: Path
    validated_params_path: Path
    oracle_surface: OracleSurface
    matlab_batch_dir: Path
    matlab_vector_paths: dict[str, Path]


@dataclass(frozen=True)
class ExactPreflightSurface:
    """Surface for an exact-route preflight check."""

    source_surface: ExactProofSourceSurface
    dest_run_root: Path
    image_shape: tuple[int, int, int]
