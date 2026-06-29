"""Unified resume path for exact-route parity pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from slavv_python.analytics.parity.constants import (
    DATASET_INPUT_DIR,
    DATASET_MANIFEST_PATH,
    EXPERIMENT_PROVENANCE_PATH,
    EXPERIMENT_REFS_DIR,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analytics.parity.oracle.surfaces import (
    ensure_dest_run_layout,
    load_dataset_surface,
    load_oracle_surface,
)
from slavv_python.analytics.parity.runs.bootstrap import _reorient_exact_input_volume
from slavv_python.analytics.parity.runs.preflight import run_exact_preflight_for_surfaces
from slavv_python.analytics.parity.utils import string_or_none
from slavv_python.engine import SlavvPipeline
from slavv_python.engine.constants import STATUS_PENDING
from slavv_python.engine.state import RunContext, load_json_dict
from slavv_python.storage import load_tiff_volume

if TYPE_CHECKING:
    from slavv_python.analytics.parity.oracle.models import DatasetSurface


def resolve_exact_run_dataset_surface(
    dest_run_root: Path,
    dataset_root: Path | None = None,
) -> DatasetSurface:
    """Resolve the catalog dataset surface for an existing exact-route run."""
    if dataset_root is not None:
        return load_dataset_surface(dataset_root)

    input_path = resolve_exact_run_input_file(dest_run_root, dataset_root=None)
    if input_path.parent.name == DATASET_INPUT_DIR.name:
        dataset_root_candidate = input_path.parent.parent
        manifest_path = dataset_root_candidate / DATASET_MANIFEST_PATH
        if manifest_path.is_file():
            return load_dataset_surface(dataset_root_candidate)

    raise ValueError(
        "could not resolve catalog dataset surface; pass --dataset-root "
        f"(input file resolved to {input_path})"
    )


def resolve_exact_run_input_file(
    dest_run_root: Path,
    dataset_root: Path | None = None,
) -> Path:
    """Resolve the TIFF path used to resume pipeline processing."""
    if dataset_root is not None:
        return load_dataset_surface(dataset_root).input_file

    snapshot = load_json_dict(dest_run_root / RUN_SNAPSHOT_PATH) or {}
    provenance = snapshot.get("provenance", {})
    raw_input = provenance.get("input_file")
    if isinstance(raw_input, str) and raw_input.strip():
        input_path = Path(raw_input).expanduser().resolve()
        if input_path.is_file():
            return input_path

    refs_dir = dest_run_root / EXPERIMENT_REFS_DIR
    tiff_candidates = sorted(refs_dir.glob("*.tif"))
    if len(tiff_candidates) == 1:
        return tiff_candidates[0].resolve()

    raise ValueError(
        "could not resolve dataset input file for resume; pass --dataset-root or ensure "
        f"run_snapshot provenance or a single TIFF exists under {refs_dir}"
    )


def resolve_exact_run_oracle_root(
    dest_run_root: Path,
    oracle_root: Path | None = None,
) -> Path:
    """Resolve oracle root from CLI arg or persisted run manifest."""
    if oracle_root is not None:
        return oracle_root.expanduser().resolve()
    manifest = load_json_dict(dest_run_root / RUN_MANIFEST_PATH) or {}
    stored = string_or_none(manifest.get("oracle_root"))
    if stored:
        return Path(stored).expanduser().resolve()
    raise ValueError(
        "oracle root not provided and run manifest does not record oracle_root; pass --oracle-root"
    )


def clear_stale_running_snapshot(dest_run_root: Path) -> bool:
    """Mark a stale running snapshot as pending so SlavvPipeline can resume."""
    context = RunContext.from_existing(dest_run_root)
    if context.snapshot.status != "running":
        return False
    context.mark_run_status(
        STATUS_PENDING,
        detail="Cleared stale running status before exact-route resume",
    )
    context.persist()
    return True


def resume_exact_run(
    dest_run_root: Path,
    *,
    dataset_root: Path | None = None,
    oracle_root: Path | None = None,
    stop_after: str | None = None,
    force_rerun_from: str | None = None,
    memory_safety_fraction: float = 0.8,
    force: bool = False,
    skip_preflight: bool = False,
    n_jobs: int | None = None,
) -> Path:
    """Resume SlavvPipeline in an init-exact-run directory after preflight checks."""
    dest_run_root = dest_run_root.expanduser().resolve()
    if not dest_run_root.is_dir():
        raise ValueError(f"run directory not found: {dest_run_root}")

    provenance = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH)
    if provenance is None:
        raise ValueError(
            f"missing {EXPERIMENT_PROVENANCE_PATH}; directory was not created by init-exact-run"
        )

    input_file = resolve_exact_run_input_file(dest_run_root, dataset_root)
    try:
        dataset_surface = resolve_exact_run_dataset_surface(dest_run_root, dataset_root)
    except ValueError:
        dataset_surface = None
    resolved_oracle_root = resolve_exact_run_oracle_root(dest_run_root, oracle_root)
    oracle_surface = load_oracle_surface(resolved_oracle_root)
    if (
        dataset_surface is not None
        and oracle_surface.dataset_hash
        and oracle_surface.dataset_hash != dataset_surface.dataset_hash
    ):
        raise ValueError(
            f"dataset and oracle hashes do not match: "
            f"{dataset_surface.dataset_hash} != {oracle_surface.dataset_hash}"
        )

    params_path = dest_run_root / VALIDATED_PARAMS_PATH
    params = load_json_dict(params_path)
    if params is None:
        raise FileNotFoundError(f"missing validated params: {params_path}")
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)

    effective_stop_after = stop_after or str(provenance.get("stop_after") or "network")

    ensure_dest_run_layout(dest_run_root)
    if not skip_preflight:
        report, _, _ = run_exact_preflight_for_surfaces(
            dest_run_root,
            dataset_surface=dataset_surface,
            oracle_surface=oracle_surface,
            params=cast("dict[str, Any]", params),
            memory_safety_fraction=memory_safety_fraction,
            force=force,
        )
        if not report.get("passed"):
            joined = ", ".join(report.get("errors", []))
            raise RuntimeError(f"exact-route preflight failed: {joined}")

    clear_stale_running_snapshot(dest_run_root)

    # Reorient identically to init-exact-run: load in [Y, X, Z] and search the
    # permutation that matches the oracle's axis order. This is self-correcting,
    # so resume can never desync from the orientation init computed (a stale
    # provenance permutation applied to a differently-loaded volume previously
    # double-permuted the full volume to [Y, Z, X]).
    image = load_tiff_volume(str(input_file))
    image, _oracle_size, _perm = _reorient_exact_input_volume(image, oracle_surface)

    SlavvPipeline().run(
        image,
        cast("dict[str, Any]", params),
        run_dir=str(dest_run_root),
        stop_after=effective_stop_after,
        force_rerun_from=force_rerun_from,
    )
    return dest_run_root


__all__ = [
    "clear_stale_running_snapshot",
    "resolve_exact_run_dataset_surface",
    "resolve_exact_run_input_file",
    "resolve_exact_run_oracle_root",
    "resume_exact_run",
]
