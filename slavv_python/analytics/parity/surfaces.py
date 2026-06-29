"""Authority surfaces for datasets, oracles, and staged slavv_python runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.io import loadmat

from slavv_python.analytics.parity import params_audit
from slavv_python.analytics.parity.constants import (
    ANALYSIS_DIR,
    ANALYSIS_TABLES_DIR,
    CHECKPOINTS_DIR,
    COMPARISON_REPORT_PATH,
    DATASET_INPUT_DIR,
    DATASET_MANIFEST_PATH,
    EXPERIMENT_PARAMS_DIR,
    EXPERIMENT_REFS_DIR,
    HASHES_DIR,
    METADATA_DIR,
    NORMALIZED_DIR,
    ORACLE_DISCOVERY_STAGES,
    ORACLE_MANIFEST_PATH,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analytics.parity.index import resolve_experiment_root, upsert_index_record
from slavv_python.analytics.parity.matlab_vector_loader import (
    find_matlab_vector_paths,
    find_single_matlab_batch_dir,
)
from slavv_python.analytics.parity.models import (
    DatasetSurface,
    ExactProofSourceSurface,
    OracleSurface,
    SourceRunSurface,
)
from slavv_python.analytics.parity.utils import (
    entity_id_from_path,
    now_iso,
    resolve_python_commit,
    string_or_none,
    write_json_with_hash,
)
from slavv_python.engine.state import fingerprint_file, load_json_dict


def ensure_dest_run_layout(dest_run_root: Path) -> None:
    """Ensure the destination run root has the required subdirectory structure."""
    for subdir in (
        ANALYSIS_DIR,
        ANALYSIS_TABLES_DIR,
        CHECKPOINTS_DIR,
        EXPERIMENT_REFS_DIR,
        EXPERIMENT_PARAMS_DIR,
        METADATA_DIR,
        HASHES_DIR,
        NORMALIZED_DIR,
    ):
        (dest_run_root / subdir).mkdir(parents=True, exist_ok=True)


def load_oracle_surface(oracle_root: Path | None) -> OracleSurface:
    """Load the authority surface for a preserved MATLAB truth package."""
    if oracle_root is None:
        return OracleSurface(
            oracle_root=None,
            manifest_path=None,
            matlab_batch_dir=None,
            matlab_vector_paths={},
            oracle_id=None,
            matlab_source_version=None,
            dataset_hash=None,
        )
    resolved_root = oracle_root.expanduser().resolve()
    manifest_path = resolved_root / ORACLE_MANIFEST_PATH
    manifest = load_json_dict(manifest_path)
    matlab_batch_dir = find_single_matlab_batch_dir(resolved_root)
    return OracleSurface(
        oracle_root=resolved_root,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        matlab_batch_dir=matlab_batch_dir,
        matlab_vector_paths=find_matlab_vector_paths(matlab_batch_dir, ORACLE_DISCOVERY_STAGES),
        oracle_id=string_or_none(manifest.get("oracle_id")) if manifest else None,
        matlab_source_version=(
            string_or_none(manifest.get("matlab_source_version")) if manifest else None
        ),
        dataset_hash=string_or_none(manifest.get("dataset_hash")) if manifest else None,
    )


def load_dataset_surface(dataset_root: Path) -> DatasetSurface:
    """Load the authority surface for a preserved dataset package."""
    resolved_root = dataset_root.expanduser().resolve()
    manifest_path = resolved_root / DATASET_MANIFEST_PATH
    manifest = load_json_dict(manifest_path)
    if manifest is None:
        raise ValueError(f"missing dataset manifest: {manifest_path}")

    stored_input = string_or_none(manifest.get("stored_input_file"))
    input_filename = string_or_none(manifest.get("input_filename"))
    dataset_hash = string_or_none(manifest.get("dataset_hash"))

    if dataset_hash is None:
        raise ValueError(f"dataset manifest is missing dataset_hash: {manifest_path}")

    input_file: Path | None = None
    if stored_input:
        input_file = Path(stored_input).expanduser().resolve()
        if not input_file.is_file():
            input_file = None

    if input_file is None and input_filename:
        input_file = resolved_root / DATASET_INPUT_DIR / input_filename
        if not input_file.is_file():
            input_file = None

    if input_file is None:
        raise ValueError(f"dataset input file not found for {dataset_hash} under {resolved_root}")

    actual_hash = fingerprint_file(input_file)
    if actual_hash != dataset_hash:
        raise ValueError(
            "dataset manifest hash does not match stored input file: "
            f"expected {dataset_hash}, found {actual_hash}"
        )

    return DatasetSurface(
        dataset_root=resolved_root,
        manifest_path=manifest_path,
        input_file=input_file,
        dataset_hash=dataset_hash,
    )


def validate_source_run_surface(source_run_root: Path) -> SourceRunSurface:
    """Validate the reusable staged slavv_python surface for a Python rerun."""
    run_root = source_run_root.resolve()
    checkpoints_dir = run_root / CHECKPOINTS_DIR
    comparison_report_path = run_root / COMPARISON_REPORT_PATH
    validated_params_path = run_root / VALIDATED_PARAMS_PATH
    run_snapshot_path = run_root / RUN_SNAPSHOT_PATH

    missing: list[Path] = []
    if not checkpoints_dir.is_dir():
        missing.append(checkpoints_dir)
    if not comparison_report_path.is_file():
        missing.append(comparison_report_path)
    if not validated_params_path.is_file():
        missing.append(validated_params_path)
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"source run root is missing required artifacts: {joined}")

    return SourceRunSurface(
        run_root=run_root,
        checkpoints_dir=checkpoints_dir,
        comparison_report_path=comparison_report_path,
        validated_params_path=validated_params_path,
        run_snapshot_path=run_snapshot_path if run_snapshot_path.is_file() else None,
    )


def validate_exact_proof_source_surface(source_run_root: Path) -> ExactProofSourceSurface:
    """Validate the authority surface for an exact-route proof against a MATLAB oracle."""
    run_root = source_run_root.resolve()
    manifest = load_json_dict(run_root / RUN_MANIFEST_PATH)
    if not manifest:
        raise ValueError(f"source run root is missing manifest: {run_root / RUN_MANIFEST_PATH}")

    validated_params_path = run_root / VALIDATED_PARAMS_PATH
    if not validated_params_path.is_file():
        raise ValueError(f"source run root is missing validated params: {validated_params_path}")

    stored_oracle_root = string_or_none(manifest.get("oracle_root"))
    if stored_oracle_root:
        oracle_surface = load_oracle_surface(Path(stored_oracle_root))
    else:
        oracle_surface = load_oracle_surface(run_root)

    return ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=run_root / CHECKPOINTS_DIR,
        validated_params_path=validated_params_path,
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
    )


def resolve_input_file(
    source_surface: SourceRunSurface,
    input_arg: str | None,
    *,
    repo_root: Path,
) -> Path:
    """Resolve the input file either from the CLI or the slavv_python run snapshot provenance."""
    if input_arg:
        candidate = Path(input_arg).expanduser()
    else:
        if source_surface.run_snapshot_path is None:
            raise ValueError(
                "source run root does not contain snapshot and no --input was provided"
            )
        snapshot = load_json_dict(source_surface.run_snapshot_path) or {}
        provenance = snapshot.get("provenance", {})
        raw_input = provenance.get("input_file")
        if not isinstance(raw_input, str):
            raise ValueError("source snapshot does not record input_file; pass --input")
        candidate = Path(raw_input)
        if not candidate.is_absolute():
            candidate = repo_root / candidate

    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"input file not found: {resolved}")
    return resolved


def copy_source_surface(source_surface: SourceRunSurface, dest_run_root: Path) -> None:
    """Copy staged checkpoints and reference metadata into a fresh destination root."""
    from shutil import copy2, copytree

    destination = dest_run_root.resolve()
    if destination.exists():
        raise ValueError(f"destination run root already exists: {destination}")

    ensure_dest_run_layout(destination)
    checkpoints_dir = destination / CHECKPOINTS_DIR
    for artifact in source_surface.checkpoints_dir.iterdir():
        target = checkpoints_dir / artifact.name
        if artifact.is_dir():
            copytree(artifact, target)
        else:
            copy2(artifact, target)

    refs_dir = destination / EXPERIMENT_REFS_DIR
    copy2(source_surface.validated_params_path, refs_dir / "source_validated_params.json")
    if source_surface.comparison_report_path.is_file():
        copy2(source_surface.comparison_report_path, refs_dir / "source_comparison_report.json")
    if source_surface.run_snapshot_path and source_surface.run_snapshot_path.is_file():
        copy2(source_surface.run_snapshot_path, refs_dir / "source_run_snapshot.json")


def write_run_manifest(
    dest_run_root: Path,
    *,
    run_kind: str,
    status: str,
    command: str,
    dataset_hash: str | None,
    oracle_surface: OracleSurface | None,
    params_payload: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the run manifest and update the central index."""
    src_manifest = load_json_dict(dest_run_root / RUN_MANIFEST_PATH) or {}
    run_id = string_or_none(src_manifest.get("run_id")) or entity_id_from_path(dest_run_root)
    exp_root = resolve_experiment_root(dest_run_root)

    ts = cast("dict[str, Any]", dict(src_manifest.get("timestamps", {})))
    created_at = (
        string_or_none(ts.get("created_at"))
        or string_or_none(src_manifest.get("created_at"))
        or now_iso()
    )
    updated_at = now_iso()

    manifest = {
        "manifest_version": 1,
        "kind": run_kind,
        "run_id": run_id,
        "command": command,
        "run_root": str(dest_run_root),
        "status": status,
        "dataset_hash": dataset_hash,
        "oracle_id": oracle_surface.oracle_id if oracle_surface else None,
        "oracle_root": str(oracle_surface.oracle_root) if oracle_surface else None,
        "python_commit": string_or_none(src_manifest.get("python_commit"))
        or resolve_python_commit(exp_root or Path.cwd()),
        "timestamps": {
            "created_at": created_at,
            "updated_at": updated_at,
            "completed_at": ts.get("completed_at"),
        },
        "retention": "disposable" if dest_run_root.parent.name == "runs" else "promoted",
        "params_path": str(dest_run_root / VALIDATED_PARAMS_PATH) if params_payload else None,
        "analysis_dir": str(dest_run_root / ANALYSIS_DIR),
        "normalized_dir": str(dest_run_root / NORMALIZED_DIR),
        "hashes_dir": str(dest_run_root / HASHES_DIR),
        "stage_metrics": src_manifest.get("stage_metrics", {}),
        "stages": src_manifest.get("stages", {}),
        "artifacts": src_manifest.get("artifacts", {}),
    }
    if extra:
        manifest.update(extra)

    write_json_with_hash(dest_run_root / RUN_MANIFEST_PATH, manifest)
    write_json_with_hash(dest_run_root / RUN_SNAPSHOT_PATH, manifest)
    upsert_index_record(exp_root, manifest)
    return manifest


def oracle_energy_size_of_image(oracle_surface: OracleSurface) -> tuple[int, int, int] | None:
    """Read Z,Y,X dimensions from the oracle energy vector artifact."""
    from slavv_python.analytics.parity.matlab_vector_loader import is_matlab_energy_hdf5

    energy_path = oracle_surface.matlab_vector_paths.get("energy")
    if energy_path is None:
        return None
    if is_matlab_energy_hdf5(energy_path):
        import h5py

        with h5py.File(energy_path, "r") as handle:
            planes = np.asarray(handle["d"])
        if planes.ndim == 4 and planes.shape[0] >= 2:
            return cast(
                "tuple[int, int, int]",
                tuple(int(value) for value in planes[1].shape),
            )

    payload = loadmat(energy_path, squeeze_me=False, struct_as_record=False)
    energy = payload.get("energy")
    if energy is not None and hasattr(energy, "shape"):
        return cast("tuple[int, int, int]", tuple(int(value) for value in energy.shape))

    size_of_image = payload.get("size_of_image")
    if size_of_image is not None:
        size_array = np.atleast_1d(np.squeeze(size_of_image))
        if size_array.ndim == 1 and len(size_array) == 3:
            return cast("tuple[int, int, int]", tuple(int(v) for v in size_array))

    return None


def _oracle_energy_size_of_image(oracle_surface: OracleSurface) -> tuple[int, int, int] | None:
    """Backward-compatible alias."""
    return oracle_energy_size_of_image(oracle_surface)


# Re-export load_params_file from params_audit for callers that import from surfaces
load_params_file = params_audit.load_params_file


__all__ = [
    "_oracle_energy_size_of_image",
    "copy_source_surface",
    "ensure_dest_run_layout",
    "load_dataset_surface",
    "load_oracle_surface",
    "load_params_file",
    "oracle_energy_size_of_image",
    "resolve_input_file",
    "validate_exact_proof_source_surface",
    "validate_source_run_surface",
    "write_run_manifest",
]
