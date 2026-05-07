"""Promotion logic for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2, copytree
from typing import TYPE_CHECKING

from slavv_python.io.matlab_exact_proof import (
    EXACT_STAGE_ORDER,
    find_matlab_vector_paths,
    load_normalized_matlab_vectors,
)
from slavv_python.runtime.run_state import fingerprint_file, fingerprint_jsonable, load_json_dict

from .constants import (
    ANALYSIS_DIR,
    DATASET_INPUT_DIR,
    DATASET_MANIFEST_PATH,
    EXPERIMENT_PARAMS_DIR,
    EXPERIMENT_REFS_DIR,
    HASHES_DIR,
    METADATA_DIR,
    NORMALIZED_DIR,
    ORACLE_DISCOVERY_STAGES,
    ORACLE_MANIFEST_PATH,
    REPORT_MANIFEST_PATH,
    RUN_MANIFEST_PATH,
)
from .index import ensure_experiment_root_layout, resolve_experiment_root, upsert_index_record
from .models import OracleSurface
from .utils import (
    entity_id_from_path,
    now_iso,
    persist_normalized_payloads,
    resolve_python_commit,
    string_or_none,
    write_hash_sidecar,
    write_json_with_hash,
    write_text_with_hash,
)

if TYPE_CHECKING:
    import argparse


def materialize_dataset_record(
    experiment_root: Path | None,
    *,
    dataset_hash: str | None,
    dataset_file: Path | None,
) -> str | None:
    """Fingerprint and catalog a dataset in the experiment tree."""
    resolved_hash = dataset_hash
    if resolved_hash is None and dataset_file is not None and dataset_file.is_file():
        resolved_hash = fingerprint_file(dataset_file)
    if experiment_root is None or resolved_hash is None:
        return resolved_hash

    ensure_experiment_root_layout(experiment_root)
    dataset_root = experiment_root / "datasets" / resolved_hash
    dataset_root.mkdir(parents=True, exist_ok=True)

    input_file: Path | None = None
    input_bytes: int | None = None
    if dataset_file is not None and dataset_file.is_file():
        input_root = dataset_root / DATASET_INPUT_DIR
        input_root.mkdir(parents=True, exist_ok=True)
        input_file = input_root / dataset_file.name
        if input_file.exists():
            existing_hash = fingerprint_file(input_file)
            if existing_hash != resolved_hash:
                raise ValueError(
                    f"Dataset root already contains a different payload for hash {resolved_hash}"
                )
        else:
            copy2(dataset_file, input_file)
        write_hash_sidecar(input_file)
        input_bytes = input_file.stat().st_size

    manifest_path = dataset_root / DATASET_MANIFEST_PATH
    existing_manifest = load_json_dict(manifest_path) or {}
    manifest_payload = {
        "manifest_version": 1,
        "kind": "dataset",
        "dataset_hash": resolved_hash,
        "dataset_root": str(dataset_root),
        "stored_input_file": str(input_file)
        if input_file
        else existing_manifest.get("stored_input_file"),
        "input_filename": dataset_file.name
        if dataset_file
        else existing_manifest.get("input_filename"),
        "input_bytes": input_bytes
        if input_bytes is not None
        else existing_manifest.get("input_bytes"),
        "timestamps": {
            "created_at": existing_manifest.get("timestamps", {}).get("created_at", now_iso()),
            "updated_at": now_iso(),
        },
    }
    write_json_with_hash(manifest_path, manifest_payload)
    upsert_index_record(
        experiment_root,
        {
            "kind": "dataset",
            "id": resolved_hash,
            "path": str(dataset_root),
            "status": "ready",
            "dataset_hash": resolved_hash,
            "updated_at": manifest_payload["timestamps"]["updated_at"],
        },
    )
    return resolved_hash


def materialize_oracle_root(
    *,
    matlab_batch_dir: Path,
    oracle_root: Path,
    dataset_hash: str | None,
    oracle_id: str | None,
    matlab_source_version: str | None,
) -> OracleSurface:
    """Stash a raw MATLAB batch into a structured oracle root with normalized artifacts."""
    oracle_root = oracle_root.expanduser().resolve()
    (oracle_root / METADATA_DIR).mkdir(parents=True, exist_ok=True)
    (oracle_root / NORMALIZED_DIR).mkdir(parents=True, exist_ok=True)
    (oracle_root / HASHES_DIR).mkdir(parents=True, exist_ok=True)

    raw_results_root = oracle_root / "01_Input" / "matlab_results"
    raw_results_root.mkdir(parents=True, exist_ok=True)
    raw_batch_dir = raw_results_root / matlab_batch_dir.name
    if not raw_batch_dir.exists():
        copytree(matlab_batch_dir, raw_batch_dir)

    vector_paths = find_matlab_vector_paths(raw_batch_dir, ORACLE_DISCOVERY_STAGES)
    normalized_payloads = load_normalized_matlab_vectors(raw_batch_dir, EXACT_STAGE_ORDER)
    normalized_artifacts = persist_normalized_payloads(
        oracle_root, group_name="oracle", payloads=normalized_payloads
    )

    raw_vector_hashes: dict[str, str] = {}
    for stage, path in vector_paths.items():
        h = fingerprint_file(path)
        raw_vector_hashes[stage] = h
        write_text_with_hash(oracle_root / HASHES_DIR / f"oracle_raw_{stage}.sha256", h)

    resolved_id = (
        oracle_id or f"{matlab_batch_dir.name}_{fingerprint_jsonable(raw_vector_hashes)[:12]}"
    )
    manifest = {
        "manifest_version": 1,
        "kind": "matlab_oracle",
        "oracle_id": resolved_id,
        "oracle_root": str(oracle_root),
        "dataset_hash": dataset_hash,
        "matlab_source_version": matlab_source_version,
        "matlab_batch_dir": str(raw_batch_dir),
        "raw_vector_hashes": raw_vector_hashes,
        "normalized_artifacts": normalized_artifacts,
        "timestamps": {"created_at": now_iso(), "updated_at": now_iso()},
    }
    write_json_with_hash(oracle_root / ORACLE_MANIFEST_PATH, manifest)

    upsert_index_record(
        resolve_experiment_root(oracle_root),
        {
            "kind": "matlab_oracle",
            "id": resolved_id,
            "path": str(oracle_root),
            "status": "ready",
            "dataset_hash": dataset_hash,
            "updated_at": manifest["timestamps"]["updated_at"],
        },
    )

    return OracleSurface(
        oracle_root=oracle_root,
        manifest_path=oracle_root / ORACLE_MANIFEST_PATH,
        matlab_batch_dir=raw_batch_dir,
        matlab_vector_paths=vector_paths,
        oracle_id=resolved_id,
        matlab_source_version=matlab_source_version,
        dataset_hash=dataset_hash,
    )


def handle_promote_oracle(args: argparse.Namespace) -> None:
    """Command handler for promoting a MATLAB batch to an oracle."""
    batch_dir = Path(args.matlab_batch_dir).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve()
    if not batch_dir.is_dir():
        raise ValueError(f"MATLAB batch directory not found: {batch_dir}")
    if oracle_root.exists():
        raise ValueError(f"Oracle root already exists: {oracle_root}")

    exp_root = resolve_experiment_root(oracle_root) or oracle_root.parent.parent
    dataset_file = Path(args.dataset_file).expanduser().resolve() if args.dataset_file else None
    ds_hash = materialize_dataset_record(
        exp_root, dataset_hash=string_or_none(args.dataset_hash), dataset_file=dataset_file
    )
    materialize_oracle_root(
        matlab_batch_dir=batch_dir,
        oracle_root=oracle_root,
        dataset_hash=ds_hash,
        oracle_id=string_or_none(args.oracle_id),
        matlab_source_version=string_or_none(args.matlab_source_version),
    )
    print(str(oracle_root / ORACLE_MANIFEST_PATH))


def handle_promote_dataset(args: argparse.Namespace) -> None:
    """Command handler for promoting a raw file to a cataloged dataset."""
    dataset_file = Path(args.dataset_file).expanduser().resolve()
    if not dataset_file.is_file():
        raise ValueError(f"Dataset file not found: {dataset_file}")

    exp_root = Path(args.experiment_root).expanduser().resolve()
    ds_hash = materialize_dataset_record(exp_root, dataset_hash=None, dataset_file=dataset_file)
    if ds_hash is None:
        raise ValueError(f"Could not fingerprint dataset: {dataset_file}")
    print(str(exp_root / "datasets" / ds_hash / DATASET_MANIFEST_PATH))


def handle_promote_report(args: argparse.Namespace) -> None:
    """Command handler for promoting a disposable run to a stable report."""
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"Run root not found: {run_root}")
    exp_root = resolve_experiment_root(run_root)
    if exp_root is None:
        raise ValueError(f"Run root is not under a structured experiment root: {run_root}")

    report_root = (
        Path(args.report_root).expanduser().resolve()
        if args.report_root
        else exp_root / "reports" / entity_id_from_path(run_root)
    )
    if report_root.exists():
        raise ValueError(f"Report root already exists: {report_root}")

    report_root.mkdir(parents=True, exist_ok=False)
    for rel in (EXPERIMENT_REFS_DIR, EXPERIMENT_PARAMS_DIR, ANALYSIS_DIR):
        src = run_root / rel
        if src.is_dir():
            copytree(src, report_root / rel)
    (report_root / METADATA_DIR).mkdir(parents=True, exist_ok=True)

    src_manifest = load_json_dict(run_root / RUN_MANIFEST_PATH) or {}
    report_manifest = {
        "manifest_version": 1,
        "kind": "promoted_report",
        "report_id": entity_id_from_path(report_root),
        "source_run_id": src_manifest.get("run_id", entity_id_from_path(run_root)),
        "source_run_root": str(run_root),
        "report_root": str(report_root),
        "dataset_hash": src_manifest.get("dataset_hash"),
        "oracle_id": src_manifest.get("oracle_id"),
        "python_commit": src_manifest.get("python_commit") or resolve_python_commit(exp_root),
        "timestamps": {"created_at": now_iso(), "updated_at": now_iso()},
    }
    write_json_with_hash(report_root / REPORT_MANIFEST_PATH, report_manifest)

    # Update source run
    src_manifest.update(
        {
            "promoted_report_root": str(report_root),
            "promotion_state": "promoted",
        }
    )
    src_manifest.setdefault("timestamps", {})["updated_at"] = now_iso()
    write_json_with_hash(run_root / RUN_MANIFEST_PATH, src_manifest)

    upsert_index_record(
        exp_root,
        {
            "kind": "promoted_report",
            "id": report_manifest["report_id"],
            "status": "promoted",
            "path": str(report_root),
            "dataset_hash": report_manifest.get("dataset_hash"),
            "oracle_id": report_manifest.get("oracle_id"),
            "updated_at": report_manifest["timestamps"]["updated_at"],
        },
    )
    print(str(report_root / REPORT_MANIFEST_PATH))
