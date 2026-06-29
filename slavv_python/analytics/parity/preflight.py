"""Exact-route preflight checks before long certification runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import psutil

from slavv_python.analytics.parity.constants import (
    EXPERIMENT_PROVENANCE_PATH,
    PREFLIGHT_EXACT_JSON_PATH,
    PREFLIGHT_EXACT_TEXT_PATH,
    RUN_SNAPSHOT_PATH,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analytics.parity.coordinator import ExactProofCoordinator
from slavv_python.analytics.parity.params_audit import build_exact_params_audit
from slavv_python.analytics.parity.reports import render_exact_preflight_report
from slavv_python.analytics.parity.surfaces import oracle_energy_size_of_image
from slavv_python.analytics.parity.utils import now_iso, write_json_with_hash, write_text_with_hash
from slavv_python.engine.state import load_json_dict

if TYPE_CHECKING:
    from pathlib import Path

    from slavv_python.analytics.parity.models import DatasetSurface, OracleSurface


def _available_system_bytes() -> int:
    return int(psutil.virtual_memory().available)


def resolve_exact_run_image_shape(
    *,
    dataset_surface: DatasetSurface | None,
    oracle_surface: OracleSurface | None,
    dest_run_root: Path,
) -> tuple[int, int, int] | None:
    """Infer Z,Y,X shape for memory estimation without loading the full volume."""
    if oracle_surface is not None:
        oracle_shape = oracle_energy_size_of_image(oracle_surface)
        if oracle_shape is not None:
            return oracle_shape

    prov = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH) or {}
    raw_size = prov.get("oracle_size_of_image")
    if isinstance(raw_size, list) and len(raw_size) == 3:
        return cast("tuple[int, int, int]", tuple(int(value) for value in raw_size))

    if dataset_surface is not None:
        try:
            import tifffile

            with tifffile.TiffFile(dataset_surface.input_file) as handle:
                shape = tuple(int(value) for value in handle.series[0].shape)
            if len(shape) == 3:
                return cast("tuple[int, int, int]", shape)
        except Exception:
            return None
    return None


def build_exact_preflight_report(
    *,
    dest_run_root: Path,
    dataset_surface: DatasetSurface | None = None,
    oracle_surface: OracleSurface | None = None,
    params: dict[str, Any] | None = None,
    memory_safety_fraction: float = 0.8,
    force: bool = False,
) -> dict[str, Any]:
    """Assemble preflight findings for an exact-route run directory."""
    warnings: list[str] = []
    errors: list[str] = []
    checks: dict[str, Any] = {}

    dest_run_root = dest_run_root.resolve()
    checks["dest_run_root"] = str(dest_run_root)
    checks["dest_exists"] = dest_run_root.is_dir()

    if dataset_surface is not None and oracle_surface is not None:
        checks["dataset_hash"] = dataset_surface.dataset_hash
        checks["oracle_id"] = oracle_surface.oracle_id
        if (
            oracle_surface.dataset_hash
            and oracle_surface.dataset_hash != dataset_surface.dataset_hash
        ):
            errors.append("dataset_oracle_hash_mismatch")

    if params is None and (dest_run_root / VALIDATED_PARAMS_PATH).is_file():
        params = load_json_dict(dest_run_root / VALIDATED_PARAMS_PATH)

    params_audit: dict[str, Any] | None = None
    if params is not None:
        params_audit = build_exact_params_audit(params)
        checks["params_audit"] = params_audit
        if not params_audit.get("passed"):
            errors.append("params_audit_failed")

    image_shape = resolve_exact_run_image_shape(
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        dest_run_root=dest_run_root,
    )
    memory_estimate: dict[str, Any] | None = None
    memory_gate: dict[str, Any] | None = None
    if image_shape is None:
        warnings.append("image_shape_unknown_memory_gate_skipped")
    else:
        memory_estimate = ExactProofCoordinator.estimate_exact_route_memory(image_shape)
        available_bytes = _available_system_bytes()
        budget_bytes = int(available_bytes * memory_safety_fraction)
        required_bytes = int(memory_estimate["estimated_required_bytes"])
        memory_gate = {
            "available_bytes": available_bytes,
            "budget_bytes": budget_bytes,
            "required_bytes": required_bytes,
            "memory_safety_fraction": memory_safety_fraction,
            "passed": required_bytes <= budget_bytes,
        }
        checks["image_shape"] = list(image_shape)
        if not memory_gate["passed"]:
            if force:
                warnings.append("memory_gate_overridden")
            else:
                errors.append("insufficient_memory")

    if dest_run_root.is_dir():
        snapshot = load_json_dict(dest_run_root / RUN_SNAPSHOT_PATH) or {}
        snapshot_status = snapshot.get("status")
        checks["run_snapshot_status"] = snapshot_status
        if snapshot_status == "running":
            warnings.append("run_snapshot_status_running")

        provenance = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH) or {}
        if provenance:
            checks["experiment_provenance"] = {
                "dataset_hash": provenance.get("dataset_hash"),
                "oracle_id": provenance.get("oracle_id"),
                "stop_after": provenance.get("stop_after"),
            }
            if dataset_surface is not None and provenance.get("dataset_hash") not in (
                None,
                dataset_surface.dataset_hash,
            ):
                errors.append("provenance_dataset_hash_mismatch")
            if oracle_surface is not None and provenance.get("oracle_id") not in (
                None,
                oracle_surface.oracle_id,
            ):
                errors.append("provenance_oracle_id_mismatch")

    passed = not errors
    return {
        "passed": passed,
        "forced": force,
        "warnings": warnings,
        "errors": errors,
        "checks": checks,
        "params_audit": params_audit,
        "memory_estimate": memory_estimate,
        "memory_gate": memory_gate,
        "generated_at": now_iso(),
    }


def persist_exact_preflight_report(
    dest_run_root: Path,
    report_payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Write JSON and text preflight artifacts under the run root."""
    json_path = dest_run_root / PREFLIGHT_EXACT_JSON_PATH
    text_path = dest_run_root / PREFLIGHT_EXACT_TEXT_PATH
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_with_hash(json_path, report_payload)
    write_text_with_hash(text_path, render_exact_preflight_report(report_payload))
    return json_path, text_path


def run_exact_preflight_for_surfaces(
    dest_run_root: Path,
    *,
    dataset_surface: DatasetSurface | None = None,
    oracle_surface: OracleSurface | None = None,
    params: dict[str, Any] | None = None,
    memory_safety_fraction: float = 0.8,
    force: bool = False,
    persist: bool = True,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """Run preflight checks and optionally persist reports under dest_run_root."""
    dest_run_root = dest_run_root.resolve()
    report = build_exact_preflight_report(
        dest_run_root=dest_run_root,
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        params=params,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
    )
    if not persist:
        return report, None, None
    if dest_run_root.is_dir():
        json_path, text_path = persist_exact_preflight_report(dest_run_root, report)
        return report, json_path, text_path
    return report, None, None


__all__ = [
    "build_exact_preflight_report",
    "persist_exact_preflight_report",
    "resolve_exact_run_image_shape",
    "run_exact_preflight_for_surfaces",
]
