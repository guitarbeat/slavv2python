"""Deterministic, compact diagnostics for failed exact-artifact proofs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .utils import write_json_with_hash, write_text_with_hash

if TYPE_CHECKING:
    from pathlib import Path


def persist_mismatch_diagnostics(
    dest_run_root: Path,
    *,
    report: dict[str, Any],
    matlab_artifacts: dict[str, dict[str, Any]],
    python_artifacts: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[Path, Path] | tuple[None, None]:
    """Persist a first-failure diagnosis suitable for a reproducible next probe."""
    stage = report.get("first_failing_stage")
    if not isinstance(stage, str) or not stage:
        return None, None
    diagnosis = build_mismatch_diagnostics(
        stage,
        matlab_artifacts.get(stage, {}),
        python_artifacts.get(stage, {}),
        params,
    )
    analysis_dir = dest_run_root / "03_Analysis"
    json_path = analysis_dir / f"exact_mismatch_{stage}.json"
    text_path = analysis_dir / f"exact_mismatch_{stage}.txt"
    write_json_with_hash(json_path, diagnosis)
    write_text_with_hash(text_path, render_mismatch_diagnostics(diagnosis))
    return json_path, text_path


def build_mismatch_diagnostics(
    stage: str,
    matlab_payload: dict[str, Any],
    python_payload: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Describe mismatched fields in deterministic Fortran-order coordinates."""
    fields: list[dict[str, Any]] = []
    for field, matlab_value in matlab_payload.items():
        if field not in python_payload:
            fields.append({"field": field, "kind": "missing", "python": "<missing>"})
            continue
        matlab_array = np.asarray(matlab_value)
        python_array = np.asarray(python_payload[field])
        if matlab_array.shape != python_array.shape:
            fields.append(
                {
                    "field": field,
                    "kind": "shape_mismatch",
                    "matlab_shape": list(matlab_array.shape),
                    "python_shape": list(python_array.shape),
                }
            )
            continue
        if matlab_array.dtype.kind not in "biufc" or python_array.dtype.kind not in "biufc":
            if not np.array_equal(matlab_array, python_array):
                fields.append({"field": field, "kind": "value_mismatch"})
            continue
        equal = np.equal(matlab_array, python_array)
        if np.all(equal):
            continue
        indices = np.flatnonzero((~equal).ravel(order="F"))
        first_linear = int(indices[0])
        coordinate = list(np.unravel_index(first_linear, matlab_array.shape, order="F"))
        finite = np.isfinite(matlab_array) & np.isfinite(python_array)
        delta = np.abs(matlab_array[finite] - python_array[finite])
        fields.append(
            {
                "field": field,
                "kind": "numeric_mismatch",
                "mismatch_count": int(indices.size),
                "first_fortran_linear_index": first_linear,
                "first_coordinate": coordinate,
                "matlab_value": _json_scalar(matlab_array[tuple(coordinate)]),
                "python_value": _json_scalar(python_array[tuple(coordinate)]),
                "max_abs_delta": float(np.max(delta)) if delta.size else None,
            }
        )
    diagnosis: dict[str, Any] = {"stage": stage, "fields": fields}
    if stage == "energy":
        diagnosis["energy_context"] = _energy_context(
            matlab_payload, python_payload, fields, params
        )
    return diagnosis


def _energy_context(
    matlab_payload: dict[str, Any],
    python_payload: dict[str, Any],
    fields: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    first = next((field for field in fields if "first_coordinate" in field), None)
    if first is None:
        return {}
    coordinate = first["first_coordinate"]
    context: dict[str, Any] = {
        "coordinate_zyx": coordinate,
        "probe_function": "probe_exact_energy_voxel_at_octave",
    }
    matlab_scales = matlab_payload.get("scale_indices")
    python_scales = python_payload.get("scale_indices")
    if matlab_scales is not None and python_scales is not None:
        matlab_scale = int(np.asarray(matlab_scales)[tuple(coordinate)])
        python_scale = int(np.asarray(python_scales)[tuple(coordinate)])
        context.update(
            {
                "matlab_winner_scale": matlab_scale,
                "python_winner_scale": python_scale,
                "winner_scale_disagrees": matlab_scale != python_scale,
            }
        )
    return context


def render_mismatch_diagnostics(diagnosis: dict[str, Any]) -> str:
    """Render a concise human-readable failed-proof summary."""
    lines = [f"Exact mismatch diagnosis: {diagnosis['stage']}"]
    for field in diagnosis.get("fields", []):
        lines.append(f"- {field['field']}: {field['kind']}")
        if "mismatch_count" in field:
            lines.append(
                f"  count={field['mismatch_count']} first={field['first_coordinate']} "
                f"matlab={field['matlab_value']} python={field['python_value']}"
            )
    context = diagnosis.get("energy_context")
    if isinstance(context, dict) and context:
        lines.extend(("", "Energy context:"))
        lines.extend(f"- {key}: {value}" for key, value in context.items())
    return "\n".join(lines)


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    "build_mismatch_diagnostics",
    "persist_mismatch_diagnostics",
    "render_mismatch_diagnostics",
]
