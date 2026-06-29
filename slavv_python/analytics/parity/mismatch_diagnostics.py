"""Deterministic, compact diagnostics for failed exact-artifact proofs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.analytics.parity.utils import write_json_with_hash, write_text_with_hash

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
        python_value = python_payload[field]
        # Variable-length fields (e.g. edge traces: a list of ragged per-edge arrays)
        # are inhomogeneous; np.asarray raises or yields an object array. Compare
        # lengths only instead of crashing the whole diagnosis.
        try:
            matlab_array = np.asarray(matlab_value)
            python_array = np.asarray(python_value)
        except ValueError:
            matlab_array = python_array = None
        if (
            matlab_array is None
            or python_array is None
            or matlab_array.dtype == object
            or python_array.dtype == object
        ):
            matlab_len = len(matlab_value) if hasattr(matlab_value, "__len__") else None
            python_len = len(python_value) if hasattr(python_value, "__len__") else None
            fields.append(
                {
                    "field": field,
                    "kind": "length_mismatch" if matlab_len != python_len else "sequence_field",
                    "matlab_len": matlab_len,
                    "python_len": python_len,
                }
            )
            continue
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
        mismatch_mask = ~equal
        ulp_stats = _ulp_mismatch_stats(
            matlab_array[mismatch_mask],
            python_array[mismatch_mask],
        )
        entry: dict[str, Any] = {
            "field": field,
            "kind": "numeric_mismatch",
            "mismatch_count": int(indices.size),
            "first_fortran_linear_index": first_linear,
            "first_coordinate": coordinate,
            "matlab_value": _json_scalar(matlab_array[tuple(coordinate)]),
            "python_value": _json_scalar(python_array[tuple(coordinate)]),
            "max_abs_delta": float(np.max(delta)) if delta.size else None,
        }
        entry.update(ulp_stats)
        fields.append(entry)
    diagnosis: dict[str, Any] = {"stage": stage, "fields": fields}
    if stage == "energy":
        diagnosis["energy_context"] = _energy_context(
            matlab_payload, python_payload, fields, params
        )
        diagnosis["energy_scale_agreeing_mismatch"] = _energy_scale_agreeing_mismatch(
            matlab_payload, python_payload
        )
    return diagnosis


def _ordered_float64_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = bits >> np.uint64(63)
    ordered_bits: np.ndarray = np.where(sign, np.uint64(0xFFFFFFFFFFFFFFFF) - bits, bits)
    return ordered_bits


def _ulp_mismatch_stats(matlab_values: np.ndarray, python_values: np.ndarray) -> dict[str, Any]:
    matlab_f = np.asarray(matlab_values, dtype=np.float64)
    python_f = np.asarray(python_values, dtype=np.float64)
    equal = matlab_f == python_f
    ordered: np.ndarray = _ordered_float64_bits(matlab_f).astype(np.int64)
    ordered_py: np.ndarray = _ordered_float64_bits(python_f).astype(np.int64)
    ulp = np.abs(ordered - ordered_py)
    ulp[equal] = 0
    histogram: dict[str, int] = {}
    for bucket in range(9):
        histogram[str(bucket)] = int(np.count_nonzero(ulp == bucket))
    histogram["9_plus"] = int(np.count_nonzero(ulp > 8))
    return {
        "max_ulp": int(ulp.max()) if ulp.size else 0,
        "ulp_histogram": histogram,
        "ulp_p50": float(np.percentile(ulp, 50)) if ulp.size else 0.0,
        "ulp_p90": float(np.percentile(ulp, 90)) if ulp.size else 0.0,
    }


def _energy_scale_agreeing_mismatch(
    matlab_payload: dict[str, Any],
    python_payload: dict[str, Any],
) -> dict[str, Any]:
    matlab_energy = matlab_payload.get("energy")
    python_energy = python_payload.get("energy")
    matlab_scales = matlab_payload.get("scale_indices")
    python_scales = python_payload.get("scale_indices")
    if matlab_energy is None or python_energy is None:
        return {}
    matlab_energy_a = np.asarray(matlab_energy, dtype=np.float64)
    python_energy_a = np.asarray(python_energy, dtype=np.float64)
    if matlab_scales is None or python_scales is None:
        return {}
    scale_equal = np.asarray(matlab_scales) == np.asarray(python_scales)
    energy_mismatch = matlab_energy_a != python_energy_a
    mask = scale_equal & energy_mismatch
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"mismatch_count": 0}
    stats = _ulp_mismatch_stats(matlab_energy_a[mask], python_energy_a[mask])
    finite = np.isfinite(matlab_energy_a[mask]) & np.isfinite(python_energy_a[mask])
    delta = np.abs(matlab_energy_a[mask][finite] - python_energy_a[mask][finite])
    stats["mismatch_count"] = count
    stats["max_abs_delta"] = float(np.max(delta)) if delta.size else 0.0
    return stats


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
            ulp_note = ""
            if "max_ulp" in field:
                ulp_note = f" max_ulp={field['max_ulp']}"
            lines.append(
                f"  count={field['mismatch_count']} first={field['first_coordinate']} "
                f"matlab={field['matlab_value']} python={field['python_value']}{ulp_note}"
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
