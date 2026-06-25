"""Compare normalized MATLAB and Python exact-route artifacts."""

from __future__ import annotations

from collections import Counter
from hashlib import sha1
from typing import Any

import numpy as np

from slavv_python.analytics.parity.energy_ulp_proof import (
    EnergyFloatGateOptions,
    evaluate_energy_float_gate,
)
from slavv_python.analytics.parity.exact_proof_contract import EXACT_STAGE_FIELDS


def compare_exact_artifacts(
    matlab_artifacts: dict[str, dict[str, Any]],
    python_artifacts: dict[str, dict[str, Any]],
    stages: tuple[str, ...],
    *,
    energy_float_options: EnergyFloatGateOptions | None = None,
    float_tol: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compare normalized MATLAB and Python artifacts stage by stage.

    ``float_tol`` is an optional ``(rtol, atol)`` applied to continuous
    floating-point fields (ADR 0011); ``None`` means strict bit-identical
    comparison everywhere (regression / ``--strict-floats``). The ``energy.energy``
    field has its own scale-aware gate via ``energy_float_options``.
    """
    stage_summaries: dict[str, dict[str, Any]] = {}
    first_failure: dict[str, Any] | None = None

    energy_float_gate: dict[str, Any] | None = None
    for stage in stages:
        matlab_payload = matlab_artifacts[stage]
        python_payload = python_artifacts[stage]
        if stage == "energy" and energy_float_options is not None:
            mismatch, energy_float_gate = _compare_energy_stage(
                matlab_payload,
                python_payload,
                energy_float_options,
                float_tol=float_tol,
            )
        else:
            mismatch = _compare_dict(
                matlab_payload,
                python_payload,
                path=stage,
                float_tol=float_tol,
            )
        stage_summaries[stage] = {
            "passed": mismatch is None,
            "field_count": len(EXACT_STAGE_FIELDS[stage]),
        }
        if mismatch is not None:
            stage_summaries[stage]["first_failure"] = mismatch
            if first_failure is None:
                first_failure = mismatch

    result: dict[str, Any] = {
        "passed": first_failure is None,
        "stages": list(stages),
        "stage_summaries": stage_summaries,
        "first_failing_stage": first_failure["stage"] if first_failure is not None else None,
        "first_failing_field_path": first_failure["field_path"]
        if first_failure is not None
        else None,
        "first_failure": first_failure,
    }
    if energy_float_gate is not None:
        result["energy_float_gate"] = energy_float_gate
    return result


def _compare_energy_stage(
    matlab_payload: dict[str, Any],
    python_payload: dict[str, Any],
    energy_float_options: EnergyFloatGateOptions,
    *,
    float_tol: tuple[float, float] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Compare Energy stage with ADR 0011 float policy on energy.energy only."""
    gate_summary = evaluate_energy_float_gate(
        np.asarray(matlab_payload["energy"]),
        np.asarray(python_payload["energy"]),
        np.asarray(matlab_payload["scale_indices"]),
        np.asarray(python_payload["scale_indices"]),
        options=energy_float_options,
    )
    for key in EXACT_STAGE_FIELDS["energy"]:
        if key == "energy":
            if gate_summary["scale_mismatch_count"]:
                mismatch = _compare_value(
                    matlab_payload["scale_indices"],
                    python_payload["scale_indices"],
                    path="energy",
                    field_path="energy.scale_indices",
                    float_tol=float_tol,
                )
                if mismatch is not None:
                    return mismatch, gate_summary
            if not gate_summary["passed"]:
                return (
                    _mismatch(
                        "energy",
                        "energy.energy",
                        "energy float gate",
                        matlab_payload["energy"],
                        python_payload["energy"],
                    ),
                    gate_summary,
                )
            continue
        if key == "scale_indices":
            continue
        mismatch = _compare_value(
            matlab_payload[key],
            python_payload[key],
            path="energy",
            field_path=f"energy.{key}",
            float_tol=float_tol,
        )
        if mismatch is not None:
            return mismatch, gate_summary
    return None, gate_summary


def _compare_dict(
    matlab_payload: dict[str, Any],
    python_payload: dict[str, Any],
    *,
    path: str,
    float_tol: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    for key, matlab_value in matlab_payload.items():
        field_path = f"{path}.{key}"
        if key not in python_payload:
            return _mismatch(
                path,
                field_path,
                "missing field",
                matlab_value,
                "<missing>",
            )
        mismatch = _compare_value(
            matlab_value,
            python_payload[key],
            path=path,
            field_path=field_path,
            float_tol=float_tol,
        )
        if mismatch is not None:
            return mismatch
    return None


def _compare_value(
    matlab_value: Any,
    python_value: Any,
    *,
    path: str,
    field_path: str,
    float_tol: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    if isinstance(matlab_value, dict):
        if not isinstance(python_value, dict):
            return _mismatch(path, field_path, "value mismatch", matlab_value, python_value)
        return _compare_dict(matlab_value, python_value, path=field_path, float_tol=float_tol)

    if isinstance(matlab_value, list):
        if not isinstance(python_value, list):
            return _mismatch(path, field_path, "value mismatch", matlab_value, python_value)
        if len(matlab_value) != len(python_value):
            return _mismatch(path, field_path, "shape mismatch", matlab_value, python_value)
        if _lists_equal(matlab_value, python_value):
            return None
        if _list_ordering_equivalent(matlab_value, python_value):
            return _mismatch(path, field_path, "ordering mismatch", matlab_value, python_value)
        for index, (matlab_item, python_item) in enumerate(zip(matlab_value, python_value)):
            mismatch = _compare_value(
                matlab_item,
                python_item,
                path=path,
                field_path=f"{field_path}[{index}]",
                float_tol=float_tol,
            )
            if mismatch is not None:
                return mismatch
        return _mismatch(path, field_path, "value mismatch", matlab_value, python_value)

    matlab_array = np.asarray(matlab_value)
    python_array = np.asarray(python_value)
    if matlab_array.shape != python_array.shape:
        return _mismatch(path, field_path, "shape mismatch", matlab_array, python_array)
    # ADR 0011: when float_tol is set, continuous float fields compare with
    # np.allclose (cross-library BLAS/libm drift is bounded, not a logic
    # difference); integer/topological fields stay strict. float_tol=None forces
    # strict bit-identical comparison everywhere (--strict-floats / regression).
    # The energy.energy field has its own scale-aware gate and never reaches here.
    if (
        float_tol is not None
        and matlab_array.dtype.kind == "f"
        and python_array.dtype.kind == "f"
    ):
        rtol, atol = float_tol
        if matlab_array.size == 0 or np.allclose(
            python_array, matlab_array, rtol=rtol, atol=atol, equal_nan=True
        ):
            return None
        return _mismatch(path, field_path, "value mismatch (float tol)", matlab_array, python_array)
    if np.array_equal(matlab_array, python_array):
        return None
    if _array_ordering_equivalent(matlab_array, python_array):
        return _mismatch(path, field_path, "ordering mismatch", matlab_array, python_array)
    return _mismatch(path, field_path, "value mismatch", matlab_array, python_array)


def _lists_equal(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    return all(_value_equals(left_item, right_item) for left_item, right_item in zip(left, right))


def _value_equals(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_value_equals(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return _lists_equal(left, right)
    return np.array_equal(np.asarray(left), np.asarray(right))


def _list_ordering_equivalent(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    return Counter(_value_signature(item) for item in left) == Counter(
        _value_signature(item) for item in right
    )


def _array_ordering_equivalent(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.ndim == 0:
        return False
    if left.ndim == 1:
        return np.array_equal(np.sort(left), np.sort(right))
    if left.ndim == 2:
        return Counter(_row_signature(row) for row in left) == Counter(
            _row_signature(row) for row in right
        )
    return False


def _value_signature(value: Any) -> tuple[Any, ...]:
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _value_signature(inner_value)) for key, inner_value in value.items()),
        )
    if isinstance(value, list):
        return ("list", tuple(_value_signature(item) for item in value))
    array = np.asarray(value)
    return (
        "ndarray",
        tuple(array.shape),
        str(array.dtype),
        sha1(np.ascontiguousarray(array).tobytes()).hexdigest(),
    )


def _row_signature(row: np.ndarray) -> tuple[Any, ...]:
    row_array = np.asarray(row)
    return (
        tuple(row_array.shape),
        str(row_array.dtype),
        sha1(np.ascontiguousarray(row_array).tobytes()).hexdigest(),
    )


def _mismatch(
    stage: str,
    field_path: str,
    mismatch_type: str,
    matlab_value: Any,
    python_value: Any,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "field_path": field_path,
        "mismatch_type": mismatch_type,
        "matlab_preview": _preview_value(matlab_value),
        "python_preview": _preview_value(python_value),
    }


def _preview_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {"keys": list(value.keys())}
    if isinstance(value, list):
        return {
            "length": len(value),
            "first": _preview_value(value[0]) if value else None,
        }
    array = np.asarray(value)
    preview_values = array.reshape(-1)[:6].tolist() if array.size else []
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "values": preview_values,
    }


__all__ = ["compare_exact_artifacts"]
