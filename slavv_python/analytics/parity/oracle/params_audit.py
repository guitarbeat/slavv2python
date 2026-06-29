"""Exact-route parameter audit and structured param persistence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.analytics.parity.constants import (
    EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS,
    EXACT_REQUIRED_PARAMETER_VALUES,
    EXACT_SHARED_METHOD_PARAMETER_KEYS,
    PARAM_DIFF_PATH,
    PYTHON_DERIVED_PARAMS_PATH,
    SHARED_PARAMS_PATH,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analytics.parity.utils import write_json_with_hash
from slavv_python.engine.state import fingerprint_jsonable, load_json_dict

if TYPE_CHECKING:
    from slavv_python.analytics.parity.oracle.models import (
        ExactProofSourceSurface,
        SourceRunSurface,
    )


def normalize_param_value(value: Any) -> Any:
    """Normalize numpy and container values for JSON comparison."""
    if isinstance(value, np.ndarray):
        return normalize_param_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [normalize_param_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_param_value(item) for key, item in value.items()}
    return value


def _normalize_param_value(value: Any) -> Any:
    """Backward-compatible alias for normalize_param_value."""
    return normalize_param_value(value)


def build_exact_params_audit(params: dict[str, Any]) -> dict[str, Any]:
    """Build a fairness audit for exact-route parameter payloads."""
    param_keys = {str(key) for key in params}
    shared_method_keys = sorted(param_keys & EXACT_SHARED_METHOD_PARAMETER_KEYS)
    orchestration_keys = sorted(param_keys & EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS)
    disallowed_python_only_keys = sorted(key for key in param_keys if key.startswith("parity_"))

    required_exact_mismatches: list[dict[str, Any]] = []
    for key, expected_value in EXACT_REQUIRED_PARAMETER_VALUES.items():
        actual_value = normalize_param_value(params.get(key))
        normalized_expected = normalize_param_value(expected_value)
        if actual_value != normalized_expected:
            required_exact_mismatches.append(
                {"key": key, "expected": normalized_expected, "found": actual_value}
            )

    known_keys = (
        EXACT_SHARED_METHOD_PARAMETER_KEYS
        | EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS
        | set(EXACT_REQUIRED_PARAMETER_VALUES)
    )
    unclassified_keys = sorted(
        key
        for key in param_keys
        if key not in known_keys and key not in disallowed_python_only_keys
    )

    return {
        "passed": not disallowed_python_only_keys and not required_exact_mismatches,
        "shared_method_param_count": len(shared_method_keys),
        "shared_method_params": {
            key: normalize_param_value(params[key]) for key in shared_method_keys if key in params
        },
        "allowed_orchestration_params": {
            key: normalize_param_value(params[key]) for key in orchestration_keys if key in params
        },
        "required_exact_values": {
            key: normalize_param_value(value)
            for key, value in EXACT_REQUIRED_PARAMETER_VALUES.items()
        },
        "required_exact_mismatches": required_exact_mismatches,
        "disallowed_python_only_keys": disallowed_python_only_keys,
        "unclassified_keys": unclassified_keys,
    }


def persist_param_storage(dest_run_root: Path, params: dict[str, Any]) -> dict[str, Any]:
    """Persist structured parameters for exact-route runs."""
    audit = build_exact_params_audit(params)
    shared_params = cast("dict[str, Any]", dict(audit.get("shared_method_params", {})))
    derived_keys = sorted(set(params) - set(shared_params))

    orchestration_params = {
        key: normalize_param_value(params[key])
        for key in derived_keys
        if key in EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS
    }
    python_only_params = {
        key: normalize_param_value(params[key]) for key in derived_keys if key.startswith("parity_")
    }
    unclassified_params = {
        key: normalize_param_value(params[key])
        for key in derived_keys
        if key not in orchestration_params and key not in python_only_params
    }

    python_derived = {
        "orchestration_params": orchestration_params,
        "python_only_params": python_only_params,
        "unclassified_params": unclassified_params,
    }

    param_diff = {
        "shared_param_count": len(shared_params),
        "shared_param_keys": sorted(shared_params),
        "derived_param_keys": derived_keys,
        "required_exact_values": cast(
            "dict[str, Any]", dict(audit.get("required_exact_values", {}))
        ),
        "required_exact_mismatches": cast(
            "list[dict[str, Any]]", list(audit.get("required_exact_mismatches", []))
        ),
        "disallowed_python_only_keys": cast(
            "list[str]", list(audit.get("disallowed_python_only_keys", []))
        ),
        "shared_params_hash": fingerprint_jsonable(shared_params),
        "python_derived_params_hash": fingerprint_jsonable(python_derived),
    }

    write_json_with_hash(dest_run_root / SHARED_PARAMS_PATH, shared_params)
    write_json_with_hash(dest_run_root / PYTHON_DERIVED_PARAMS_PATH, python_derived)
    write_json_with_hash(dest_run_root / PARAM_DIFF_PATH, param_diff)
    write_json_with_hash(dest_run_root / VALIDATED_PARAMS_PATH, params)

    return {
        "shared_params": shared_params,
        "python_derived_params": python_derived,
        "param_diff": param_diff,
    }


def load_params_file(
    source_surface: SourceRunSurface | ExactProofSourceSurface,
    params_arg: str | None,
) -> dict[str, Any]:
    """Load the JSON parameters either from the CLI or the slavv_python run metadata."""
    if params_arg:
        path = Path(params_arg).expanduser().resolve()
    else:
        path = source_surface.validated_params_path

    payload = load_json_dict(path)
    if payload is None:
        raise ValueError(f"expected JSON object in params file: {path}")

    if payload.get("comparison_exact_network") is True:
        audit = build_exact_params_audit(payload)
        if not audit["passed"]:
            mismatches = audit.get("required_exact_mismatches", [])
            disallowed = audit.get("disallowed_python_only_keys", [])
            msg = "disallowed exact parameters:"
            if mismatches:
                msg += f" mismatches={mismatches}"
            if disallowed:
                msg += f" disallowed Python-only parity keys={disallowed}"
            raise ValueError(msg)

    return cast("dict[str, Any]", payload)


__all__ = [
    "_normalize_param_value",
    "build_exact_params_audit",
    "load_params_file",
    "normalize_param_value",
    "persist_param_storage",
]
