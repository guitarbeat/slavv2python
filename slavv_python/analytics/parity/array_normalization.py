"""Array normalization primitives for exact-route Oracle comparison."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def _normalize_python_bridge_payload(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return {
            "connections": np.empty((0, 2), dtype=np.int64),
            "traces": [],
            "scale_traces": [],
            "energy_traces": [],
            "energies": np.empty((0,), dtype=np.float64),
        }
    payload = cast("dict[str, Any]", raw_payload)
    return {
        "connections": _normalize_connection_array(payload.get("connections")),
        "traces": _normalize_float_matrix_list(payload.get("traces"), columns=3),
        "scale_traces": _normalize_float_vector_list(payload.get("scale_traces")),
        "energy_traces": _normalize_float_vector_list(payload.get("energy_traces")),
        "energies": _normalize_float_vector(payload.get("energies")),
    }


def _normalize_python_strands(value: Any) -> list[np.ndarray]:
    items = _coerce_sequence_items(value)
    strands: list[np.ndarray] = []
    for item in items:
        strands.append(_normalize_int_vector(item))
    return strands


def _normalize_matlab_strands(value: Any) -> list[np.ndarray]:
    connections = _normalize_matlab_connections(value)
    return [row.astype(np.int64, copy=False) for row in connections]


def _normalize_matlab_connections(value: Any) -> np.ndarray:
    return _normalize_connection_array(value, one_based=True)


def _normalize_optional_matlab_connections(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> np.ndarray:
    value = _optional_field(payload, *field_names)
    return _normalize_matlab_connections(value)


def _normalize_matlab_int_vector(value: Any) -> np.ndarray:
    return _normalize_int_vector(value, one_based=True)


def _normalize_optional_matlab_int_vector(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> np.ndarray:
    return _normalize_int_vector(_optional_field(payload, *field_names), one_based=True)


def _normalize_matlab_spatial_matrix(value: Any) -> np.ndarray:
    return _normalize_spatial_matrix(value, one_based=True)


def _normalize_optional_matlab_float_matrix(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
    *,
    columns: int,
) -> np.ndarray:
    return _normalize_float_matrix(
        _optional_field(payload, *field_names), columns=columns, one_based=True
    )


def _normalize_optional_matlab_spatial_matrix(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> np.ndarray:
    return _normalize_spatial_matrix(_optional_field(payload, *field_names), one_based=True)


def _normalize_matlab_spatial_matrix_list(value: Any) -> list[np.ndarray]:
    return _normalize_spatial_matrix_list(value, one_based=True)


def _normalize_optional_matlab_float_matrix_list(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
    *,
    columns: int,
) -> list[np.ndarray]:
    return _normalize_float_matrix_list(
        _optional_field(payload, *field_names), columns=columns, one_based=True
    )


def _normalize_optional_matlab_spatial_matrix_list(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> list[np.ndarray]:
    return _normalize_spatial_matrix_list(_optional_field(payload, *field_names), one_based=True)


def _normalize_matlab_float_vector_list(value: Any) -> list[np.ndarray]:
    return _normalize_float_vector_list(value, one_based=True)


def _normalize_matlab_spatial_scale_matrix_list(value: Any) -> list[np.ndarray]:
    return _normalize_spatial_scale_matrix_list(value, one_based=True)


def _normalize_optional_matlab_float_vector_list(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> list[np.ndarray]:
    return _normalize_float_vector_list(_optional_field(payload, *field_names), one_based=True)


def _normalize_connection_array(value: Any, *, one_based: bool = False) -> np.ndarray:
    array = np.asarray([] if value is None else value, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    normalized = np.atleast_2d(array).reshape(-1, 2).astype(np.int64, copy=False)
    if not one_based:
        return cast("np.ndarray", normalized)
    adjusted = normalized.copy()
    positive = adjusted > 0
    adjusted[positive] -= 1
    adjusted[~positive] = -1
    return cast("np.ndarray", adjusted)


def _normalize_int_vector(value: Any, *, one_based: bool = False) -> np.ndarray:
    array = np.asarray([] if value is None else value, dtype=np.float64).reshape(-1)
    if array.size:
        array = np.rint(array)
        if one_based:
            array = array - 1.0
    return cast("np.ndarray", array.astype(np.int64, copy=False))


def _normalize_int_array(value: Any, *, one_based: bool = False) -> np.ndarray:
    array = np.asarray([] if value is None else value, dtype=np.int64)
    if one_based and array.size:
        array = array - 1
    return cast("np.ndarray", array.astype(np.int64, copy=False))


def _normalize_float_vector(value: Any, *, one_based: bool = False) -> np.ndarray:
    array = np.asarray([] if value is None else value, dtype=np.float64).reshape(-1)
    if one_based and array.size:
        array = array - 1.0
    return cast("np.ndarray", array.astype(np.float64, copy=False))


def _normalize_float_array(value: Any) -> np.ndarray:
    array = np.asarray([] if value is None else value, dtype=np.float64)
    return cast("np.ndarray", array.astype(np.float64, copy=False))


def _normalize_float_matrix(value: Any, *, columns: int, one_based: bool = False) -> np.ndarray:
    array = np.asarray([] if value is None else value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float64)
    normalized = np.asarray(array, dtype=np.float64).reshape(-1, columns)
    if one_based:
        normalized = normalized - 1.0
    return cast("np.ndarray", normalized.astype(np.float64, copy=False))


def _normalize_spatial_matrix(value: Any, *, one_based: bool = False) -> np.ndarray:
    normalized = _normalize_float_matrix(value, columns=3, one_based=False)
    if normalized.size == 0:
        return normalized
    if one_based:
        normalized = normalized - 1.0
    # MATLAB spatial is [Y, X, Z]. We want Python [Z, Y, X].
    normalized = normalized[:, [2, 0, 1]]
    return cast("np.ndarray", normalized.astype(np.float64, copy=False))


def _normalize_float_vector_list(value: Any, *, one_based: bool = False) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for item in _coerce_sequence_items(value):
        vectors.append(_normalize_float_vector(item, one_based=one_based))
    return vectors


def _normalize_spatial_vector_list(value: Any) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for item in _coerce_sequence_items(value):
        normalized = _normalize_float_matrix(item, columns=3, one_based=False)
        vectors.append(normalized.astype(np.float64, copy=False))
    return vectors


def _normalize_float_matrix_list(
    value: Any,
    *,
    columns: int,
    one_based: bool = False,
) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for item in _coerce_sequence_items(value):
        matrices.append(_normalize_float_matrix(item, columns=columns, one_based=one_based))
    return matrices


def _normalize_spatial_matrix_list(value: Any, *, one_based: bool = False) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for item in _coerce_sequence_items(value):
        matrices.append(_normalize_spatial_matrix(item, one_based=one_based))
    return matrices


def _normalize_spatial_scale_matrix_list(
    value: Any, *, one_based: bool = False
) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for item in _coerce_sequence_items(value):
        normalized = _normalize_float_matrix(item, columns=4, one_based=False)
        if normalized.size == 0:
            matrices.append(np.empty((0, 4), dtype=np.float64))
            continue
        reordered = normalized
        if one_based:
            # Only the spatial voxel columns (Y, X, Z) are 1-based and need the
            # 1->0 shift. The scale column (S) is a 1-based scale index in BOTH the
            # MATLAB strand_subscripts and the Python output — the network strand
            # smoothing offsets the 0-based edge scale by +1 to match MATLAB's
            # output convention — so the scale column must NOT be shifted.
            reordered = reordered.copy()
            reordered[:, :3] = reordered[:, :3] - 1.0
        # MATLAB spatial is [Y, X, Z, S]. We want Python [Z, Y, X, S].
        reordered = reordered[:, [2, 0, 1, 3]]
        matrices.append(reordered.astype(np.float64, copy=False))
    return matrices


def _coerce_sequence_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim == 0:
            item: Any = value.item()
            return [] if item is None else [item]
        return list(value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    array = np.asarray(value)
    if array.size == 0:
        return []
    return [value]


def _require_key(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing MATLAB vector field: {key}")
    return payload[key]


def _optional_field(payload: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name in payload:
            return payload[field_name]
    return None
