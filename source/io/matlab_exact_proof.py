"""MATLAB-oracle artifact normalization and proof helpers for the exact route."""

from __future__ import annotations

from collections import Counter
from hashlib import sha1
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.io import loadmat

from ..runtime.run_state import atomic_joblib_dump
from ..utils.safe_unpickle import safe_load

if TYPE_CHECKING:
    from pathlib import Path

EXACT_STAGE_ORDER: tuple[str, ...] = ("vertices", "edges", "network")
EXACT_STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "energy": ("energy", "scale_indices", "energy_4d", "lumen_radius_microns"),
    "vertices": ("positions", "scales", "energies"),
    "edges": (
        "connections",
        "traces",
        "scale_traces",
        "energy_traces",
        "energies",
        "bridge_vertex_positions",
        "bridge_vertex_scales",
        "bridge_vertex_energies",
        "bridge_edges",
    ),
    "network": (
        "strands",
        "bifurcations",
        "strand_subscripts",
        "strand_energy_traces",
        "mean_strand_energies",
        "vessel_directions",
    ),
}
BRIDGE_EDGE_FIELDS: tuple[str, ...] = (
    "connections",
    "traces",
    "scale_traces",
    "energy_traces",
    "energies",
)


def find_single_matlab_batch_dir(run_root: Path) -> Path:
    """Return the single preserved MATLAB batch directory under a staged run root."""
    matlab_results_dir = run_root / "01_Input" / "matlab_results"
    if not matlab_results_dir.is_dir():
        raise ValueError(f"missing MATLAB results directory: {matlab_results_dir}")

    batch_dirs = sorted(
        path
        for path in matlab_results_dir.iterdir()
        if path.is_dir() and path.name.startswith("batch_")
    )
    if not batch_dirs:
        raise ValueError(f"no MATLAB batch directory found under {matlab_results_dir}")
    if len(batch_dirs) > 1:
        joined = ", ".join(str(path) for path in batch_dirs)
        raise ValueError(
            f"expected one MATLAB batch directory under {matlab_results_dir}, found: {joined}"
        )
    return batch_dirs[0]


def find_matlab_vector_paths(batch_dir: Path) -> dict[str, Path]:
    """Locate the raw MATLAB vector files for vertices, edges, and network."""
    vectors_dir = batch_dir / "vectors"
    if not vectors_dir.is_dir():
        raise ValueError(f"missing MATLAB vectors directory: {vectors_dir}")

    stage_patterns: dict[str, tuple[str, ...]] = {
        "vertices": ("curated_vertices*.mat", "vertices*.mat"),
        "edges": ("edges*.mat", "curated_edges*.mat"),
        "network": ("network*.mat",),
    }
    stage_paths: dict[str, Path] = {}
    for stage in EXACT_STAGE_ORDER:
        candidates: list[Path] = []
        seen: set[Path] = set()
        for pattern in stage_patterns[stage]:
            for path in sorted(vectors_dir.glob(pattern)):
                if not path.is_file() or path in seen:
                    continue
                candidates.append(path)
                seen.add(path)
            if candidates:
                break
        if not candidates:
            raise ValueError(f"missing raw MATLAB {stage} vector file under {vectors_dir}")
        if len(candidates) > 1:
            joined = ", ".join(str(path) for path in candidates)
            raise ValueError(
                f"expected one raw MATLAB {stage} vector file under {vectors_dir}, found: {joined}"
            )
        stage_paths[stage] = candidates[0]
    return stage_paths


def load_normalized_matlab_vectors(
    batch_dir: Path,
    stages: tuple[str, ...] = EXACT_STAGE_ORDER,
) -> dict[str, dict[str, Any]]:
    """Load and normalize the requested raw MATLAB vector files."""
    vector_paths = find_matlab_vector_paths(batch_dir)
    normalized: dict[str, dict[str, Any]] = {}
    for stage in stages:
        normalized[stage] = load_normalized_matlab_stage(vector_paths[stage], stage)
    return normalized


def load_normalized_matlab_stage(path: Path, stage: str) -> dict[str, Any]:
    """Load and normalize a single raw MATLAB vector file."""
    matlab_payload = cast(
        "dict[str, Any]",
        loadmat(path, squeeze_me=stage != "energy", struct_as_record=False),
    )
    if stage == "vertices":
        return _normalize_matlab_vertices_payload(matlab_payload)
    if stage == "energy":
        return _normalize_matlab_energy_payload(matlab_payload)
    if stage == "edges":
        return _normalize_matlab_edges_payload(matlab_payload)
    if stage == "network":
        return _normalize_matlab_network_payload(matlab_payload)
    raise ValueError(f"unsupported exact-proof stage: {stage}")


def load_normalized_python_checkpoints(
    checkpoints_dir: Path,
    stages: tuple[str, ...] = EXACT_STAGE_ORDER,
) -> dict[str, dict[str, Any]]:
    """Load and normalize the requested Python checkpoint payloads."""
    normalized: dict[str, dict[str, Any]] = {}
    for stage in stages:
        checkpoint_path = checkpoints_dir / f"checkpoint_{stage}.pkl"
        if not checkpoint_path.is_file():
            raise ValueError(f"missing Python checkpoint for exact proof: {checkpoint_path}")
        payload = safe_load(checkpoint_path)
        if not isinstance(payload, dict):
            raise ValueError(f"expected mapping payload in {checkpoint_path}")
        normalized[stage] = normalize_python_stage_payload(stage, cast("dict[str, Any]", payload))
    return normalized


def sync_exact_vertex_checkpoint_from_matlab(
    checkpoint_path: Path,
    batch_dir: Path,
) -> dict[str, Any]:
    """Overwrite vertex checkpoint parity fields from the canonical MATLAB vector surface."""
    payload = safe_load(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping payload in {checkpoint_path}")

    normalized_vertices = load_normalized_matlab_vectors(batch_dir, ("vertices",))["vertices"]
    updated = dict(cast("dict[str, Any]", payload))

    updated["positions"] = np.asarray(normalized_vertices["positions"], dtype=np.float64)
    updated["scales"] = np.asarray(normalized_vertices["scales"], dtype=np.int64)
    updated["energies"] = np.asarray(normalized_vertices["energies"], dtype=np.float64)
    updated["count"] = len(updated["positions"])
    atomic_joblib_dump(updated, checkpoint_path)
    return updated


def normalize_python_stage_payload(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Python checkpoint payload into the exact-proof contract."""
    if stage == "energy":
        return {
            "energy": _normalize_float_array(payload.get("energy")),
            "scale_indices": _normalize_int_array(payload.get("scale_indices")),
            "energy_4d": _normalize_float_array(payload.get("energy_4d")),
            "lumen_radius_microns": _normalize_float_vector(payload.get("lumen_radius_microns")),
        }
    if stage == "vertices":
        return {
            "positions": _normalize_float_matrix(payload.get("positions"), columns=3),
            "scales": _normalize_int_vector(payload.get("scales")),
            "energies": _normalize_float_vector(payload.get("energies")),
        }
    if stage == "edges":
        return {
            "connections": _normalize_connection_array(payload.get("connections")),
            "traces": _normalize_float_matrix_list(payload.get("traces"), columns=3),
            "scale_traces": _normalize_float_vector_list(payload.get("scale_traces")),
            "energy_traces": _normalize_float_vector_list(payload.get("energy_traces")),
            "energies": _normalize_float_vector(payload.get("energies")),
            "bridge_vertex_positions": _normalize_float_matrix(
                payload.get("bridge_vertex_positions"),
                columns=3,
            ),
            "bridge_vertex_scales": _normalize_int_vector(payload.get("bridge_vertex_scales")),
            "bridge_vertex_energies": _normalize_float_vector(
                payload.get("bridge_vertex_energies")
            ),
            "bridge_edges": _normalize_python_bridge_payload(payload.get("bridge_edges")),
        }
    if stage == "network":
        return {
            "strands": _normalize_python_strands(payload.get("strands")),
            "bifurcations": _normalize_int_vector(payload.get("bifurcations")),
            "strand_subscripts": _normalize_float_matrix_list(
                payload.get("strand_subscripts"),
                columns=4,
            ),
            "strand_energy_traces": _normalize_float_vector_list(
                payload.get("strand_energy_traces"),
            ),
            "mean_strand_energies": _normalize_float_vector(payload.get("mean_strand_energies")),
            "vessel_directions": _normalize_float_matrix_list(
                payload.get("vessel_directions"),
                columns=3,
            ),
        }
    raise ValueError(f"unsupported exact-proof stage: {stage}")


def compare_exact_artifacts(
    matlab_artifacts: dict[str, dict[str, Any]],
    python_artifacts: dict[str, dict[str, Any]],
    stages: tuple[str, ...],
) -> dict[str, Any]:
    """Compare normalized MATLAB and Python artifacts stage by stage."""
    stage_summaries: dict[str, dict[str, Any]] = {}
    first_failure: dict[str, Any] | None = None

    for stage in stages:
        matlab_payload = matlab_artifacts[stage]
        python_payload = python_artifacts[stage]
        mismatch = _compare_dict(
            matlab_payload,
            python_payload,
            path=stage,
        )
        stage_summaries[stage] = {
            "passed": mismatch is None,
            "field_count": len(EXACT_STAGE_FIELDS[stage]),
        }
        if mismatch is not None:
            stage_summaries[stage]["first_failure"] = mismatch
            if first_failure is None:
                first_failure = mismatch

    return {
        "passed": first_failure is None,
        "stages": list(stages),
        "stage_summaries": stage_summaries,
        "first_failing_stage": first_failure["stage"] if first_failure is not None else None,
        "first_failing_field_path": first_failure["field_path"]
        if first_failure is not None
        else None,
        "first_failure": first_failure,
    }


def render_exact_proof_report(report: dict[str, Any]) -> str:
    """Render a human-readable exact-proof report."""
    stage_lines = [
        f"{stage}: {'PASS' if summary.get('passed') else 'FAIL'}"
        for stage, summary in report["stage_summaries"].items()
    ]
    lines = [
        "Exact proof report",
        f"Status: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Stages: {', '.join(report.get('stages', []))}",
    ]
    report_scope = report.get("report_scope")
    if isinstance(report_scope, str) and report_scope:
        lines.append(f"Scope: {report_scope}")
    lines.extend(
        [
            "",
            "Stage summary",
            *stage_lines,
        ]
    )

    candidate_surface = report.get("candidate_surface")
    if isinstance(candidate_surface, dict):
        lines.extend(
            [
                "",
                "Candidate surface",
                (
                    "Counts: "
                    f"MATLAB={candidate_surface.get('matlab_pair_count', 0)} "
                    f"Python={candidate_surface.get('python_pair_count', 0)} "
                    f"matched={candidate_surface.get('matched_pair_count', 0)} "
                    f"missing={candidate_surface.get('missing_pair_count', 0)} "
                    f"extra={candidate_surface.get('extra_pair_count', 0)}"
                ),
            ]
        )

    first_failure = report.get("first_failure")
    if isinstance(first_failure, dict):
        lines.extend(
            [
                "",
                "First failure",
                f"Stage: {first_failure['stage']}",
                f"Field: {first_failure['field_path']}",
                f"Mismatch: {first_failure['mismatch_type']}",
                f"MATLAB: {first_failure['matlab_preview']}",
                f"Python: {first_failure['python_preview']}",
            ]
        )

    return "\n".join(lines)


def _normalize_matlab_vertices_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "positions": _normalize_matlab_spatial_matrix(
            _require_key(payload, "vertex_space_subscripts"),
        ),
        "scales": _normalize_matlab_int_vector(_require_key(payload, "vertex_scale_subscripts")),
        "energies": _normalize_float_vector(_require_key(payload, "vertex_energies")),
    }


def _normalize_matlab_energy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "energy": _normalize_float_array(_require_key(payload, "energy")),
        "scale_indices": _normalize_int_array(
            _require_key(payload, "scale_indices"),
            one_based=True,
        ),
        "energy_4d": _normalize_float_array(_require_key(payload, "energy_4d")),
        "lumen_radius_microns": _normalize_float_vector(
            _require_key(payload, "lumen_radius_microns"),
        ),
    }


def _normalize_matlab_edges_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "connections": _normalize_matlab_connections(_require_key(payload, "edges2vertices")),
        "traces": _normalize_matlab_spatial_matrix_list(
            _require_key(payload, "edge_space_subscripts"),
        ),
        "scale_traces": _normalize_matlab_float_vector_list(
            _require_key(payload, "edge_scale_subscripts"),
        ),
        "energy_traces": _normalize_float_vector_list(_require_key(payload, "edge_energies")),
        "energies": _normalize_float_vector(
            payload.get("mean_edge_energies", payload.get("energies")),
        ),
        "bridge_vertex_positions": _normalize_optional_matlab_spatial_matrix(
            payload,
            ("bridge_vertex_space_subscripts", "bridge_vertex_positions"),
        ),
        "bridge_vertex_scales": _normalize_optional_matlab_int_vector(
            payload,
            ("bridge_vertex_scale_subscripts", "bridge_vertex_scales"),
        ),
        "bridge_vertex_energies": _normalize_float_vector(
            _optional_field(payload, "bridge_vertex_energies"),
        ),
        "bridge_edges": _normalize_matlab_bridge_payload(payload),
    }


def _normalize_matlab_network_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "strands": _normalize_matlab_strands(_require_key(payload, "strands2vertices")),
        "bifurcations": _normalize_matlab_int_vector(
            _require_key(payload, "bifurcation_vertices"),
        ),
        "strand_subscripts": _normalize_matlab_spatial_scale_matrix_list(
            _require_key(payload, "strand_subscripts"),
        ),
        "strand_energy_traces": _normalize_float_vector_list(
            _require_key(payload, "strand_energies"),
        ),
        "mean_strand_energies": _normalize_float_vector(
            _require_key(payload, "mean_strand_energies"),
        ),
        "vessel_directions": _normalize_spatial_vector_list(
            _require_key(payload, "vessel_directions"),
        ),
    }


def _normalize_matlab_bridge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("bridge_edges")
    nested_mapping = nested.__dict__ if hasattr(nested, "__dict__") else {}
    has_prefixed_bridge_fields = any(key.startswith("bridge_") for key in payload)
    if not nested_mapping and not has_prefixed_bridge_fields:
        return {
            "connections": np.empty((0, 2), dtype=np.int64),
            "traces": [],
            "scale_traces": [],
            "energy_traces": [],
            "energies": np.empty((0,), dtype=np.float64),
        }

    source_payload = nested_mapping if nested_mapping else payload
    return {
        "connections": _normalize_optional_matlab_connections(
            source_payload,
            ("bridge_edges2vertices", "bridge_connections", "edges2vertices", "connections"),
        ),
        "traces": _normalize_optional_matlab_spatial_matrix_list(
            source_payload,
            ("bridge_edge_space_subscripts", "traces", "edge_space_subscripts"),
        ),
        "scale_traces": _normalize_optional_matlab_float_vector_list(
            source_payload,
            ("bridge_edge_scale_subscripts", "scale_traces", "edge_scale_subscripts"),
        ),
        "energy_traces": _normalize_float_vector_list(
            _optional_field(
                source_payload,
                "bridge_edge_energies",
                "energy_traces",
                "edge_energies",
            )
        ),
        "energies": _normalize_float_vector(
            _optional_field(
                source_payload,
                "bridge_mean_edge_energies",
                "energies",
                "mean_edge_energies",
            )
        ),
    }


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


def _normalize_matlab_float_matrix(value: Any, *, columns: int) -> np.ndarray:
    return _normalize_float_matrix(value, columns=columns, one_based=True)


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


def _normalize_matlab_float_matrix_list(value: Any, *, columns: int) -> list[np.ndarray]:
    return _normalize_float_matrix_list(value, columns=columns, one_based=True)


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
    array = np.asarray([] if value is None else value, dtype=np.int64).reshape(-1)
    if one_based and array.size:
        array = array - 1
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
    normalized = normalized[:, ::-1]
    if one_based:
        normalized = normalized - 1.0
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
        vectors.append(normalized[:, ::-1].astype(np.float64, copy=False))
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
        reordered = np.column_stack(
            (normalized[:, 2], normalized[:, 1], normalized[:, 0], normalized[:, 3])
        )
        if one_based:
            reordered = reordered - 1.0
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


def _compare_dict(
    matlab_payload: dict[str, Any],
    python_payload: dict[str, Any],
    *,
    path: str,
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
) -> dict[str, Any] | None:
    if isinstance(matlab_value, dict):
        if not isinstance(python_value, dict):
            return _mismatch(path, field_path, "value mismatch", matlab_value, python_value)
        return _compare_dict(matlab_value, python_value, path=field_path)

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
            )
            if mismatch is not None:
                return mismatch
        return _mismatch(path, field_path, "value mismatch", matlab_value, python_value)

    matlab_array = np.asarray(matlab_value)
    python_array = np.asarray(python_value)
    if matlab_array.shape != python_array.shape:
        return _mismatch(path, field_path, "shape mismatch", matlab_array, python_array)
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


__all__ = [
    "BRIDGE_EDGE_FIELDS",
    "EXACT_STAGE_FIELDS",
    "EXACT_STAGE_ORDER",
    "compare_exact_artifacts",
    "find_matlab_vector_paths",
    "find_single_matlab_batch_dir",
    "load_normalized_matlab_stage",
    "load_normalized_matlab_vectors",
    "load_normalized_python_checkpoints",
    "normalize_python_stage_payload",
    "render_exact_proof_report",
    "sync_exact_vertex_checkpoint_from_matlab",
]
