"""Fail-fast parity helpers for the native-first exact route."""

from __future__ import annotations

import json
from collections import Counter
from importlib import resources
from typing import Any, cast

import numpy as np

from ..core.edge_candidates_internal.common import (
    _build_matlab_global_watershed_lut,
    _candidate_endpoint_pair_set,
)

LUT_FIXTURE_RESOURCE = "matlab_global_watershed_luts.json"
DEBUG_MAP_FIELDS: tuple[str, ...] = (
    "energy_map",
    "vertex_index_map",
    "pointer_map",
    "raw_pointer_map",
    "d_over_r_map",
    "branch_order_map",
)


def load_builtin_lut_fixture() -> dict[str, Any]:
    """Load the checked-in exact-LUT fixture payload."""
    fixture_text = (
        resources.files("slavv_python.io.fixtures")
        .joinpath(LUT_FIXTURE_RESOURCE)
        .read_text(encoding="utf-8")
    )
    payload = json.loads(fixture_text)
    if not isinstance(payload, dict):
        raise ValueError("expected mapping payload in built-in LUT fixture")
    return cast("dict[str, Any]", payload)


def compare_lut_fixture_payload(
    fixture_payload: dict[str, Any],
    *,
    size_of_image: tuple[int, int, int],
    microns_per_voxel: np.ndarray,
    lumen_radius_microns: np.ndarray,
) -> dict[str, Any]:
    """Compare the shared Python LUT builder against a checked-in fixture payload."""
    stage_summaries: dict[str, dict[str, Any]] = {}
    first_failure: dict[str, Any] | None = None

    expected_shape = tuple(int(value) for value in fixture_payload.get("size_of_image", []))
    actual_shape = tuple(int(value) for value in size_of_image)
    if expected_shape != actual_shape:
        first_failure = {
            "stage": "luts",
            "field_path": "luts.inputs.size_of_image",
            "mismatch_type": "value mismatch",
            "expected_preview": list(expected_shape),
            "actual_preview": list(actual_shape),
        }

    expected_mpv = np.asarray(
        fixture_payload.get("microns_per_voxel", []),
        dtype=np.float32,
    ).reshape(-1)
    actual_mpv = np.asarray(microns_per_voxel, dtype=np.float32).reshape(-1)
    if first_failure is None and not np.array_equal(expected_mpv, actual_mpv):
        first_failure = {
            "stage": "luts",
            "field_path": "luts.inputs.microns_per_voxel",
            "mismatch_type": _array_mismatch_type(expected_mpv, actual_mpv),
            "expected_preview": expected_mpv.tolist(),
            "actual_preview": actual_mpv.tolist(),
        }

    expected_radii = np.asarray(
        fixture_payload.get("lumen_radius_microns", []),
        dtype=np.float32,
    ).reshape(-1)
    actual_radii = np.asarray(lumen_radius_microns, dtype=np.float32).reshape(-1)
    if first_failure is None and not np.array_equal(expected_radii, actual_radii):
        first_failure = {
            "stage": "luts",
            "field_path": "luts.inputs.lumen_radius_microns",
            "mismatch_type": _array_mismatch_type(expected_radii, actual_radii),
            "expected_preview": _preview_array(expected_radii),
            "actual_preview": _preview_array(actual_radii),
        }

    scales_payload = fixture_payload.get("scales", {})
    if not isinstance(scales_payload, dict):
        raise ValueError("expected mapping payload for LUT fixture scales")

    for scale_key in sorted(scales_payload, key=lambda value: int(value)):
        scale_index = int(scale_key)
        expected_scale_payload = _normalize_lut_fixture_entry(
            cast("dict[str, Any]", scales_payload[scale_key])
        )
        actual_scale_payload = _build_matlab_global_watershed_lut(
            scale_index,
            size_of_image=size_of_image,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=1.0,
        )
        mismatch = _compare_lut_scale_payload(
            expected_scale_payload,
            actual_scale_payload,
            field_path=f"luts.scales[{scale_index}]",
        )
        stage_summaries[str(scale_index)] = {
            "passed": mismatch is None,
            "entry_count": len(expected_scale_payload["linear_offsets"]),
        }
        if mismatch is not None:
            stage_summaries[str(scale_index)]["first_failure"] = mismatch
            if first_failure is None:
                first_failure = mismatch

    return {
        "passed": first_failure is None,
        "report_scope": "exact LUT parity only",
        "size_of_image": list(actual_shape),
        "scale_count": len(scales_payload),
        "stage_summaries": stage_summaries,
        "first_failure": first_failure,
    }


def render_lut_proof_report(report_payload: dict[str, Any]) -> str:
    """Render a compact LUT-proof report."""
    status = (
        "SKIP"
        if report_payload.get("skipped")
        else ("PASS" if report_payload.get("passed") else "FAIL")
    )
    lines = [
        "Exact LUT proof report",
        f"Status: {status}",
        f"Image shape: {report_payload.get('size_of_image')}",
        f"Scale count: {report_payload.get('scale_count')}",
    ]
    if report_payload.get("skipped"):
        lines.append(f"Reason: {report_payload.get('skip_reason')}")
        fixture_inputs = report_payload.get("fixture_inputs")
        source_inputs = report_payload.get("source_inputs")
        if isinstance(fixture_inputs, dict):
            lines.append(f"Fixture inputs: {fixture_inputs}")
        if isinstance(source_inputs, dict):
            lines.append(f"Source inputs: {source_inputs}")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "Scale summary",
        ]
    )
    for scale_key, summary in report_payload.get("stage_summaries", {}).items():
        lines.append(f"scale {scale_key}: {'PASS' if summary.get('passed') else 'FAIL'}")
    first_failure = report_payload.get("first_failure")
    if isinstance(first_failure, dict):
        lines.extend(
            [
                "",
                "First failure",
                f"Field: {first_failure.get('field_path')}",
                f"Type: {first_failure.get('mismatch_type')}",
                f"Expected: {first_failure.get('expected_preview')}",
                f"Actual: {first_failure.get('actual_preview')}",
            ]
        )
    return "\n".join(lines)


def build_candidate_snapshot_payload(
    candidates: dict[str, Any],
    *,
    include_debug_maps: bool = False,
) -> dict[str, Any]:
    """Return a slim candidate payload suitable for replay and coverage checks."""
    snapshot = {
        "connections": np.asarray(
            candidates.get("connections", np.empty((0, 2))), dtype=np.int32
        ).reshape(-1, 2),
        "traces": [np.asarray(trace, dtype=np.float32) for trace in candidates.get("traces", [])],
        "scale_traces": [
            np.asarray(trace, dtype=np.float32).reshape(-1)
            for trace in candidates.get("scale_traces", [])
        ],
        "energy_traces": [
            np.asarray(trace, dtype=np.float32).reshape(-1)
            for trace in candidates.get("energy_traces", [])
        ],
        "metrics": np.asarray(candidates.get("metrics", np.empty((0,))), dtype=np.float32).reshape(
            -1
        ),
        "origin_indices": np.asarray(
            candidates.get("origin_indices", np.empty((0,))),
            dtype=np.int32,
        ).reshape(-1),
        "connection_sources": [
            str(value) for value in list(candidates.get("connection_sources", []))
        ],
        "diagnostics": dict(candidates.get("diagnostics", {})),
        "candidate_source": str(candidates.get("candidate_source", "unknown")),
        "matlab_global_watershed_exact": bool(
            candidates.get("matlab_global_watershed_exact", False)
        ),
    }
    if include_debug_maps:
        for field_name in DEBUG_MAP_FIELDS:
            if field_name in candidates:
                snapshot[field_name] = np.asarray(candidates[field_name]).copy()
    return snapshot


def build_candidate_coverage_report(
    matlab_edges_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compare raw Python candidate endpoint pairs against final MATLAB edge pairs."""
    matlab_pair_set = _candidate_endpoint_pair_set(
        np.asarray(matlab_edges_payload.get("connections", np.empty((0, 2))), dtype=np.int64)
    )
    python_pair_set = _candidate_endpoint_pair_set(
        np.asarray(candidate_payload.get("connections", np.empty((0, 2))), dtype=np.int64)
    )
    matlab_pairs = _sorted_pair_list(matlab_pair_set)
    python_pairs = _sorted_pair_list(python_pair_set)
    matched_pairs = _sorted_pair_list(matlab_pair_set & python_pair_set)
    missing_pairs = _sorted_pair_list(matlab_pair_set - python_pair_set)
    extra_pairs = _sorted_pair_list(python_pair_set - matlab_pair_set)

    return {
        "passed": not missing_pairs and not extra_pairs,
        "report_scope": "candidate coverage only",
        "matlab_pair_count": len(matlab_pairs),
        "python_pair_count": len(python_pairs),
        "matched_pair_count": len(matched_pairs),
        "missing_pair_count": len(missing_pairs),
        "extra_pair_count": len(extra_pairs),
        "missing_pairs": missing_pairs,
        "extra_pairs": extra_pairs,
        "matched_pair_samples": matched_pairs[:10],
        "missing_pair_samples": missing_pairs[:10],
        "extra_pair_samples": extra_pairs[:10],
        "top_missing_vertices": _vertex_counter_payload(missing_pairs),
        "top_extra_vertices": _vertex_counter_payload(extra_pairs),
    }


def render_candidate_coverage_report(report_payload: dict[str, Any]) -> str:
    """Render a compact candidate coverage report."""
    lines = [
        "Candidate coverage report",
        f"Status: {'PASS' if report_payload.get('passed') else 'FAIL'}",
        (
            "Counts: "
            f"MATLAB={report_payload.get('matlab_pair_count', 0)} "
            f"Python={report_payload.get('python_pair_count', 0)} "
            f"matched={report_payload.get('matched_pair_count', 0)} "
            f"missing={report_payload.get('missing_pair_count', 0)} "
            f"extra={report_payload.get('extra_pair_count', 0)}"
        ),
    ]
    missing_samples = report_payload.get("missing_pair_samples", [])
    extra_samples = report_payload.get("extra_pair_samples", [])
    if missing_samples:
        lines.append(f"Missing pair samples: {missing_samples}")
    if extra_samples:
        lines.append(f"Extra pair samples: {extra_samples}")
    top_missing = report_payload.get("top_missing_vertices", [])
    if top_missing:
        lines.append(f"Top missing vertices: {top_missing}")
    top_extra = report_payload.get("top_extra_vertices", [])
    if top_extra:
        lines.append(f"Top extra vertices: {top_extra}")
    return "\n".join(lines)


def _normalize_lut_fixture_entry(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "linear_offsets": np.asarray(payload.get("linear_offsets", []), dtype=np.int64).reshape(-1),
        "local_subscripts": np.asarray(payload.get("local_subscripts", []), dtype=np.int32).reshape(
            -1, 3
        ),
        "r_over_R": np.asarray(payload.get("r_over_R", []), dtype=np.float32).reshape(-1),
        "unit_vectors": np.asarray(payload.get("unit_vectors", []), dtype=np.float32).reshape(
            -1, 3
        ),
    }


def _compare_lut_scale_payload(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    field_path: str,
) -> dict[str, Any] | None:
    for field_name in ("linear_offsets", "local_subscripts", "r_over_R", "unit_vectors"):
        expected_array = np.asarray(expected[field_name])
        actual_array = np.asarray(actual[field_name])
        if np.array_equal(expected_array, actual_array):
            continue
        return {
            "stage": "luts",
            "field_path": f"{field_path}.{field_name}",
            "mismatch_type": _array_mismatch_type(expected_array, actual_array),
            "expected_preview": _preview_array(expected_array),
            "actual_preview": _preview_array(actual_array),
        }
    return None


def _array_mismatch_type(expected: np.ndarray, actual: np.ndarray) -> str:
    if expected.shape != actual.shape:
        return "shape mismatch"
    if expected.ndim == 1:
        if np.array_equal(np.sort(expected), np.sort(actual)):
            return "ordering mismatch"
        return "value mismatch"

    if expected.ndim == 2:
        expected_rows = Counter(tuple(row.tolist()) for row in expected)
        actual_rows = Counter(tuple(row.tolist()) for row in actual)
        if expected_rows == actual_rows:
            return "ordering mismatch"
    return "value mismatch"


def _preview_array(value: np.ndarray, *, max_rows: int = 5) -> list[Any]:
    array = np.asarray(value)
    if array.ndim <= 1:
        return cast("list[Any]", array[:max_rows].tolist())
    return cast("list[Any]", array[:max_rows].tolist())


def _sorted_pair_list(pairs: set[tuple[int, int]]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in sorted(pairs)]


def _vertex_counter_payload(pairs: list[list[int]]) -> list[dict[str, int]]:
    counter: Counter[int] = Counter()
    for start_vertex, end_vertex in pairs:
        counter[int(start_vertex)] += 1
        counter[int(end_vertex)] += 1
    return [
        {"vertex_index": int(vertex_index), "count": int(count)}
        for vertex_index, count in counter.most_common(10)
    ]


__all__ = [
    "DEBUG_MAP_FIELDS",
    "build_candidate_coverage_report",
    "build_candidate_snapshot_payload",
    "compare_lut_fixture_payload",
    "load_builtin_lut_fixture",
    "render_candidate_coverage_report",
    "render_lut_proof_report",
]
