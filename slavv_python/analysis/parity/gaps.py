"""Diagnostic logic for candidate gaps in native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from slavv_python.runtime.run_state import load_json_dict
from slavv_python.runtime.run_tracking.io import atomic_write_json, atomic_write_text

from .constants import (
    CANDIDATE_COVERAGE_JSON_PATH,
    EDGE_CANDIDATE_AUDIT_PATH,
    GAP_DIAGNOSIS_JSON_PATH,
    GAP_DIAGNOSIS_TEXT_PATH,
)
from .execution import ensure_dest_run_layout
from .utils import (
    now_iso,
    write_hash_sidecar,
)

if TYPE_CHECKING:
    from pathlib import Path


def _build_gap_hotspot_rows(
    top_vertices: list[dict[str, Any]],
    per_origin_summary: dict[str, dict[str, Any]],
    *,
    gap_kind: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Join candidate coverage hotspots with origin-level diagnostic counters."""
    if not top_vertices:
        return []

    vertex_frame = pd.DataFrame(top_vertices).head(limit)
    vertex_frame["rank"] = range(1, len(vertex_frame) + 1)

    origin_records = []
    for origin_index, payload in per_origin_summary.items():
        record = {"origin_index": int(origin_index)}
        record.update(payload.get("counters", {}))
        origin_records.append(record)

    if not origin_records:
        vertex_frame["gap_kind"] = gap_kind
        return cast("list[dict[str, Any]]", vertex_frame.to_dict(orient="records"))

    origin_frame = pd.DataFrame(origin_records)

    # Ensure origin_index is int for merging
    vertex_frame["origin_index"] = pd.to_numeric(
        vertex_frame["origin_index"], errors="coerce"
    ).astype("Int64")
    origin_frame["origin_index"] = pd.to_numeric(
        origin_frame["origin_index"],
        errors="coerce",
    ).astype("Int64")

    merged = vertex_frame.merge(
        origin_frame, how="left", on="origin_index", suffixes=("", "_audit")
    )
    merged["gap_kind"] = gap_kind

    columns = [
        "gap_kind",
        "rank",
        "origin_index",
        "gap_count",
        "candidate_connection_count",
        "fallback_candidate_count",
        "frontier_candidate_count",
        "geodesic_candidate_count",
        "watershed_candidate_count",
        "candidate_endpoint_pair_count",
    ]
    available_columns = [column for column in columns if column in merged.columns]
    merged = merged.reindex(columns=available_columns)
    merged = merged.replace({pd.NA: None})
    return cast("list[dict[str, Any]]", merged.to_dict(orient="records"))


def build_gap_diagnosis_report(
    run_root: Path,
    *,
    limit: int = 10,
    coverage_payload: dict[str, Any] | None = None,
    audit_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a joined gap diagnosis report payload."""
    normalized_run_root = run_root.expanduser().resolve()

    coverage = (
        coverage_payload or load_json_dict(normalized_run_root / CANDIDATE_COVERAGE_JSON_PATH) or {}
    )
    audit = audit_payload or load_json_dict(normalized_run_root / EDGE_CANDIDATE_AUDIT_PATH) or {}

    per_origin_summary = cast("dict[str, dict[str, Any]]", audit.get("per_origin_summary", {}))
    diagnostic_counters = cast("dict[str, Any]", (audit or {}).get("diagnostic_counters", {}))
    pair_source_breakdown = cast("dict[str, Any]", (audit or {}).get("pair_source_breakdown", {}))

    warnings: list[str] = []
    if not audit:
        warnings.append("No origin-level candidate audit was found; hotspot joins are limited.")
    if audit and audit.get("use_frontier_tracer") is False:
        warnings.append("Exact frontier tracer was disabled in this recording.")
    if int(diagnostic_counters.get("watershed_total_pairs", 0)) == 0:
        warnings.append("No watershed candidate pairs were recorded.")
    if (
        int(pair_source_breakdown.get("fallback_only_pair_count", 0)) > 0
        and int(pair_source_breakdown.get("frontier_only_pair_count", 0)) == 0
        and int(pair_source_breakdown.get("watershed_only_pair_count", 0)) == 0
        and int(pair_source_breakdown.get("geodesic_only_pair_count", 0)) == 0
    ):
        warnings.append("All recorded candidate endpoint pairs were fallback-only.")

    top_missing_vertices = cast("list[dict[str, Any]]", coverage.get("top_missing_vertices", []))
    top_extra_vertices = cast("list[dict[str, Any]]", coverage.get("top_extra_vertices", []))
    hotspot_limit = max(1, int(limit))

    top_missing_hotspots = _build_gap_hotspot_rows(
        top_missing_vertices,
        per_origin_summary,
        gap_kind="missing",
        limit=hotspot_limit,
    )
    top_extra_hotspots = _build_gap_hotspot_rows(
        top_extra_vertices,
        per_origin_summary,
        gap_kind="extra",
        limit=hotspot_limit,
    )

    return {
        "run_root": str(normalized_run_root),
        "created_at": now_iso(),
        "report_scope": "candidate gap diagnosis",
        "passed": bool(coverage.get("passed")),
        "missing_pair_count": int(coverage.get("missing_pair_count", 0)),
        "extra_pair_count": int(coverage.get("extra_pair_count", 0)),
        "matched_pair_count": int(coverage.get("matched_pair_count", 0)),
        "matlab_pair_count": int(coverage.get("matlab_pair_count", 0)),
        "python_pair_count": int(coverage.get("python_pair_count", 0)),
        "missing_pair_samples": cast("list[list[int]]", coverage.get("missing_pair_samples", [])),
        "extra_pair_samples": cast("list[list[int]]", coverage.get("extra_pair_samples", [])),
        "missing_pairs": cast("list[list[int]]", coverage.get("missing_pairs", [])),
        "extra_pairs": cast("list[list[int]]", coverage.get("extra_pairs", [])),
        "warnings": warnings,
        "audit_present": bool(audit),
        "origin_summary_count": len(per_origin_summary),
        "diagnostic_counters": diagnostic_counters,
        "pair_source_breakdown": pair_source_breakdown,
        "top_missing_vertices": top_missing_vertices[:hotspot_limit],
        "top_extra_vertices": top_extra_vertices[:hotspot_limit],
        "top_missing_hotspots": top_missing_hotspots,
        "top_extra_hotspots": top_extra_hotspots,
    }


def render_gap_diagnosis_report(report_payload: dict[str, Any]) -> str:
    """Render a concise gap diagnosis focused on actionable hotspots."""
    lines = [
        "Gap diagnosis report",
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

    warnings = cast("list[str]", report_payload.get("warnings", []))
    if warnings:
        lines.extend(["", "Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)

    missing_hotspots = cast("list[dict[str, Any]]", report_payload.get("top_missing_hotspots", []))
    if missing_hotspots:
        lines.extend(["", "Top Missing Hotspots"])
        for row in missing_hotspots:
            lines.append(
                "- "
                f"origin={row.get('origin_index')} gap_count={row.get('gap_count')} "
                f"candidates={row.get('candidate_connection_count')} "
                f"fallback={row.get('fallback_candidate_count')} "
                f"frontier={row.get('frontier_candidate_count')} "
                f"watershed={row.get('watershed_candidate_count')}"
            )

    extra_hotspots = cast("list[dict[str, Any]]", report_payload.get("top_extra_hotspots", []))
    if extra_hotspots:
        lines.extend(["", "Top Extra Hotspots"])
        for row in extra_hotspots:
            lines.append(
                "- "
                f"origin={row.get('origin_index')} gap_count={row.get('gap_count')} "
                f"candidates={row.get('candidate_connection_count')} "
                f"fallback={row.get('fallback_candidate_count')} "
                f"frontier={row.get('frontier_candidate_count')} "
                f"watershed={row.get('watershed_candidate_count')}"
            )

    missing_samples = cast("list[list[int]]", report_payload.get("missing_pair_samples", []))
    if missing_samples:
        lines.append("")
        lines.append(f"Missing pair samples: {missing_samples[:5]}")
    extra_samples = cast("list[list[int]]", report_payload.get("extra_pair_samples", []))
    if extra_samples:
        lines.append(f"Extra pair samples: {extra_samples[:5]}")
    return "\n".join(lines)


def persist_gap_diagnosis_report(
    run_root: Path,
    *,
    limit: int = 10,
    coverage_payload: dict[str, Any] | None = None,
    audit_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a joined gap diagnosis report under 03_Analysis."""
    normalized_run_root = run_root.expanduser().resolve()
    ensure_dest_run_layout(normalized_run_root)
    report_payload = build_gap_diagnosis_report(
        normalized_run_root,
        limit=limit,
        coverage_payload=coverage_payload,
        audit_payload=audit_payload,
    )

    atomic_write_json(normalized_run_root / GAP_DIAGNOSIS_JSON_PATH, report_payload)
    write_hash_sidecar(normalized_run_root / GAP_DIAGNOSIS_JSON_PATH)

    report_text = render_gap_diagnosis_report(report_payload)
    atomic_write_text(normalized_run_root / GAP_DIAGNOSIS_TEXT_PATH, report_text)
    write_hash_sidecar(normalized_run_root / GAP_DIAGNOSIS_TEXT_PATH)

    return report_payload
