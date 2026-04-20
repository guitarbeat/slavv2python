"""Shared-neighborhood diagnostic reports for parity workflow iteration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ._persistence import (
    load_json_dict_or_empty,
    write_json_file,
    write_text_file,
)
from .run_layout import resolve_run_layout

if TYPE_CHECKING:
    from pathlib import Path

_DIAGNOSTIC_JSON_NAME = "shared_neighborhood_diagnostics.json"
_DIAGNOSTIC_MARKDOWN_NAME = "shared_neighborhood_diagnostics.md"
_EDGE_CANDIDATES_SURFACES = [
    "source/slavv/core/edge_candidates.py: candidate-manifest ordering and terminal rejection logic",
    "source/slavv/core/edge_candidates.py: parent/child branch invalidation paths",
]
_EDGE_SELECTION_SURFACES = [
    "source/slavv/core/edge_selection.py: local partner-choice scoring and conflict resolution",
    "source/slavv/core/edge_selection.py: final cleanup paths that drop emitted candidates",
]


@dataclass
class NeighborhoodDivergence:
    """Neighborhood-level divergence surfaced from the shared-neighborhood audit."""

    neighborhood_id: int | str
    divergence_type: str
    severity: str
    matlab_partner_choices: list[int] = field(default_factory=list)
    python_partner_choices: list[int] = field(default_factory=list)
    matlab_claim_count: int = 0
    python_claim_count: int = 0
    candidate_coverage_delta: int = 0
    notes: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    first_divergence_stage: str = ""
    first_divergence_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary view."""
        return asdict(self)


@dataclass
class SharedNeighborhoodDiagnosticReport:
    """Canonical persisted shared-neighborhood diagnostic report."""

    run_root: str
    generated_at: str
    matlab_edges_count: int
    python_edges_count: int
    edge_count_delta: int
    claim_ordering_differences: int
    branch_invalidation_differences: int
    partner_choice_differences: int
    divergent_neighborhoods: list[NeighborhoodDivergence] = field(default_factory=list)
    matlab_candidate_coverage: dict[str, int] = field(default_factory=dict)
    python_candidate_coverage: dict[str, int] = field(default_factory=dict)
    coverage_delta: dict[str, int] = field(default_factory=dict)
    top_divergence_patterns: list[str] = field(default_factory=list)
    recommended_investigations: list[str] = field(default_factory=list)
    edge_candidates_to_review: list[str] = field(default_factory=list)
    edge_selection_logic_to_review: list[str] = field(default_factory=list)
    source_artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary view."""
        payload = asdict(self)
        payload["divergent_neighborhoods"] = [
            divergence.to_dict() for divergence in self.divergent_neighborhoods
        ]
        return payload


def _load_comparison_payload(run_root: Path) -> dict[str, Any]:
    layout = resolve_run_layout(run_root)
    report_path = layout["analysis_dir"] / "comparison_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Comparison report not found at {report_path}. Run comparison analysis first."
        )
    report = load_json_dict_or_empty(report_path)
    if not report:
        raise FileNotFoundError(
            f"Comparison report at {report_path} could not be loaded as a JSON object."
        )
    return report


def _load_shared_audit(analysis_dir: Path, comparison_report: dict[str, Any]) -> dict[str, Any]:
    if audit := (
        comparison_report.get("edges", {})
        .get("diagnostics", {})
        .get("shared_neighborhood_audit", {})
    ):
        return audit

    audit_path = analysis_dir / "shared_neighborhood_audit.json"
    if audit_path.exists():
        audit = load_json_dict_or_empty(audit_path)
        if audit:
            return audit

    raise FileNotFoundError(
        f"Shared-neighborhood audit is not available in the comparison report or at {audit_path}."
    )


def _classify_divergence(neighborhood: dict[str, Any]) -> str:
    stage = str(neighborhood.get("first_divergence_stage", "")).strip().lower()
    reason = str(neighborhood.get("first_divergence_reason", "")).strip().lower()

    if "cleanup" in stage:
        return "partner_choice"
    if any(token in stage for token in ("partner", "substitution")) or any(
        token in reason for token in ("partner", "substitution", "extra candidate pair")
    ):
        return "partner_choice"
    if (
        any(
            token in reason
            for token in (
                "parent_has_child",
                "child_has_parent",
                "branch",
                "rejected_",
                "manifest",
            )
        )
        or stage == "pre_manifest_rejection"
    ):
        return "branch_invalidation"
    return "claim_ordering"


def _classify_severity(neighborhood: dict[str, Any]) -> str:
    missing_candidate = int(neighborhood.get("missing_matlab_incident_endpoint_pair_count", 0))
    missing_final = int(neighborhood.get("missing_final_endpoint_pair_count", 0))
    extra_candidate = int(neighborhood.get("extra_candidate_endpoint_pair_count", 0))
    score = missing_candidate + missing_final + extra_candidate
    if score >= 3:
        return "high"
    return "medium" if score >= 1 else "low"


def _recommend_actions(divergence_type: str) -> list[str]:
    if divergence_type == "branch_invalidation":
        return [
            "Inspect candidate-manifest rejection order for parent/child branch invalidation.",
            "Trace candidate lifecycle events around rejected terminal hits in edge_candidates.py.",
        ]
    if divergence_type == "partner_choice":
        return [
            "Inspect local partner scoring and conflict resolution in edge_selection.py.",
            "Compare emitted candidates that survive manifesting but disappear during final cleanup.",
        ]
    return [
        "Inspect claim ordering around candidate admission for the divergent seed origin.",
        "Compare candidate coverage before final cleanup to isolate missing MATLAB endpoint pairs.",
    ]


def _build_divergence(neighborhood: dict[str, Any]) -> NeighborhoodDivergence:
    divergence_type = _classify_divergence(neighborhood)
    matlab_pairs = neighborhood.get("missing_matlab_incident_endpoint_pair_samples", [])
    python_pairs = neighborhood.get("candidate_endpoint_pair_samples", [])
    return NeighborhoodDivergence(
        neighborhood_id=int(neighborhood.get("origin_index", -1)),
        divergence_type=divergence_type,
        severity=_classify_severity(neighborhood),
        matlab_partner_choices=[
            int(pair[1])
            for pair in matlab_pairs
            if isinstance(pair, list) and len(pair) >= 2 and int(pair[0]) >= 0
        ],
        python_partner_choices=[
            int(pair[1])
            for pair in python_pairs
            if isinstance(pair, list) and len(pair) >= 2 and int(pair[0]) >= 0
        ],
        matlab_claim_count=int(neighborhood.get("matlab_incident_endpoint_pair_count", 0)),
        python_claim_count=int(neighborhood.get("final_chosen_endpoint_pair_count", 0)),
        candidate_coverage_delta=int(neighborhood.get("candidate_endpoint_pair_count", 0))
        - int(neighborhood.get("matlab_incident_endpoint_pair_count", 0)),
        notes=[
            str(neighborhood.get("first_divergence_reason", "")).strip(),
            (
                "Selection sources: "
                + ", ".join(str(source) for source in neighborhood.get("selection_sources", []))
            ).strip(),
        ],
        recommended_actions=_recommend_actions(divergence_type),
        first_divergence_stage=str(neighborhood.get("first_divergence_stage", "")),
        first_divergence_reason=str(neighborhood.get("first_divergence_reason", "")),
    )


def _summarize_top_patterns(divergences: list[NeighborhoodDivergence]) -> list[str]:
    category_counts = {
        "claim_ordering": 0,
        "branch_invalidation": 0,
        "partner_choice": 0,
    }
    for divergence in divergences:
        category_counts[divergence.divergence_type] = (
            category_counts.get(divergence.divergence_type, 0) + 1
        )

    top_patterns = [
        f"{category.replace('_', ' ')}: {count} neighborhood(s)"
        for category, count in sorted(
            category_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count > 0
    ]

    top_patterns.extend(
        f"origin {divergence.neighborhood_id}: {divergence.severity} {divergence.divergence_type.replace('_', ' ')}"
        for divergence in divergences[:2]
    )
    return top_patterns


def _recommended_investigations(divergences: list[NeighborhoodDivergence]) -> list[str]:
    categories = {divergence.divergence_type for divergence in divergences}
    recommendations: list[str] = []
    if "branch_invalidation" in categories:
        recommendations.append(
            "Start with branch invalidation in edge_candidates.py because MATLAB endpoint pairs are rejected before they survive the candidate manifest."
        )
    if "partner_choice" in categories:
        recommendations.append(
            "Inspect partner-choice and conflict resolution in edge_selection.py because emitted candidates diverge from MATLAB's local pairing."
        )
    if "claim_ordering" in categories:
        recommendations.append(
            "Inspect claim ordering in edge_candidates.py where MATLAB incident endpoint pairs are missing from the Python candidate surface."
        )
    if not recommendations:
        recommendations.append(
            "Start with candidate-endpoint coverage before edge or strand diffs."
        )
    return recommendations


def _build_coverage_maps(
    coverage: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    matlab_candidate_coverage = {
        "endpoint_pairs": int(coverage.get("matlab_endpoint_pair_count", 0)),
        "matched_endpoint_pairs": int(coverage.get("matched_matlab_endpoint_pair_count", 0)),
        "missing_endpoint_pairs": int(coverage.get("missing_matlab_endpoint_pair_count", 0)),
    }
    python_candidate_coverage = {
        "candidate_endpoint_pairs": int(coverage.get("candidate_endpoint_pair_count", 0)),
        "final_endpoint_pairs": int(coverage.get("python_endpoint_pair_count", 0)),
        "extra_candidate_endpoint_pairs": int(
            coverage.get("extra_candidate_endpoint_pair_count", 0)
        ),
    }
    coverage_delta = {
        "candidate_minus_matlab_pairs": python_candidate_coverage["candidate_endpoint_pairs"]
        - matlab_candidate_coverage["endpoint_pairs"],
        "final_minus_matlab_pairs": python_candidate_coverage["final_endpoint_pairs"]
        - matlab_candidate_coverage["endpoint_pairs"],
        "matched_minus_missing_pairs": matlab_candidate_coverage["matched_endpoint_pairs"]
        - matlab_candidate_coverage["missing_endpoint_pairs"],
    }
    return matlab_candidate_coverage, python_candidate_coverage, coverage_delta


def generate_shared_neighborhood_diagnostics(
    run_root: Path,
    *,
    matlab_edges_path: Path | None = None,
    python_edges_path: Path | None = None,
) -> SharedNeighborhoodDiagnosticReport:
    """Generate and persist the canonical shared-neighborhood diagnostic report."""
    del matlab_edges_path, python_edges_path

    layout = resolve_run_layout(run_root)
    analysis_dir = layout["analysis_dir"]
    comparison_report = _load_comparison_payload(layout["run_root"])
    audit = _load_shared_audit(analysis_dir, comparison_report)
    coverage = (
        comparison_report.get("edges", {})
        .get("diagnostics", {})
        .get("candidate_endpoint_coverage", {})
    )

    divergences = [
        _build_divergence(neighborhood)
        for neighborhood in audit.get("neighborhoods", [])
        if isinstance(neighborhood, dict)
    ]
    matlab_candidate_coverage, python_candidate_coverage, coverage_delta = _build_coverage_maps(
        coverage
    )

    report = SharedNeighborhoodDiagnosticReport(
        run_root=str(layout["run_root"]),
        generated_at=datetime.now().isoformat(),
        matlab_edges_count=int(
            comparison_report.get("edges", {}).get(
                "matlab_count",
                comparison_report.get("matlab", {}).get("edges_count", 0),
            )
        ),
        python_edges_count=int(
            comparison_report.get("edges", {}).get(
                "python_count",
                comparison_report.get("python", {}).get("edges_count", 0),
            )
        ),
        edge_count_delta=int(
            comparison_report.get("edges", {}).get(
                "python_count",
                comparison_report.get("python", {}).get("edges_count", 0),
            )
        )
        - int(
            comparison_report.get("edges", {}).get(
                "matlab_count",
                comparison_report.get("matlab", {}).get("edges_count", 0),
            )
        ),
        claim_ordering_differences=sum(
            divergence.divergence_type == "claim_ordering" for divergence in divergences
        ),
        branch_invalidation_differences=sum(
            divergence.divergence_type == "branch_invalidation" for divergence in divergences
        ),
        partner_choice_differences=sum(
            divergence.divergence_type == "partner_choice" for divergence in divergences
        ),
        divergent_neighborhoods=divergences,
        matlab_candidate_coverage=matlab_candidate_coverage,
        python_candidate_coverage=python_candidate_coverage,
        coverage_delta=coverage_delta,
        top_divergence_patterns=_summarize_top_patterns(divergences),
        recommended_investigations=_recommended_investigations(divergences),
        edge_candidates_to_review=list(_EDGE_CANDIDATES_SURFACES),
        edge_selection_logic_to_review=list(_EDGE_SELECTION_SURFACES),
        source_artifacts={
            "comparison_report": str(analysis_dir / "comparison_report.json"),
            "shared_neighborhood_audit": str(analysis_dir / "shared_neighborhood_audit.json"),
        },
    )
    persist_shared_neighborhood_diagnostics(layout["run_root"], report)
    return report


def persist_shared_neighborhood_diagnostics(
    run_root: Path,
    report: SharedNeighborhoodDiagnosticReport,
) -> tuple[Path, Path]:
    """Persist the canonical shared-neighborhood diagnostic report artifacts."""
    analysis_dir = resolve_run_layout(run_root)["analysis_dir"]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_path = analysis_dir / _DIAGNOSTIC_JSON_NAME
    markdown_path = analysis_dir / _DIAGNOSTIC_MARKDOWN_NAME

    write_json_file(json_path, report.to_dict())
    write_text_file(markdown_path, render_shared_neighborhood_markdown(report))
    return json_path, markdown_path


def load_shared_neighborhood_diagnostics(
    run_root: Path,
) -> SharedNeighborhoodDiagnosticReport | None:
    """Load a persisted shared-neighborhood diagnostic report when available."""
    json_path = resolve_run_layout(run_root)["analysis_dir"] / _DIAGNOSTIC_JSON_NAME
    if not json_path.exists():
        return None

    payload = load_json_dict_or_empty(json_path)
    if not payload:
        return None
    divergences = [
        NeighborhoodDivergence(**divergence)
        for divergence in payload.get("divergent_neighborhoods", [])
    ]
    return SharedNeighborhoodDiagnosticReport(
        run_root=str(payload.get("run_root", "")),
        generated_at=str(payload.get("generated_at", "")),
        matlab_edges_count=int(payload.get("matlab_edges_count", 0)),
        python_edges_count=int(payload.get("python_edges_count", 0)),
        edge_count_delta=int(payload.get("edge_count_delta", 0)),
        claim_ordering_differences=int(payload.get("claim_ordering_differences", 0)),
        branch_invalidation_differences=int(payload.get("branch_invalidation_differences", 0)),
        partner_choice_differences=int(payload.get("partner_choice_differences", 0)),
        divergent_neighborhoods=divergences,
        matlab_candidate_coverage=dict(payload.get("matlab_candidate_coverage", {})),
        python_candidate_coverage=dict(payload.get("python_candidate_coverage", {})),
        coverage_delta=dict(payload.get("coverage_delta", {})),
        top_divergence_patterns=list(payload.get("top_divergence_patterns", [])),
        recommended_investigations=list(payload.get("recommended_investigations", [])),
        edge_candidates_to_review=list(payload.get("edge_candidates_to_review", [])),
        edge_selection_logic_to_review=list(payload.get("edge_selection_logic_to_review", [])),
        source_artifacts=dict(payload.get("source_artifacts", {})),
    )


def render_shared_neighborhood_markdown(report: SharedNeighborhoodDiagnosticReport) -> str:
    """Render a human-readable Markdown report."""
    lines = [
        "# Shared Neighborhood Diagnostics",
        "",
        f"- Run root: `{report.run_root}`",
        f"- Generated at: `{report.generated_at}`",
        f"- MATLAB/Python edge counts: `{report.matlab_edges_count}/{report.python_edges_count}`",
        f"- Edge count delta: `{report.edge_count_delta}`",
        "",
        "## Top Divergence Patterns",
        "",
    ]
    lines.extend(f"- {pattern}" for pattern in report.top_divergence_patterns)
    lines.extend(
        [
            "",
            "## Candidate Coverage",
            "",
            f"- MATLAB endpoint pairs: `{report.matlab_candidate_coverage.get('endpoint_pairs', 0)}`",
            f"- Python candidate endpoint pairs: `{report.python_candidate_coverage.get('candidate_endpoint_pairs', 0)}`",
            f"- Python final endpoint pairs: `{report.python_candidate_coverage.get('final_endpoint_pairs', 0)}`",
            f"- Candidate minus MATLAB delta: `{report.coverage_delta.get('candidate_minus_matlab_pairs', 0)}`",
            "",
            "## Highest Severity Neighborhoods",
            "",
        ]
    )
    lines.extend(
        f"- Origin `{divergence.neighborhood_id}`: {divergence.severity} {divergence.divergence_type.replace('_', ' ')}; MATLAB/Python claims `{divergence.matlab_claim_count}/{divergence.python_claim_count}`; reason `{divergence.first_divergence_reason}`"
        for divergence in sorted(
            report.divergent_neighborhoods,
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}.get(item.severity, 3),
                (
                    int(item.neighborhood_id)
                    if str(item.neighborhood_id).lstrip("-").isdigit()
                    else 999999
                ),
            ),
        )[:5]
    )
    lines.extend(["", "## Recommended Investigation Order", ""])
    lines.extend(f"- {recommendation}" for recommendation in report.recommended_investigations)
    lines.extend(["", "## Code Surfaces", ""])
    lines.extend(
        f"- {surface}"
        for surface in report.edge_candidates_to_review + report.edge_selection_logic_to_review
    )
    return "\n".join(lines) + "\n"


def format_shared_neighborhood_summary(report: SharedNeighborhoodDiagnosticReport) -> str:
    """Render a compact CLI summary for an existing diagnostic report."""
    top_origins = (
        ", ".join(
            f"{divergence.neighborhood_id} ({divergence.divergence_type}/{divergence.severity})"
            for divergence in report.divergent_neighborhoods[:3]
        )
        or "none"
    )
    return (
        "Shared-neighborhood diagnostics ready: "
        f"{len(report.divergent_neighborhoods)} divergent neighborhood(s); "
        f"claim-ordering={report.claim_ordering_differences}, "
        f"branch-invalidation={report.branch_invalidation_differences}, "
        f"partner-choice={report.partner_choice_differences}. "
        f"Top origins: {top_origins}."
    )


def recommend_diagnostics_if_needed(
    *,
    run_root: Path,
    edges_parity_ok: bool,
    network_gate_parity_ok: bool | None,
) -> str | None:
    """Return the next diagnostic step when the remaining gap is still in edges."""
    if edges_parity_ok:
        return None

    analysis_dir = resolve_run_layout(run_root)["analysis_dir"]
    report_path = analysis_dir / _DIAGNOSTIC_JSON_NAME
    if report_path.exists():
        report = load_shared_neighborhood_diagnostics(run_root)
        if report is not None:
            return format_shared_neighborhood_summary(report)

    message = (
        "Edges parity still diverges. Generate shared-neighborhood diagnostics in "
        "`03_Analysis/shared_neighborhood_diagnostics.{json,md}` to inspect claim ordering, "
        "branch invalidation, and partner choice."
    )
    if network_gate_parity_ok:
        message += (
            " The stage-isolated network gate already achieved exact parity, so the remaining "
            "gap is isolated to edge candidate generation/selection."
        )
    return message
