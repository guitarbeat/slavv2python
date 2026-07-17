"""Parity campaign series — figure paint KPI mirror only.

Authoritative live status: docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth
Update these numbers only when ONE TRUTH moves, then regenerate claim figures.
Figure drawers in generate_parity_claim_figures.py only paint; they do not invent counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Shared campaign constants
# ---------------------------------------------------------------------------

CROP_N_MATLAB_FINAL_PAIRS = 15_511
CANONICAL_MATLAB_EDGES = 69_500
CANONICAL_MATLAB_STRANDS = 48_049
# Network ADR 0012 residual on claim surface (Python - MATLAB strand multiset)
NETWORK_STRAND_RESIDUAL = -1
CANONICAL_PYTHON_STRANDS = CANONICAL_MATLAB_STRANDS + NETWORK_STRAND_RESIDUAL
# Edge multiset residual count (connection multiset Δ = 1; strict count can still match)
EDGE_PAIR_MULTISET_RESIDUAL = 1

ORACLE_CROP_ID = "180709_E_crop_M_v2"
ORACLE_FULL_VOLUME = "180709_E"
# Latest full-volume claim/audit surface (Phase 1 still open — Network multiset FAIL)
CANONICAL_CLAIM_RUN = "canonical_full_v16"

# Log-scale floor so zero residual remains visible as a bar/point
LOG_FLOOR_COUNT = 0.85

# Retired 80% candidate-overlap gate → still-missing pairs at the gate
RETIRED_GATE_OVERLAP_FRAC = 0.80
RETIRED_GATE_MISSING = round(CROP_N_MATLAB_FINAL_PAIRS * (1.0 - RETIRED_GATE_OVERLAP_FRAC))

# Full-volume residual class (generation join displacement → degree-excess)
# Crop final pair multiset is closed on re-selection (guard only).
FULL_RESIDUAL_MATLAB_PAIR = (34897, 38584)
FULL_RESIDUAL_PYTHON_PAIR = (26444, 38584)

ColorKey = Literal["accent", "muted", "amber", "green", "steel"]
# Which y-series an annotation arrow anchors to (view must not infer this from color)
AnnotationSeries = Literal["point", "edge", "network", "extra", "missing"]


@dataclass(frozen=True)
class Annotation:
    """Callout owned by a series record (identity is the record, not a magic index)."""

    text: str
    text_x: float  # absolute x in series-index units
    text_y: float  # absolute y in data units
    color_key: ColorKey = "muted"
    bold: bool = False
    series: AnnotationSeries = "point"


# ---------------------------------------------------------------------------
# Trajectory — generation residual (MATLAB final pairs absent from candidates)
# ---------------------------------------------------------------------------

TrajectoryKind = Literal["null", "leap", "polish", "closed"]


@dataclass(frozen=True)
class TrajectoryStep:
    id: str
    label: str
    missing: int
    kind: TrajectoryKind
    annotation: Annotation | None = None


# Claims kept short for wrap-width titles; full narrative lives in captions.
TRAJECTORY_CLAIM = "One LUT fix recovered ~6k missing edges"
TRAJECTORY_FOOTNOTE = (
    f"Crop harness n = {CROP_N_MATLAB_FINAL_PAIRS:,} MATLAB final pairs "
    f"(oracle {ORACLE_CROP_ID}). "
    "Y shows generation residual (pairs never emitted as candidates), not final selection."
)

TRAJECTORY_STEPS: list[TrajectoryStep] = [
    TrajectoryStep(
        id="baseline",
        label="Base-\nline",
        missing=6_532,
        kind="null",
        annotation=Annotation(
            text="queue:\nno recovery",
            text_x=0.55,
            text_y=900,
            color_key="muted",
            series="point",
        ),
    ),
    TrajectoryStep(
        id="sorted_queue",
        label="Sorted\nqueue",
        missing=6_532,
        kind="null",
    ),
    TrajectoryStep(
        id="directional_lut",
        label="LUT +\nsuppress",
        missing=417,
        kind="leap",
        annotation=Annotation(
            text="\u22126,115\nin one step",
            text_x=2.75,
            text_y=2_200,
            color_key="accent",
            bold=True,
            series="point",
        ),
    ),
    TrajectoryStep(
        id="vertex_inf_sentinel",
        label="Vertex\n-Inf",
        missing=19,
        kind="polish",
    ),
    TrajectoryStep(
        id="trace_match",
        label="Trace\nmatch",
        missing=0,
        kind="closed",
    ),
]


# ---------------------------------------------------------------------------
# Funnel — crop residual collapse (missing vs extra)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunnelPhase:
    id: str
    label: str
    missing: int
    # None = not applicable at that stage (e.g. generation-only residual)
    extra: int | None
    annotation: Annotation | None = None


FUNNEL_CLAIM = "Crop residual collapsed to a closed pair multiset"
FUNNEL_FOOTNOTE = (
    f"Crop residual vs MATLAB oracle pairs ({ORACLE_CROP_ID}). "
    f"Middle columns: generation extras \u2248 candidates \u2212 "
    f"{CROP_N_MATLAB_FINAL_PAIRS:,}; later columns are final missing/extra after Edge Selection. "
    "Crop re-selection closes the pair multiset (regression guard); full-volume residual is separate."
)

FUNNEL_PHASES: list[FunnelPhase] = [
    FunnelPhase(
        id="baseline_generation",
        label="Base\ngen.",
        missing=6_532,
        extra=None,
    ),
    FunnelPhase(
        id="lut_unlocked",
        label="LUT\nunlock",
        missing=417,
        extra=3_772,  # ~19,283 candidates - 15,511
    ),
    FunnelPhase(
        id="trace_match_overselect",
        label="Trace +\nover-sel.",
        missing=149,
        extra=365,
        annotation=Annotation(
            text="extras displace\noracle pairs",
            text_x=0.55,
            text_y=22,
            color_key="muted",
            series="extra",
        ),
    ),
    FunnelPhase(
        id="post_watershed",
        label="Post-\nWS",
        missing=1,
        extra=1,
        annotation=Annotation(
            text="1-pair\nckpt",
            text_x=2.45,
            text_y=14,
            color_key="muted",
            series="extra",
        ),
    ),
    FunnelPhase(
        id="reselection_closed",
        label="Re-sel.\nclosed",
        missing=0,
        extra=0,
        annotation=Annotation(
            text="closed\n(guard)",
            text_x=3.55,
            text_y=10,
            color_key="green",
            bold=True,
            series="missing",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Canonical signed residual (Python - MATLAB) across audits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalAudit:
    id: str
    label: str
    edge_delta: int  # Python - MATLAB connections
    network_delta: int  # Python - MATLAB strands
    note: str
    # When True, show network residual label (near-zero edge residual region)
    show_network_label: bool = False
    annotation: Annotation | None = None

    @property
    def near_zero_edge(self) -> bool:
        """Use end-of-series label layout when edge residual is exact match."""
        return self.edge_delta == 0


AGREEMENT_CLAIM = (
    f"Edges under → over → match; Network still {NETWORK_STRAND_RESIDUAL:+d}"
)
AGREEMENT_FOOTNOTE = (
    f"Canonical full {ORACLE_FULL_VOLUME} audits "
    f"(MATLAB edges = {CANONICAL_MATLAB_EDGES:,}; strands = {CANONICAL_MATLAB_STRANDS:,}). "
    f"v15/v16 edge residual 0; Network still {NETWORK_STRAND_RESIDUAL:+d} strand — "
    "ADR 0012 multiset FAIL (open ship gate)."
)

CANONICAL_AUDITS: list[CanonicalAudit] = [
    CanonicalAudit(
        id="v4",
        label="v4",
        edge_delta=-9_287,
        network_delta=-8_426,
        note="pre-fix",
    ),
    CanonicalAudit(
        id="v6",
        label="v6",
        edge_delta=-4_064,
        network_delta=-3_454,
        note="own. PASS",
        annotation=Annotation(
            text="own. PASS\nstill \u22124k",
            text_x=0.15,
            text_y=-7_200,
            color_key="muted",
            series="edge",
        ),
    ),
    CanonicalAudit(
        id="v7",
        label="v7",
        edge_delta=-3_276,
        network_delta=-2_632,
        note="gen. up",
        annotation=Annotation(
            text="Net tracks\nEdges",
            text_x=2.45,
            text_y=-7_800,
            color_key="steel",
            series="network",
        ),
    ),
    CanonicalAudit(
        id="v10",
        label="v10",
        edge_delta=+747,
        network_delta=+534,
        note="over-sel.",
        annotation=Annotation(
            text="sign flip\n(axis fix)",
            text_x=4.15,
            text_y=1_900,
            color_key="amber",
            bold=True,
            series="edge",
        ),
    ),
    CanonicalAudit(
        id="v15",
        label="v15",
        edge_delta=0,
        network_delta=NETWORK_STRAND_RESIDUAL,
        note="edges 0",
        show_network_label=True,
    ),
    CanonicalAudit(
        id="v16",
        label="v16",
        edge_delta=0,
        network_delta=NETWORK_STRAND_RESIDUAL,
        note="Net FAIL",
        show_network_label=True,
    ),
]


# ---------------------------------------------------------------------------
# Cert / mismatch budget table
# ---------------------------------------------------------------------------

CertTone = Literal["closed", "residual", "neutral"]


@dataclass(frozen=True)
class CertRow:
    stage: str
    quantity: str
    residual_display: str
    denom: str
    reading: str
    tone: CertTone


CERT_CLAIM = "Network still fails ADR 0012 by one strand"
CERT_TAKEAWAY = (
    f"Ship gate open: Network multiset FAIL "
    f"({CANONICAL_PYTHON_STRANDS:,}/{CANONICAL_MATLAB_STRANDS:,}) on {CANONICAL_CLAIM_RUN}; "
    f"downstream of Edge Set residual "
    f"(MATLAB {list(FULL_RESIDUAL_MATLAB_PAIR)} vs Python {list(FULL_RESIDUAL_PYTHON_PAIR)})."
)

CERT_ROWS: list[CertRow] = [
    CertRow(
        stage="Energy scale-index",
        quantity="mismatched voxels",
        residual_display="0",
        denom="16.8M",
        reading="exact field",
        tone="closed",
    ),
    CertRow(
        stage="Energy float",
        quantity="max |\u0394|",
        residual_display="~2e-11",
        denom="allclose",
        reading="FP drift only",
        tone="closed",
    ),
    CertRow(
        stage="Vertices",
        quantity="position / scale",
        residual_display="0",
        denom="exact",
        reading="bit-identical",
        tone="closed",
    ),
    CertRow(
        stage="Edges ownership",
        quantity="disagreed voxels",
        residual_display="~8",
        denom="5.84M",
        reading="map ~100%",
        tone="neutral",
    ),
    CertRow(
        stage="Edges connections",
        quantity="pair multiset \u0394",
        residual_display=str(EDGE_PAIR_MULTISET_RESIDUAL),
        denom=f"{CANONICAL_MATLAB_EDGES:,}",
        reading="join displace.",
        tone="residual",
    ),
    CertRow(
        stage="Network strands",
        quantity="strand multiset \u0394",
        residual_display=str(abs(NETWORK_STRAND_RESIDUAL)),
        denom=f"{CANONICAL_MATLAB_STRANDS:,}",
        reading="ADR 0012 FAIL",
        tone="residual",
    ),
]
