"""Parity campaign series — single edit surface for figure constants.

Update these numbers when docs/reference/core/EXACT_PROOF_FINDINGS.md moves.
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
# Edge multiset residual count (equal-metric swap class; count delta is 0, multiset Δ = 1)
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

# Crop single-pair swap residual (degree-pruning hub)
CROP_SWAP_MATLAB_PAIR = (4212, 6281)
CROP_SWAP_PYTHON_PAIR = (4043, 6281)

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


TRAJECTORY_CLAIM = "One directional-LUT fix recovered ~6,000 missing MATLAB edges"
TRAJECTORY_FOOTNOTE = (
    f"Crop harness n = {CROP_N_MATLAB_FINAL_PAIRS:,} MATLAB final pairs "
    f"(oracle {ORACLE_CROP_ID}). "
    "Y shows generation residual (pairs never emitted as candidates), not final selection."
)

TRAJECTORY_STEPS: list[TrajectoryStep] = [
    TrajectoryStep(
        id="baseline",
        label="Baseline\nfrontier",
        missing=6_532,
        kind="null",
        annotation=Annotation(
            text="queue cosmetics:\nno recovery",
            text_x=0.5,
            text_y=1_200,
            color_key="muted",
            series="point",
        ),
    ),
    TrajectoryStep(
        id="sorted_queue",
        label="Sorted queue\n(no effect)",
        missing=6_532,
        kind="null",
    ),
    TrajectoryStep(
        id="directional_lut",
        label="Directional LUT\n+ suppression",
        missing=417,
        kind="leap",
        annotation=Annotation(
            text="\u22126,115 pairs\nin one step",
            text_x=2.85,
            text_y=2_500,
            color_key="accent",
            bold=True,
            series="point",
        ),
    ),
    TrajectoryStep(
        id="vertex_inf_sentinel",
        label="Vertex -Inf\nsentinel",
        missing=19,
        kind="polish",
    ),
    TrajectoryStep(
        id="trace_match",
        label="Trace match",
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


FUNNEL_CLAIM = "Crop residual collapsed from thousands to a single pair swap"
FUNNEL_FOOTNOTE = (
    f"Crop final residual vs MATLAB oracle pairs ({ORACLE_CROP_ID}). "
    f"Middle columns use published checkpoints: generation extras \u2248 candidates \u2212 "
    f"{CROP_N_MATLAB_FINAL_PAIRS:,}; later columns are final missing/extra after selection."
)

FUNNEL_PHASES: list[FunnelPhase] = [
    FunnelPhase(
        id="baseline_generation",
        label="Baseline\ngeneration",
        missing=6_532,
        extra=None,
    ),
    FunnelPhase(
        id="lut_unlocked",
        label="LUT unlocked\ncandidates",
        missing=417,
        extra=3_772,  # ~19,283 candidates - 15,511
    ),
    FunnelPhase(
        id="trace_match_overselect",
        label="Trace match +\nover-select",
        missing=149,
        extra=365,
        annotation=Annotation(
            text="Once generation closed,\nextras displaced MATLAB\npairs in faithful cleanup",
            text_x=1.15,
            text_y=40,
            color_key="muted",
            series="extra",
        ),
    ),
    FunnelPhase(
        id="post_watershed",
        label="Post-watershed\nfinalization",
        missing=1,
        extra=1,
        annotation=Annotation(
            text="equal-count\n1-pair swap",
            text_x=3.35,
            text_y=12,
            color_key="green",
            bold=True,
            series="extra",
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
    f"Full-volume Edges under-, over-, then matched - Network still "
    f"{NETWORK_STRAND_RESIDUAL:+d} strand"
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
        note="pre-fix\naudit",
    ),
    CanonicalAudit(
        id="v6",
        label="v6",
        edge_delta=-4_064,
        network_delta=-3_454,
        note="Edges ADR\nPASS; Net FAIL",
        annotation=Annotation(
            text="ownership PASS\nwhile still \u22124k edges",
            text_x=0.05,
            text_y=-7_000,
            color_key="muted",
            series="edge",
        ),
    ),
    CanonicalAudit(
        id="v7",
        label="v7",
        edge_delta=-3_276,
        network_delta=-2_632,
        note="generation\nimproved",
        annotation=Annotation(
            text="Network tracks Edges\n(no independent bug)",
            text_x=2.55,
            text_y=-8_200,
            color_key="steel",
            series="network",
        ),
    ),
    CanonicalAudit(
        id="v10",
        label="v10",
        edge_delta=+747,
        network_delta=+534,
        note="sign flip\n(over-select)",
        annotation=Annotation(
            text="axis/finalization fix\nflipped the sign",
            text_x=3.45,
            text_y=2_400,
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
        note="edges exact;\n1 strand",
        show_network_label=True,
    ),
    CanonicalAudit(
        id="v16",
        label="v16",
        edge_delta=0,
        network_delta=NETWORK_STRAND_RESIDUAL,
        note="Edges PASS;\nNet FAIL",
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


CERT_CLAIM = "On 180M voxels Network still fails ADR 0012 by one strand"
CERT_TAKEAWAY = (
    "Takeaway: Energy/Vertices closed; Edges ownership/count evaluated PASS on "
    f"{CANONICAL_CLAIM_RUN}.\n"
    "Open ship gate: Network strand multiset FAIL "
    f"({CANONICAL_PYTHON_STRANDS:,} / {CANONICAL_MATLAB_STRANDS:,}) — "
    "one degree-pruning pair swap\n"
    f"(crop: MATLAB {list(CROP_SWAP_MATLAB_PAIR)} vs Python {list(CROP_SWAP_PYTHON_PAIR)}). "
    "Not an independent Network bug."
)

CERT_ROWS: list[CertRow] = [
    CertRow(
        stage="Energy scale-index",
        quantity="mismatched voxels",
        residual_display="0",
        denom="16.8M",
        reading="exact discrete field",
        tone="closed",
    ),
    CertRow(
        stage="Energy float",
        quantity="max |\u0394|",
        residual_display="~2e-11",
        denom="allclose",
        reading="library FP drift only",
        tone="closed",
    ),
    CertRow(
        stage="Vertices",
        quantity="position / scale",
        residual_display="0",
        denom="exact",
        reading="bit-identical geometry",
        tone="closed",
    ),
    CertRow(
        stage="Edges ownership",
        quantity="disagreed voxels",
        residual_display="~8",
        denom="5.84M claimed",
        reading="99.9999% map agree",
        tone="neutral",
    ),
    CertRow(
        stage="Edges connections",
        quantity="pair multiset \u0394",
        residual_display=str(EDGE_PAIR_MULTISET_RESIDUAL),
        denom=f"{CANONICAL_MATLAB_EDGES:,}",
        reading="one equal-metric swap",
        tone="residual",
    ),
    CertRow(
        stage="Network strands",
        quantity="strand multiset \u0394",
        residual_display=str(abs(NETWORK_STRAND_RESIDUAL)),
        denom=f"{CANONICAL_MATLAB_STRANDS:,}",
        reading="ADR 0012 FAIL (open gate)",
        tone="residual",
    ),
]
