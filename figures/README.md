# Publication figures

## In short

These four pictures tell the **ship** story (Phase 1 closed), not the extra
“identical last digits” leftover. Live pass/fail is [ONE TRUTH](../docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

All committed **publication-oriented** assets for the PhD proposal appendix and
methods write-ups. Distinct from runtime plotting (`slavv_python/visualization/`).

| Location | Role |
|----------|------|
| **[`figures/claim/`](claim/)** | Exact-parity claim figures + regenerator |
| **[`figures/research/`](research/)** | Data-backed energy ULP / speedup drafts |

```text
figures/
├── README.md                 # this inventory
├── claim/                    # Phase 1 ship / proposal appendix
│   ├── README.md
│   ├── campaign_series.py    # KPI mirror (edit when ONE TRUTH moves)
│   ├── generate.py           # paints series data to PDF/PNG
│   ├── crop_missing_edges.{pdf,png}
│   ├── crop_leftover_funnel.{pdf,png}
│   ├── full_signed_residual.{pdf,png}
│   └── mismatch_budget.{pdf,png}
└── research/                 # Energy ULP / speedup
    ├── README.md
    ├── generate.py
    ├── energy_ulp_histogram.png
    ├── energy_parity_composition.png
    └── energy_octave1_speedup.png
```

## MATLAB→Python exact-parity figures

**Design rule:** four **standalone** claim-driven figures. Each answers one
non-trivial question. Prefer residual / signed delta / absolute counts over
flat “all green” dashboards.

**Wrap-first layout:** canvases are ~3.3 in wide so type stays legible at
manuscript wrap width (`0.48\textwidth` / `\figWidthWrap`). Claims and callouts
are short; long narrative lives in LaTeX captions, not in-figure footnotes.
They still scale cleanly inside `fullwidthfigure` when needed.

| File | Claim | Why it is interesting |
|------|-------|------------------------|
| [`crop_missing_edges`](claim/crop_missing_edges.pdf) | One lookup-table fix recovered ~6k missing MATLAB edges | Log-scale *missing* pairs; queue cosmetics did nothing; only the LUT step jumped |
| [`crop_leftover_funnel`](claim/crop_leftover_funnel.pdf) | Crop leftover collapsed from thousands to a closed pair set | Missing vs extra; extras during cleanup → re-selection closes crop (guard) |
| [`full_signed_residual`](claim/full_signed_residual.pdf) | Full-volume Edges under-, over-, then matched; historical Network miss of one strand | Signed leftover across `v4→v16`; Network tracks Edges. **Live:** Phase 1 CLOSED on `v18` (ONE TRUTH) |
| [`mismatch_budget`](claim/mismatch_budget.pdf) | Phase 1 CLOSED: Network bags match on `v18` | Absolute leftover budget; historical `v16` one-strand miss is closed. **Live** status: [ONE TRUTH](../docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) |

| Script | Role |
|--------|------|
| [`campaign_series.py`](claim/campaign_series.py) | **Edit this when findings KPIs move** — all counts, labels, callouts, cert tones |
| [`generate.py`](claim/generate.py) | View layer — paints series data to PDF/PNG |

Prefer **PDF** for Word/LaTeX (vector text); PNG is 600 dpi for preview/slides.

**Regenerate:**

```powershell
.\.venv\Scripts\python.exe figures\claim\generate.py
```

### Suggested captions

**Crop missing edges**

> How many MATLAB edges Python was still missing on the small real cut-out
> (*n* = 15,511 MATLAB final pairs). Reordering the queue recovered zero pairs.
> One lookup-table + suppression change recovered 6,115 pairs and cleared the
> old 80% gate.

**Crop leftover funnel**

> The crop leftover collapsing. Early work was “Python never found ~6.5k
> edges.” Once candidates covered MATLAB’s set, leftover missing pairs were
> crowded out by extras during cleanup; re-selection closes the crop pair set
> (a regression guard). The old full-volume leftover was a different class and
> is closed on the live claim surface.

**Full signed residual**

> Python minus MATLAB on the full photo across audits. Edges under-selected
> through `v7`, over-selected at `v10`, then matched at `v15`/`v16`. Network
> leftover tracked Edges — not a separate Network bug. At historical `v16`,
> Edges passed while Network still missed one strand. **Live:** Phase 1 CLOSED
> on `canonical_full_v18` ([ONE TRUTH](../docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk)).

**Mismatch budget**

> Absolute leftover on the full-volume claim surface (live run root in
> findings). Energy and vertices are closed; Edges pass the who-owns-which-voxel
> bar. The historical `v16` miss was one Network strand, downstream of ranking —
> now closed. Approximate strand % is not the ship bar. Live pair/strand
> numbers: EXACT_PROOF_FINDINGS only. Close enough is not identical last digits.

**Methodology backdrop:** [PARITY_METHODOLOGY.md](../docs/reference/core/PARITY_METHODOLOGY.md),
[ADR 0011](../docs/adr/0011-energy-float-certification-policy.md),
[ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md),
[EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md).

When numbers in findings move, update constants in
[`campaign_series.py`](claim/campaign_series.py) and re-run the generator.

## Figure ↔ documentation story map

Live KPIs and stage pass/fail live only in
[EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md).
These figures are a **publication highlight reel** of the residual campaign—not
the status log and not the methodology paper alone.

| Figure | Story beat | Primary docs |
|------|------------|--------------|
| Crop missing edges | Missing edges on the crop: cosmetics flat; lookup-table leap closes candidates | Findings watershed iteration log; residual analysis hypothesis |
| Crop leftover funnel | Crop leftover collapse after generation closed | Findings funnel/cleanup + [PHASE1 residual framing](../docs/reference/workflow/PHASE1_RESIDUAL_EXPERIMENT_ANALYSIS.md) |
| Full signed residual | Full under→over→Edges matched; Network tracks Edges; historical `v16` miss | Findings canonical audit ladder; ADR 0012 “Network downstream of edge set”. **Live:** CLOSED on the claim root |
| Mismatch budget | Absolute leftover budget; Phase 1 CLOSED on `v18` | Findings executive status; ADR 0012 ship vs stretch |

**Told by these four:** watershed leftover campaign (generation → selection → historical Network miss, now closed).

**Not told here (complementary sets):**

| Gap | Where it lives |
|-----|----------------|
| Why `allclose` not pure ULP | [ADR 0011](../docs/adr/0011-energy-float-certification-policy.md), [research figures](research/README.md) |
| Bit-exact energy parallelism / speedup | [research figures](research/README.md), post-parity paper notes |
| Why ownership ≥60% not pair-set equality | [ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md), [PARITY_METHODOLOGY](../docs/reference/core/PARITY_METHODOLOGY.md) |
| Operator next action | [HANDOFF](../.claude/HANDOFF.md), [TODO](../docs/TODO.md) |

## PhD proposal manuscript (live include)

Appendix `.tex` includes live under the PhD-Writing manuscript, not in this
repo (`figures/include/` is not checked in here). Source stems below are the
PDFs/PNGs generated in `figures/claim/`.

| Manuscript asset | Source stem |
|------------------|-------------|
| `fig-appendix-parity-trajectory` | `claim/crop_missing_edges` |
| `fig-appendix-parity-funnel` | `claim/crop_leftover_funnel` |
| `fig-appendix-parity-agreement` | `claim/full_signed_residual` |
| `fig-appendix-parity-cert-table` | `claim/mismatch_budget` |

Macros: `PhD-Writing/manuscript/config/figure-assets.tex`.
Prose: `sections/30-backmatter/appendix/370-analytical-development.tex`.

**After regenerating here**, re-copy:

```powershell
$dst = "D:\2P_Data\Aaron\PhD-Writing\manuscript\figures"
Copy-Item -Force figures\claim\crop_missing_edges.pdf     "$dst\fig-appendix-parity-trajectory.pdf"
Copy-Item -Force figures\claim\crop_missing_edges.png     "$dst\fig-appendix-parity-trajectory.png"
Copy-Item -Force figures\claim\crop_leftover_funnel.pdf   "$dst\fig-appendix-parity-funnel.pdf"
Copy-Item -Force figures\claim\crop_leftover_funnel.png   "$dst\fig-appendix-parity-funnel.png"
Copy-Item -Force figures\claim\full_signed_residual.pdf   "$dst\fig-appendix-parity-agreement.pdf"
Copy-Item -Force figures\claim\full_signed_residual.png   "$dst\fig-appendix-parity-agreement.png"
Copy-Item -Force figures\claim\mismatch_budget.pdf        "$dst\fig-appendix-parity-cert-table.pdf"
Copy-Item -Force figures\claim\mismatch_budget.png        "$dst\fig-appendix-parity-cert-table.png"
```
