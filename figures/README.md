# Publication figures

All committed **publication-oriented** assets for the PhD proposal appendix and
methods write-ups. Distinct from runtime plotting (`slavv_python/visualization/`).

| Location | Role |
|----------|------|
| **`figures/` (claim charts)** | Exact-parity claim figures + regenerator |
| [`figures/research/`](research/) | Data-backed energy ULP / speedup drafts (`scripts/make_report_figures.py`) |

## MATLAB→Python exact-parity figures

**Design rule:** four **standalone** claim-driven figures. Each answers one
non-trivial question. Prefer residual / signed delta / absolute counts over
flat “all green” dashboards.

| File | Claim | Why it is interesting |
|------|-------|------------------------|
| [`parity_trajectory`](parity_trajectory.pdf) | One directional-LUT fix recovered ~6k missing MATLAB edges | Log-scale *missing* pairs; queue cosmetics flatlined; only the LUT step is a leap |
| [`parity_funnel`](parity_funnel.pdf) | Crop residual collapsed from thousands to a 1-pair swap | Missing vs extra side-by-side; after generation closed, extras displaced MATLAB pairs in faithful cleanup |
| [`parity_agreement`](parity_agreement.pdf) | Full-volume Edges under-, over-, then matched - Network still -1 strand | Signed residual across `v4→v16`; Network tracks Edges; at v16 Edges PASS while Network multiset FAIL |
| [`parity_cert_table`](parity_cert_table.pdf) | On 180M voxels Network still fails ADR 0012 by one strand | Absolute mismatch budget; Network multiset FAIL is the open ship gate |

| Script | Role |
|--------|------|
| [`parity_campaign_series.py`](parity_campaign_series.py) | **Edit this when findings KPIs move** — all counts, labels, callouts, cert tones |
| [`generate_parity_claim_figures.py`](generate_parity_claim_figures.py) | View layer — paints series data to PDF/PNG |

Prefer **PDF** for Word/LaTeX (vector text); PNG is 600 dpi for preview/slides.

**Regenerate:**

```powershell
.\.venv\Scripts\python.exe figures\generate_parity_claim_figures.py
```

### Suggested captions

**Trajectory**

> Generation residual on the crop harness (*n* = 15,511 MATLAB final pairs):
> pairs still absent from Python candidates after successive frontier fixes.
> Queue-order cosmetics recovered zero pairs; a single directional-LUT +
> suppression change recovered 6,115 pairs and cleared the retired 80% gate.

**Funnel**

> Crop residual collapse. Early work was a generation gap (~6.5k missing).
> Once candidates covered the oracle set, remaining missing pairs were
> displaced by extra candidates during faithful degree/cycle cleanup; after
> post-watershed finalization the residual is an equal-count 1-pair swap.

**Agreement**

> Signed residual (Python − MATLAB) on full `180709_E` across closure audits.
> Edges under-selected through `v7`, over-selected at `v10` after the axis /
> finalization fix, then matched at `v15`/`v16`. Network strand residual
> tracks the edge-set residual throughout — evidence against an independent
> Network-stage defect. At `v16`, Edges PASS while Network still fails multiset
> equality by one strand (open ship gate).

**Mismatch budget**

> Absolute residual on the full-volume surface (`canonical_full_v16` proofs).
> Energy and vertices are closed; Edges ownership/count pass ADR 0012. The open
> ship gate is Network strand multiset FAIL (48,048 / 48,049), downstream of
> one equal-metric degree-pruning swap (crop: MATLAB `[4212, 6281]` vs Python
> `[4043, 6281]`). Approximate strand % is not the ADR 0012 bar.

**Methodology backdrop:** [PARITY_METHODOLOGY.md](../docs/reference/core/PARITY_METHODOLOGY.md),
[ADR 0011](../docs/adr/0011-energy-float-certification-policy.md),
[ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md),
[EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md).

When numbers in findings move, update constants in
[`parity_campaign_series.py`](parity_campaign_series.py) and re-run the generator.

## Figure ↔ documentation story map

Live KPIs and stage pass/fail live only in
[EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md).
These figures are a **publication highlight reel** of the residual campaign—not
the status log and not the methodology paper alone.

| Figure | Story beat | Primary docs |
|------|------------|--------------|
| Trajectory | Generation residual: cosmetics flat; LUT leap closes candidates | Findings watershed iteration log; residual analysis hypothesis |
| Funnel | Final residual collapse: extras after gen closed → 1-pair swap | Funnel/cleanup notes in findings; [PHASE1 residual](../docs/reference/workflow/PHASE1_RESIDUAL_EXPERIMENT_ANALYSIS.md) |
| Agreement | Full under→over→Edges matched; Network tracks Edges; still open | Findings canonical audit ladder; ADR 0012 “Network downstream of edge set” |
| Cert table | Absolute residual budget; Network multiset FAIL = ship gate | Findings executive status; ADR 0012 ship vs stretch |

**Told by these four:** watershed residual campaign (generation → selection → open Network gate).

**Not told here (complementary sets):**

| Gap | Where it lives |
|-----|----------------|
| Why `allclose` not pure ULP | [ADR 0011](../docs/adr/0011-energy-float-certification-policy.md), [research figures](research/README.md) |
| Bit-exact energy parallelism / speedup | [research figures](research/README.md), post-parity paper notes |
| Why ownership ≥60% not pair-set equality | [ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md), [PARITY_METHODOLOGY](../docs/reference/core/PARITY_METHODOLOGY.md) |
| Operator next action | [HANDOFF](../.claude/HANDOFF.md), [TODO](../docs/TODO.md) |

## PhD proposal manuscript (live include)

| Manuscript asset | Source stem | Include |
|------------------|-------------|---------|
| `fig-appendix-parity-trajectory` | `parity_trajectory` | `figures/include/appendix-parity-trajectory.tex` |
| `fig-appendix-parity-funnel` | `parity_funnel` | `figures/include/appendix-parity-funnel.tex` |
| `fig-appendix-parity-agreement` | `parity_agreement` | `figures/include/appendix-parity-agreement.tex` |
| `fig-appendix-parity-cert-table` | `parity_cert_table` | `figures/include/appendix-parity-cert-table.tex` |

Macros: `PhD-Writing/manuscript/config/figure-assets.tex`.
Prose: `sections/30-backmatter/appendix/370-analytical-development.tex`.

**After regenerating here**, re-copy:

```powershell
$dst = "D:\2P_Data\Aaron\PhD-Writing\manuscript\figures"
Copy-Item -Force figures\parity_trajectory.pdf  "$dst\fig-appendix-parity-trajectory.pdf"
Copy-Item -Force figures\parity_trajectory.png  "$dst\fig-appendix-parity-trajectory.png"
Copy-Item -Force figures\parity_funnel.pdf      "$dst\fig-appendix-parity-funnel.pdf"
Copy-Item -Force figures\parity_funnel.png      "$dst\fig-appendix-parity-funnel.png"
Copy-Item -Force figures\parity_agreement.pdf   "$dst\fig-appendix-parity-agreement.pdf"
Copy-Item -Force figures\parity_agreement.png   "$dst\fig-appendix-parity-agreement.png"
Copy-Item -Force figures\parity_cert_table.pdf  "$dst\fig-appendix-parity-cert-table.pdf"
Copy-Item -Force figures\parity_cert_table.png  "$dst\fig-appendix-parity-cert-table.png"
```
