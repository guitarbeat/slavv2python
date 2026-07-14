# Proposal / methods figures

Committed **publication-oriented** figures for the PhD proposal appendix and
related methods write-ups. Distinct from:

| Location | Role |
|----------|------|
| **`figures/` (this folder)** | Proposal / methods multipanels; regenerate from checked-in scripts |
| [`docs/research/figures/`](../docs/research/figures/) | Data-backed energy ULP / speedup drafts from run artifacts (`scripts/make_report_figures.py`) |
| `slavv_python/visualization/` | Runtime plotting API (`NetworkVisualizer`), not paper figures |

## MATLAB→Python exact-parity journey

| File | Use |
|------|-----|
| [`matlab_python_parity_journey.pdf`](matlab_python_parity_journey.pdf) | **Preferred for Word/LaTeX** (vector text) |
| [`matlab_python_parity_journey.png`](matlab_python_parity_journey.png) | Preview / slides (600 dpi) |
| [`generate_matlab_python_parity_journey.py`](generate_matlab_python_parity_journey.py) | Generator |

**Design:** three pure-data panels (no schematic dashboard chrome):

| Panel | Content | Primary evidence |
|-------|---------|------------------|
| **(a)** | Crop candidate-pair overlap trajectory vs MATLAB | [EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md) watershed iteration log |
| **(b)** | Waterfall of MATLAB edge-pair recovery (generation vs final cleanup residual) | Findings current crop funnel / overlap counts |
| **(c)** | Full-volume strict counts (MATLAB vs Python) + certification metrics table | latest canonical audit / ADR 0011–0012 |

**Regenerate:**

```powershell
.\.venv\Scripts\python.exe figures\generate_matlab_python_parity_journey.py
```

**Suggested caption** (proposal appendix):

> **Figure X. Quantitative exact-parity validation of the SLAVV MATLAB→Python port.**
> **(a)** Fraction of MATLAB final edge pairs present among Python watershed candidates
> on the crop harness (*n* = 15,511), across iterative frontier-backend fixes. Shaded
> region marks the now-retired 80% threshold for historical context.
> **(b)** Waterfall of MATLAB edge-pair recovery on the same crop after generation fixes:
> candidate generation covers 15,511 of 15,511 MATLAB final pairs, while final cleanup
> currently retains 15,362 of 15,511 with 149 missing and 365 extra Python pairs.
> **(c)** Strict stage counts on full volume `180709_E` (left) versus certification
> metrics under ADR 0011/0012 spatial and float bars (right). Network mismatch is
> entirely downstream of the residual edge-set mismatch.

**Methodology backdrop:** [PARITY_METHODOLOGY.md](../docs/reference/core/PARITY_METHODOLOGY.md),
[ADR 0011](../docs/adr/0011-energy-float-certification-policy.md),
[ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md), and the
[Phase 1 residual experiment analysis](../docs/reference/workflow/PHASE1_RESIDUAL_EXPERIMENT_ANALYSIS.md).

When numbers in [EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md)
move, update the constants in the generator and re-run the script before pasting
into the proposal.

## Figure improvement plan for parity iteration

The committed parity figure is proposal-ready, but the active engineering loop
also needs a progress dashboard. Treat figures as a decision aid:

1. **Refresh the existing journey figure** after any findings change:
   - label the 80% crop-overlap gate as retired / historical;
   - make Network ADR 0012 the visible open ship gate;
   - keep Edges ownership pass separate from strict-field connection gap;
   - show that Network failure is downstream of the residual edge-set mismatch.
2. **Add a residual-iteration figure** when the next metric moves. Suggested file:
   `watershed_residual_iteration_map.{pdf,png}`. Suggested panels:
   - crop golden-trace status (current crop trace matches end-to-end);
   - crop final missing/extra gap (`149` missing, `365` extra current);
   - full edge over/under count feeding full Network strand gap.
3. **Move constants into a shared data file** before adding more figures. Suggested
   file: `figures/parity_metrics_current.json`. Both figure generators should read
   from it so captions and panels do not drift.
4. **Regenerate only when metrics move**:
   - frontier trace status regresses or moves to a new trace surface;
   - crop final edge gap drops;
   - candidate-generation gap regresses from zero;
   - successor full edge/network counts change;
   - evaluated Network ADR 0012 changes status.

Do not use the figures as live status. Live status remains
[EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md); figures
are publication and planning summaries generated from that record.

## PhD proposal manuscript (live include)

The figure is **wired into** the dissertation proposal appendix
(Analytical Development) as:

| Manuscript path | Role |
|-----------------|------|
| `PhD-Writing/manuscript/figures/fig-appendix-matlab-python-parity.{pdf,png}` | Asset copies (PDF preferred) |
| `.../figures/include/appendix-matlab-python-parity.tex` | Caption + `\label{fig:appendix-matlab-python-parity}` |
| `.../sections/30-backmatter/appendix/370-analytical-development.tex` | Prose + `\inputfigure{...}` |
| `.../config/figure-assets.tex` | `\FigAppendixMatlabPythonParity` stem |

**After regenerating here**, re-copy into the manuscript:

```powershell
Copy-Item -Force figures\matlab_python_parity_journey.pdf `
  "D:\2P_Data\Aaron\New folder\PhD-Writing\manuscript\figures\fig-appendix-matlab-python-parity.pdf"
Copy-Item -Force figures\matlab_python_parity_journey.png `
  "D:\2P_Data\Aaron\New folder\PhD-Writing\manuscript\figures\fig-appendix-matlab-python-parity.png"
```

Then rebuild the standalone appendix PDF (`appendix.tex` / project Makefile).
