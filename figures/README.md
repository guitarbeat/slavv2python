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
| **(b)** | Waterfall of MATLAB edge-pair recovery (generation vs crop/residual) | Findings 2026-07-08 funnel / overlap counts |
| **(c)** | Full-volume strict counts (MATLAB vs Python) + certification metrics table | `canonical_full_v6` / ADR 0011–0012 |

**Regenerate:**

```powershell
.\.venv\Scripts\python.exe figures\generate_matlab_python_parity_journey.py
```

**Suggested caption** (proposal appendix):

> **Figure X. Quantitative exact-parity validation of the SLAVV MATLAB→Python port.**
> **(a)** Fraction of MATLAB final edge pairs present among Python watershed candidates
> on the crop harness (*n* = 15,511), across iterative frontier-backend fixes. Shaded
> region marks the 80% launch gate for the full-volume writer.
> **(b)** Waterfall of MATLAB edge-pair recovery on the same crop after generation fixes:
> residual losses partition into a generation gap (−417) and crop/residual selection
> (−980); 14,114 of 15,511 MATLAB pairs are recovered in the Python final set.
> **(c)** Strict stage counts on full volume `180709_E` (left) versus certification
> metrics under ADR 0011/0012 spatial and float bars (right). Network shortfall is
> entirely downstream of the residual edge-generation gap.

**Methodology backdrop:** [PARITY_METHODOLOGY.md](../docs/reference/core/PARITY_METHODOLOGY.md),
[ADR 0011](../docs/adr/0011-energy-float-certification-policy.md),
[ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md).

When numbers in [EXACT_PROOF_FINDINGS](../docs/reference/core/EXACT_PROOF_FINDINGS.md)
move, update the constants in the generator and re-run the script before pasting
into the proposal.
