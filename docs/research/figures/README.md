# Report figures (data-backed)

Draft figures for the SLAVV port/optimization paper, generated **from real run
artifacts** (no synthetic data) by
[`scripts/make_report_figures.py`](../../../scripts/make_report_figures.py).

**Related (proposal appendix, methods multipanel):** the quantitative
MATLAB→Python exact-parity journey figure lives at repo-root
[`figures/`](../../../figures/) (PDF + PNG + generator). See
[figures/README.md](../../../figures/README.md). That figure summarizes crop
overlap trajectory, edge-pair recovery waterfall, and full-volume counts /
certification bars; it is maintained separately from the energy ULP/speedup
set below.

Regenerate energy/speedup drafts:

```powershell
python scripts/make_report_figures.py `
  --ulp-json workspace\runs\oracle_180709_E\crop_M_exact\03_Analysis\exact_proof_energy_ulp.json `
  --run-log <run-log-with-joblib-progress> `
  --out-dir docs\research\figures
```

| Figure | Shows | Data source |
|---|---|---|
| `energy_ulp_histogram.png` | ULP-distance distribution on mismatching energy voxels: p50=4, p90=13, with a long ≥9 tail — yet max \|Δ\| = 1.99×10⁻¹¹. A pure-ULP gate (≤48 ULP) would flag 37,174 voxels, motivating the `np.allclose` policy ([ADR 0011](../../adr/0011-energy-float-certification-policy.md)). | `exact_proof_energy_ulp.json` |
| `energy_parity_composition.png` | Composition of all 4,194,304 energy voxels: 9.16% bitwise-exact, 89.95% within ≤48 ULP (\|Δ\|≤2e-11), 0.89% over the ULP gate (all `allclose`-pass). Scale-index mismatches = **0** (discrete field exact). | `exact_proof_energy_ulp.json` |
| `energy_speedup.png` | Octave-1 energy throughput, parallel (`n_jobs=6`, ~11 s/chunk) vs serial reference (~44 s/chunk): **~4.1× speedup, bit-exact**. Left: chunks completed vs wall-clock; right: per-chunk time. | joblib `Done N tasks \| elapsed` lines in the run log; serial baseline measured from the `n_jobs=1` run |

**Notes for the paper:**
- The ULP histogram + composition are the empirical backbone of the
  certification-policy argument (strict on discrete, `allclose` on continuous).
- The speedup figure pairs with the bit-impact matrix in
  [post-parity-optimization-and-paper.md](../post-parity-optimization-and-paper.md):
  the parallelism is bit-exact because reduction order is fixed.
- The serial baseline is a measured reference (~44 s/chunk); for a final
  strong-scaling figure, run the A/B at `n_jobs ∈ {1,2,4,6,8}` and plot the curve.
- For an appendix overview of the port (overlap, funnel, full-volume status),
  use the standalone set under [figures/](../../../figures/) (see [figures/README.md](../../../figures/README.md)).
