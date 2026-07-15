# Research figures (data-backed)

Draft figures for the SLAVV port/optimization paper, generated **from real run
artifacts** (no synthetic data) by
[`scripts/make_report_figures.py`](../../scripts/make_report_figures.py).

Lives under **`figures/research/`** so all publication assets share one tree
with the proposal claim figures in [`figures/`](../README.md).

**Related (proposal appendix):** four standalone MATLAB→Python exact-parity
claim figures — see [figures/README.md](../README.md).

## Regenerate

```powershell
python scripts/make_report_figures.py `
  --ulp-json workspace\runs\oracle_180709_E\crop_M_exact\03_Analysis\exact_proof_energy_ulp.json `
  --run-log <run-log-with-joblib-progress> `
  --out-dir figures\research
```

| Figure | Shows | Data source |
|---|---|---|
| `energy_ulp_histogram.png` | ULP-distance distribution on mismatching energy voxels: p50=4, p90=13, with a long ≥9 tail — yet max \|Δ\| = 1.99×10⁻¹¹. A pure-ULP gate (≤48 ULP) would flag 37,174 voxels, motivating the `np.allclose` policy ([ADR 0011](../../docs/adr/0011-energy-float-certification-policy.md)). | `exact_proof_energy_ulp.json` |
| `energy_parity_composition.png` | Composition of all 4,194,304 energy voxels: 9.16% bitwise-exact, 89.95% within ≤48 ULP (\|Δ\|≤2e-11), 0.89% over the ULP gate (all `allclose`-pass). Scale-index mismatches = **0** (discrete field exact). | `exact_proof_energy_ulp.json` |
| `energy_speedup.png` | Octave-1 energy throughput, parallel (`n_jobs=6`, ~11 s/chunk) vs serial reference (~44 s/chunk): **~4.1× speedup, bit-exact**. | joblib progress log; serial baseline from `n_jobs=1` |

**Notes for the paper:**
- ULP histogram + composition back the ADR 0011 certification-policy argument.
- Speedup pairs with [post-parity-optimization-and-paper.md](../../docs/research/post-parity-optimization-and-paper.md).
- Claim figures (trajectory / funnel / agreement / cert table): [figures/README.md](../README.md).
