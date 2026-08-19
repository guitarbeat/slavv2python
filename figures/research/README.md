# Research figures (data-backed)

## In short

These charts show why Energy **ship** uses “close enough” (`allclose`): last-digit
drift is tiny, but a strict ULP cutoff would still flag many voxels. They are
**not** the extra 100% / identical-bits leftover on the current crop dest.

Draft figures for the SLAVV port/optimization paper, generated **from real run
artifacts** (no synthetic data) by [`generate.py`](generate.py).

Lives under **`figures/research/`** so all publication assets share one tree
with the proposal claim figures in [`figures/claim/`](../claim/).

**Related (proposal appendix):** four standalone MATLAB→Python exact-parity
claim figures — see [figures/README.md](../README.md).

## Regenerate

```powershell
python figures/research/generate.py `
  --ulp-json workspace\runs\oracle_180709_E\crop_M_exact_v3\03_Analysis\exact_proof_energy_ulp.json `
  --run-log <run-log-with-joblib-progress>
```

| Figure | Shows | Data source |
|---|---|---|
| `energy_ulp_histogram.png` | How far mismatching Energy voxels sit in last-digit units (ULP): typical 4, 90th percentile 13, yet the largest real gap is `1.99×10⁻¹¹`. A pure-ULP cutoff would still flag 37,174 voxels — that is why the ship bar is `np.allclose` ([ADR 0011](../../docs/adr/0011-energy-float-certification-policy.md)). | `exact_proof_energy_ulp.json` |
| `energy_parity_composition.png` | All 4,194,304 Energy voxels: ~9% identical bits, ~90% within the ULP cutoff (still `allclose`-pass), ~1% over that ULP cutoff (still `allclose`-pass). Scale-index mismatches = **0**. | `exact_proof_energy_ulp.json` |
| `energy_octave1_speedup.png` | Octave-1 Energy throughput, parallel (`n_jobs=6`) vs serial: **~4.1× faster, same bits**. | joblib progress log; serial baseline from `n_jobs=1` |

**Notes for the paper:**
- ULP histogram + composition back the ADR 0011 certification-policy argument.
- Speedup pairs with [post-parity-optimization-and-paper.md](../../docs/research/post-parity-optimization-and-paper.md).
- Claim figures (crop missing edges / leftover funnel / signed residual / mismatch budget): [figures/README.md](../README.md).
