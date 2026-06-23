# ADR 0010: Random Component Parity Suite

## Status
Accepted

## Context
The three-tier [Parity Pre-Gate](0009-parity-pre-gate-tiers.md) (ADR 0009) still requires multi-hour crop or canonical runs to exercise Energy-stage building blocks end-to-end. Developers needed a **fast, reproducible** MATLAB R2019a vs Python differential loop for isolated components (linspace meshes, `interp3`, Hessian intermediates) without weakening the strict `prove-exact` certification bar on crop or canonical volumes.

Investigation showed that live strict `float64` equality on IFFT-derived Hessian floats is **not achievable** cross-language even with identical complex spectra: NumPy and MATLAB MKL FFT libraries differ by at least **1 ULP** on `ifftn(..., 'symmetric')` outputs. Additional drift comes from `scipy.special.jv` vs MATLAB `besselj` in the matching-kernel path unless Python loads a promoted MATLAB reference mesh.

## Decision
Adopt a **seeded random-component differential suite** as a developer and self-hosted-CI diagnostic surface:

1. **Versioned corpus** — six PCG64-seeded `uint16` white-noise TIFFs plus 128 fixed linspace contexts in `tests/support/fixtures/matlab_random_component_corpus.json`. Randomness is confined to input generation; queries and seeds are pinned in the manifest.
2. **Live differential compare** — `tests/support/random_component_parity.py` materializes inputs, runs MATLAB R2019a (`random_component_reference.m`) and Python, and writes per-case plus aggregate JSON reports.
3. **Structural strict gate** — bit-identical compare on:
   - all 128 linspace contexts;
   - 16 `interp3` queries per case (integer, half-integer lattice, boundary/OOB probes);
   - Energy `padded_shape_yxz`, sample `coordinate_yxz`, and `valid`.
4. **Advisory Hessian diagnostics** — curvatures, gradient, laplacian, and energy floats are computed and summarized (`hessian_diagnostics.max_ulp_distance`) but **do not gate** CI. The workflow prints an advisory summary; it does not relax certification thresholds.
5. **Promoted reference fixtures** (linspace pattern):
   - `matlab_random_linspace_reference.json` — MATLAB linspace meshes;
   - `matlab_random_matching_reference.json` — MATLAB matching-kernel DFT for iso (`1,1,1`) and aniso (`1,1,2`) spacing on the corpus padded grid.
6. **Self-hosted CI** — `.github/workflows/matlab-random-component-parity.yml` on `[self-hosted, windows, matlab-r2019a]`. Ubuntu `regression-gate.yml` unchanged.

MATLAB energy samples use `principal_energy_from_derivatives` on sampled Hessian components (aligned with Python `compute_principal_energy`), not a direct `energy_filter_V200` volume lookup.

## Considered Options
- **Strict live compare on all Hessian floats** — rejected: ≥1 ULP FFT library floor with identical spectra; matching-kernel libm drift without promoted fixtures.
- **Synthetic certification claim on random noise** — rejected: no promoted oracle; suite is diagnostic only.
- **Skip MATLAB reference fixtures; fix scipy bessel in Python** — rejected for this suite: crop/canonical certification remains the production gate; random suite uses promoted meshes for the Python runner only.

## Consequences
- Operator workflow: [PARITY_RANDOM_COMPONENT_SUITE.md](../reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md).
- Unit tests: `tests/unit/parity/test_random_component_parity.py`.
- Regenerating matching fixtures requires MATLAB R2019a and `tests/support/export_random_matching_reference.py`.
- Passing the random-component gate does **not** satisfy Phase 1 **Certification** or crop harness `prove-exact` claims (ADR 0009 unchanged).
- [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md) records IFFT and matching-kernel evidence used to justify the structural vs advisory split.