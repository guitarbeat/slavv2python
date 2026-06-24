# Random Component Parity Suite

[Up: Reference Docs](../README.md) · [Parity Pre-Gate](PARITY_PRE_GATE.md) · [ADR 0010](../../adr/0010-random-component-parity-suite.md)

Fast MATLAB R2019a vs Python differential diagnostics on pinned white-noise inputs. This suite is a **developer loop** and self-hosted CI check; it is **not** a certification claim on crop or canonical volumes (see [ADR 0009](../../adr/0009-parity-pre-gate-tiers.md) and [PARITY_CERTIFICATION_GUIDE.md](PARITY_CERTIFICATION_GUIDE.md)).

---

## What it exercises

| Component | Compare mode | Notes |
|-----------|--------------|-------|
| Linspace (128 contexts) | Strict `float64` | Python uses promoted `matlab_random_linspace_reference.json` |
| `interp3` (16 queries × 6 cases) | Strict `float64` | Integer/half-integer lattice + boundary/OOB only |
| Energy `padded_shape_yxz` | Strict integer | From `fourier_transform_V2` / `_fourier_transform_input`; included in structural mode |
| Energy sample coordinates / `valid` | Strict integer / bool | Diagnostics mode only |
| Curvatures, gradient, laplacian, energy | **Advisory ULP** | Diagnostics mode only; emitted in reports and does not fail the gate |

---

## Corpus and fixtures

| Artifact | Path |
|----------|------|
| Corpus manifest | `tests/support/fixtures/matlab_random_component_corpus.json` |
| Linspace reference | `tests/support/fixtures/matlab_random_linspace_reference.json` |
| Matching-kernel reference | `tests/support/fixtures/matlab_random_matching_reference.json` |
| MATLAB driver | `tests/support/matlab/random_component_reference.m` |
| Python runner | `tests/support/random_component_parity.py` |

Six cases: three isotropic and three anisotropic spacing configs on `16×32×32` Z×Y×X noise volumes.

---

## Local run

Requires MATLAB R2019a and `external/Vectorization-Public`.

```powershell
$env:MATLAB_EXE = "C:\Program Files\MATLAB\R2019a\bin\matlab.exe"
python -m tests.support.random_component_parity `
  --output-dir workspace\scratch\random_component_parity `
  --matlab-exe $env:MATLAB_EXE
```

The default `--mode structural` is the fast PR path: it checks linspace,
interpolation, and padded shapes but skips per-sample Hessian FFT work. Use
`--mode diagnostics` only when triaging a numerical mismatch; it emits the
Hessian/gradient/Energy ULP telemetry for all six cases.

Outputs under `<output-dir>`:

- `manifest.json` — resolved corpus with TIFF paths and queries
- `matlab_reference.mat` — MATLAB v7 results
- `random_component_parity_report.json` — aggregate report
- `random_component_parity_report.txt` — compact structural + advisory summary for logs/artifacts
- `reports/<case_id>.json` — per-case report (includes `hessian_diagnostics`)

Print advisory Hessian summary from a saved report:

```powershell
python -m tests.support.random_component_parity `
  --print-hessian-summary workspace\scratch\random_component_parity\random_component_parity_report.json
```

Unit tests (no MATLAB required):

```powershell
python -m pytest tests/unit/parity/test_random_component_parity.py
```

---

## CI

Workflow: `.github/workflows/matlab-random-component-parity.yml`

- Trigger: `pull_request` on Energy/parity paths and `workflow_dispatch`
- Runner: `[self-hosted, windows, matlab-r2019a]`
- Fast precheck: `tests/unit/parity/test_random_component_parity.py`
- Structural gate: differential suite exit code (`--mode structural`)
- Manual diagnostics: `workflow_dispatch` exposes `mode=diagnostics` to collect Hessian samples without changing the certification claim
- Advisory step: prints `hessian_diagnostics.max_ulp_distance` to logs and `GITHUB_STEP_SUMMARY` without failing the job

The workflow is intentionally non-blocking while the corpus establishes its
baseline. Keep it out of required branch-protection checks until maintainers
explicitly promote it.

---

## Regenerating promoted fixtures

**Linspace** (from an existing MATLAB reference run):

```powershell
python -m tests.support.export_random_linspace_overrides `
  --manifest workspace\scratch\random_component_parity\manifest.json `
  --matlab-mat workspace\scratch\random_component_parity\matlab_reference.mat
```

**Matching kernels** (MATLAB R2019a export + Python merge):

```powershell
python -m tests.support.random_component_parity --output-dir workspace\scratch\matching_export
& $env:MATLAB_EXE -batch "addpath('tests/support/matlab'); export_random_matching_reference('workspace/scratch/matching_export/manifest.json','workspace/scratch/matching_export/matching_reference.mat')"
python -m tests.support.export_random_matching_reference `
  --matlab-mat workspace\scratch\matching_export\matching_reference.mat
```

---

## Why Hessian floats are advisory only

Live strict compare on IFFT-derived Hessian samples fails cross-language even when the complex spectrum is byte-identical:

1. **FFT libraries** — NumPy vs MATLAB MKL `ifftn(..., 'symmetric')` differs by ≥1 ULP on identical input.
2. **Bessel libm** — `scipy.special.jv` vs MATLAB `besselj` in the spherical matching-kernel term drifts without the promoted matching reference on the Python side.

Bit-identical Hessian certification remains on crop/canonical `prove-exact` oracles. See [EXACT_PROOF_FINDINGS.md § Random component parity](../core/EXACT_PROOF_FINDINGS.md#random-component-parity-2026-06-22).

---

## Related docs

- [tests/README.md](../../../tests/README.md) — test pyramid placement
- [PARITY_PRE_GATE.md](PARITY_PRE_GATE.md) — tier 1 component entry points
- [ADR 0010](../../adr/0010-random-component-parity-suite.md) — decision record
- [Implementation Hardening Spec](../../plans/random-component-parity-hardening-spec.md) — active plan for refactoring the suite for maintainability while preserving the structural gate (created 2026-06-24)
