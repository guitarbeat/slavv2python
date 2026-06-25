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

## Internal structure (post-hardening refactor)

The Random Component Parity Suite and full SLAVV pipeline were refactored (see [random-component-parity-hardening-spec.md](../../plans/random-component-parity-hardening-spec.md) and the architecture deepening plan) for maintainability, depth, and exact MATLAB parity while preserving behavior.

- **Python SLAVV facade over MATLAB source**: See `slavv_python/pipeline/slavv_vectorize.py` for `vectorize_python` (high-level orchestrator equivalent to MATLAB `vectorize_V200.m`, delegating to the exact-parity stage managers). The module also exposes thin `get_*_python` convenience wrappers (`get_energy_v202_python`, `get_vertices_v200_python`, `get_edges_by_watershed_python`, etc.); `get_energy_v202_python` delegates to the real manager, while the vertices/edges/choose/network wrappers are simplified demonstration shims. The exact-parity implementations live in `pipeline/{energy,vertices,edges,network}/` and the `matlab_get_*` modules.
  - Delegates to stage managers for production use (`EnergyManager`, `VertexManager`, etc.).
  - Supports both Paper Path and Exact Route (via `PipelinePolicy`).

- **Public API for Random Component Parity Suite** (preferred for new code/tests): `run_structural_gate`, `collect_hessian_diagnostics`, `build_structural_report`, `build_diagnostics_report`, `StructuralGateResult`, `Mismatch` (exported from `tests/support/random_component/`).

- **Structural gate** (`random_component/gate.py`): pure, zero knowledge of energy/Hessian. Always used for the structural gate.
- **Diagnostics** (`random_component/diagnostics.py`): separate, only for `--mode diagnostics`.
- **Builders** (`random_component/reports.py`): produce the legacy report dict shapes for compatibility.
- **Main orchestration** (`random_component_parity.py`): thin (CLI, MATLAB driver, report writing, `compare_references` as compat shim). Reference computation logic is being further deepened (see references deepening plan).

### How to hack on the suite / full SLAVV Python

- For the full SLAVV pipeline (Energy → Vertices → Edges → Network matching MATLAB): Use `slavv_python/pipeline/slavv_vectorize.vectorize_python(...)` or the individual stage managers. See docstrings for MATLAB function mappings.
- For new structural checks or gate logic in the Random Component Suite → edit `gate.py` + `models.py`.
- For Hessian ULP / diagnostics changes → `diagnostics.py`.
- Report shape changes → `reports.py` (keep legacy compat).
- Reference computation (linspace, `interp3`, energy samples, full `get_energy_V202` style) → `slavv_energy_filter.py` (clean Python port) + `matlab_get_energy_v202_chunked.py` (exact/chunked) or production `energy/` modules. Future: dedicated deep `references.py` module per architecture plan.
- Full pipeline orchestration / MATLAB source mapping → `slavv_vectorize.py` (high-level facade) + the exact-parity submodules in `pipeline/{energy,vertices,edges,network}/`.
- Add public helpers to `random_component/__init__.py`.
- Keep the main file lean (<1000 lines); extract to package (ongoing per architecture review).
- `compare_references` is the old entry for tests/compat — prefer the new public builders/gate in fresh code.
- Always verify structural gate against the Phase 0 baseline after changes (see Phase 5 in the spec).
- For exact parity work: Use the random component suite + `prove-exact` before/after long runs. See pre-gate tiers.

The Python SLAVV pipeline (the stage managers) is the equivalent of the MATLAB source in `external/Vectorization-Public/source/`, with the same stages, chunking, Fourier-domain filtering, principal energy, watershed/tracing, etc. `slavv_vectorize.py` provides the high-level mapping and orchestration; the exact-parity engines are the stage managers and `matlab_get_*` modules.

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
