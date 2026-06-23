# Test Organization

Keep tests under `tests/` organized by the owning package surface.

## Placement Rules

Tests mirror `slavv_python/` exactly — same folder name, same depth:

| Source module | Test location |
|:--------------|:--------------|
| `slavv_python/pipeline/` | `tests/unit/pipeline/` |
| `slavv_python/analytics/` | `tests/unit/analytics/` |
| `slavv_python/analytics/parity/` | `tests/unit/parity/` |
| `slavv_python/engine/` | `tests/unit/engine/` |
| `slavv_python/interface/` | `tests/unit/interface/` |
| `slavv_python/storage/` | `tests/unit/storage/` |
| `slavv_python/schema/` | `tests/unit/schema/` |
| `slavv_python/utils/` | `tests/unit/utils/` |
| `slavv_python/visualization/` | `tests/unit/visualization/` |
| `slavv_python/workflows/` | `tests/unit/workflows/` |

- `tests/integration/` for cross-component workflows and end-to-end pipeline behavior
- `tests/integration/parity/` for parity pre-gate integration tests (ADR 0009 tiers 1–2)
- `tests/ui/` for Streamlit- and visualization-facing behavior

## Notes

- Prefer moving a misfiled test into the matching owner directory instead of reshaping production code around the old location.
- Keep regression intent in test names, markers, and assertions.
- Reuse the shared builders under `tests/support/` when a test needs synthetic payloads, run snapshots, checkpoints, or reusable network fixtures.
- Keep exact-route parity tests under `tests/unit/parity/` (mirrors `slavv_python/analytics/parity/`).
- Do not create new task-history or workstream-specific test directories; place tests by owner surface, not by the temporary project that introduced them.

## MATLAB–Python Bit Parity Testing

Exact-route parity uses a **test pyramid**. Lower layers catch regressions in seconds; `prove-exact` remains the certification capstone.

| Layer | Location | Proves | CI |
|-------|----------|--------|-----|
| Component fixtures | `tests/unit/pipeline/energy/test_matlab_linspace_table.py` | MATLAB `linspace` meshes on crop chunks | Yes (checked-in fixture) |
| Voxel probes | `tests/unit/pipeline/energy/test_voxel_probe*.py` | One-voxel exact-route Energy vs oracle | Optional (needs crop TIFF + oracle) |
| Harness scripts | `tests/support/parity_harness.py` | Batch probes, JSONL export/compare | Local |
| Seeded random components | `tests/support/random_component_parity.py` | MATLAB R2019a/Python white-noise mesh, interpolation, and Energy intermediates | Self-hosted advisory CI |
| Tier 1 smoke | `tests/integration/parity/test_parity_pre_gate_tier1.py` | Pipeline wiring + LUT seam | Yes |
| Tier 2 surface | `tests/integration/parity/test_parity_pre_gate_tier2.py` | Oracle loadability | Optional |
| Certification | `slavv parity prove-exact` | Full-volume strict zero | Developer gate |

### Shared harness

[`tests/support/parity_harness.py`](support/parity_harness.py) centralizes:

- Crop harness paths (`SLAVV_CROP_TIFF`, `SLAVV_CROP_ORACLE_ROOT` overrides)
- Strict float64 assertions (`assert_bit_parity_energy`, `assert_bit_parity_scale`)
- One-voxel probe execution and oracle comparison
- JSONL probe export for MATLAB cross-validation (`ProbeRequest` / `ProbeResult` schema)

Checked-in fixtures live under [`tests/support/fixtures/`](support/fixtures/):

- `crop_energy_voxel_regression.json` — parametrized voxel oracle cases
- `matlab_linspace_probe_table.json` — full crop mesh table (2463 contexts)

### Pytest commands

```powershell
# CI-safe: unit + fixture-backed parity (no workspace oracles)
python -m pytest -m "unit and parity" tests/unit/pipeline/energy/test_matlab_linspace_table.py tests/unit/parity/test_parity_harness.py

# Local crop harness: voxel probes (requires promoted 180709_E_crop_M oracle)
python -m pytest -m "parity" tests/unit/pipeline/energy/test_voxel_probe.py tests/unit/pipeline/energy/test_voxel_probe_regression.py

# Integration pre-gate
python -m pytest tests/integration/parity/
```

### Script commands (local evaluation)

```powershell
# Run checked-in voxel regression fixture against oracle
python -m tests.support.parity_harness regression

# Probe top prove-exact mismatch groups
python -m tests.support.batch_energy_mismatch_probe --mode mismatch-groups `
  --probe-requests workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/energy_probe_requests.json

# Export probes for MATLAB-side execution, then compare JSONL responses
python -m tests.support.parity_harness export-jsonl --output workspace/scratch/python_probes.jsonl
python -m tests.support.parity_harness compare-jsonl `
  --matlab workspace/scratch/matlab_probes.jsonl `
  --python workspace/scratch/python_probes.jsonl
```

### Seeded random-component differential suite

Documented in [PARITY_RANDOM_COMPONENT_SUITE.md](../docs/reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md) ([ADR 0010](../docs/adr/0010-random-component-parity-suite.md)).

The suite is a fast diagnostic, not a synthetic certification claim. It writes six deterministic `uint16` white-noise TIFFs from the versioned manifest, then runs MATLAB R2019a and Python over the same inputs.

**Structural strict gate:** linspace (128), `interp3` (lattice queries), Energy `padded_shape_yxz`, sample coordinates, `valid`.

**Advisory only:** Hessian float fields (`hessian_diagnostics.max_ulp_distance` in reports; CI prints summary without failing).

```powershell
$env:MATLAB_EXE = "C:\Program Files\MATLAB\R2019a\bin\matlab.exe"
python -m tests.support.random_component_parity `
  --output-dir workspace\scratch\random_component_parity `
  --matlab-exe $env:MATLAB_EXE

python -m tests.support.random_component_parity `
  --print-hessian-summary workspace\scratch\random_component_parity\random_component_parity_report.json
```

Self-hosted workflow: `.github/workflows/matlab-random-component-parity.yml`.

Fixtures: `tests/support/fixtures/matlab_random_component_corpus.json`, `matlab_random_linspace_reference.json`, `matlab_random_matching_reference.json`.

Bit-parity rules:

- **Scale indices:** exact integer match everywhere.
- **Voxel probe vs promoted oracle:** scale exact; energy within 8 ULP (`assert_oracle_energy_parity`).
- **Python vs live MATLAB probe JSONL:** strict `float64` equality (`assert_bit_parity_energy`).
- **Full volume `prove-exact`:** strict `np.equal` on stage vectors (certification capstone).

After lower layers pass, run `prove-exact` per [PARITY_CERTIFICATION_GUIDE.md](../docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md).

When a regression locks a fix, add cases to `tests/support/fixtures/crop_energy_voxel_regression.json` rather than hard-coding coordinates in multiple test files.
