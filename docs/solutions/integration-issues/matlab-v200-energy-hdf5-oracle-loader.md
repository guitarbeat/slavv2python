---
title: MATLAB V200 energy HDF5 bundles for promote-oracle
date: 2026-05-28
category: integration-issues
module: slavv_python.analytics.parity
problem_type: integration_issue
component: testing_framework
symptoms:
  - "promote-oracle fails with ValueError missing MATLAB vector field energy when only metadata exists in energy_*.mat"
  - "find_matlab_vector_paths resolves energy_*.mat instead of the extensionless HDF5 volume companion"
root_cause: wrong_api
resolution_type: code_fix
severity: high
tags:
  - matlab-oracle
  - energy-hdf5
  - promote-oracle
  - prove-exact
  - parity-exact
  - crop-harness
related_components:
  - tooling
---

# MATLAB V200 energy HDF5 bundles for promote-oracle

## Problem

MATLAB `vectorize_V200` crop batches store energy in a **split layout**: a metadata-only `.mat` beside an extensionless HDF5 file. The parity harness assumed all energy fields lived in the `.mat`, so `promote-oracle` failed before normalized oracle artifacts could be written.

## Symptoms

- `python scripts/parity_experiment.py promote-oracle ...` raises `ValueError: missing MATLAB vector field: energy`.
- `data/energy_<timestamp>_<name>.mat` contains only `size_of_image`, `energy_runtime_in_seconds`, and `intensity_limits` — no `energy` or `scale_indices` arrays.
- Crop harness oracle promotion (`180709_E_crop_M`) blocked after MATLAB vectorization completed successfully.

## What Didn't Work

- Treating `data/energy_*.mat` as the authoritative energy artifact when an extensionless HDF5 companion exists.
- Expecting `energy_4d` inside the MATLAB crop export (only projected 3D energy and per-voxel scale indices are stored in HDF5).

## Solution

Teach `slavv_python/analytics/parity/matlab_exact_proof.py` to detect and load MATLAB energy HDF5 bundles.

**Batch layout (crop example `batch_260527-220010`):**

| Path | Role |
|------|------|
| `data/energy_<ts>_<name>.mat` | Metadata only (`size_of_image`, etc.) |
| `data/energy_<ts>_<name>` | Extensionless HDF5; dataset `d` with shape `(2, Z, Y, X)` |
| `settings/energy_<ts>.mat` | `lumen_radius_in_microns_range` for the lumen lookup table |

**HDF5 plane semantics (`dataset d`):**

| Plane | Field in exact-proof contract |
|-------|------------------------------|
| `[0]` | `scale_indices` (0-based indices into the lumen radius table) |
| `[1]` | `energy` (projected 3D energy, negative floats) |

**Code changes (commit `79ff949a`):**

- `_is_matlab_energy_hdf5`, `_matlab_energy_hdf5_companion`, `_find_matlab_energy_path` — prefer HDF5 over metadata-only `.mat`.
- `_load_normalized_matlab_energy_from_hdf5` — assemble normalized energy payload; `energy_4d` is intentionally empty (empty arrays compare equal when Python also omits 4D storage).
- `oracle_energy_size_of_image` in `surfaces.py` reads Z×Y×X from HDF5 plane `[1]`.

**Verify promotion:**

```powershell
python scripts/parity_experiment.py promote-oracle `
  --matlab-batch-dir workspace/scratch/matlab_crop_batches/batch_260527-220010 `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --dataset-file workspace/datasets/0cdf88e930482e9eb818963da22846c43b53b531582bf3aed83678b549863d06/01_Input/180709_E_crop_M.tif `
  --oracle-id 180709_E_crop_M
```

**Regression test:**

```powershell
pytest tests/unit/analysis/parity/test_matlab_exact_proof.py::test_find_matlab_vector_paths_prefers_hdf5_energy_companion -v
```

## Why This Works

The harness now matches MATLAB's on-disk contract: heavy volumes in HDF5, workflow metadata in `.mat`, lumen table in `settings/energy_<ts>.mat`. `promotion.materialize_oracle_root` calls the same `find_matlab_vector_paths` / `load_normalized_matlab_vectors` path as `prove-exact`, so discovery stays consistent end-to-end.

## Prevention

- After MATLAB vectorization, confirm `data/energy_<stem>` (no extension) opens in `h5py` with dataset `d` shaped `(2, Z, Y, X)`.
- Do not spatially crop a full-volume oracle in Python; run MATLAB on the identical subvolume (see [PARITY_PRE_GATE.md](../../reference/workflow/PARITY_PRE_GATE.md)).
- Distinguish **MATLAB oracle HDF5** (this doc) from **Python run energy storage** (`npy` / `zarr` on `init-exact-run`; see [ZARR_ENERGY_STORAGE.md](../../reference/backends/ZARR_ENERGY_STORAGE.md)).

## Known follow-up (not fixed by this change)

`prove-exact-sequence` on `workspace/runs/oracle_180709_E/crop_M_exact` still **failed the energy stage** after promotion and a successful pipeline run. Later diagnostics narrowed the failure beyond oracle discovery: same-scale values were effectively exact, but Python created false octave winners by finite-only interpolation where MATLAB `interp3` propagated `Inf` through invalid/nonnegative coarse-energy neighbors. The exact-route code has been patched; the next proof result depends on the active crop rerun. Treat any remaining mismatch as an energy computation issue, not a promotion/discovery bug.

```powershell
python scripts/parity_experiment.py prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M
```

## Related Issues

- [PARITY_PRE_GATE.md](../../reference/workflow/PARITY_PRE_GATE.md) — tier-2 crop harness workflow
- [PARITY_CERTIFICATION_GUIDE.md](../../reference/workflow/PARITY_CERTIFICATION_GUIDE.md) — `prove-exact-sequence` gates
- `slavv_python/analytics/parity/matlab_exact_proof.py` — loader implementation
- `scripts/matlab/vectorize_180709_E_crop_M.m` — headless crop vectorization driver
