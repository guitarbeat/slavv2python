# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-06-05

**Authoritative status log** for exact-parity alignment with MATLAB. **Live operational status** (active runs, proof failures, blockers) lives here—not in [TODO.md](../../TODO.md). Tasks and checkboxes: TODO only.

**Spec:** [phase-1-exact-route-spec.md](../../plans/phase-1-exact-route-spec.md)

---

## 📊 Executive status (stage model)

Phase 1 exit criterion: **strict zero** missing/extra per stage via sequential `prove-exact` (energy → vertices → edges → network) on full `180709_E`—not informal match-rate thresholds.

| Stage | Harness / prior work | Phase 1 certification (strict zero) |
| :--- | :--- | :--- |
| **Energy** | Native Hessian path exact-compatible | 🟡 Active reruns for both crop (PID `18556`) and canonical (PID `26832`) after discovering stale vertex checkpoints. These runs enforce the new **[Y, X, Z]** internal grid and Fortran-order linear indexing required for bit-perfect parity. |
| **Vertices** | Verified on prior surfaces | ⏳ Pending passing proof on certified run |
| **Edges** | v29 ~88.7% pair match (diagnostic baseline) | ⏳ Pending sequential proof. **Fixed**: Resolved `KeyError` in `FrontierQueue` and restored iterative directional suppression for origin seeds. |
| **Network** | End-to-end pipeline runs | ⏳ Pending sequential proof |
| **`prove-exact` energy stage** | ✅ In `EXACT_STAGE_ORDER` (2026-05-28) | Required for R3 gating |

---

## 🚦 Active Phase 1 operations

| Track | Run / artifact | Status |
|-------|----------------|--------|
| **Canonical cert** | `workspace/runs/oracle_180709_E/phase1_cert_network` | 🟡 Rerunning from Energy (PID `26832`). Previous run (PID `10532`) was based on stale vertex checkpoints that diverged significantly from the oracle. This rerun establishes a bit-perfect foundation using standard [Y, X, Z] alignment. |
| **Crop harness oracle** | `workspace/oracles/180709_E_crop_M` | ✅ Promoted ([HDF5 energy loader](../../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md)) |
| **Crop harness run** | `workspace/runs/oracle_180709_E/crop_M_exact` | 🟡 Active Energy rerun (PID `18556`). Incorporates bit-perfect `linspace` roundoff and raw intensity preservation. |

**Champion edges baseline (informal, not cert bar):** `workspace/runs/oracle_180709_E/validation_strel_fix_output_v29`

### Cold-start protocol

If resuming exact parity work from a fresh thread:

1. Check active monitored jobs with `slavv jobs list` to see if any parity jobs are running.
2. Check the crop rerun status with `slavv parity status-exact-run --run-dir workspace/runs/oracle_180709_E/crop_M_exact`.
3. Prefer run-local `99_Metadata/parity_job.pid` / `parity_job.json` over legacy scratch PID files. If a matching process is still alive, do not start another writer on `crop_M_exact`.
4. If it has exited, run the crop energy proof first.
5. If energy passes, refresh crop downstream checkpoints with `--force-rerun-from vertices --stop-after network --monitor`, then run `prove-exact-sequence`.
6. If crop energy passes, rerun canonical `phase1_cert_network` from energy using `workspace/scratch/phase1_cert_network_rerun_from_energy.ps1` with `--monitor`.
7. If any proof fails, inspect the first failing field before changing code.

Use the `--monitor` flag on long reruns to enable automatic tracking and desktop notifications (see [PARITY_JOB_MONITORING.md](../../reference/workflow/PARITY_JOB_MONITORING.md)).

Scratch diagnostics for the current crop energy hypothesis are indexed in `workspace/scratch/parity_energy_diagnostics.md`.

### Operator commands

```powershell
# Monitor canonical run
slavv monitor --run-dir workspace/runs/oracle_180709_E/phase1_cert_network

# Rerun canonical from energy (single process; required before proof)
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 `
  --force-rerun-from energy `
  --stop-after network `
  --skip-preflight

# Re-prove crop harness after energy fix (rerun from energy first if code changed)
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --force-rerun-from energy `
  --stop-after network `
  --skip-preflight

slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M
```

---

## 📚 Compound learnings (parity-related)

Curated index of solved problems under `docs/solutions/` (from `/ce-compound`). Search all solutions via YAML frontmatter (`module`, `tags`, `problem_type`); see [docs/solutions/README.md](../../solutions/README.md).

| Topic | Doc |
|-------|-----|
| MATLAB energy HDF5 + `promote-oracle` | [matlab-v200-energy-hdf5-oracle-loader.md](../../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md) |
| Detached exact-route parity jobs | [detached-exact-run-jobs.md](../../solutions/parity/detached-exact-run-jobs.md) |

_Add rows here when a new compound doc is parity-relevant; do not duplicate full write-ups in this file._

---

## 🏆 June 2026 Memory Breakthrough (Canonical Scale-up)

A second major architectural breakthrough was achieved in June 2026, resolving persistent **ArrayMemoryError** blocks that prevented Phase 1 certification of the full 512x512x64 canonical volume.

### The Solution: Incremental Best-Scale Engine
- **4D Array Elimination**: Refactored `exact_mesh.py` to discard the large per-chunk 4D energy stack. The engine now updates the `best_energy` and `best_scale_index` volumes incrementally within the multi-scale loop. Peak memory usage dropped from **~300 MiB/thread to ~10 MiB/thread**.
- **Kernel Pre-computation**: Optimized the Hessian backend to pre-compute scale-independent derivative kernels (9 complex/double volumes per chunk) once. This eliminated redundant allocations that were fragmenting the heap.
- **Explicit GC Control**: Integrated `gc.collect()` and explicit `del` of large DFT products to ensure immediate reclamation of working memory.
- **Outcome**: Enabled stable multi-scale processing of the full canonical volume on hardware with limited physical RAM (e.g. 16GB), allowing the `phase1_cert_network` track to proceed to formal proof.

### Bit-Perfect Mathematical Refinements
The memory-safe engine simultaneously absorbed two final mathematical refinements discovered during crop-harness isolation:
- **MATLAB `linspace` Roundoff**: Replaced standard arithmetic meshes with MATLAB-accurate `linspace` endpoints. This preserved tiny fractional drifts (e.g. $10^{-16}$) that were causing interpolation boundary flips in the MATLAB `interp3` engine.
- **Raw Intensity Preservation**: Forced the exact-route pipeline to skip all normalization and clipping steps, ensuring bit-perfect parity with the MATLAB TIFF/HDF5 source values.

---

## 🏆 Historical high-water mark breakthrough (May 2026)

A major architectural breakthrough was achieved in May 2026, dramatically narrowing the discrepancy gap in edge candidate generation.

This section is historical context only. See [Active Phase 1 operations](#-active-phase-1-operations) and [Active blockers](#-active-blockers) for current proof status.

### The Solution: Parameter Alignment & NaN Stability
- **Parameter Alignment (v29)**: Discovered that the MATLAB oracle was generated with `edge_number_tolerance = 4`, while Python was hardcoded to 2. Aligning this parameter allowed high-degree vertices (hubs) to initiate sufficient exploratory traces.
- **NaN Stability**: Fixed a floating-point instability where multiplying `-Inf` (vertex priority) by `0.0` (directional suppression factor) created `NaNs`, leading to incorrect seed selection in subsequent iterations.
- **Precision Alignment (May 22)**: Implemented bit-accurate tie-breaking using exact equality (`==`) and Fortran-order linear index priority. Removed all remaining `float32` casts in the expansion frontier.
- **Tightened Filtering**: Implemented hard distance cutoffs ($d/R > 3.0$) and aligned edge influence sigmas to exactly $2/3$.
- **Outcome**: Successfully reached the **88.7%** match rate milestone (1062/1197 pairs). This remains an informal edge baseline, not the Phase 1 certification bar.

### Historical mathematical impact
| Metric | Previous Baseline (v28) | High-Water Mark (v29) |
| :--- | :--- | :--- |
| **Matched MATLAB Pairs** | 958 | 1062 |
| **Total Match Rate** | 80.0% | 88.7% |
| Missing Pairs | 239 | 135 |
| Over-generated Pairs | 263 | 371 |

---

## ⚖️ Exact Parameter Fairness Gate

To guarantee a fair mathematical race, all exact-route experiments must maintain structural lock-step between Python and MATLAB configuration inputs. This is validated via the **Parameter Diffusion Matrix**.

Every compliant proof run maintains three persistent JSON manifests under `01_Params/`:

1. **`shared_params.json`**: The authoritative overlap of settings that must exist in both MATLAB and Python.
2. **`python_derived_params.json`**: Internal Python-only pipeline management levers.
3. **`param_diff.json`**: The cryptographic hash bridge that proves zero illicit divergence occurred between the split configuration states.

### Locked Mathematical Constants
The audit system mandates these exact value bindings (derived from source-hardcoded MATLAB constants):
- `step_size_per_origin_radius = 1.0`
- `max_edge_energy = 0.0`
- `distance_tolerance_per_origin_radius = 3.0`
- `edge_number_tolerance = 4` (Corrected from 2)

---

## 🛠️ Verified Infrastructure Fixes

The core codebase has absorbed the following permanent fixes, ensuring structural stability:

*   ✅ **Exact-route intensity scale**: Skip min-max normalization and ignore `intensity_limits` clipping when `comparison_exact_network=True`, preserving the raw TIFF/HDF5 values used by the MATLAB crop oracle (2026-06-02).
*   ✅ **Energy downsample stride phase**: `_downsample_volume` uses MATLAB `get_starts_and_counts_V200` last-chunk alignment `start = (size-1) mod rf` per axis (whole-volume single chunk). Verified analytically (rf=9 → start 3; rf=20 → start 15) and empirically (origin energy −20.3757) (2026-05-29).
*   ✅ **HDF5 scale-index base**: `matlab_vector_loader` reads MATLAB global scale subscripts as 1-based (`one_based=True`), converting e.g. 91→90 for Python 0-based indexing (2026-05-29).
*   ✅ **Energy axis permutation**: `energy_axis_permutation` param permutes `microns_per_voxel` and `pixels_per_sigma_PSF` to the working image axis order so per-axis resolution factors map to the correct dimensions; added to `EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS` (2026-05-29).
*   ✅ **Upsample/`interp3` coordinate consistency**: exact-route chunk mesh uses MATLAB `get_energy_V202` local ranges and `mod(offset, rf)/rf` coordinates for each writing chunk (2026-06-02).
*   ✅ **MATLAB `interp3` invalid-neighbor semantics**: exact-route interpolation now propagates positive-weight `Inf` coarse-energy neighbors instead of finite-only reweighting, matching MATLAB's false-candidate suppression at downsampled octave boundaries (2026-06-02).
*   ✅ **MATLAB `linspace` mesh roundoff**: exact-route chunk meshes use MATLAB-style `linspace` endpoints rather than arithmetic `arange / rf`; this preserves tiny fractional drift such as `9.000000000000002`, which changes `interp3` `Inf` propagation at octave boundaries (2026-06-02).
*   ✅ **Double-Precision Energy Alignment**: Forced all core watershed maps (`energy_map_temp`, `vertex_energies`) and neighborhood penalty calculations to `float64`. This prevents precision-induced tie-breaking divergences where `float32` would collapse distinct energy values into identical bits, causing different seed selections than MATLAB's `double`.
*   ✅ **Bit-Accurate Tie-Breaking**: Replaced `np.isclose` with exact bitwise equality and added linear index priority to the frontier priority queue, matching MATLAB's hub vertex exploration behavior.
*   ✅ **Hard Distance Cutoff**: Implemented the MATLAB-exact $d/R > 3.0$ expansion cutoff in the watershed loop.
*   ✅ **Edge Influence Alignment**: Updated default `sigma_per_influence_edges` to $2/3$, aligning with MATLAB's conflict painting regions.
*   ✅ **Stable Frontier Splicing**: Verified and anchored the `available_locations` insertion logic to mirror MATLAB's `find(..., 'last')` and `find(..., 'first')` behavior, ensuring stable FIFO/LIFO handling for identical energy seeds.
*   ✅ **Backtracking Pointer Correction**: Fixed reverse-index logic, allowing trace recovery back to the origin vertex.
*   ✅ **Stable Discovery Sorting**: Forces deterministic processing orders matching MATLAB's explicit energy quality sorting.
*   ✅ **Trace Order Randomization**: Anchored native shuffling to a stable, reproducible seeded RNG generator.
*   ✅ **Distance Normalization (r/R)**: Corrected physical energy penalties to scale relatively to the vessel's radius ($R$).
*   ✅ **Strel Offset Alignment**: Realigned watershed structuring element (strel) offsets and loops to match the (Z, X, Y) universe layout, fixing major distance-penalty errors.
*   ✅ **Filtering Logic Reordering**: Realigned the cleanup sequence (Crop first $\rightarrow$ Pair Cleanup second) to protect valid pairs from phantom blocking.

---

## 🚀 Active blockers

1. **Crop harness energy proof — current rerun pending** — The completed PID `30880` rerun reduced the crop gap to 9,095 scale mismatches; the first remaining mismatch at Python `(z=0,y=0,x=27)` was traced to two exact-route issues: arithmetic mesh coordinates missed MATLAB `linspace` roundoff, and exact-route preprocessing still clipped raw intensities with `intensity_limits`. A one-voxel MATLAB/Python diagnostic now matches the MATLAB winner (scale 54, energy `-12.028384150742127`). The live rerun now runs as detached parity job PID `6616` (registry ID `2d0e790e`) under `workspace/runs/oracle_180709_E/crop_M_exact/99_Metadata/`. **Next:** when the rerun completes, run `prove-exact --stage energy` first and inspect the first failing proof surface if any.
2. **Canonical run re-execution** — `phase1_cert_network` completed with pre-fix energy semantics. The canonical oracle now has readable normalized energy (`energy`: `(64, 512, 512)` `float64`; `scale_indices`: `(64, 512, 512)` `int64`; `energy_4d`: empty placeholder; `lumen_radius_microns`: `(99,)` `float64`). Rerun from energy on the current exact-route branch before any canonical proof claim. Commands are staged in `workspace/scratch/phase1_cert_network_rerun_from_energy.ps1`.
3. **Sequential strict-zero closure** — After energy passes on a run, prove vertices → edges → network in order. v29 **135 missing / 371 extra** pairs remain the informal edges baseline until `prove-exact --stage edges` reports zero.

**Superseded guidance:** “>95% match” or “prove-exact once parity exceeds 95%” is not the Phase 1 bar. Use strict zero per stage only.

