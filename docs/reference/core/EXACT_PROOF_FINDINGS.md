# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-06-25 (oracle v2; **Energy + Vertices CERTIFIED** under ADR 0011 gate; **Edges in progress**)

**Authoritative status log** for exact-parity alignment with MATLAB. **Live operational status** (active runs, proof failures, blockers) lives here—not in [TODO.md](../../TODO.md). Tasks and checkboxes: TODO only.

**Spec:** [phase-1-exact-route-spec.md](../../plans/phase-1-exact-route-spec.md)

---

## 📊 Executive status (stage model)

Phase 1 exit criterion: **strict zero** missing/extra per stage via sequential `prove-exact` (energy → vertices → edges → network) on full `180709_E`—not informal match-rate thresholds.

| Stage | Harness / prior work | Phase 1 certification (strict zero) |
| :--- | :--- | :--- |
| **Energy** | Native Hessian path exact-compatible | 🟢 `prove-exact --stage energy` vs **`180709_E_crop_M_v2`** **PASS** (ADR 0011 `np.allclose` gate, rtol=1e-7/atol=1e-9). `scale_indices` **0**; `energy` max \|Δ\|=1.99×10⁻¹¹; `lumen_radius_microns` max \|Δ\|=7.1×10⁻¹⁵. Cross-library float drift is bounded, not a logic difference. Strict `np.equal` available via `--strict-floats`. |
| **Vertices** | Verified on prior surfaces | 🟢 **PASS** vs `180709_E_crop_M_v2` (`prove-exact --stage vertices` exit 0): positions + scales match MATLAB **exactly** (13,706 = 13,706; 0 missing/extra) after the SE fix (`ellipsoid_offsets` ports MATLAB `construct_structuring_element.m` float-radius membership); `energies` certify under the ADR 0011 `np.allclose` policy after the loader recovers true energies from the raw `vertices.mat` (curated artifact stored a rank ramp). |
| **Edges** | v29 ~88.7% pair match (baseline) | 🟡 Partially closed (~52% shared). Fixes `edge_number_tolerance`→2 and exact-route conflict-painting-off moved Python **9,429 → 13,775** (vs MATLAB 15,511). **Frontier-ordering ruled out** (faithful sorted-list closed only ~3% at 10× cost; reverted). **Diagnosis (per-edge, 2026-06-25): early trace termination on long paths** — 5,774/7,164 missing edges are **never traced** (Python over-traces 18,378 candidates but mis-connects), and never-traced edges are systematically longer (mean len 9.1 vs 5.3 shared). **Penalty/distance/tolerance math verified line-equivalent to MATLAB.** **Size-penalty reference fix** (use origin-vertex scale per `get_edges_by_watershed.m:243`, not the current voxel's) applied as a faithfulness correction but **ruled out** (no metric change — scale rarely drifts here; degree cap now matches at 4). **`microns_per_voxel` ordering RULED OUT**: making the forward-direction LUT use `mpv_matlab` (to match the strel) *regressed* shared 8,135→7,437 — the `mpv_matlab` (strel) / raw (forward-LUT) split is intentional/correct. **Three hypotheses now ruled out** (frontier ordering, size-reference, directional frame). Residual is a deep tracing-geometry divergence (Python over-traces ~18.4k candidates but mis-connects long edges); needs single-edge trace instrumentation or a 2-vertex synthetic isolation, not more parameter guesses. **Edges paused** pending that fresh approach. |
| **Network** | End-to-end pipeline runs | ⏳ Pending sequential proof |

---

## 🚦 Active Phase 1 operations

| Track | Run / artifact | Status |
|-------|----------------|--------|
| **Crop harness oracle** | `workspace/oracles/180709_E_crop_M_v2` | ✅ Fresh MATLAB batch `batch_260624-105705` (lattice-6000). Use v2 for all new proofs. v1 (`180709_E_crop_M`) stale on scale plane. |
| **Oracle artifact readiness** | `180709_E_crop_M_v2`, `180709_E_batch_190910-103039` | ✅ `ensure-oracle-artifacts --stage all` passes on v2. |
| **Crop harness run** | `workspace/runs/oracle_180709_E/crop_M_exact` | **Energy writer completed** (job `75188cc2`). `inspect-energy-evidence` **valid**. `prove-exact --stage energy` vs v2: ✅ **PASS** (allclose gate, exit 0) — `scale_indices` 0; `energy` max \|Δ\| 1.99×10⁻¹¹; `lumen_radius` max \|Δ\| 7.1×10⁻¹⁵. **Vertices/Edges/Network now unblocked** (sequential). |
| **Canonical cert** | `workspace/runs/oracle_180709_E/phase1_cert_network` | ⏸️ Default paused for cert claim until crop tier-2 sequence passes (ADR 0009: canonical may run in parallel when memory allows — not the Phase 1 claim surface until crop + canonical proofs pass). |

Evidence template: [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md)

### Current crop Energy evidence guard (2026-06-24)

- **Freshness command:** `slavv parity inspect-energy-evidence --run-root workspace/runs/oracle_180709_E/crop_M_exact`
- **Current result:** **valid** (checkpoint at `02_Output/python_results/checkpoints/checkpoint_energy.pkl`; parity `stage_metrics.energy` completed).
- **Proof oracle:** `workspace/oracles/180709_E_crop_M_v2` (`batch_260624-105705`).

### Latest crop Energy proof vs oracle v2 (2026-06-24)

- **MATLAB vectorization:** `batch_260624-105705` (~3h, lattice-6000) → promoted `180709_E_crop_M_v2`.
- **Proof:** `prove-exact --stage energy --oracle-root workspace/oracles/180709_E_crop_M_v2` → **FAIL** (`exact_proof_energy.json`).
- **`scale_indices`:** **0** mismatches (strict-zero vs v2).
- **`energy`:** **3,810,126** mismatches under strict `np.equal`; first `(0,0,0)` scale 90 agree; \|Δ\|≈1.1×10⁻¹⁴ (3 ULP).
- **ULP triage** (`workspace/scratch/energy_ulp_triage_v2.json`): on scale-agreeing mismatches — median **4 ULP**, p90 **13 ULP**, max \|Δ\| **1.99×10⁻¹¹**; **384,178** voxels bit-identical. Stored `best_energy.npy` matches single-octave Python replay on sampled mismatches (writer persistence ruled out).
- **Voxel probes vs v2:** `(0,0,0)` ≤8 ULP pass; `(40,83,116)` 47 ULP fail (stored==replay); `(12,0,0)` stored vs replay 8 ULP but both disagree with MATLAB.
- **Classification:** accumulated **NumPy vs MATLAB MKL** float drift at matching scales ([ADR 0010](../../adr/0010-random-component-parity-suite.md) documents ≥1 ULP IFFT floor; crop volume shows median 4 ULP). **No localized Python fix** — this is library-level FP non-associativity, not a logic bug.
- **✅ Resolved ([ADR 0011](../../adr/0011-energy-float-certification-policy.md) accepted 2026-06-24):** certification uses `np.allclose(rtol=1e-7, atol=1e-9)` on continuous float fields, strict on `scale_indices`/topology. `prove-exact --stage energy` now **PASSES** (exit 0, pass_rate 1.0). Pure ULP was rejected — it explodes near zero (36,074 false fails at 48 ULP, max 72,343 ULP, despite \|Δ\|≤2×10⁻¹¹). The generic comparator also applies `np.allclose` to other float fields (e.g. `lumen_radius_microns`, max \|Δ\|=7.1×10⁻¹⁵).
- **Advisory ULP gate** (`slavv parity prove-energy-ulp --max-ulps N`): strict `scale_indices`, bounded float ULP — **not** certification. Crop vs v2 @ `max_ulps=8`: **FAIL** (~755k voxels >8 ULP). @ `max_ulps=48`: **99.11%** pass rate (37,174 failures — denormal/near-zero energies with large bit-space ULP, \|Δ\| still ≤2×10⁻¹¹).
- **Downstream oracle v2:** `ensure-oracle-artifacts --stage all` passes (vertices 13,706; edges 15,511; network 10,722 strands).
- **Policy:** [ADR 0011](../../adr/0011-energy-float-certification-policy.md) **Accepted** — Option B refined to `np.allclose` (rtol=1e-7, atol=1e-9) on continuous floats; strict scales/topology.
- **Next:** run `prove-exact --stage vertices` vs `180709_E_crop_M_v2` (now unblocked), then edges, network → crop tier-2 sequence.

### Latest crop Energy proof (2026-06-22)

- **Writer:** `resume-exact-run --force-rerun-from energy --stop-after energy --skip-preflight --n-jobs 1`; `max_voxels_per_node_energy: 6000` (lattice `[3,3,2]`, 821 chunks).
- **Proof:** `prove-exact --stage energy` → **FAIL** (`03_Analysis/exact_proof_energy.json`).
- **First failure:** `energy.energy` @ `(0,0,0)` — scales agree (90); energy differs at ULP (~10⁻¹⁴) under strict `np.equal`.
- **Mismatch counts** (`exact_mismatch_energy.json`):
  - `energy`: **3,823,893** (max |Δ|≈26.4) — ~3,804,481 voxels have **matching scale** but fail bit-identical float compare (median |Δ|≈1.4×10⁻¹⁴).
  - `scale_indices`: **19,412** (0.46% of volume; max |Δ|=72) — first @ `(61,81,0)` MATLAB 44 vs Python 46.
  - `lumen_radius_microns`: **8** (machine epsilon).
- **Probe note:** One-voxel probe at `(12,0,0)` passes at `atol=1e-10` but fails strict `np.equal` on full volume — certification bar is bit-identical, not probe tolerance.
- **Adaptive probes:** `03_Analysis/energy_probe_requests.json` (3650 mismatch groups).
- **Next:** Triage scale-winner disagreements first; then float64 bit-identical path for scale-agreeing voxels. No downstream crop refresh.

### Fresh scale-winner triage (2026-06-24, post-writer)

Cross-octave Python replay on all **31** `scale_indices` mismatches (`workspace/scratch/fresh_scale_mismatch_triage.json`):

- **31/31** `cross_octave_reduction` — replayed Python winner matches **stored** `best_scale.npy` (not stale checkpoint drift).
- **0/31** replay matches MATLAB oracle winner.
- Scale delta histogram vs MATLAB: **14** at −1, **13** at +1 (remaining outliers at −5, −2, +7, +26).
- Example `(33,80,133)`: MATLAB 14, Python stored/replay **13**; octave-1 candidate wins replay (−20.15) over octave-5 scale 92 (−18.25).
- **Implication:** remaining scale gaps are live cross-octave winner math vs MATLAB, not writer persistence. Next fix surface: reduction/tie-break + invalid-octave candidate handling (probe shows `global_scale=-1`, energy `0.0` on octaves 3–4).

### MATLAB-backed cross-octave probe (top 3 voxels, 2026-06-24)

Batch replay: `workspace/scratch/cross_octave_top3/cross_octave_reduction.json` (R2019a, 15 per-octave probes).

| Voxel (Z,Y,X) | Oracle `scale_indices` | Python stored/replay | MATLAB live replay | Python↔MATLAB replay |
| --- | --- | --- | --- | --- |
| (33,80,133) | **14** | 13 | 13 | **agree** |
| (40,83,116) | **13** | 12 | 12 | **agree** |
| (33,83,131) | **13** | 12 | 12 | **agree** |

All three classify as **`matlab_oracle_state_path`**: promoted oracle plane is **exactly +1** vs fresh MATLAB batch replay on the same crop config; Python stored `best_scale.npy` matches live MATLAB replay on every sample.

**Revised implication:** the 31 `scale_indices` strict-zero failures may be dominated by **oracle vector indexing / promotion drift** (±1 vs live `get_energy_V202` replay), not Python cross-octave math. Next: audit oracle `scale_indices` plane convention vs `matlab_vector_loader` one-based shift and crop batch vintage before another Energy writer rerun.

### Oracle HDF5 scale-plane trace (top 3 voxels, 2026-06-24)

Source: `batch_260527-220010` HDF5 plane 0 (`get_energy_V202` writes `energy_chunk_scale_min + sum(octave_at_scales < current_octave)` — **1-based global** per `external/Vectorization-Public/source/get_energy_V202.m`).

| Voxel (Z,Y,X) | Raw HDF5 plane0 | After loader `−1` | Python stored | MATLAB live replay |
| --- | --- | --- | --- | --- |
| (33,80,133) | 15 | 14 | 13 | 13 |
| (40,83,116) | 14 | 13 | 12 | 12 |
| (33,83,131) | 14 | 13 | 12 | 12 |

**Loader is not double-subtracting:** `matlab_vector_loader` applies exactly one `one_based` shift. Identity: `raw_hdf5 − 2 == python_stored == matlab_live_replay` (0-based global); `raw_hdf5 − 1 == prove-exact oracle surface`.

**Conclusion:** promoted oracle `scale_indices` reflects the **May 27 full-volume MATLAB batch** (pre–lattice-6000 / pre–IFFT-fix crop writer), which is **+1 (0-based)** vs current Python and fresh MATLAB batch replay on the same voxels. Remediation path: **promote a fresh crop oracle** from a lattice-`6000` MATLAB run (or accept probe-surface proof) before chasing Python reduction code.

Historical crop Energy evidence (2026-06-21, superseded writer state):

- **Coarse-source / interpolation-mesh contract**: MATLAB `get_energy_V202` local ranges use `1 + floor(offset/rf) : 1 + ceil((offset + write_count - 1)/rf)` on the **padded FFT grid** (`fourier_transform_V2` output), not the strided read shape. Python had been clamping to `original_chunk.shape`, dropping one padded row on crop octaves 3–5 (e.g. octave 4 chunk 0: requested `(27,27,14)` vs retained `(26,26,13)`). Fixed via `_matlab_coarse_local_slices` in `matlab_get_energy_v202_chunked.py`.
- **Resolution-factor rounding (authoritative)**: MATLAB `get_energy_V202` line 116 uses `round(worst_resolution_to_downsample ./ resolutions_at_octave)` (positive half-up). Not `floor(L/(v*2.5))`.
- **Manual source audit (2026-06-22)**: MATLAB V202 and the Python exact route agree on octave consolidation, chunk geometry, symmetric even FFT padding, padded-grid local ranges, interpolation mesh construction, negative-only Energy handling, and first-winner min projection. V200's active principal-Energy code clips a positive third component and then uses an **unweighted** sum; the `[1.5, 1, 0.5]` weighted expression is commented out. Do not claim magnitude-descending eigenvalue sorting: MATLAB calls `eig`, while Python calls `eigh`; their ordering remains an empirical crop-probe and strict-proof check.
- **Coarse-range invariant**: Python now raises if a MATLAB-requested local range exceeds the padded FFT extent rather than silently shortening the interpolation source. The crop boundary regression covers the MATLAB-requested `(27,27,14)` support from raw `(26,26,13)` input.
- **Probe artifacts**: Python `workspace/scratch/crop_coarse_slice_probe_python.json`; MATLAB companion `workspace/scratch/matlab/probe_coarse_slice.m`.
- **Regression coverage**: `tests/unit/pipeline/energy/test_hessian_downsample.py::{test_exact_crop_coarse_slice_retains_padded_fft_support_not_strided_read,test_exact_crop_coarse_slice_octave4_chunk0_matches_matlab_local_ranges}`.
- **Prior proof (1M-chunk run, 2026-06-21)**: `prove-exact --stage energy` **FAIL** — 4,010,103 voxel diffs, max |Δ|≈135.4. First scale mismatch: `(12,0,0)` scale 54 / −13.52 (MATLAB) vs 61 / −16.45 (Python), octave `rf=[2,5,5]`.
- **Chunk-lattice root cause (2026-06-21)**: `max_voxels_per_node_energy` was `1_000_000` (lattice `[1,1,1]`, 1 chunk) vs MATLAB oracle `6_000` (lattice `[3,3,2]`, 18 chunks). Run-local `validated_params.json` restored to `6000`. With `6000`, one-voxel probe at `(12,0,0)` matches oracle: scale 54, energy −13.52067537392248.
- **Probe artifacts (voxel)**: `workspace/scratch/crop_voxel_12_0_0_probe_python.json`, `workspace/scratch/crop_voxel_12_0_0_probe_matlab.json`; helper `slavv_python/pipeline/energy/parity_energy_voxel_probe.py`.
- **Regression coverage (voxel + lattice)**: `tests/unit/pipeline/energy/test_voxel_probe.py` (3 tests, incl. lattice `[3,3,2]` and oracle match at `(12,0,0)`).
- **Ruled out for crop**: octave `unique(...,'last')` consolidation (scale groupings identical); coarse-slice truncation alone (fix did not move winner on 1M-chunk run).
- **Stale broadcast failure resolved**: earlier `(64,27,8)` into `(65,27,8)` crash came from unbounded slices; padded-bound slices stay inside the FFT grid on `(64,256,256)`.

### 🟡 2026-06-17: Energy Parity Discoveries (Crop Harness)

- **Eigenvalue Ordering**: The active V200 source calls `eig` and applies its special third-component clip in returned order; it does not sort eigenvalues by magnitude. Python uses `np.linalg.eigh` in the same returned-component role. Ordering is not certified by source inspection alone and remains covered by the crop one-voxel probe and strict Energy proof.
- **Resolution Factors**: MATLAB `get_energy_V202` uses `round(1/2.5 ./ resolutions_at_octave)` (positive half-up). Python now uses `floor(x+0.5)` in `matlab_energy_filter_v200.py`.
- **Validation Whitelist**: Identified that `validate_parameters` in `validation.py` was stripping `comparison_exact_network` and other orchestration keys during the `RunContext.prepare` phase, causing the pipeline to fall back to the standard "Paper" route. Whitelisted these keys to ensure the "Innovation" path is correctly triggered.
- **FFT Symmetry**: Verified that `_ifftn_matlab_symmetric` manual enforcement matches `np.fft.ifftn().real` and correctly handles Fortran-order raveling for conjugate-pair matching.

### Random component parity (2026-06-22)

Seeded white-noise differential suite ([ADR 0010](../../adr/0010-random-component-parity-suite.md), [PARITY_RANDOM_COMPONENT_SUITE.md](../workflow/PARITY_RANDOM_COMPONENT_SUITE.md)):

- **Structural gate (green):** 128 linspace contexts; 16 lattice/boundary `interp3` queries per case; Energy `padded_shape_yxz`, sample coordinates, and `valid`.
- **IFFT floor:** With a **byte-identical** MATLAB complex spectrum loaded in Python, `_ifftn_matlab_symmetric` vs MATLAB `ifftn(...,'symmetric')` differs by **1 ULP** at sample voxels — NumPy vs MKL FFT, not the symmetry mask.
- **Matching kernel:** `scipy.special.jv` vs MATLAB `besselj` drifts without the promoted `matlab_random_matching_reference.json` fixture on the Python runner.
- **Hessian floats:** Reported as advisory `hessian_diagnostics` (max ULP per case); they do **not** gate CI. Crop/canonical `prove-exact` remains the strict Energy certification surface.

*Status*: Incorporated into the current worktree, but not yet certified. See [Latest crop Energy proof (2026-06-22)](#latest-crop-energy-proof-2026-06-22).

**Implementation hardening:** Active plan at [random-component-parity-hardening-spec.md](../../plans/random-component-parity-hardening-spec.md) (Phase 0 complete: spec landed + baseline captured + unit tests green). Future changes to the suite (decomposition, models, separation of structural gate from advisory) should follow that spec. Baseline artifacts in `workspace/scratch/random_component_baseline/`.

**Champion edges baseline (informal, not cert bar):** `workspace/runs/oracle_180709_E/validation_strel_fix_output_v29`

### Cold-start protocol

If resuming exact parity work from a fresh thread:

1. Check active monitored jobs with `slavv jobs list` to see if any parity jobs are running.
2. Check the crop rerun status with `slavv parity status-exact-run --run-dir workspace/runs/oracle_180709_E/crop_M_exact`.
3. Prefer run-local `99_Metadata/parity_job.pid` / `parity_job.json` over legacy scratch PID files. If a matching process is still alive, do not start another writer on `crop_M_exact`.
4. Verify oracle surfaces with `slavv parity ensure-oracle-artifacts --oracle-root <oracle> --stage all --no-repair` if there is any doubt about artifact readiness.
5. If no writer is alive and Energy artifacts are missing, rerun crop Energy before attempting proof.
6. If Energy artifacts exist, run the crop energy proof first.
7. If energy passes, refresh crop downstream checkpoints with `--force-rerun-from vertices --stop-after network --monitor`, then run `prove-exact-sequence`.
8. If crop energy passes, rerun canonical `phase1_cert_network` from energy using `workspace/scratch/phase1_cert_network_rerun_from_energy.ps1` with `--monitor`.
9. If any proof fails, capture evidence per [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md) and inspect the first failing field before changing code.
10. If `status-exact-run` reports `interrupted` (dead PID), reconcile is automatic; rerun foreground diagnostic before detaching another writer.

Use the `--monitor` flag on long reruns to enable automatic tracking and desktop notifications (see [PARITY_JOB_MONITORING.md](../../reference/workflow/PARITY_JOB_MONITORING.md)).

Scratch diagnostics: `workspace/scratch/crop_coarse_slice_probe_python.json`, `workspace/scratch/matlab/probe_coarse_slice.m`, `workspace/scratch/crop_voxel_12_0_0_probe_{python,matlab}.json`.

### Operator commands

```powershell
# Rerun crop Energy first; do not start canonical while this is pending.
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M_v2 `
  --force-rerun-from energy `
  --stop-after energy `
  --skip-preflight

slavv parity prove-exact `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M_v2 `
  --stage energy

# Only after crop Energy passes, refresh downstream crop checkpoints.
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M_v2 `
  --force-rerun-from vertices `
  --stop-after network `
  --skip-preflight

slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M_v2

# After crop sequence passes, rerun canonical from Energy.
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 `
  --force-rerun-from energy `
  --stop-after network `
  --skip-preflight

slavv monitor --run-dir workspace/runs/oracle_180709_E/phase1_cert_network
```

---

## 📚 Compound learnings (parity-related)

Curated index of solved problems under `docs/solutions/` (from `/ce-compound`). Search all solutions via YAML frontmatter (`module`, `tags`, `problem_type`); see [docs/solutions/README.md](../../solutions/README.md).

| Topic | Doc |
|-------|-----|
| MATLAB energy HDF5 + `promote-oracle` | [matlab-v200-energy-hdf5-oracle-loader.md](../../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md) |
| Detached exact-run jobs | [detached-exact-run-jobs.md](../../solutions/parity/detached-exact-run-jobs.md) |
| Run/proof evidence template | [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md) |
| Sparse Meshgrid Memory Optimization | [sparse-meshgrid-memory-optimization.md](../../solutions/parity/sparse-meshgrid-memory-optimization.md) |
| MATLAB Stride Phase Lead | [matlab-stride-phase-lead.md](../../solutions/parity/matlab-stride-phase-lead.md) |

_Add rows here when a new compound doc is parity-relevant; do not duplicate full write-ups in this file._

---

## 🏆 June 2026 Memory Breakthrough (Canonical Scale-up)

A second major architectural breakthrough was achieved in June 2026, resolving persistent **ArrayMemoryError** blocks that prevented Phase 1 certification of the full 512x512x64 canonical volume.

### The Solution: Incremental Best-Scale Engine
- **4D Array Elimination**: Refactored `matlab_get_energy_v202_chunked.py` to discard the large per-chunk 4D energy stack. The engine now updates the `best_energy` and `best_scale_index` volumes incrementally within the multi-scale loop. Peak memory usage dropped from **~300 MiB/thread to ~10 MiB/thread**.
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

*   ✅ **Watershed Orientation Realignment**: Standardized the Edges stage on internal [Y, X, Z] orientation with Fortran contiguity. Input volumes are now explicitly transposed before watershed processing, and result maps are re-mapped to physical [Z, Y, X] for artifact persistence (2026-06-13).
*   ✅ **Vertex Painting Rounding**: Replaced Python's built-in `round()` (round-to-even) with bit-perfect round-half-up logic in the painting and candidate filtering loops, preventing selection divergence at .5 coordinate boundaries (2026-06-13).
*   ✅ **Mesh Offset Alignment**: Fixed a 3-pixel coordinate shift in energy interpolation by removing saturated subtraction from chunk offsets and updating `_matlab_zero_based_linspace` to handle explicit local starts (2026-06-13).
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

## 2026-06-14 Update: Systemic float64 Enforcement & Spatial Alignment

*   **Systemic Precision Alignment**: Identified and replaced all `float32` and `np.float32` casts with `float64` across Energy, Vertices, and Edges stages. This resolves precision-induced divergences where rounding (e.g., `np.floor(pos + 0.5)`) or normalization would deviate from MATLAB's `double`.
*   **Bessel Sum Chunking**: Implemented a chunked computation loop for `jv` sums in `_matching_kernel_dft` to keep peak memory footprint minimal during kernel generation, preventing `ArrayMemoryError` on canonical volumes.
*   **Preprocessing Parity**: Fixed `preprocess_image` to respect `comparison_exact_network=True` by using `float64` and skipping min-max normalization, ensuring raw TIFF/HDF5 values are preserved for the Hessian engine.
*   **Spatial Shift Resolution**: Discovered a (0, 15, 15) pixel shift in the energy map. Root cause: MATLAB's "Last Chunk Alignment" rule in `get_starts_and_counts_V200` shifts the reading start to align with the final pixel. For a single-chunk volume (like the crop volume at coarse octaves), this results in a 15-pixel lead in the coarse grid. Reintroduced `sat_sub` in Python's `matlab_get_energy_v202_chunked.py` to correctly replicate this shifting behavior.

---

## 🚀 Active blockers

1. **Crop Energy strict-zero proof** — Current `crop_M_exact` Energy evidence is valid after the 2026-06-24 Energy writer. Scale winners now agree (`scale_indices` 0 mismatches); the active blocker is strict float64 Energy drift (~3.81M ULP-level mismatches) plus the epsilon-radius tail. Use `exact_proof_energy_ulp.json` / `exact_mismatch_energy.json` as the current diagnostic surface, not stale `exact_proof_energy.json`.
2. **Sequential strict-zero closure** — Blocked on crop Energy proof. After pass: refresh vertices → network on `crop_M_exact`, then `prove-exact-sequence`. v29 **135 missing / 371 extra** pairs remain the informal edges baseline until `prove-exact --stage edges` reports zero.
3. **Canonical run re-execution** — Phase 1 **cert claim** stays on full `180709_E` only. Crop is the fast iteration gate (ADR 0009). Rerun canonical from Energy after crop Energy passes, or in parallel when preflight shows safe memory headroom (one long Energy writer at a time on 16GB).

**Superseded guidance:** “>95% match” or “prove-exact once parity exceeds 95%” is not the Phase 1 bar. Use strict zero per stage only.

