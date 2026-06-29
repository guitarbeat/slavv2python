---
title: Canonical Energy Divergence at Multi-Chunk Downsampled Octaves
module: pipeline/energy
tags: [energy, parity, canonical, octaves, chunking, open]
problem_type: parity
resolution_type: investigation_open
---

# Canonical Energy Divergence at Multi-Chunk Downsampled Octaves

**Status: OPEN investigation** (root cause not yet located).

## Problem
After the resume-orientation fix, the canonical full-`180709_E` energy proof runs
cleanly (correct `(64,512,512)` orientation) but **does not certify**:
`scale_indices` differs from the MATLAB oracle at **39,494 / 16,777,216 voxels
(0.235%)**. ADR 0011 requires strict zero on `scale_indices`, so energy FAILs.

## Evidence / characterization
- Not float ties: at mismatch voxels energy `|Δ|` median **1.79** (max 29);
  **0%** within the `allclose` band. Genuinely different energy + scale winner.
- `|Δscale|` spans **1→90** (large jumps), concentrated in **octaves 3–4**
  (scales 54–89; peak octave 4, scales 72–89).
- The crop (`180709_E_crop_M`) **certified** energy (0 scale mismatches). Per-octave
  chunk counts explain why:

  | octave | rf (z,y,x) | crop chunks | full chunks |
  |---|---|---|---|
  | 3 | [2,5,5] | 18 | 72 |
  | **4** | **[5,10,10]** | **1** | **9** |

  The crop's octave 4 is a **single chunk**; the full volume's is **9 chunks**.
  The mismatches peak exactly where the full volume goes multi-chunk.

## Ruled out (decisive)
- **`n_jobs` parallelism.** Crop energy at `n_jobs=6` is **byte-identical** to the
  certified `n_jobs=1` crop (same SHA, 0 differences, max|Δ|=0), across all the
  crop's octaves. The chunk merge is an order-independent min; threading is
  bit-exact. (This also validates the n_jobs bit-exactness claim beyond octave 1.)
- **Chunk seams.** Only 0.9% of mismatches sit on an octave-4 write-window border
  (17% within 10 voxels — *less* than a random voxel's ~30%). The divergence is in
  chunk **interiors**, not at overlap seams.

## Leading hypothesis (NOT confirmed)
A per-chunk computation at heavily-downsampled octaves differs from MATLAB when the
full volume splits an octave into many small chunks (vs the crop's single chunk):
the per-chunk **FFT padded-shape / matched-filter kernel** or the **`interp3`
upsample** at rf≫1 produces slightly different interior energy, flipping scale
winners over large vessels. Why crop octave 3 (18 chunks) certifies but full
octave 3 (72 chunks) does not is unexplained — so this is not a simple
"multi-chunk is wrong" story.

## Isolation result (computation vs selection) — COMPUTATION
At the 39,494 mismatch voxels: `E_python − E_matlab` median **+0.71** (mean +1.16);
Python's winner is **less negative than MATLAB's 70.5%** of the time. Under
min-projection, identical per-scale fields would pick the same winner and yield
identical final energy — the ~0.7–1.2 gap proves the **per-scale energies genuinely
differ** (computation, not a selection/tie-break flip). Directionally, **Python
under-computes the downsampled octave-3/4 matched-filter energy for large vessels**:
MATLAB's winners concentrate in octaves 3–4 (17,802 + 17,090) while Python's octave-3
winners are roughly half (9,098), with Python scattering to weaker winners in
octaves 1–2 and 5. So Python's octave-3/4 energy is too weak and loses the argmin.

## Ground-truth result (MATLAB harness) — sign flip, not weak magnitude
The MATLAB ground-truth harness now exists and is validated:
`workspace/scratch/matlab_energy_instr/` (see its `README.md`). It replays
`get_energy_V202`'s per-chunk math (`energy_filter_V200` + `interp3`) at the 16
target voxels on **both** sides and lines up the per-scale upsampled energy:
- `probe_canonical_targets.m` (R2019a) — MATLAB per-scale. Validated: its
  downsampled-octave argmin reproduces the **oracle's** stored scale *and* energy at
  the targets (e.g. voxel (0,94,390): scale 54, energy −3.5122, matching the oracle to
  6 decimals).
- `probe_python_targets.py` — Python per-scale via `parity_energy_voxel_probe`.
  Validated: its winner reproduces the **canonical run's** stored `scale_indices` on
  targets whose winner is in the computed (downsampled) octaves.
- `compare.py` — per-scale diff. **Crucially, the resolution factor per scale index is
  identical on both sides** (scale 54 → rf=[2,5,5], scale 72 → rf=[5,10,10]); only the
  octave *label* differs by 1 (Python merges the two rf=[1,1,1] octaves). So the
  comparison is apples-to-apples: same scale, same downsampling, different energy.

**Finding (validated on all 16 targets — 16/16):** at the scale MATLAB chooses,
**Python's energy is exactly `0.0` in every case** while MATLAB's is strongly negative
(−2.76 to −33.88). The MATLAB harness winner also reproduced the oracle's stored scale
for all 16 targets, fully validating the MATLAB side.

| voxel (z,y,x) | MATLAB scale | E_matlab@scale | E_python@scale | Python winner |
|---|---|---|---|---|
| (0, 94, 390) | 54 | −3.5122 | **0.0000** | 98 |
| (6, 210, 383) | 85 | −9.0542 | **0.0000** | 83 |
| (18, 390, 228) | 58 | −33.8819 | **0.0000** | 96 |
| (40, 319, 200) | 88 | −22.2610 | **0.0000** | 54 |
| … (12 more) | | all −ve | **all 0.0000** | |

`0.0` is the clamp firing (`energy >= 0 → 0`): Python's matched-filter energy at that
scale/voxel is **non-negative** — a positive Laplacian / positive principal energy,
i.e. "not a vessel" — so it is zeroed and cannot win the argmin, and Python settles on
a different scale. This reframes the bug: it is **not** a magnitude under-computation
or a tie-break, it is a **sign / principal-curvature error** in the downsampled-octave
matched filter (rf≫1). The earlier "Python energy ~0.7–1.2 weaker" characterization
was the aggregate shadow of these zeroed scales losing the argmin.

## Next step (the open work) — now narrowed to the filter internals
The divergent step is inside the per-chunk `energy_filter_V200` equivalent at rf≫1,
where Python's output sign is wrong. Suspects, in order: the **principal-energy /
Hessian-curvature assembly** (sign of the projected curvatures → Laplacian validity
mask), then the per-chunk **FFT padded-shape / frequency grid** and **kernel
normalization at large downsample**. Instrument one octave-4 chunk at voxel
(0,94,390), scale 54, on both sides and compare the intermediate Laplacian /
principal-curvature fields (not just the final energy) to localize the sign flip.

## Notes
- A clean single-writer rerun (v4) would **reproduce** this (energy is
  deterministic; n_jobs ruled out) — so the v4 rerun is paused pending root cause.
- The v3 run had a concurrent-writer incident (a stray `init` was not killed before
  the `n_jobs=6` resume); its checkpoints were verified single-run + correct
  orientation, and its snapshot was repaired to reflect completion. See
  [resume-energy-orientation](resume-energy-orientation.md).
