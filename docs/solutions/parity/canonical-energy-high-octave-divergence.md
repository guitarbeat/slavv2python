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

## Next step (the open work) — now targeted
Since MATLAB uses the **same** 9-chunk octave-4 geometry yet gets the strong energy,
this is a **Python-vs-MATLAB per-chunk computation difference at rf≫1** (not seams,
not n_jobs). Prime suspects, in order: per-chunk **FFT padded-shape / frequency
grid**, **kernel normalization at large downsample**, the **`interp3` upsample**.
MATLAB ground-truth: for one octave-4 chunk, dump MATLAB's per-scale energy at a few
large-vessel voxels (instrumented `get_energy_V202`, R2019a) and compare to Python's
to find which step weakens Python's energy. (Oracle `energy_4d` is empty, so a
MATLAB run is required.)

## Notes
- A clean single-writer rerun (v4) would **reproduce** this (energy is
  deterministic; n_jobs ruled out) — so the v4 rerun is paused pending root cause.
- The v3 run had a concurrent-writer incident (a stray `init` was not killed before
  the `n_jobs=6` resume); its checkpoints were verified single-run + correct
  orientation, and its snapshot was repaired to reflect completion. See
  [resume-energy-orientation](resume-energy-orientation.md).
