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

## Ground-truth result (MATLAB harness) — Python energy zeroed at MATLAB's scale
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

`0.0` is the clamp firing (invalid/Inf → 0). The original guess was a sign error in the
matched filter — **that was wrong** (see root cause below).

## ROOT CAUSE (verified) — upsample-mesh roundoff at a coarse-cell boundary
Instrumenting the intermediate Hessian/Laplacian fields for one octave-4 chunk
(voxel (0,94,390), scale 54, rf=[2,5,5]) localized the bug precisely. Tools:
`energy_filter_V200_instr.m` (instrumented copy of the oracle filter, source untouched),
`probe_laplacian_octave4.m` / `.py`.

The Hessian, Laplacian, principal curvatures, and principal energy are computed
**correctly and identically** on both sides — there is **no** sign error. The coarse
energy at the relevant valid corners matches: Python's own coarse field has
`E(Y1,X10,Z0)=−2.846`, `E(Y2,X10,Z0)=−3.679`, exactly as MATLAB.

The divergence is purely the **coarse→fine upsample mesh**. At this voxel MATLAB's
`interp3` mesh coordinate is exactly `x=10.0`; Python's `_matlab_zero_based_linspace`
yields `x=9.999999999999998` (**1.78e-15** low). That sub-ULP drift floors the trilinear
interp **base cell** from X=10 down to X=9 — and the X=9 coarse corner is *invalid*
(`Laplacian=+2.95 ≥ 0`, genuinely "not a vessel", agreed by both sides). MATLAB,
sitting exactly on the integer grid line, samples only the valid X=10 column and gets
`0.2·(−2.846) + 0.8·(−3.679) = −3.5122`. Python straddles into the invalid X=9 column;
the `Inf` corner poisons the trilinear blend and the energy collapses to `0.0`.

**Decisive check:** feeding Python's own valid corners with `x=10.0` (instead of the
drifted value) reproduces `−3.512245`, matching MATLAB to 6 decimals. So the −3.51 is
recoverable from Python's own (correct) coarse field — only the mesh coordinate is off.

This is the same class as the documented "preserve MATLAB `linspace` roundoff" issue,
but here the port does **not** match at integer-valued sample points: MATLAB `linspace`
lands exactly on the grid integer; the Python mesh is one ULP short, which only matters
when the adjacent coarse cell is invalid (so it never showed up away from vessel/octave
boundaries). The 16/16 "energy = 0 at MATLAB's scale" symptom is consistent with this
mechanism (octave-3/4 winners sit next to invalid downsampled cells); mechanism verified
in full on (0,94,390)/scale 54.

## Fix applied + result — mesh snap clears the interp-clamp class (39,494 → 11,793)
`_snap_mesh_to_grid_integers` (in `matlab_get_energy_v202_chunked.py`) snaps upsample-mesh
coordinates within 1e-9 of an integer to it, so the Python mesh lands on grid lines like
MATLAB `linspace` and `interp3` no longer floors across a coarse-cell boundary into an
invalid cell. Full test suite green (595 passed); the 16 probe voxels now reproduce
MATLAB's scale **and** energy exactly.

Canonical energy re-run (v4, parallel n_jobs=6, then `prove-exact --stage energy`):
`scale_indices` mismatches dropped **39,494 → 11,793** (~70%), but the energy gate still
**FAILS** ADR 0011 strict-zero. Energy floats pass `allclose`; the blocker is the residual
`scale_indices`.

### The residual (11,793) is a SECOND, distinct class — genuine per-scale energy diffs
Probing a residual voxel `[41,310,0]` (MATLAB winner scale 95, Python 74) with the fixed
code: Python `E@74 = −6.485`, `E@95 = −5.956` — **both valid negatives, neither clamped**.
So this is *not* the interp-clamp/Inf mechanism; the per-scale matched-filter energies
themselves differ between Python and MATLAB enough to flip the argmin between non-adjacent
scales. This is the "computation differs" signal from the isolation section, now cleanly
separated from the (fixed) interp-clamp class.

## Next step (the open work) — root-cause the residual per-scale energy difference
Instrument the per-scale energy at a residual voxel on **both** sides (extend the MATLAB
`probe_canonical_targets.m` / Python `probe_python_targets.py` to the residual voxel set,
e.g. `[41,310,0]` scales 74 & 95) and compare the matched-filter / Hessian intermediates
(as in `probe_laplacian_octave4.*`) to find which downsampled-octave filter step yields the
small numeric energy difference. Suspects: FFT padded-shape / frequency grid or kernel
normalization at rf≫1 (the interp/Hessian-assembly is already cleared).

## Notes
- A clean single-writer rerun (v4) would **reproduce** this (energy is
  deterministic; n_jobs ruled out) — so the v4 rerun is paused pending root cause.
- The v3 run had a concurrent-writer incident (a stray `init` was not killed before
  the `n_jobs=6` resume); its checkpoints were verified single-run + correct
  orientation, and its snapshot was repaired to reflect completion. See
  [resume-energy-orientation](resume-energy-orientation.md).
