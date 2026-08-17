---
title: Crop Energy stretch float isolation (E11–E20)
module: pipeline/energy
tags: [energy, stretch, strict-floats, matlab-engine, blocked_float_path]
problem_type: parity
resolution_type: diagnosis
---

# Crop Energy stretch float isolation (E11–E20)

## Problem

True zero-tolerance stretch needs crop Energy bit-equal under
`prove-exact --stage energy --strict-floats`. Phase 1 ADR 0011 allclose is
already CLOSED and is **not** this bar. After MATLAB owned the per-chunk
filter + `interp3` + scale-min body, crop Energy was still not bit-identical.

## Evidence

- v1 dest `crop_M_stretch_engine_v1` (MATLAB `energy_filter_V200`, Python
  `interp3`): 2,623,250 / 4,194,304 bit-identical (62.5%).
- v2 dest `crop_M_stretch_engine_v2` (`stretch_energy_chunk_v202`):
  3,786,847 / 4,194,304 (90.3%); 407,457 mismatches; 0 scale mismatches;
  max abs delta `1e-10`; ULP p50=3, p90=9.
- Proof: `workspace/runs/oracle_180709_E/crop_M_stretch_engine_v2/03_Analysis/exact_proof_energy.json`
- Status: `stretch_status.json` on that dest is `blocked_float_path`.

Ruled out on cheap fixtures (engine skip = `incomplete_infra`, not fail):

- E12: py37 `npy` → list → `matlab.double` roundtrip is bit-identical.
- E13: linspace mesh, Inf `interp3`, and tiny chunk-vs-full bit-match.
- E14: whole-crop MATLAB `get_energy_V202` is octave-chunked (726 chunks on
  octave 2 of 6). Aborted as `incomplete_infra`. Not a cheap probe.
- One production chunk (mismatch ZYX `(13, 0, 0)`, winner scale 43, octave 2,
  chunk 0, write window ZYX `[0:21, 0:51, 0:51]`): octave-owned and the
  mismatch voxel are **re-run == v2 and both ≠ oracle**. Full write-window
  re-run ≠ v2 is min-merge from other octaves, not packaging. Scratch:
  `workspace/scratch/stretch_one_production_chunk.json`.
- Lattice/params vs original MATLAB `get_energy_V202`: octave-**index** 2 is
  Python 75 vs MATLAB-formula 726, but that is `unique()` id labeling.
  **rf-matched lattices are identical** (both 821 chunks). Core params match
  (`max_voxels=6000`, ratios, `scales_per_octave=6`, microns raw, radii/PSF
  allclose after YXZ align). Residual is helper **body** vs original MATLAB
  chunk math on the same lattice. Scratch:
  `workspace/scratch/stretch_lattice_params_isolation.json`.

## Root Cause

Named leftover after E13, one-chunk, and lattice/params: crop-scale FFT/filter /
`interp3` helper **body** vs the original MATLAB batch `get_energy_V202` on the
**same** rf-matched lattice (not merge/packaging, not a 75-vs-726 lattice bug).
The exact ULP source is **not** closed. Not an Energy unlock.

## Solution

Record **`blocked_float_path`**. Do not emit an Energy unlock. Do not treat
90.3% or default allclose as stretch success. Do not relaunch v2. Do not
overwrite `canonical_full_v18` or `crop_M_exact_v3`. U5/U6 stay gated.

```powershell
slavv parity inspect-proof --path workspace\runs\oracle_180709_E\crop_M_stretch_engine_v2\03_Analysis\exact_proof_energy.json
```

`(512, 64, 512)` vs oracle `(64, 512, 512)` is `incomplete_infra` before ULP
(`classify_stretch_energy_orientation` in Energy compare).

## Verification

- E12/E13/E17/E18/E19 unit tests under `tests/unit/pipeline/energy/` and
  `tests/unit/parity/` (engine tests skip without py37 + R2019a).
- One-chunk mapping/interpret: `tests/unit/pipeline/energy/test_stretch_one_production_chunk.py`.
  Operator: `scripts/stretch_one_production_chunk.py`.
- Lattice/params: `tests/unit/pipeline/energy/test_stretch_lattice_params_isolation.py`.
  Operator: `scripts/stretch_lattice_params_isolation.py`.
- E11 prove already ran on v2; do not re-run the writer to “verify” this note.

## Follow-Up

Helper **body** vs original MATLAB chunk math remains (`blocked_float_path`).
rf-matched lattices match (821=821); do not treat octave-index 75 vs 726 as a
production lattice bug. Do not relaunch v2. Do not start U5/U6. Live status
stays in dest `stretch_status.json`, not ONE TRUTH.
