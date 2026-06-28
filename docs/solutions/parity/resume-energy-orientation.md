---
title: Resume/Init Input Reorientation Mismatch (Energy Axis Order)
module: analytics/parity
tags: [orientation, energy, resume, exact-route, certification, parity]
problem_type: parity
resolution_type: code_fix
---

# Resume/Init Input Reorientation Mismatch (Energy Axis Order)

## Problem
The first full-volume canonical certification run (`180709_E`, all four stages)
completed, but `prove-exact-sequence` crashed at the energy gate:

```
ValueError: Energy and scale arrays must share shape
```

Python energy was `(512, 64, 512)` (`[Y, Z, X]`) while the oracle was
`(64, 512, 512)` (`[Z, Y, X]`). The crop tier-2 run certified fine — only the
full volume failed.

## Evidence
- `checkpoint_energy.pkl` shape `(512, 64, 512)`; oracle normalized energy
  `(64, 512, 512)`; input TIF `(64, 512, 512)`.
- `(512,64,512) == np.transpose((64,512,512), [2,0,1])` — the stored
  `energy_axis_permutation`/`input_axis_permutation` `[2,0,1]` was applied to the
  *volume*.
- Crop provenance: `input_axis_permutation = None` (so it never triggered the
  bug). Full provenance: `[2,0,1]`.

## Root Cause
The **init** path (`cli_runs.handle_init_exact_run`) loads the TIF with
`load_tiff_volume(...)` (`transpose_to_yxz=True` → `[Y,X,Z]`) and reorients with
`_reorient_exact_input_volume`, which **searches** the permutation that matches
the oracle axis order (records `[2,0,1]`).

The **resume** path (`resume.resume_exact_run`) instead loaded
`load_tiff_volume(..., transpose_to_yxz=False)` (raw `[Z,Y,X]`) and then
**replayed** the stored permutation via `_reorient_volume_from_provenance`. For a
volume whose stored permutation was `[2,0,1]` (derived against the `[Y,X,Z]`
load), replaying it on the already-`[Z,Y,X]` raw load **double-permuted** the
volume to `[Y,Z,X]` `(512,64,512)`. The anisotropic matched filter then paired
the wrong physical voxel sizes per axis (Z=1.997 vs Y/X=0.916 µm), so the energy
values were genuinely wrong — not merely transposed. The crop dodged it because
its permutation was `None` (no-op replay).

## Solution
Make resume reorient **identically to init** — self-correcting, so it can never
desync from the orientation init computed (`slavv_python/analytics/parity/resume.py`):

```python
image = load_tiff_volume(str(input_file))                      # transpose_to_yxz=True
image, _oracle_size, _perm = _reorient_exact_input_volume(image, oracle_surface)
```

This drops `transpose_to_yxz=False` and the provenance-replay helper
(`_reorient_volume_from_provenance` removed) in favor of the search-based
`_reorient_exact_input_volume`, which always matches the oracle axis order
regardless of loader heuristics or stale provenance.

## Verification
- Replaying resume's load+reorient on the full volume now yields `(64, 512, 512)`
  (was `(512, 64, 512)`).
- Regression test `tests/unit/analytics/parity/test_preflight_resume.py::
  test_resume_reorients_input_to_oracle_axis_order` mocks `load_tiff_volume` to
  honor `transpose_to_yxz`, so the pre-fix double-permute would hand the pipeline
  `(8,6,4)` instead of the oracle's `(4,8,6)` and fail.
- Full suite: 573 passed; ruff + mypy clean.

## Follow-Up
- The full-volume canonical sequence is being recomputed with the fix
  (`canonical_full_v3`, `n_jobs=6`) — energy values must be regenerated since the
  bad orientation corrupted them.
- Lesson: symmetric test volumes (crop Y=X) can mask axis-order bugs; the
  certification volume must have distinct Z/Y/X to exercise reorientation.
