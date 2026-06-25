---
title: Curated MATLAB vertices store a rank-ramp energy; source true energies from raw vertices.mat
module: analytics/parity
tags: [vertices, energies, matlab-loader, curated, oracle, parity]
problem_type: integration
resolution_type: code_fix
---

# Curated MATLAB vertices store a rank-ramp energy

## Problem
After vertex positions+scales matched MATLAB exactly, `prove-exact --stage
vertices` still failed on `vertices.energies` — every value differed by a huge
amount (max |Δ| ≈ 6.5e4), not a tolerance/ULP issue.

## Evidence
- Loaded `energies` (via the normalized loader) were `-65532.2, -65527.4, …` —
  perfectly evenly spaced (step 4.7816 ≈ 65535/13706), i.e. a rank ramp to the
  uint16 range, not physical energy.
- The raw `vectors/vertices_*.mat` `vertex_energies` were `-364.45, -363.41, …`
  and matched Python's computed energies exactly.
- `find_matlab_vector_paths` prefers `curated_vertices*.mat` over `vertices*.mat`;
  the curated artifact's `vertex_energies` are overwritten by MATLAB curation with
  the rank ramp (its `edge_curation`/curation step is a display normalization).

## Root Cause
The proof compared Python's true vertex energies against MATLAB's **curated**
(rank-remapped) energies. Positions/scales correctly come from the curated
(post-choose) artifact, but its energies are a non-physical display value.

## Solution
`matlab_vector_loader.py`: for `curated_vertices*.mat`, recover true energies from
the raw `vertices*.mat` sibling, matched by **exact integer voxel subscript**
(positions/scales still from curated). Falls back to curated energies if the raw
sibling is absent or a curated vertex is missing from it.

## Verification
`prove-exact --stage vertices` vs `180709_E_crop_M_v2` → PASS (exit 0); energies
certify under the ADR 0011 `np.allclose` gate. Unit tests
`test_vertex_energy_backfill.py` cover recovery + the no-raw-sibling fallback.
Commit `4a78b870`.

## Follow-Up
Watch for the same curation remap on other curated MATLAB artifacts (e.g.
`curated_edges*.mat`); compare against the raw sibling when a field looks like an
evenly-spaced ramp.
