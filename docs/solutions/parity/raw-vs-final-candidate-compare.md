---
title: Compare raw watershed candidates to raw, finals to finals
module: pipeline/edges
tags: [edges, watershed, candidates, sort_edges, residual, experiment]
problem_type: parity
resolution_type: diagnosis
---

# Compare raw watershed candidates to raw, finals to finals

## Problem

Network ADR 0012 fails by one strand because degree-excess keeps `(26444, 38584)`
and drops MATLAB’s `(34897, 38584)`. Weeks of join-emission / tie-scan work
assumed MATLAB never emits the extra pair.

## Evidence

- MATLAB oracle `edges_*.mat` / normalized `edges.pkl` = **69,500 finals**.
- Python `candidates.pkl` / MATLAB `raw_full_candidates.mat` = **84,650 raw**.
- Undirected pair-set compare of those two raw dumps: **intersection 84,650,
  only_py=0, only_mat=0**. Extra pair is in **both**.
- After MATLAB `sort_edges` (`global_presort_candidates.mat`): extra at row
  79596, oracle at 32325. MATLAB extra raw `max=0.0`, oracle `max=−0.239`
  (claimed `energy_map`). Python stored traces: extra `−9.24`, oracle `−7.73`
  (original field). Spatial traces are the same 9 voxels.
- Cheap reproduction: `tests/unit/pipeline/test_watershed_energy_map_sort_experiments.py`.

## Root Cause

`edges2vertices` in the promoted oracle is **post-cleanup**. Treating it as
watershed emission invents a “MATLAB never joins” story. Live MATLAB join is
`min(current_strel_energies)` (`get_edges_by_watershed.m` L530). `sort_edges`
uses `max` of the map that received L445 penalized claim writes (L846).

## Solution

1. Compare raw↔raw and final↔final only.
2. Sample `claim_map.energy_map` when assembling watershed energy traces.
3. Run MATLAB `sort_edges` (raw `max`, ascend) after crop and before resampled
   `clean_edge_pairs`.
4. Do not use join-partner `find(...,'last')` or endpoint tertiary keys as the
   ship-gate change.

## Verification

- Crop pair sets 19,225/19,225; crop re-selection 15,511/15,511 on existing
  `crop_M_exact_v3` candidates after the sort change.
- Unit experiments above all pass.
- Full Edges regen + evaluated `prove-exact --stage edges` and `--stage network`
  on a **new** claim root still required (old checkpoints have original-field
  traces). Do not claim `canonical_full_v17`.

## Follow-Up

Keep crop as a regression guard. New successor run: Edges→Network only from
certified Energy/Vertices.
