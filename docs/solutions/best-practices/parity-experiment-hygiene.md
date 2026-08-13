---
title: Parity experiment hygiene
module: analytics/parity
tags: [experiment, oracle, run-root, scratch, notes]
problem_type: workflow
resolution_type: runbook
---

# Parity experiment hygiene

## Problem

Parity agents launched full-volume writers, compared the wrong MATLAB artifact
class, cited proof JSON that belonged to another run folder, and treated
contaminated roots (`canonical_full_v17`, `180709_E_crop_M_v2_old`) as live.

## Evidence

Workspace audit 2026-08-13: live oracles pass
`ensure-oracle-artifacts --no-repair`; `v16` proofs match ONE TRUTH; `v17`
has a stale “energy running” snapshot, no local E/V checkpoints, and a job
that “succeeded” on permission denied; `crop_M_exact_v3/exact_proof_network.json`
has `dest_run_root=crop_M_exact`; `raw_watershed_candidates_canonical.mat` is
crop-shaped `(64,256,256)`.

## Root Cause

No shared rule for (1) cheap vs expensive experiments, (2) which disk objects
are claim / audit / junk, (3) pairing a proof file to its folder.

## Solution

**Enforced by** `slavv_python.analytics.parity.experiments` (`require_cheap_loop`,
`compare_same_class_pair_sets`, `load_edge_artifact`, `load_proof_record`).
Cite proofs with `slavv parity inspect-proof --path <json>`. Load pkl/mat
connections through `load_edge_artifact` — do not add another `_pair_set`.

**Loop:** unit/synthetic → crop pair-set → no-writer re-selection → full writer
only if the cheap layer cannot falsify the hypothesis.

**Artifact class:** raw Candidate Set ≠ Edge Set. See
[raw-vs-final-candidate-compare.md](../parity/raw-vs-final-candidate-compare.md).

**Data classes (do not delete claim/audit without explicit approval):**

| Class | Examples | Use |
|---|---|---|
| Live oracle | `180709_E_full_v2`, `180709_E_crop_M_v2` | Proofs only |
| Claim run | `canonical_full_v16` | Status numbers |
| Lineage seed | `canonical_full_v4` Energy/Vertices, `v8` energy `.npy` | Seed successors |
| Crop guard | `crop_M_exact_v3` candidates | Regression, not unevaluated proof JSON |
| Audit history | `v5`–`v15` completed writers | Keep |
| Failed shells | `v9`, `v11`, `v12`, `v13`, `v14` | Ignore; empty or stale “running” |
| Contaminated | `canonical_full_v17` | Do not claim |
| Duplicate oracle | `180709_E_crop_M_v2_old` | Ignore (points at v2 batch) |
| Labeled scratch | `matlab_edge_dump/raw_full_candidates.mat` | Raw MATLAB compare |

**Proof pairing:** `dest_run_root` must equal the directory you opened.
`adr0012_evaluated: true` required for Edges/Network closure.

**Successor seed:** copy Energy/Vertices/params/provenance onto a **new** dest.
Exclude Edges/Network checkpoints, proof JSON, and old `writer_lease` /
`parity_job.*`. Pass `--dataset-root` (`00_Refs` may be empty). On Windows
start the writer with `Start-Process resume-exact-run`, not
`launch-exact-run`. Do not block on `slavv jobs list`.

**Notes:** status only in ONE TRUTH; commands in HANDOFF; checkboxes in TODO;
verified runbooks in `docs/solutions`. Promote a scratch probe to a unit test
or a solution note; do not add another unlabeled `probe_*.py`.

## Verification

`slavv parity ensure-oracle-artifacts --oracle-root workspace/oracles/180709_E_full_v2 --stage all --no-repair` (and crop v2) both `passed: true`.

## Follow-Up

Delete failed shells / `crop_M_v2_old` only with an explicit user “delete”
instruction. Rename mislabeled scratch dumps in place rather than copying.
