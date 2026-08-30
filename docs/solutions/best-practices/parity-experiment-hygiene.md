---
title: Parity experiment hygiene
module: analytics/parity
tags: [experiment, oracle, run-root, scratch, notes]
problem_type: workflow
resolution_type: runbook
---

# Parity experiment hygiene

## In short

Compare like with like (raw to raw, final to final). Do not start a second
writer on a live folder. A proof JSON can sit in one folder and still belong to
another — cite through `inspect-proof`. Cheap tests before overnight writers.

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
| Claim run | `canonical_full_v18` | Live claim (Phase 1 closed) |
| Crop guard | `crop_M_exact_v3` candidates | Regression, not unevaluated proof JSON |
| Stretch dest | `crop_M_stretch_engine_v2` | Crop Energy leftover; do not overwrite |
| Archived proofs | `workspace/reports/phase1_volume_archive/` | `v4` Energy/Vertices JSON; `v16` Network FAIL record; other audit proofs. Volumes removed 2026-08-18 — do not resurrect |
| Failed shells | `v9`, `v11`, `v12`, `v13`, `v14` | Already gone |
| Contaminated | `canonical_full_v17` | Do not claim; deleted 2026-08-13 |
| Duplicate oracle | `180709_E_crop_M_v2_old` | Ignore (points at v2 batch) |
| Labeled scratch | `matlab_edge_dump/README.md` | Dump mats deleted 2026-08-18; README remains |

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

## Testable parity experiments portfolio (E1–E10)

Ranked falsifiers from
`docs/plans/2026-08-14-001-feat-testable-parity-experiments-plan.md`.
**R2 non-claim for every row:** results are not Phase 1 Certification unless the
named surface is an evaluated ADR 0012 proof (`adr0012_evaluated: true`) cited via
`slavv parity inspect-proof`.

### Residual ladder (F2) — no full Edges writer for ranking / pair-set

| Ei | Tier | Entrypoint | Notes |
|---|---|---|---|
| E1 | `synthetic/unit` | `pytest tests/unit/pipeline/test_watershed_energy_map_sort_experiments.py -k e1` | Claimed-map vs original-field ranking |
| E2 | `synthetic/unit` | same file `-k e2` | Degree-excess keeps earlier row under tie |
| E3 | `crop` | same file `-k e3` | Raw↔raw only; skip/blocked if crop dumps absent |
| E4 | `full no-writer` | `python scripts/persist_full_edges_selection.py` (+ unit tests in `test_full_no_writer_reselection_experiment.py`) | Claimed-map hub adapter; default `canonical_full_v16`; **never** starts a writer |
| E5 | `crop` (escalate only if needed) | `python scripts/network_matlab_edge_isolation.py` (+ `tests/unit/parity/test_network_matlab_edge_isolation_experiment.py`) | MATLAB edges → Python Network; isolation ≠ Phase 1 closure. Crop may **fail** multiset (hypothesis falsified on that surface) — still not a Network rewrite mandate; escalate `--oracle-root` to full only if crop cannot speak to the claim. |

Forbidden for E1–E4 ranking / pair-set questions: `resume-exact-run --force-rerun-from edges`
or any full Edges writer. Writer CLI does **not** yet call `require_cheap_loop`
(follow-up); enforce via this ladder + E10 unit gate.

### Audit honesty track (F3) after report refresh

| Ei | Tier | Entrypoint |
|---|---|---|
| E6 | unit | `tests/unit/test_parity_experiment_portfolio_docs.py` + `tests/unit/test_synthetic_validator.py` / `test_compile_audit_report.py` (skip if workspace audit tools absent) |
| E7 | unit | live `matrix_coverage` 13/13 in synthetic_validator tests |
| E8 | unit | no static-only `GENUINE_BEHAVIORAL_DIVERGENCE` (same validator tests) |
| E9 | unit | `tests/unit/test_parity_module_map.py` shared ParityModuleMap seam |
| E10 | unit | `tests/unit/parity/test_parity_experiment_module.py` — RANKING / ARTIFACT_CLASS / PAIR_SET refuse `FULL_WRITER` |

No portfolio CLI in v1 (`slavv parity experiment run E{n}` is deferred).

## Verification

`slavv parity ensure-oracle-artifacts --oracle-root workspace/oracles/180709_E_full_v2 --stage all --no-repair` (and crop v2) both `passed: true`.

## Follow-Up

Delete failed shells / `crop_M_v2_old` only with an explicit user “delete”
instruction. Rename mislabeled scratch dumps in place rather than copying.
