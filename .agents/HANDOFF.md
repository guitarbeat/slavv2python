# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-06-22

This is the single successor brief for the current exact-route effort. Do not use
dated agent passovers, PID snapshots, or parallel-work checklists as current
status.

## Canonical records

| Need | Source of truth |
|---|---|
| Active work and checkboxes | [docs/TODO.md](../docs/TODO.md) |
| Verified run status, proof evidence, and blockers | [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) |
| Phase 1 requirements | [phase-1-exact-route-spec.md](../docs/plans/phase-1-exact-route-spec.md) |
| Run commands and evidence format | [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md), [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md) |
| Repository and parity guardrails | [AGENTS.md](../AGENTS.md) |

## Current decision point

Phase 1 remains blocked at the crop-harness Energy gate.

- The `crop_M_exact` Energy writer completed on 2026-06-22 using lattice
  `6000`, `n_jobs=1`; `best_energy.npy` and `best_scale.npy` exist.
- The subsequent strict `prove-exact --stage energy` failed. The verified report
  has 19,412 scale-winner mismatches and 3,823,893 Energy-value mismatches.
- Crop Vertices, Edges, and Network must not be refreshed until the Energy proof
  is strict-zero. The canonical `180709_E` certification run remains paused for
  the same reason.

The first work is not another blind rerun. Reproduce and explain representative
scale-winner disagreements, then isolate bit-identical float64 drift for voxels
whose winning scale already agrees. The evidence and probe ledger are recorded
in the findings document.

## Operating sequence

1. Before any writer action, check `slavv jobs list` and
   `slavv parity status-exact-run --run-dir workspace/runs/oracle_180709_E/crop_M_exact`.
   Never introduce a second writer on that root.
2. Make a MATLAB-backed diagnosis for a scale-winner mismatch and add the
   minimal regression only after the discrepancy is reproduced.
3. Rerun crop Energy only for a specific, tested hypothesis; capture proof
   evidence with `prove-exact --stage energy`.
4. After crop Energy is strict-zero, refresh Crop Vertices → Edges → Network and
   run `prove-exact-sequence`.
5. Only after the crop sequence passes, resume the full canonical sequence.

## Relevant uncommitted scope

The worktree includes uncommitted parity lifecycle, probe/harness, linspace
mesh, Energy, test, and documentation changes. Review `git status` and the
targeted tests before committing; do not assume all local changes belong to one
atomic fix.

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above. Their historical outcome is preserved: an earlier job died, the
replacement Energy writer completed, and the strict Energy proof failed.
