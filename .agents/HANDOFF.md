# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-06-24

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

- **Current evidence is stale.** `inspect-energy-evidence` reports
  `checkpoint_energy.pkl` missing and `energy_status=failed` on
  `crop_M_exact`. Historical `exact_proof*.json` artifacts are diagnostic
  history only until a completed Energy checkpoint exists again.
- The last Energy attempt (`2026-06-23`) died with heap-fragmentation OOM while
  allocating small `complex128` buffers inside `_ifftn_matlab_symmetric` after
  ~5h of preprocessing/chunk work. Do not trust pre-2026-06-23 proof counts
  until a fresh writer completes.
- Prior strict proof (when checkpoint existed) reported 19,412 scale-winner
  mismatches and 3,823,893 float64 value mismatches. Probe triage now
  distinguishes `winner_scale_disagreement` from `matching_winner_ulp_drift`;
  cross-octave reduction implicates `python_stored_state_path` for at least
  some voxels where per-octave probes agree with MATLAB but stored winners do
  not.
- Crop Vertices, Edges, and Network must not be refreshed until Energy proof is
  strict-zero. Canonical `180709_E` remains paused.

**Energy writer completed (2026-06-24):** job `75188cc2`, 821/821 chunks,
`inspect-energy-evidence` valid. Fresh `prove-exact --stage energy` **FAIL**:
3,810,130 energy ULP mismatches; **31** scale-index mismatches (was 19,412).
First scale gap: `(40,83,116)` MATLAB 13 vs Python 12.

Next: triage the 31 scale winners (`energy_probe_requests.json` refreshed) →
fix root cause → rerun Energy only if needed → strict-zero before downstream.

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

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above. Their historical outcome is preserved: an earlier job died, the
replacement Energy writer completed, and the strict Energy proof failed.
