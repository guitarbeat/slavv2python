---
title: Mac R2024b Parity Certification - Plan
type: feat
date: 2026-09-01
topic: mac-r2024b-parity-certification
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-brainstorm
execution: code
plan_enriched_by: ce-plan
product_contract_preservation: unchanged
---

# Mac R2024b Parity Certification - Plan

## Goal Capsule

- **Objective:** Achieve evaluated MATLAB↔Python exact-route parity on this Mac for crop `180709_E_crop_M` and canonical full `180709_E`, using R2024b oracles and dests on LoveSSD, verified by ADR 0011/0012 `prove-exact` on both volumes.
- **Product authority:** Windows Phase 1 closure on frozen claim roots (`canonical_full_v18`, `180709_E_full_v2`, etc.) remains the historical official record until an explicit post-pass decision updates ONE TRUTH. This plan owns the Mac R2024b lineage only; parity automation loops are supporting infrastructure, not separate product scope.
- **Open blockers:** Full-volume MATLAB Energy still running (`batch_260831-211917`, octave 2/6); crop Python SIGSTOP at 25/821; full oracle not yet promoted; neither evaluated prove-exact pass exists for r2024b dests.
- **Stop when:** AE1 and AE2 both satisfied with evaluated proof JSON on disk; frozen roots integrity confirmed; operator decides OQ1.

---

## Product Contract

### Summary

Python exact-route parity is certified on Windows claim roots under ONE TRUTH. This Mac must reproduce that certification independently: fresh R2024b MATLAB oracles, fresh run dests (`crop_M_r2024b`, `canonical_full_r2024b`), and evaluated `prove-exact` greens for Energy/Vertices (ADR 0011) and Edges/Network (ADR 0012 evaluated gate) — without overwriting protected Windows artifacts.

Automation (`parity_orchestrator.sh`, tier-3 CONT, auto-prove wiring) exists to advance the pipeline hands-off while MATLAB holds RAM. Success is evidence-backed parity on both volumes, not automation completeness alone.

### Problem Frame

| Pain | Impact |
|------|--------|
| Windows claim roots frozen on different host/toolchain | Mac R2024b lineage cannot cite Windows proofs as its own certification |
| 8 GB RAM serializes MATLAB then Python | Crop writer must SIGSTOP during full MATLAB Energy; long wall-clock wait |
| R2024b MATLAB FFT instability | Energy runs require `maxNumCompThreads(1)`; no meaningful parallel speedup |
| Oracle promotion gate | Full oracle promotion blocked until MATLAB batch vectors complete |
| Proof pairing discipline | Bootstrap oracle during crop write vs promoted oracle at prove time must not be conflated |

### Requirements

**Lineage and naming**

- R1. Use oracle ids `180709_E_crop_M_r2024b` and `180709_E_full_r2024b` only for Mac R2024b lineage.
- R2. Use run dests `workspace/runs/oracle_180709_E/crop_M_r2024b` and `workspace/runs/oracle_180709_E/canonical_full_r2024b`.
- R3. Do not overwrite or mutate frozen Windows oracles or claim roots listed in `PROTECTED_DEST_NAMES` / freeze JSON (`180709_E_crop_M_v2`, `180709_E_full_v2`, `crop_M_exact_v3`, `canonical_full_v18`).

**Oracle generation (MATLAB)**

- R4. Crop oracle is already promoted from `batch_260831-023332`; treat as complete unless promotion integrity fails audit.
- R5. Full oracle promotion runs only when `batch_260831-211917/vectors/` is complete — never promote segfault-aborted or partial batches.
- R6. Full MATLAB Energy uses serialized threads (`maxNumCompThreads(1)`) on this 8 GB host for stability.

**Python writers**

- R7. Crop writer resumes from existing checkpoint after tier-3 CONT; do not `--force-rerun-from energy` on in-progress dest.
- R8. Full canonical writer launches only after full oracle promotion and crop evaluated prove-exact passes (sequential gate).
- R9. All exact-route processing uses Fortran-order `[Y, X, Z]` grid alignment per AGENTS.md mandate.

**Proof and certification bars**

- R10. Crop certification requires evaluated `prove-exact --stage all` (or edges + network with evaluated ADR 0012 gate) against `180709_E_crop_M_r2024b`.
- R11. Full certification requires the same evaluated pass against `180709_E_full_r2024b`.
- R12. Edges/Network pass uses ADR 0012 spatial bars (ownership-map, trace tolerance, strand/bifurcation multisets) — not strict-field `connections` equality.
- R13. Energy/Vertices pass uses ADR 0011 (discrete equality + `np.allclose` on floats).
- R14. Cite proofs via `slavv parity inspect-proof --path <json> --require-evaluated`.

**Automation (supporting)**

- R15. Single orchestrator (`parity_orchestrator.sh`) owns CONT, prove, and promote wiring; legacy watchers stay retired.
- R16. CONT is SIGCONT only; `do_not_resume_exact_run_on_cont` remains true.
- R17. Tier-3 CONT fires when full MATLAB exits; crop resumes same orchestrator tick.
- R18. Auto-invoke `PROVE_CROP_R2024B_WHEN_READY.sh` and `PROMOTE_AND_PROVE_FULL_R2024B_WHEN_READY.sh` idempotently when gates open.

### Flows

**F1. Full MATLAB oracle path (in progress)**

1. MATLAB `vectorize_180709_E_full.m` completes all six Energy octaves → vectors in batch dir.
2. Orchestrator or operator promotes `180709_E_full_r2024b` via existing WHEN_READY script.
3. Preflight `canonical_full_r2024b` from certified crop lineage or prior seed as documented in scratch scripts.

**F2. Crop Python → prove (blocked on RAM)**

1. Tier-3 SIGCONT crop writer when MATLAB PID exits.
2. Crop completes Energy → Vertices → Edges → Network (4 stage checkpoints).
3. Evaluated crop prove-exact vs promoted crop oracle.

**F3. Full Python → prove (gated)**

1. Launch canonical full writer on `canonical_full_r2024b` after F1 + F2 gates pass.
2. Full pipeline completes on LoveSSD.
3. Evaluated full prove-exact vs promoted full oracle.

### Acceptance Examples

**AE1. Crop evaluated pass**

Given `crop_M_r2024b` checkpoints from current code and oracle `180709_E_crop_M_r2024b`, when `slavv parity prove-exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_r2024b --oracle-root workspace/oracles/180709_E_crop_M_r2024b --stage all` runs, then edges and network proofs have `adr0012_evaluated: true` and `passed: true`, and energy/vertices proofs pass ADR 0011.

**AE2. Full evaluated pass**

Given promoted `180709_E_full_r2024b` and completed `canonical_full_r2024b`, when full prove-exact runs with `--require-evaluated`, then both edges and network ADR 0012 gates pass with evaluated flag true.

**AE3. Frozen root integrity**

Given any automation or manual operator action during the Mac run, when complete, then byte hashes and contents of `canonical_full_v18` and Windows oracle roots are unchanged from pre-run baseline.

**AE4. No premature goal closure**

Given crop or full writer still in progress, when automation status is checked, then `do_not_mark_goal_complete` remains true and Cursor goal is not marked complete.

### Key Decisions

- KTD1. Mac R2024b is a **parallel certification lineage** until both volumes pass; ONE TRUTH update is a separate explicit operator decision after AE1+AE2.
- KTD2. **Sequential RAM policy** on 8 GB: full MATLAB first, crop SIGSTOP, tier-3 CONT on MATLAB exit — not tier-2 mid-MATLAB overlap.
- KTD3. **Serial compute** for both MATLAB (`maxNumCompThreads(1)`) and crop Python (`n_jobs=1`) on this host; do not chase parallel speedups that risk FFT segfault or RAM OOM.
- KTD4. **Automation-first advance** via orchestrator; agents poll on idle contract (≥900s) during long-wait, not every goal ping.
- KTD5. Crop writer bootstrap oracle vs promoted oracle mismatch at preflight is expected during write; prove uses promoted oracle only.

### Non-Goals

- NG1. Overwriting or relabeling Windows Phase 1 claim roots as Mac results.
- NG2. True zero-tolerance stretch (bit-identical Energy floats) — separate stretch plan.
- NG3. Strict-field Edges `connections` count/order equality as ship gate.
- NG4. Re-opening Windows Phase 1 closure narrative or ADR 0013 ranking residual investigation.
- NG5. Streamlit GUI as canonical overnight parity watcher.

### Outstanding Questions

- OQ1. After AE1+AE2 pass, does operator want ONE TRUTH updated to name Mac r2024b claim roots, or maintain dual lineage documentation?
- OQ2. If full prove-exact passes but crop regresses on re-run, is crop re-prove required before declaring Mac lineage complete?

### How This Work Fits Together

| Related work | Relationship |
|--------------|--------------|
| `docs/plans/2026-09-01-001-feat-parity-automation-loops-plan.md` | Supporting — orchestrator/CONT/idle already largely implemented |
| `docs/plans/phase-1-exact-route-spec.md` | Inherited bars — same ADR 0011/0012 certification semantics |
| ONE TRUTH / `canonical_full_v18` | Historical Windows closure — not overwritten by Mac run |
| ce-optimize `parity-loop-tuning` | Tuning automation latency; does not substitute for prove-exact pass |

---

## Planning Contract

**Product Contract preservation:** unchanged — enrichment adds execution units only.

**Approach:** Execute the Mac R2024b lineage as a gated operator pipeline on LoveSSD. Most units are long-running jobs orchestrated by `workspace/scratch/parity_orchestrator.sh` and idempotent WHEN_READY scripts — not new package code. Code changes are limited to fixing automation regressions discovered during the run; parity-sensitive pipeline logic is frozen unless prove-exact fails.

**Sequencing:** U1 (MATLAB wait) → U2 (full oracle promote) after vectors; U3 after tier-3 CONT; U4 after U2 + crop prove; U5 audit after U4.

**Patterns to follow:**

- `docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md` — prove-exact three-root pairing
- `docs/plans/phase-1-exact-route-spec.md` — ADR 0011/0012 bars
- `slavv parity inspect-proof --require-evaluated` — only evaluated ADR 0012 counts

**Deferred:** OQ1 ONE TRUTH update policy; OQ2 crop re-prove on regression.

---

## Implementation Units

### U1. Long-wait — MATLAB full Energy completion

**Covers:** R5, R6, F1 step 1, KTD2, KTD4

**Owner surface:** `workspace/scratch/matlab/vectorize_180709_E_full.m`, batch `workspace/scratch/matlab_full_batches/batch_260831-211917`

**Work:**

- Keep orchestrator alive (`parity_orchestrator.sh detach` if dead; SIGTERM orchestrator loop only, never MATLAB/crop).
- Monitor via `bash workspace/scratch/parity_orchestrator.sh status`, `workspace/scratch/GOAL_PIPELINE_STATUS.md`, `workspace/scratch/matlab_full_batches/matlab_r2024b_full.log`.
- Confirm `vectors/` has ≥5 files before U2.

**Exit:** `batch_260831-211917/vectors/` complete; MATLAB exited cleanly.

---

### U2. Promote full R2024b oracle

**Covers:** R5, R1, F1 steps 2–3

**Owner surface:** `workspace/scratch/PROMOTE_AND_PROVE_FULL_R2024B_WHEN_READY.sh`

**Work:**

- Orchestrator or manual script run after vectors complete.
- Verify `workspace/oracles/180709_E_full_r2024b/99_Metadata/oracle_manifest.json` and `slavv parity inspect-experiment-root`.

**Exit:** `workspace/scratch/FULL_ORACLE_PROMOTED.flag` set.

---

### U3. Crop resume, complete, evaluated prove

**Covers:** R7, R10, R12–R14, R16–R18, F2, AE1

**Owner surface:** `workspace/runs/oracle_180709_E/crop_M_r2024b`, `workspace/scratch/PROVE_CROP_R2024B_WHEN_READY.sh`

**Work:**

- Tier-3 CONT via orchestrator (`TIER3 matlab_dead_immediate` in log).
- Crop completes four checkpoints; auto or manual prove via WHEN_READY script.
- Fallback: `uv run slavv parity prove-exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_r2024b --oracle-root workspace/oracles/180709_E_crop_M_r2024b --source-run-root workspace/runs/oracle_180709_E/crop_M_r2024b --stage all`

**Exit:** `CROP_PROVE_DONE.flag`; edges + network `passed: true`, `adr0012_evaluated: true`.

---

### U4. Canonical full Python writer and evaluated prove

**Covers:** R8, R11–R14, F3, AE2

**Owner surface:** `workspace/runs/oracle_180709_E/canonical_full_r2024b`, `workspace/scratch/PROMOTE_AND_PROVE_FULL_R2024B_WHEN_READY.sh`

**Preconditions:** U2 + U3 complete.

**Work:**

- Script launches writer with `--n-jobs 1`; monitor with `slavv monitor --once`.
- Full prove-exact vs `180709_E_full_r2024b`; set `FULL_PROVE_DONE.flag`.

**Exit:** Evaluated edges + network proofs pass on canonical dest.

---

### U5. Certification audit and goal closure

**Covers:** R3, AE3, AE4, OQ1

**Work:**

- Audit R1–R14, AE1–AE4 from disk evidence.
- Confirm frozen roots untouched.
- Resolve OQ1; `UpdateGoal` complete only after D1–D4.

**Exit:** Both volumes certified; proof paths documented.

---

## Verification Contract

| Check | Command / evidence | Pass criterion |
|-------|-------------------|----------------|
| V1 Orchestrator alive | `parity_orchestrator.sh status` | Running, tick fresh |
| V2 Experiment root | `slavv parity inspect-experiment-root` | No stubs; oracles present |
| V3 Full vectors | `batch_260831-211917/vectors/` | ≥5 files |
| V4 Crop checkpoints | Four files under crop dest checkpoints | All present |
| V5 Crop evaluated prove | `inspect-proof --require-evaluated` edges + network | Both pass |
| V6 Full evaluated prove | Same on canonical dest | Both pass |
| V7 Frozen roots | Protected dests unchanged | AE3 |
| V8 Tier-3 CONT | `parity_orchestrator.log` after MATLAB exit | TIER3 + SIGCONT same tick |

---

## Definition of Done

- D1. Crop evaluated prove-exact passes (AE1).
- D2. Full evaluated prove-exact passes (AE2).
- D3. Protected Windows roots untouched (AE3).
- D4. Proofs cited via `inspect-proof --require-evaluated`.
- D5. OQ1 resolved by operator.
- D6. Goal complete only after D1–D4 verified from disk.
