---
title: "Phase 1 exact-route certification on 180709_E"
type: spec
status: active
date: 2026-05-28
topic: 180709-native-parity-ship-confidence
merged_from:
  - historical brainstorm requirements
  - historical dated implementation plan
---

# Phase 1 exact-route certification on 180709_E

**Authoritative spec** for program intent, requirements, and implementation. **Tasks:** [TODO.md](../TODO.md) (checkboxes only). **Live status:** [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md) (runs, proofs, blockers).

---

# Part 1 — Program intent & requirements

## Summary

Establish **program ship confidence** by first certifying the exact parity route on canonical volume **180709_E**, then certifying the paper profile on a defined volume and oracle, using **native** runs (no oracle-injected vertices), **strict zero** missing/extra on every gated stage, and **sequential** certification through energy → vertices → edges → network.

**Definitions**

- **Program ship confidence:** Both the exact parity route and the paper profile have passed R1–R3 on their defined volumes. Phase 1 alone does not achieve program ship confidence.
- **Phase 1 milestone (exact-route certification):** Sequential `prove-exact` passes for energy, vertices, edges, and network on full `180709_E` under the exact parity route. May be communicated as “MATLAB-equivalent exact route on 180709_E”; do not claim program ship confidence or full public-pipeline equivalence until phase 2 passes.

## Problem frame

The public workflow is verified end-to-end, but the **exact parity track** on edges remains at roughly **88.7%** pair match against the preserved MATLAB oracle (v29 baseline: 135 missing, 371 extra pairs). That gap blocks a credible claim that native Python output matches MATLAB for real extraction work—not just “close enough” for demos.

Ship confidence requires a **reproducible, strict** certification on a **single canonical volume** rather than informal match-rate milestones. Recent work (e.g. strel linear-index tie-break, manager consolidation) is necessary but not sufficient until `prove-exact` reports **zero missing and zero extra** at each stage in order.

## Actors

- **A1. Maintainer / developer:** Runs parity experiments, interprets `prove-exact` reports, and decides when ship confidence is achieved.
- **A2. Public SLAVV user:** Runs `slavv run` with the **paper** profile expecting MATLAB-faithful results without running the parity harness (`matlab_compat` is out of scope unless reprioritized).
- **A3. MATLAB oracle:** Preserved truth vectors under `workspace/oracles/` derived from neurovasc-db (`180709_E` vectorization batch).

## Key flows

- **F1. Sequential exact-route certification (phase 1)**
  - **Trigger:** Maintainer initiates or resumes `init-exact-run` on full `180709_E` with oracle `180709_E_batch_190910-103039`.
  - **Actors:** A1, A3
  - **Steps:** Bootstrap dataset and oracle → run native pipeline through network → run sequential `prove-exact` gates (energy, then vertices, edges, network) → each must show zero missing/extra.
  - **Outcome:** Phase 1 exact-route certification is complete on `180709_E` for all four stages.
  - **Covered by:** R1–R6, R9

- **F2. Paper profile certification (phase 2)**
  - **Trigger:** Phase 1 complete; volume and oracle policy for paper profile resolved (see Outstanding questions).
  - **Actors:** A1, A2, A3
  - **Steps:** Define paper-profile volume and oracle → run public paper profile with **native** vertex discovery (R2) → apply same sequential `prove-exact` gates (R1, R3).
  - **Outcome:** Paper profile is certified at the same strict bar as exact route; program ship confidence is achieved.
  - **Covered by:** R1–R3, R2, R7, R8

## Requirements

**Certification bar**

- R1. Success means **`prove-exact` reports zero missing and zero extra** for the stage under test—strict set equality, not a percentage threshold.
- R2. Certification uses a **native** pipeline: Python discovers vertices; no oracle vertex checkpoint injection for the certified run.
- R3. **Sequential gating:** Energy must pass before vertices are judged; vertices before edges; edges before network. A failure at any stage blocks **certification** for **that workflow** until resolved.
- R4. **Canonical volume** for the first milestone is **`180709_E` full volume** (not center crop) for the exact parity route.

**Scope of stages and workflows**

- R5. The certified pipeline includes **energy, vertices, edges, and network**—not edges alone.
- R6. **Phase 1** delivers **exact-route certification** on full `180709_E`. Program ship confidence remains blocked until R7 is satisfied.
- R7. **Phase 2** delivers the same strict bar (R1–R3 and **R2**) for the **paper** profile on its defined volume and oracle.
- R8. Both workflows must eventually reach R1–R3; phase 2 does not start until phase 1 exact-route certification is complete unless explicitly reprioritized.

**Operational expectations**

- R9. Preserved MATLAB truth for the canonical exact-route milestone must live in **`workspace/oracles/180709_E_batch_190910-103039`** with one loadable artifact per gated stage. Current artifact readiness belongs in [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md), not this spec.
- R10. Disposable trial runs live under **`workspace/runs/`**; promoted summaries under **`workspace/reports/`** when warranted.
- R11. Incremental fixes (e.g. `strel_tiebreak_v30`) are **edges-stage experiments** under the sequential gate—they do not alone satisfy ship confidence.

**Change strategy**

- R12. Pursue the **shortest honest path** to **R1–R7**: incremental parity fixes tied to a failing `prove-exact` stage on the certified run. Shims require documented downgrade and do not satisfy R2.

## Acceptance examples

- **AE1. Energy gate blocks downstream** — Given energy `prove-exact` reports any missing/extra, vertices/edges/network are not certified.
- **AE2. Edges pass only with matching vertex surface** — Vertex mismatch propagates to edge pair missing/extra until vertices pass.
- **AE3. Phase 1 complete** — Sequential `prove-exact` passes for all four stages on the exact-route run; program ship confidence not claimed until phase 2.

## Success criteria

**Phase 1 (P0):** Exact-route native run on full `180709_E` passes sequential `prove-exact` with zero missing/extra on energy, vertices, edges, and network.

**Phase 2 (P1):** Paper profile passes the same strict bar; **program ship confidence** achieved.

## Priority

| Tier | Scope |
|------|--------|
| **P0** | Phase 1: R1–R6, R9, F1 |
| **P1** | Phase 2: R7–R8, F2 |
| **P2** | Deferred items below |

## Scope boundaries

**In scope:** `180709_E` certification program; oracle bootstrap; diagnostics after v29.

**Deferred:** All neurovasc-db volumes; CI merge-blocking parity; third-party reproduction as primary deliverable; open-ended refactors not tied to a failing `prove-exact` stage.

**Outside identity:** Replacing paper workflow with exact-route-only defaults for all users; claiming equivalence without passing `prove-exact` for that workflow.

## Key decisions

| Decision | Choice |
|----------|--------|
| Success bar | Strict zero missing/extra |
| Vertex source | Native discovery |
| Gating | Sequential by stage |
| Volume (milestone 1) | Full `180709_E` |
| Delivery order | Exact route first, then paper |

## Dependencies / assumptions

- Oracle and dataset are available locally; canonical oracle artifact readiness is verified against [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md) before proof.
- Windows certification runs may use `--energy-storage-format npy` when zarr is unreliable.

## Outstanding questions

1. **Paper profile volume** (phase 2): Full `180709_E`, `180709_EL`, or crop — and matching oracle?
2. **Paper vs exact parameter alignment** (phase 2): Oracle-derived exact params vs paper defaults?
3. ~~**Ship messaging**~~ **Resolved:** Phase 1 = exact-route only; program ship confidence requires phase 2.
4. ~~**Energy in `prove-exact`**~~ **Resolved (2026-05-28):** `energy` added to `EXACT_STAGE_ORDER`; `prove-exact-sequence` runs energy → vertices → edges → network.

---

# Part 2 — Implementation plan

## Harness & pre-gate (parallel work)

- **Parity Pre-Gate tiers:** [PARITY_PRE_GATE.md](../reference/workflow/PARITY_PRE_GATE.md), [ADR 0009](../adr/0009-parity-pre-gate-tiers.md). Crop harness (`180709_E_crop_M`) may run in parallel with canonical `phase1_cert_network`; passing crop does not satisfy Phase 1 certification.
- **Random component suite (diagnostic):** [PARITY_RANDOM_COMPONENT_SUITE.md](../reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md), [ADR 0010](../adr/0010-random-component-parity-suite.md). Fast MATLAB/Python differential on seeded noise; does not replace crop or canonical `prove-exact`.
- **Crop oracle:** MATLAB batch → `promote-oracle`; HDF5 energy layout — [matlab-v200-energy-hdf5-oracle-loader.md](../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md).

**In-scope non-goals:** Changing public paper defaults (phase 2).

## Context & research

- Parity CLI: `slavv parity`, `slavv_python/analytics/parity/cli.py`
- Proof core: `slavv_python/analytics/parity/matlab_exact_proof.py`, `coordinator.py`
- Pipeline: `slavv_python/engine/orchestrator.py`, `processing/stages/edges/`
- v29 baseline: **1062/1197** pairs, **135 missing**, **371 extra**
- Docs reconciled to strict zero bar (not >95% match rate)

## Key technical decisions

- **`energy` first in `EXACT_STAGE_ORDER`** — `prove-exact-sequence` enforces R3.
- **Certified run:** `init-exact-run --stop-after network --energy-storage-format npy` on Windows.
- **Single writer** per `--dest-run-root` — no concurrent init/resume.
- **No candidate-boundary fallback** for certification claims.
- **Stage-targeted closure:** fix vertices before edges when AE2 applies.

## Implementation units

| Unit | Goal | Notes |
|------|------|--------|
| **U1** | Extend `prove-exact` energy stage + `prove-exact-sequence` | Done |
| **U2** | Align certification docs with strict zero bar | Done |
| **U3** | Certification candidate run through network | See [findings — active operations](../reference/core/EXACT_PROOF_FINDINGS.md#-active-phase-1-operations) |
| **U4** | Sequential `prove-exact` certification | Status in findings (not here) |
| **U5** | Close parity gaps on failing stage | Active |
| **U6** | Record phase 1 milestone in findings | Pending U4 |

**Current status table:** always update [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md), not this spec.

### U4 — Sequential certification (operator)

```powershell
slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/<run> `
  --dest-run-root workspace/runs/oracle_180709_E/<run> `
  --oracle-root workspace/oracles/<oracle_id>
```

Stop if any stage fails. Do not claim phase 1 complete until all four pass on the **same** dest root.

### U5 — Closure loop

- Crop energy passes → refresh crop vertices→network checkpoints, run crop `prove-exact-sequence`, verify canonical Oracle Artifacts with `ensure-oracle-artifacts`, and rerun canonical from energy.
- Crop energy fails → inspect the first failing energy field, fix only the exact-route energy path, rerun crop from energy, then repeat the crop proof gate.
- Vertices fail → fix discovery before edges.
- Edges fail → use `diagnose-gaps`, `capture-candidates`, and `fail-fast`; record live blockers and accepted lessons in [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md).

## Risks

| Risk | Mitigation |
|------|------------|
| Long energy / stale resume | `npy`; single process; fresh run if resume unreliable |
| Vertices fail → edges fail (AE2) | Fix vertices first |
| Shim under pressure | R12: document downgrade; does not satisfy R2 |
| Doc drift | This spec + TODO.md only; ROADMAP is narrative-only |

## Sources

- [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md)
- [PARITY_CERTIFICATION_GUIDE.md](../reference/workflow/PARITY_CERTIFICATION_GUIDE.md)
- ADRs: [0008](../adr/0008-exact-proof-coordinator.md), [0009](../adr/0009-parity-pre-gate-tiers.md), [0010](../adr/0010-random-component-parity-suite.md)
