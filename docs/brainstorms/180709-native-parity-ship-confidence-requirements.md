---
date: 2026-05-28
topic: 180709-native-parity-ship-confidence
---

# Native parity ship confidence on 180709_E

## Summary

Establish **program ship confidence** that both certified workflows (exact parity route, then paper profile) are **MATLAB-equivalent** on the canonical volume **180709_E**, using **native** runs (no oracle-injected vertices), **strict zero** missing/extra on every gated stage, and **sequential** certification through energy → vertices → edges → network.

**Definitions**

- **Program ship confidence:** Both the exact parity route and the paper profile have passed R1–R3 on their defined volumes. Phase 1 alone does not achieve program ship confidence.
- **Phase 1 milestone (exact-route certification):** Sequential `prove-exact` passes for energy, vertices, edges, and network on full `180709_E` under the exact parity route. May be communicated as “MATLAB-equivalent exact route on 180709_E”; do not claim program ship confidence or full public-pipeline equivalence until phase 2 passes.

---

## Problem Frame

The public workflow is verified end-to-end, but the **exact parity track** on edges remains at roughly **88.7%** pair match against the preserved MATLAB oracle (v29 baseline: 135 missing, 371 extra pairs). That gap blocks a credible claim that native Python output matches MATLAB for real extraction work—not just “close enough” for demos.

Ship confidence requires a **reproducible, strict** certification on a **single canonical volume** rather than informal match-rate milestones. Recent work (e.g. strel linear-index tie-break, manager consolidation) is necessary but not sufficient until `prove-exact` reports **zero missing and zero extra** at each stage in order.

---

## Actors

- **A1. Maintainer / developer:** Runs parity experiments, interprets `prove-exact` reports, and decides when ship confidence is achieved.
- **A2. Public SLAVV user:** Runs `slavv run` with the **paper** profile expecting MATLAB-faithful results without running the parity harness (`matlab_compat` is out of scope unless reprioritized).
- **A3. MATLAB oracle:** Preserved truth vectors under `workspace/oracles/` derived from neurovasc-db (`180709_E` vectorization batch).

---

## Key Flows

- **F1. Sequential exact-route certification (phase 1)**
  - **Trigger:** Maintainer initiates or resumes `init-exact-run` on full `180709_E` with oracle `180709_E_batch_190910-103039`.
  - **Actors:** A1, A3
  - **Steps:** Bootstrap dataset and oracle from `D:\db` (or submodule equivalent) → run native pipeline through network → run sequential `prove-exact` gates (energy, then vertices, edges, network) → each must show zero missing/extra. Execution may compute all stages once; certification (R3) is awarded only in stage order based on `prove-exact` results for that run.
  - **Outcome:** Phase 1 exact-route certification is complete on `180709_E` for all four stages.
  - **Covered by:** R1–R6, R9

- **F2. Paper profile certification (phase 2)**
  - **Trigger:** Phase 1 complete; volume and oracle policy for paper profile resolved (see Outstanding Questions).
  - **Actors:** A1, A2, A3
  - **Steps:** Define paper-profile volume and oracle → run public paper profile on that volume with **native** vertex discovery (R2) → apply same sequential `prove-exact` gates (R1, R3).
  - **Outcome:** Paper profile is certified at the same strict bar as exact route; program ship confidence is achieved.
  - **Covered by:** R1–R3, R2, R7, R8

---

## Requirements

**Certification bar**

- R1. Success means **`prove-exact` reports zero missing and zero extra** for the stage under test—strict set equality, not a percentage threshold.
- R2. Certification uses a **native** pipeline: Python discovers vertices; no oracle vertex checkpoint injection for the certified run.
- R3. **Sequential gating:** Energy must pass before vertices are judged; vertices before edges; edges before network. A failure at any stage blocks **certification** for **that workflow** until resolved. Prior energy verification in docs does not satisfy R3 for a new certified run—each stage needs a passing `prove-exact` on **that** run.
- R4. **Canonical volume** for the first milestone is **`180709_E` full volume** (not center crop) for the exact parity route.

**Scope of stages and workflows**

- R5. The certified pipeline includes **energy, vertices, edges, and network**—not edges alone.
- R6. **Phase 1** delivers **exact-route certification** on full `180709_E` (`comparison_exact_network=True`, oracle-derived parameters). Program ship confidence remains blocked until R7 is satisfied.
- R7. **Phase 2** delivers the same strict bar (R1–R3 and **R2**) for the **paper** profile (`slavv run` paper defaults) on its defined volume and oracle.
- R8. Both workflows must eventually reach R1–R3; phase 2 does not start until phase 1 exact-route certification is complete unless explicitly reprioritized.

**Operational expectations**

- R9. Preserved MATLAB truth lives in **`workspace/oracles/180709_E_batch_190910-103039`** (promoted from neurovasc-db vectorization zip with a **single artifact per stage** in the canonical batch).
- R10. Disposable trial runs live under **`workspace/runs/`**; promoted summaries under **`workspace/reports/`** when a milestone is worth keeping.
- R11. Incremental fixes (e.g. strel tie-break experiments such as `strel_tiebreak_v30`) are **edges-stage experiments** under the sequential gate—they do not alone satisfy ship confidence.

**Change strategy**

- R12. Pursue the **shortest honest path** to **R1–R7** (and satisfy **R8** sequencing): incremental parity fixes tied to a failing `prove-exact` stage on the certified run. MATLAB-shaped shims are a last resort after a written probe; if used, downgrade public claims to “MATLAB-equivalent with documented shim at stage Z”—shim-assisted runs do not satisfy the phase 1 **native** milestone (R2).

---

## Acceptance Examples

- **AE1. Energy gate blocks downstream**
  - **Covers:** R3, R6
  - **Given** an exact-route run on `180709_E` where `prove-exact --stage energy` reports any missing or extra artifact field
  - **When** the maintainer assesses ship confidence
  - **Then** vertices, edges, and network are **not** certified and ship confidence is **not** claimed.

- **AE2. Edges pass only with matching vertex surface**
  - **Covers:** R1, R2, R3
  - **Given** native vertex discovery produced a vertex list that does not match the oracle vertex stage at strict zero
  - **When** `prove-exact --stage edges` is run
  - **Then** edges will show missing/extra pairs (vertex-index mismatch), and ship confidence remains blocked until vertices pass.

- **AE3. Phase 1 complete**
  - **Covers:** R1, R3, R5, R6
  - **Given** sequential `prove-exact` on the exact-route run reports **passed** with zero missing/extra for energy, vertices, edges, and network
  - **When** the maintainer records the milestone
  - **Then** phase 1 exact-route certification is **complete** on full `180709_E`; program ship confidence is **not** claimed until phase 2 passes; phase 2 planning may begin per R8.

---

## Success Criteria

**Phase 1 (P0)**

- Exact-route native run on full `180709_E` passes **sequential** `prove-exact` with **zero missing/extra** on energy, vertices, edges, and network.
- Maintainer may state: **MATLAB-equivalent exact parity route on full `180709_E`** (reproducible harness)—not program ship confidence or paper-profile equivalence.

**Phase 2 (P1)**

- Paper profile passes the **same strict bar** on its defined volume and oracle.
- Maintainer can state **program ship confidence**: both certified workflows are **MATLAB-equivalent** on `180709_E`—not merely “~89% pair match.”

---

## Priority

| Tier | Scope |
|------|--------|
| **P0** | Phase 1 only: R1–R6, R9, F1—exact route, full `180709_E`, one promoted run and reports |
| **P1** | Phase 2: R7–R8, F2—paper profile after OQ1–OQ2 resolved |
| **P2** | Deferred items below; open-ended architecture consolidation unless tied to a P0 stage failure |

---

## Scope Boundaries

**In scope**

- `180709_E` certification program (exact route phase 1, paper profile phase 2).
- Oracle bootstrap from neurovasc-db (`D:\db` or `external/neurovasc-db`).
- Pair-level and stage-level diagnostics to close remaining gaps after v29.

**Deferred for later**

- Parity across **all** neurovasc-db vectorized volumes.
- **CI merge-blocking** parity gate on every PR.
- **Publishable third-party reproduction** as the primary deliverable (documentation may follow ship confidence).
- Do not require network exact proof on volumes other than `180709_E` until edges are closed on `180709_E`.
- Open-ended architecture consolidation not tied to a P0 `prove-exact` failure.

**Outside this product's identity**

- Replacing the public paper workflow with exact-route-only parameters for all users.
- Claiming MATLAB equivalence without a passing `prove-exact` report for the workflow being claimed.

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary outcome | Ship confidence | User optimizes for trusting public SLAVV, not publication or CI gates first |
| Success bar | Strict 100% (zero missing/extra) | User rejected threshold-based “good enough” |
| Vertex source | Native discovery | User chose full native stack over oracle-injected vertices |
| Gating | Sequential by stage | User chose hard gates; edges-only headline rejected |
| Volume scope (milestone 1) | `180709_E` alone | Sufficient for first certification |
| Workflows | Both exact and paper | Both must reach strict bar; phased delivery |
| Delivery order | Exact route first, then paper | Resolves profile/volume mismatch; shortest honest path |

---

## Dependencies / Assumptions

- **Oracle availability:** `180709_E` raw TIFF and vectorization zip are available locally (e.g. `D:\db/data/raw/scans/180709_E.tif`, `180709_E_vector.zip`).
- **Canonical batch:** Oracle promotion uses one MATLAB artifact per stage (timestamp-matched set: vertices `172151`, edges `225419`, network `225419`, energy `103039`) plus full `settings/` tree.
- **Energy stage:** Prior docs list energy as verified; sequential gates still require a fresh `prove-exact` pass for the certified run.
- **Windows storage:** Zarr energy artifacts may fail on Windows; `npy` (or another stable format) is acceptable for certification runs if zarr is unreliable.
- **Approach breadth:** Team is open to architecture or shims if native closure stalls—document the honest path if shims are used.

---

## Outstanding Questions

1. **Paper profile volume:** Full `180709_E`, `180709_EL`, or `180709_EL_center_crop_24x256x256` (CI synthetic path)—and matching oracle promotion?
2. **Paper vs exact parameter alignment:** Should paper profile adopt oracle-derived exact params on `180709_E`, or maintain paper defaults with a separate oracle?
3. ~~**Ship messaging:** After phase 1 only…~~ **Resolved:** Phase 1 messaging is exact-route only; program ship confidence requires phase 2 (see Definitions and Success Criteria).

4. **Harness: energy in `prove-exact`:** Today `prove-exact --stage` accepts `vertices`, `edges`, `network`, or `all`—not `energy`. Phase 1 planning must either extend the CLI or define an equivalent energy gate (e.g. `all` plus documented energy comparison).

---

## Resolve Before Planning

- Resolve **Outstanding Question 1** and **Question 4** before phase 2 planning; phase 1 planning may proceed with full `180709_E` exact route only.
