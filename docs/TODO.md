# SLAVV Developer Dashboard

**Single entry point** for what to do next, where plans live, and where to put new thoughts so they do not scatter across chat, ad-hoc notes, and stale markdown.

> **Rule of thumb:** Checkboxes only here. **Status** → [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md). **Specs** → [plans/](plans/). **Fixes** → [solutions/](solutions/) (`/ce-compound`). **Operator brief** → [.claude/HANDOFF.md](../.claude/HANDOFF.md) (re-synthesize when findings top banner changes).

---

## Where things live

| Kind | Location | Put it here when… |
|------|----------|-------------------|
| **Active tasks** | **This file (`docs/TODO.md`)** | Concrete next actions with checkboxes |
| **Specs (requirements + plan)** | [plans/](plans/) | One `*-spec.md` per active initiative; index in [plans/README.md](plans/README.md) |
| **Ideas (pre-plan)** | [brainstorms/](brainstorms/) | Before a spec exists; promote into `plans/` when scoped |
| **Solved problems & runbooks** | [solutions/](solutions/) | `/ce-compound` writes here; parity index in [findings](reference/core/EXACT_PROOF_FINDINGS.md#-compound-learnings-parity-related) |
| **Architecture decisions** | [adr/](adr/) | Load-bearing design choice (do not re-litigate in TODO) |
| **Live exact-parity status** | [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md) | **Only place** for active run status, proof results, blockers (not TODO) |
| **Operator workflows** | [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md), [PARITY_CERTIFICATION_GUIDE.md](reference/workflow/PARITY_CERTIFICATION_GUIDE.md), [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md) | How to run pre-gate / certification; evidence template after writers/proofs |
| **Investigation archives** | [investigations/](investigations/) | Deep dives that are context, not the task list |

**Do not duplicate:** Status tables and run state → [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md). This file = checkboxes + links only.

**Phase 1:** [phase-1-exact-route-spec.md](plans/phase-1-exact-route-spec.md) · **Status:** [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md#-active-phase-1-operations) · **Handoff:** [.claude/HANDOFF.md](../.claude/HANDOFF.md)

---

## Checklist — do now

### Phase 1 exact route (canonical + crop)

### 🎯 Phase 1 Certification Gates

- [x] **Crop Energy writer** — Lattice `6000` rerun completed `2026-06-22`; evidence in findings / [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).
- [x] **Crop Energy proof** — vs `180709_E_crop_M_v2`: **PASS** (ADR 0011).
- [x] **Energy certification policy** — [ADR 0011](adr/0011-energy-float-certification-policy.md) **ACCEPTED**.
- [x] **Crop Vertices Proof** — **PASS** vs `180709_E_crop_M_v2`.
- [x] **Crop Edges Proof (ADR 0012)** — **PASS** (ownership bar) on crop; residual stretch is strict-field / generation.
- [x] **Canonical full oracle** — `180709_E_full_v2` + ownership map present.
- [x] **Canonical Energy + Vertices** — `canonical_full_v4` **CERTIFIED**.
- [x] **Crop candidate-overlap ≥80% gate** — **97.31%** (15,094 / 15,511) on `crop_M_exact_v3` (2026-07-07). **Retired as launch gate.**
- [x] **Crop-edge truncation parity** — `uint16` floor vs `np.rint` fixed in `_matlab_crop_edges_V200`; crop final edges 14,403 → **14,922** (gap 589).
- [x] **Canonical `v6` evaluated Edges ADR 0012** — **PASS** (ownership **96.02%**, `adr0012_evaluated: true`).
- [x] **Residual watershed generation moved** — `-Inf` sentinel + queue insertion fixes moved crop first diverge **13,761 → 23,005** and crop candidate generation gap **417 → 0**. Refreshed crop final strict gap is **502**. See [HANDOFF](../.claude/HANDOFF.md) § A.
- [x] **Canonical `v8` audit run** — full Edges ADR 0012 still **PASS**, but strict full counts regressed vs `v7` (Edges 66,057 vs 66,224; Network 45,254 vs 45,417). Keep `v7` as better full baseline.
- [ ] **Canonical Network ADR 0012** — **FAIL** on `v8` and still best on `v7` (strands 45,417 vs 48,049). **Downstream of edge gap only.** Continue crop funnel/selection residual loop before a successor full run. **This is the only open Phase 1 ship gate.**
- [ ] **Phase 1 closure** — Energy ✅ Vertices ✅ Edges ✅ Network ⬜ on a fresh evaluated full-volume run; evidence in findings + [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).

### 🛠️ Hardening & Infrastructure (done — keep as archive checkboxes)

- [x] **PipelinePolicy Architecture**, unified math kernel, lattice utils, oracle manifest sync, padded-FFT coarse-slice contract.
- [x] **Canonical energy octave-3/4 divergence** — bit-exact MATLAB `linspace`; full energy certified.
- [x] **Exact-route energy parallelism** — bit-exact `--n-jobs`; resume reorientation fix.
- [x] **Numba painting + systemic float64** on Innovation path.
- [x] **`slavv_vectorize` facade cleanup** — no scipy/skimage demonstration shims as parity engine.
- [x] **Parity job lifecycle** — dead-PID reconciliation tests.

### Harness & ops

- [x] Energy memory safety, internal **[Y,X,Z]** F-order, SortedFrontier default, fail-loud ADR 0012 when maps missing.
- [x] **Experiment-analysis template** — Added [EXPERIMENT_ANALYSIS_TEMPLATE.md](reference/workflow/EXPERIMENT_ANALYSIS_TEMPLATE.md) and normalized the residual analysis entry point.
- [x] **Phase 1 → Phase 2 transition spec** — Added [phase-1-to-phase-2-transition-spec.md](plans/phase-1-to-phase-2-transition-spec.md); transition remains gated on Network ADR 0012.
- [ ] **Parity change verification** — For each tested diagnosis, run focused parity tests + Ruff before a long rerun; record proof result with [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).
- [ ] **Doc freshness** — When findings top banner moves, same-session update of [HANDOFF](../.claude/HANDOFF.md) + this checklist (avoid multi-day operator drift).

**Operational guardrails:** `preflight-exact` before recovery launch; never concurrent writers on one `--dest-run-root`; use `.venv\Scripts\slavv.exe` after `pip install -e .`.

---

## Checklist — next (after Phase 1 gates)

- [x] **O(log N) frontier** — SortedFrontier / heap backends.
- [x] **API reference** — public pipeline / manager docstrings.
- [x] **Sparse Meshgrids** — sparse interp3 meshes for canonical memory.
- [ ] **Freeze Phase 1 baseline after Network green** — record closure run root, proof hashes, release evidence, and figure metrics before Phase 2 starts.
- [ ] **Phase 1 → Phase 2 handoff execution** — only after Network ADR 0012 green; follow [transition spec](plans/phase-1-to-phase-2-transition-spec.md), do not unwind Fortran emulation early.
- [ ] **Paper-profile certification** — phase-1-spec F2 / R7 (volume + oracle TBD).
- [ ] **neurovasc-db** — additional volumes after Phase 1 closed.
- [ ] **Strict-field stretch (optional)** — exact connections / order-sensitive fields on crop after ship gate.

---

## Strategy notes (meta — keep short)

1. **Ship gate is Network multiset on full volume**, not ownership % (already met) and not `prove-exact-sequence`.
2. **Generation gap drives Network** — treat Network red as an edges-generation problem until isolation with MATLAB edges fails.
3. **Crop is the iteration surface**; full volume is the claim surface. Prefer golden-trace + funnel probes over new scratch scripts.
4. **Anti-patterns** → [UNPRODUCTIVE_LOOPS.md](reference/core/UNPRODUCTIVE_LOOPS.md).

---

## Historical context (superseded — do not treat as current tasks)

Older dashboard text referred to **v10 / 76% match**, **>95% edge match rate**, **57.89% crop overlap**, and **80% gate before v6**. Phase 1 now uses **ADR 0011** + **ADR 0012 evaluated per-stage proofs** on the canonical volume. Edge **88.7%** (v29) pair overlap is **deprecated**. Strict-field sequence failure is stretch signal only.

---

## Maintenance

- [x] Contributor guide — `docs/CONTRIBUTING.md`
- [x] Parity run evidence template — [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md)
- [x] Glossary / architecture — `GLOSSARY.md`, `TECHNICAL_ARCHITECTURE.md`
- [x] Parity pre-gate & certification guides
- [x] Planning hub — this file; status + compound index in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)
- [x] **2026-07-12 meta realignment** — HANDOFF, ROADMAP, TODO, AGENTS operating sequence, ADR 0012 post-v6 addendum synced to findings
