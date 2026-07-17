# SLAVV Developer Dashboard

**Single entry point** for what to do next, where plans live, and where to put new thoughts so they do not scatter across chat, ad-hoc notes, and stale markdown.

> **Rule of thumb:** Checkboxes only here. **Status** → [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk). **Specs** → [plans/](plans/). **Fixes** → [solutions/](solutions/) (`/ce-compound`). **Operator brief** → [.claude/HANDOFF.md](../.claude/HANDOFF.md) (re-synthesize when ONE TRUTH moves).

---

## Where things live

| Kind | Location | Put it here when… |
|------|----------|-------------------|
| **Active tasks** | **This file (`docs/TODO.md`)** | Concrete next actions with checkboxes |
| **Specs (requirements + plan)** | [plans/](plans/) | One `*-spec.md` per active initiative; index in [plans/README.md](plans/README.md) |
| **Ideas (pre-plan)** | [brainstorms/](brainstorms/) | Before a spec exists; promote into `plans/` when scoped |
| **Solved problems & runbooks** | [solutions/](solutions/) | `/ce-compound` writes here; parity index in [findings](reference/core/EXACT_PROOF_FINDINGS.md#-compound-learnings-parity-related) |
| **Architecture decisions** | [adr/](adr/) | Load-bearing design choice (do not re-litigate in TODO) |
| **Live exact-parity status** | [ONE TRUTH in EXACT_PROOF_FINDINGS](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) | **Only place** for active run status, proof results, blockers (not TODO) |
| **Operator workflows** | [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md), [PARITY_CERTIFICATION_GUIDE.md](reference/workflow/PARITY_CERTIFICATION_GUIDE.md), [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md) | How to run pre-gate / certification; evidence template after writers/proofs |
| **Investigation archives** | [investigations/](investigations/) | Deep dives that are context, not the task list |

**Do not duplicate:** Status tables and run state → [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md). This file = checkboxes + links only.

**Phase 1:** [phase-1-exact-route-spec.md](plans/phase-1-exact-route-spec.md) · **Status:** [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) · **Handoff:** [.claude/HANDOFF.md](../.claude/HANDOFF.md)

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
- [x] **Crop-axis finalization parity** — align MATLAB-order voxel spacing to Python `[Z,Y,X]` traces before edge smoothing/crop. Crop final overlap improved to **15,361 / 15,511** with **150** missing and **367** extra pairs; later LUT unit-vector refresh leaves the current overlap at **15,362 / 15,511** with **149** missing and **365** extra pairs.
- [x] **MATLAB post-watershed finalization parity** — raw MATLAB watershed candidates and Python candidates match exactly (**19,225 / 19,225**). Python now mirrors MATLAB `resample_vectors` → map-resampled energy/size → smoothing/crop unsigned casts → cleanup. Refreshed crop final edges are **15,511** vs MATLAB **15,511**, with **15,510 / 15,511** overlap (**1** missing, **1** extra).
- [x] **Canonical `v10` audit run** — full Edges ADR 0012 still **PASS** (`70,247` vs MATLAB `69,500`, ownership **99.9867%**), but Network ADR 0012 still **FAIL** and now over-selects (`48,583` vs MATLAB `48,049` strands).
- [x] **Canonical `v15` audit run** — full Edges ADR 0012 **PASS** evaluated with exact strict count (**69,500 / 69,500**), ownership **99.999863%**, and **0** trace failures. Network ADR 0012 still **FAILS** by one strand (**48,048 / 48,049**).
- [x] **Python claim-state trace hardening** — `strel_state` now separates pre-claim and post-claim `vertex_index` / `pointer` / `d_over_r` / `size` values; no-writer probes still show crop candidate generation **15,511 / 15,511**.
- [x] **MATLAB claim-state trace instrumentation** — opt-in state rows added in `external/Vectorization-Public/source/get_edges_by_watershed.m`; scratch Edges-only trace confirms iter **13,761** strel state matches Python and iter **23,005** diverges before strel claim.
- [x] **Frontier action tracing** — compact push / join-reset / discard target rows added for Python and MATLAB; target history shows Python bad-pop location `2844114` is pushed by both implementations at iter **19,247** and removed by MATLAB at iter **22,421**.
- [x] **Reject shared priority-map restore hypothesis** — restoring popped vertex origins in `energy_temp_flat` moved the local split but regressed live crop generation to **14,936 / 15,511** with **575** missing MATLAB pairs; do not carry that patch forward.
- [x] **Join-reset predecessor-state fix** — orthogonal direction factor handling fixes MATLAB's iter-**22,421** join reset for `2844114` without regressing the **15,511 / 15,511** crop generation baseline; the next split moved to **25,495**.
- [x] **LUT unit-vector frontier trace parity** — using MATLAB strel LUT `unit_vectors` fixes the iter-**25,495** tiny-positive direction case. Crop golden frontier trace now matches MATLAB end-to-end; refreshed candidates are **15,511 / 15,511** with **19,225** candidates and **3,714** extras.
- [x] **MATLAB cleanup row-order parity** — `prepare_candidate_indices_for_cleanup()` now sorts by double-precision `max(edge_energies)` like MATLAB `get_edge_metric`; `scripts/compare_clean_edge_pairs_matlab.py` shows **0** row-index mismatches for `clean_edge_pairs`, degree pruning, and cycle pruning on the crop candidate surface.
- [x] **Reject crop chunk-eligibility hypothesis** — `scripts/edge_selection_funnel_probe.py --apply-matlab-chunk-eligibility` emulates MATLAB `get_edges_V300` read/write chunk emission; crop is single-chunk and drops **0 / 19,225** candidates, so chunk windows do not explain candidate extras.
- [x] **Bounded golden-trace regression** — after stopping stale trace writers, `scripts/watershed_frontier_diff.py --stop-after-iteration 30000` reports `bounded_match`, well past the retired iter-**13,761** split.
- [x] **Quantify degree/cycle displacement** — funnel aggregate output shows degree loss **103** MATLAB pairs (**99** with incident surviving extras, **97** with better-metric extras) and cycle loss **32** pairs (**32** with earlier/better incident extras).
- [x] **Reject broad boundary suppression** — final extras skew boundary-adjacent, but a geometry-only boundary candidate filter worsens overlap (threshold 1 → **14,984** overlap / **527** missing); even oracle zero-degree-boundary suppression only reaches **15,377** overlap / **134** missing.
- [x] **Crop final edge-set residual** — re-selection via `select_and_finalize_edge_set` / `scripts/persist_crop_edges_selection.py` on `crop_M_exact_v3` yields **15,511 / 15,511** undirected pair overlap vs `180709_E_crop_M_v2` (both keep `[4043, 6281]`; crop one-pair swap closed on current main). Generation/frontier/cleanup regression guards remain green.
- [x] **Full residual localized (not fixed in production)** — funnel: degree-excess only; equal post-resample max; Python extra `cand 46698` `(26444,38584)` displaces oracle `(34897,38584)`. MATLAB cleanup ≡ Python on same surface. **Ablation:** drop `46698` only → **69,500 / 69,500** pair overlap. See findings 2026-07-15 banner.
- [ ] **Full production fix: suppress/displace-safe the `(26444→38584)` watershed join** — match MATLAB join/emission so the extra candidate is not generated (or not earlier-ranked under equal post-resample max). Do **not** reintroduce cleanup endpoint secondary keys.
- [ ] **Canonical Network ADR 0012** — **FAIL** on `v16` proofs on disk (48,048 / 48,049 strands). Expected green once full Edge Set multiset matches after generation fix + Edges→Network re-proof. **Open Phase 1 ship gate.**
- [ ] **Phase 1 closure** — Energy ✅ Vertices ✅ Edges ✅ Network ⬜ on evaluated full-volume proofs; evidence in findings + [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).

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

1. **Ship gate is Network multiset on full volume**, not ownership % and not `prove-exact-sequence`.
2. **Edge Set multiset drives Network** — crop generation/selection are regression guards; full residual is Candidate Set join displacement (see findings), not a Network rewrite.
3. **Crop = guard; full = claim.** Prefer funnel / cleanup comparator / `select_and_finalize_edge_set` over new selection forks.
4. **Anti-patterns** → [UNPRODUCTIVE_LOOPS.md](reference/core/UNPRODUCTIVE_LOOPS.md). **Authority map** → [docs/README.md](README.md#documentation-authority-map-one-concept--one-home).

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
- [x] **2026-07-15 docs consolidate** — authority map in docs/README; residual narrative → findings-only KPIs; ROADMAP/AGENTS/ADR0012/PHASE1 residual/figures aligned to join-displacement residual (not crop pair swap)
- [x] **2026-07-16 ONE TRUTH** — revalidated `canonical_full_v16` edges/network JSON on disk; collapsed findings banners into single validated table + session diary; HANDOFF/README/TODO/phase-1-spec point only at ONE TRUTH for live status
