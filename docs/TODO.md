# SLAVV Developer Dashboard

**Single entry point** for what to do next, where plans live, and where to put new thoughts so they do not scatter across chat, ad-hoc notes, and stale markdown.

> **Rule of thumb:** Checkboxes only here. **Status** → [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md). **Specs** → [plans/](plans/). **Fixes** → [solutions/](solutions/) (`/ce-compound`).

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

**Phase 1:** [phase-1-exact-route-spec.md](plans/phase-1-exact-route-spec.md) · **Status:** [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md#-active-phase-1-operations)

---

## Checklist — do now

### Phase 1 exact route (canonical + crop)

### 🎯 Phase 1 Certification Gates

- [x] **Crop Energy writer** — Lattice `6000` rerun completed `2026-06-22` (~7h 44m); `best_energy.npy` + `best_scale.npy` present. Evidence: [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).
- [ ] **Crop Energy proof** — vs `180709_E_crop_M_v2`: **scale_indices 0**, **energy 3,810,126** strict ULP failures (median 4 ULP, max \|Δ\|≈2×10⁻¹¹). Status: [findings § v2 proof](reference/core/EXACT_PROOF_FINDINGS.md#latest-crop-energy-proof-vs-oracle-v2-2026-06-24).
- [x] **Diagnose Energy scale winners** — Resolved via fresh MATLAB `batch_260624-105705` → oracle v2; 0 scale mismatches.
- [x] **Diagnose Energy float64 drift** — Triage complete (`workspace/scratch/energy_ulp_triage_v2.json`): cross-library NumPy/MKL drift at matching scales; no localized Python fix without gate change.
- [ ] **Energy certification policy** — Decide strict `np.equal` vs documented ULP tolerance for Phase 1 claim ([ADR 0010](adr/0010-random-component-parity-suite.md) advisory precedent only).
- [ ] **Audit downstream proof surfaces** — Verify crop Vertex, Edge, and Network oracle/checkpoint fields and ordering before the Energy gate opens; record commands and evidence requirements in the maintained parity workflow docs.
- [ ] **Crop Vertices Proof** — Blocked on crop Energy strict-zero proof (not writer).
- [ ] **Crop Edges Proof** — Blocked on crop Vertices strict-zero.
- [ ] **Canonical Energy Proof** — Cert claim gate: after crop Energy proof passes; may run in parallel per ADR 0009 when memory allows ([findings](reference/core/EXACT_PROOF_FINDINGS.md#-active-phase-1-operations)).
- [ ] **Canonical Sequence** — Full `prove-exact-sequence` on `180709_E` after crop tier-2 gate passes.

### 🛠️ Hardening & Infrastructure
- [x] **PipelinePolicy Architecture** — Implemented declarative Baseline vs Innovation control for Energy, Vertices, and Edges.
- [x] **Unified Math Kernel** — Centralized bit-perfect EIGH math in `energy/math.py`.
- [x] **Unified Lattice Logic** — Created `utils/lattice.py` to prevent rounding errors in 3D chunking.
- [x] **Oracle artifact manifest sync** — `ensure-oracle-artifacts` now reconciles readable normalized artifacts back into `oracle_manifest.json`.
- [x] **Padded-FFT coarse-slice contract** — `_matlab_coarse_local_slices` + regression; Energy proof still open (see findings).
- [x] **Numba Painting Fix** — Ported round-half-up logic to `_choose_vertices_loop_numba`.
- [x] **Systemic float64 Enforcement** — Upgraded all pipeline intermediates to float64 for Innovation path.
- [ ] **Crop tier-2 gate** — After crop Energy passes, `prove-exact-sequence` on `180709_E_crop_M` (all four stages strict-zero).
- [ ] **Canonical tier-3 gate** — All four stages pass on full `180709_E`; promote summary to `workspace/reports/`.

### Harness & ops

- [x] **Energy memory safety** — Removed large 4D chunk arrays in `matlab_get_energy_v202_chunked.py`; peak memory reduced 30x.
- [x] **Internal Grid Alignment** — Anchored pipeline to **[Y, X, Z]** with Fortran (F) memory order to match MATLAB tie-breaking.
- [x] **Watershed Robustness** — Resolved `KeyError` in `FrontierQueue` and restored directional suppression parity.
- [x] **Parity job lifecycle reconciliation** — Dead-PID + running snapshot → persisted `interrupted`; terminal `parity_job.json` metadata. Tests: `tests/unit/parity/test_parity_job_lifecycle.py`.
- [ ] **Parity change verification** — For each tested diagnosis, run the focused parity tests and Ruff checks before a long rerun; record the exact proof result using [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).

**Operational guardrails:** run `preflight-exact` before a recovery launch and never start concurrent writers for one `--dest-run-root`.

---

## Checklist — next (after Phase 1 gates)

- [x] **O(log N) frontier** — `heapq` / `SortedList` for `available_locations` in `matlab_get_edges_by_watershed.py` (performance, not cert blocker).
- [x] **API reference** — Public `SlavvPipeline` and internal `Manager` class docstrings.
- [x] **Sparse Meshgrids** — Refactor `_interp3_matlab_linear_inf` to accept sparse coordinate meshes, saving >400MB for canonical volumes.
- [ ] **neurovasc-db** — Import and verify additional volumes when Phase 1 is closed.

---

## Historical context (superseded — do not treat as current tasks)

Older dashboard text referred to **v10 / 76% match** and **>95% edge match rate** as the certification bar. Phase 1 now uses **strict zero** per stage via `prove-exact-sequence` on the canonical volume. Edge **88.7%** (v29) remains a useful baseline narrative in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md), not the Phase 1 exit criterion.

---

## Maintenance

- [x] Contributor guide — `CONTRIBUTING.md`
- [x] Parity run evidence template — [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md)
- [x] Glossary / architecture — `GLOSSARY.md`, `TECHNICAL_ARCHITECTURE.md`
- [x] Parity pre-gate & certification guides — `PARITY_PRE_GATE.md`, `PARITY_CERTIFICATION_GUIDE.md`
- [x] Planning hub — this file; status + compound index in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)
