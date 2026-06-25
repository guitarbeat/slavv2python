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
- [x] **Crop Energy proof** — vs `180709_E_crop_M_v2`: **PASS** under the ADR 0011 `np.allclose` gate (rtol=1e-7, atol=1e-9). `scale_indices` **0** mismatches; `energy` max \|Δ\|=**1.99×10⁻¹¹**; `lumen_radius_microns` max \|Δ\|=**7.1×10⁻¹⁵**. `prove-exact --stage energy` exit 0 (2026-06-24). Status: [findings § v2 proof](reference/core/EXACT_PROOF_FINDINGS.md#latest-crop-energy-proof-vs-oracle-v2-2026-06-24).
- [x] **Diagnose Energy scale winners** — Resolved via fresh MATLAB `batch_260624-105705` → oracle v2; 0 scale mismatches.
- [x] **Diagnose Energy float64 drift** — Triage complete (`workspace/scratch/energy_ulp_triage_v2.json`): cross-library NumPy/MKL drift at matching scales; no localized Python fix without gate change.
- [x] **Energy certification policy** — [ADR 0011](adr/0011-energy-float-certification-policy.md) **ACCEPTED** (2026-06-24): Option B refined to `np.allclose(rtol=1e-7, atol=1e-9)` on continuous float fields (`energy.energy`, `lumen_radius_microns`, …); strict `scale_indices`/topology. Pure ULP rejected — it explodes near zero (36,074 false fails at 48 ULP despite \|Δ\|≤2×10⁻¹¹).
- [ ] **Audit downstream proof surfaces** — Verify crop Vertex, Edge, and Network oracle/checkpoint fields and ordering; record commands and evidence requirements in the maintained parity workflow docs.
- [x] **Crop Vertices Proof** — **PASS** vs `180709_E_crop_M_v2` (`prove-exact --stage vertices` exit 0): positions + scales match MATLAB **exactly** (13,706 = 13,706; 0 missing/extra) after the SE fix; `energies` now certify under the ADR 0011 `np.allclose` policy after sourcing true energies from the raw `vertices.mat`. (Note: the CLI evidence-freshness guard now reports the energy checkpoint as stale because the vertices writer reran — operational only; re-run the energy writer or the full sequence to refresh.)
- [x] **Decide vertex `energies` certification source** — Chose **(A)**: the loader recovers true physical energies from the raw `vertices.mat` (matched by exact integer position) instead of the curated rank ramp. Generalized: the comparator now applies `np.allclose` to **all** continuous float fields via `float_tol` (strict for integer/topological), with `--strict-floats` forcing strict everywhere.
- [ ] **Crop Edges Proof** — First proof (pre-fix) showed a large divergence (MATLAB 15,511 vs Python 9,429; 5,109 shared). Investigation found **two MATLAB-faithful parity bugs**, now fixed: (1) `edge_number_tolerance` honored the param (4) but MATLAB `get_edges_V300.m:100` **hard-codes 2** for watershed seeds → fixed (hard-code 2 in `matlab_get_edges_by_watershed.py`); (2) the exact route ran conflict painting (paper-profile `comparison_exact_network_use_conflict_painting=True`) but MATLAB **comments out `choose_edges_V200`** → fixed (force off on the exact route in `selection.py`). After the fixes: Python **9,429 → 13,775** edges, shared **5,109 → 8,135** (vs MATLAB 15,511) — materially closer but still ~52% shared. **Frontier-ordering hypothesis RULED OUT**: a faithful sorted-list `available_locations` reimplementation (correct vertex order + seed tie-break) closed only ~3% of the gap at ~10× slowdown → reverted (kept the heap). **Per-edge evidence diagnosis (2026-06-25):** the residual is **early trace termination on long paths**, not ordering or cleanup. Of 7,164 missing MATLAB edges, **5,774 (80%) Python never traces at all** (only 1,390 are traced-then-dropped); the never-traced set is systematically **longer** (mean trace len **9.1 vs 5.3** for shared, p90 15). Python over-traces overall (18,378 candidates → 13,797 final) but mis-connects: it links vertices to closer neighbors and fails the longer MATLAB connections. **Prime suspect:** cumulative distance penalty `d_over_r` / energy-tolerance gate (`_matlab_frontier_adjusted_neighbor_energies`, `_matlab_global_watershed_tolerance_mask`) killing long traces before they reach the far vertex (no hard length cap exists on either side). **Secondary cleanup bug:** Python max vertex degree **5** vs MATLAB **4** (degree-excess cap not fully enforced). **Next:** instrument one long never-traced edge and compare `d_over_r`/adjusted-energy/tolerance step-by-step vs MATLAB to pinpoint the termination divergence.
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
- [ ] **Reconcile `slavv_vectorize.py` convenience wrappers** — The standalone `get_vertices_v200_python` / `get_edges_by_watershed_python` / `choose_edges_v200_python` / `get_network_v190_python` in `slavv_python/pipeline/slavv_vectorize.py` are simplified `scipy`/`skimage` demonstration shims, **not** exact-parity code (the real logic is in the stage managers + `matlab_get_*` modules). Either route them through the managers or remove them so the facade can't be mistaken for the parity engine. Docs softened 2026-06-24.
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

- [x] Contributor guide — `docs/CONTRIBUTING.md`
- [x] Parity run evidence template — [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md)
- [x] Glossary / architecture — `GLOSSARY.md`, `TECHNICAL_ARCHITECTURE.md`
- [x] Parity pre-gate & certification guides — `PARITY_PRE_GATE.md`, `PARITY_CERTIFICATION_GUIDE.md`
- [x] Planning hub — this file; status + compound index in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)
