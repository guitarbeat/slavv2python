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
- [x] **Audit downstream proof surfaces** — Verify crop Vertex, Edge, and Network oracle/checkpoint fields and ordering; record commands and evidence requirements in the maintained parity workflow docs.
- [x] **Crop Vertices Proof** — **PASS** vs `180709_E_crop_M_v2` (`prove-exact --stage vertices` exit 0): positions + scales match MATLAB **exactly** (13,706 = 13,706; 0 missing/extra) after the SE fix; `energies` now certify under the ADR 0011 `np.allclose` policy after sourcing true energies from the raw `vertices.mat`. (Note: the CLI evidence-freshness guard now reports the energy checkpoint as stale because the vertices writer reran — operational only; re-run the energy writer or the full sequence to refresh.)
- [x] **Decide vertex `energies` certification source** — Chose **(A)**: the loader recovers true physical energies from the raw `vertices.mat` (matched by exact integer position) instead of the curated rank ramp. Generalized: the comparator now applies `np.allclose` to **all** continuous float fields via `float_tol` (strict for integer/topological), with `--strict-floats` forcing strict everywhere.
- [x] **Crop Edges Proof** — **PASS** vs `180709_E_crop_M_v2` under ADR 0012: ownership map agreement 63.51% (above 60.00% threshold) and trace-level match PASS. Parity fixes implemented: (1) `edge_number_tolerance` set to 2; (2) conflict painting disabled on exact route. Residual is emergent watershed order-sensitivity, certified under spatial bars.
- [x] **Canonical full oracle** — Fresh MATLAB batch `batch_260626-125646` promoted to `180709_E_full_v2` (energy size `(64,512,512)`).
- [x] **Exact-route energy parallelism** — `--n-jobs` threaded chunk energy is bit-exact (~4×); see [solution note](solutions/parity/exact-energy-chunk-parallelism.md). Default cert runs use `--n-jobs 6`.
- [x] **Resume reorientation bug** — Resume double-permuted the full volume (energy `(512,64,512)` vs oracle `(64,512,512)`); crop dodged it via Y=X symmetry. **Fixed** (resume reorients like init) + regression test. See [solution note](solutions/parity/resume-energy-orientation.md).
- [/] **Canonical Sequence** — `canonical_full_v4` sequence ran 2026-07-04 (`n_jobs=6`): **Energy ✅ + Vertices ✅ CERTIFIED** on full `180709_E`; **Edges ⛔ (60,213 vs 69,500) + Network ⛔ (39,623 vs 48,049 strands) FAIL strict-field.** Blocked on the watershed generation gap below. Status: [findings](reference/core/EXACT_PROOF_FINDINGS.md#-active-phase-1-operations).
- [ ] **🔑 Watershed edge candidate-generation adjacency gap (PRIMARY Phase-1 blocker)** — Debug session (2026-07-04) localized the Edges/Network strict-field failure to **candidate *generation*, not selection**: 43% of MATLAB's final crop edges are never proposed as Python candidates (only 916 of the gap is pruning); Python wires vertices to different neighbors (more connected vertices, higher degree, but ~57% pair overlap). Fix surface: `matlab_get_edges_by_watershed.py` / `matlab_watershed_heap.py` (basin-meeting order / tie-break). **Next:** instrumented watershed trace on the crop against MATLAB adjacency (H1–H5 in [findings § root cause](reference/core/EXACT_PROOF_FINDINGS.md#-2026-07-04-edge-shortfall-root-cause-generation-gap-not-prune-gap)). Diagnostics: `workspace/scratch/edge_gap_split.py`, `edge_funnel_probe.py`.

### 🛠️ Hardening & Infrastructure
- [x] **PipelinePolicy Architecture** — Implemented declarative Baseline vs Innovation control for Energy, Vertices, and Edges.
- [x] **Unified Math Kernel** — Centralized bit-perfect EIGH math in `energy/math.py`.
- [x] **Unified Lattice Logic** — Created `utils/lattice.py` to prevent rounding errors in 3D chunking.
- [x] **Oracle artifact manifest sync** — `ensure-oracle-artifacts` now reconciles readable normalized artifacts back into `oracle_manifest.json`.
- [x] **Padded-FFT coarse-slice contract** — `_matlab_coarse_local_slices` + regression.
- [x] **Canonical energy octave-3/4 divergence — root cause + fix** — root-caused (MATLAB ground-truth harness `workspace/scratch/matlab_energy_instr/`) to the coarse→fine **upsample mesh not bit-matching MATLAB `linspace`** at coarse-cell boundaries (a ~1-ULP mesh drift floors `interp3` into the wrong valid↔Inf cell, flipping the scale argmin). Fix `ca709a8d` (bit-exact MATLAB `linspace`: mod-based `d1`, multiply-then-divide, forced endpoints, integer phase term) verified to <1e-17 vs MATLAB on integer- and sub-integer-landing voxels; 595 tests green. **Canonical `prove-exact --stage energy` CERTIFIED — 0 scale mismatches across all 16,777,216 voxels** (39,494 → 0). See [findings](reference/core/EXACT_PROOF_FINDINGS.md) + [solution note](solutions/parity/canonical-energy-high-octave-divergence.md).
- [x] **Numba Painting Fix** — Ported round-half-up logic to `_choose_vertices_loop_numba`.
- [x] **Systemic float64 Enforcement** — Upgraded all pipeline intermediates to float64 for Innovation path.
- [x] **Reconcile `slavv_vectorize.py` convenience wrappers** — The standalone `get_vertices_v200_python` / `get_edges_by_watershed_python` / `choose_edges_v200_python` / `get_network_v190_python` `scipy`/`skimage` demonstration shims were removed from `slavv_python/pipeline/slavv_vectorize.py` (2026-07-03). The facade now exposes only `vectorize_python` + `get_energy_v202_python`, both delegating to the stage managers, so it can't be mistaken for the parity engine.
- [x] **Crop tier-2 gate** — After crop Energy passes, `prove-exact-sequence` on `180709_E_crop_M` (all four stages pass sequence proof under ADR 0011/0012).
- [ ] **Canonical tier-3 gate** — All four stages pass on full `180709_E`; promote summary to `workspace/reports/`. **Status (2026-07-04):** Energy + Vertices pass strict; Edges + Network blocked on the watershed candidate-generation adjacency gap (see PRIMARY blocker above).

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

Older dashboard text referred to **v10 / 76% match** and **>95% edge match rate** as the certification bar. Phase 1 now uses **strict zero** per stage (energy/vertices) plus the [ADR 0012](adr/0012-edge-watershed-parity-bar.md) spatial bars for edges/network, via `prove-exact-sequence` on the canonical volume. Edge **88.7%** (v29) is **deprecated** — that pair-overlap figure was inflated by a since-fixed double-transpose bug; the edge bar is voxel ownership-map (~63.5%) + per-edge trace tolerance, not pair-set equality.

---

## Maintenance

- [x] Contributor guide — `docs/CONTRIBUTING.md`
- [x] Parity run evidence template — [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md)
- [x] Glossary / architecture — `GLOSSARY.md`, `TECHNICAL_ARCHITECTURE.md`
- [x] Parity pre-gate & certification guides — `PARITY_PRE_GATE.md`, `PARITY_CERTIFICATION_GUIDE.md`
- [x] Planning hub — this file; status + compound index in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)
