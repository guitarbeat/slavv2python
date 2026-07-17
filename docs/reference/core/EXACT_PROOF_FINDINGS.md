# Exact Proof Findings

[Up: Reference Docs](../README.md) · [Authority map](../../README.md#documentation-authority-map-one-concept--one-home) · [HANDOFF](../../../.claude/HANDOFF.md) · [TODO](../../TODO.md)

**Last Updated:** 2026-07-16  
**Role:** **Only** live source of truth for exact-route MATLAB↔Python parity status (runs, proofs, blockers, residual claim).  
**Not here:** task checkboxes ([TODO](../../TODO.md)), operator commands ([HANDOFF](../../../.claude/HANDOFF.md)), figure paint constants ([parity_campaign_series.py](../../../figures/parity_campaign_series.py) — mirror KPIs only).

---

## ONE TRUTH — Phase 1 parity (validated from disk)

> **Answer:** We do **not** have 100% end-to-end MATLAB≡Python certification.  
> **Phase 1 is OPEN.** Three of four stages pass their certification bars on the claim surface; Network fails ADR 0012 multiset equality by **one strand**.

| Stage | Verdict | Claim surface / evidence | Notes |
| :--- | :--- | :--- | :--- |
| **Energy** | ✅ **PASS** (ADR 0011) | Full-volume proof lineage: `canonical_full_v4` `03_Analysis/exact_proof_energy.json` (`passed: true`). Seeded into later claim roots. | Discrete scale indices exact; continuous under `np.allclose`. |
| **Vertices** | ✅ **PASS** (ADR 0011) | `canonical_full_v4` `exact_proof_vertices.json` (`passed: true`). | Positions/scales exact. |
| **Edges** | ✅ **PASS** (ADR 0012 evaluated) | **`canonical_full_v16`** `03_Analysis/exact_proof_edges.json` | Connections **69,500 / 69,500**; ownership **5,843,205 / 5,843,213** (**99.999863%**); trace failures **0** / 69,499; `adr0012_evaluated: true`. |
| **Network** | ❌ **FAIL** (ADR 0012) | **`canonical_full_v16`** `03_Analysis/exact_proof_network.json` | Strand endpoint-pair multiset: Python **48,048** vs MATLAB **48,049**. `release_evidence.json` `proof_passed: false`. **Open ship gate.** |

**Oracle:** `workspace/oracles/180709_E_full_v2` (batch `batch_260626-125646`).  
**Claim run root:** `workspace/runs/oracle_180709_E/canonical_full_v16`.  
**Phase 1 closes only when** evaluated Edges **and** Network both pass on a fresh full claim root (see [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md)).

### Disk revalidation stamp

**2026-07-16** — re-read JSON on disk (no re-run). Confirmed:

- `exact_proof_edges.json`: `passed=true`, `edges_adr0012_gate.adr0012_evaluated=true`, `n_python_connections=n_matlab_connections=69500`, `ownership_map_agreement_rate=0.9999986308902311`, `trace_n_failures=0`.
- `exact_proof_network.json`: `passed=false`, first failure `network.strands` strand endpoint-pair multiset mismatch, shapes `[48048,2]` vs `[48049,2]`.
- Energy/Vertices stage JSON live under `canonical_full_v4` (lineage seed); not re-proved on `v16` path because those stages were carried forward unchanged.

### Active residual (why Network is red)

- **Crop guard closed:** `crop_M_exact_v3` re-selection undirected pair overlap **15,511 / 15,511** vs `180709_E_crop_M_v2`.
- **Full residual:** Edge Selection degree-excess drops oracle pair `(34897, 38584)` in favor of earlier equal-metric **extra** candidate **`46698`** `(26444, 38584)`. Cleanup Python≡MATLAB on the same Candidate Set (0 row-index mismatches on comparator).
- **Ablation:** drop only `cand 46698` → undirected pair overlap **69,500 / 69,500**. Production fix = stop emitting / displacing that join (watershed join emission), **not** a cleanup secondary-key hack.
- Scratch localization artifact: `workspace/scratch/full_residual_pair_raw.npz` (indices `46698`, `56786`).

**Do not claim:** “100% parity”, “all four stages certified on full volume”, approximate strand-count % as Network pass, or Phase 1 closed from Edges-only.

**Figure KPI mirror:** update [`figures/parity_campaign_series.py`](../../../figures/parity_campaign_series.py) only when the table above moves; then regenerate claim figures.

**Spec:** [phase-1-exact-route-spec.md](../../plans/phase-1-exact-route-spec.md)

---

## 📊 Executive status (stage model)

**Numbers live only in [ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk) above.** This section is the stage model + pointers, not a second status table.

Phase 1 exit criterion ([ADR 0012 post-v6 addendum](../../adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-12-post-v6-residual--network-is-the-open-ship-gate)): **evaluated** ADR 0012 on full `180709_E` for **both** Edges and Network. Claim root: `canonical_full_v16`. Operator brief: [.claude/HANDOFF.md](../../../.claude/HANDOFF.md). Proposal figures: [figures/README.md](../../../figures/README.md).

| Stage | Certification bar | Full-volume claim (see ONE TRUTH) |
| :--- | :--- | :--- |
| **Energy** | ADR 0011 strict discrete + `np.allclose` | ✅ PASS (lineage seed `v4`) |
| **Vertices** | ADR 0011 | ✅ PASS (lineage seed `v4`) |
| **Edges** | ADR 0012 ownership-map + trace (evaluated) | ✅ PASS on `v16` |
| **Network** | ADR 0012 strand/bifurcation multisets | ❌ FAIL on `v16` — **open ship gate** |

Strict-field connection equality remains stretch after Network multiset passes.

---

## 🚦 Active Phase 1 operations

> **Closure policy:** Phase 1 closes when a fresh full-volume run passes **evaluated** ADR 0012 for **Edges and Network**. Latest audit **`canonical_full_v16`**: Edges ✅ Network ❌ (one strand). Prior audits `v6`–`v15` preserved. Crop overlap ≥80% is **cleared**; generation gap is **0**. See [HANDOFF](../../../.claude/HANDOFF.md).

| Track | Run / artifact | Status |
|-------|----------------|--------|
| **Crop harness oracle** | `workspace/oracles/180709_E_crop_M_v2` | ✅ Fresh MATLAB batch `batch_260624-105705` (lattice-6000). Use v2 for all new proofs. |
| **Oracle artifact readiness** | `180709_E_crop_M_v2`, `180709_E_full_v2` | ✅ `ensure-oracle-artifacts --stage all` passes. ✅ Full oracle **`watershed_ownership_map.mat`** present (`batch_260626-125646/data/`). |
| **Crop harness run (audit)** | `workspace/runs/oracle_180709_E/crop_M_exact` | ✅ Per-stage ADR 0012 certified (2026-07-01). Stale for stretch KPI (pre–PR #103). |
| **Crop stretch run** | `workspace/runs/oracle_180709_E/crop_M_exact_v3` | ✅ Candidate generation **100%** (15,511 / 15,511); frontier **match**. **2026-07-15 re-selection:** final undirected pair overlap **15,511 / 15,511** (crop residual closed). Keep as regression guard; full hub-**38584** swap is the active residual. |
| **Canonical full oracle** | `workspace/oracles/180709_E_full_v2` | ✅ Batch `batch_260626-125646`; energy `(64,512,512)`. ✅ `watershed_ownership_map.mat` (2026-07-07 MATLAB harness). |
| **Canonical full run (audit)** | `workspace/runs/oracle_180709_E/canonical_full_v4` | ✅ Energy + Vertices **CERTIFIED** (2026-07-04). Stale edges/network (pre–PR #103). |
| **Canonical closure run (audit)** | `workspace/runs/oracle_180709_E/canonical_full_v5` | ✅ Writer succeeded (2026-07-06, ~2.1h). ⛔ Proof invalid ADR 0012 (`adr0012_evaluated: false`); strict-field edges 60,287 vs 69,500. Preserve as audit. |
| **Canonical closure run (v6)** | `workspace/runs/oracle_180709_E/canonical_full_v6` | ✅ Writer **COMPLETED** (2026-07-08, edges ~91min + network ~4min). Edges ADR 0012 ✅ **PASS** (`adr0012_evaluated: true`, ownership 96.02%). Network ADR 0012 ❌ **FAIL** (strand multiset 44,595 vs 48,049; downstream of edge gap). Phase 1 open. |
| **Canonical closure run (v7)** | `workspace/runs/oracle_180709_E/canonical_full_v7` | ✅ Writer **COMPLETED** (2026-07-13) seeded from `v6` Energy/Vertices; `parity_include_debug_maps=true`. Edges ADR 0012 ✅ **PASS** (`adr0012_evaluated: true`, ownership 99.89%; Python 66,224 vs MATLAB 69,500). Network ADR 0012 ❌ **FAIL** (strand multiset 45,417 vs 48,049). Phase 1 open, but residual improved. |
| **Canonical closure run (v10)** | `workspace/runs/oracle_180709_E/canonical_full_v10` | ✅ Writer **COMPLETED** (2026-07-13) seeded from `v8` Energy/Vertices after crop-axis finalization fix. Edges ADR 0012 ✅ **PASS** (`adr0012_evaluated: true`, ownership 99.9867%; Python 70,247 vs MATLAB 69,500). Network ADR 0012 ❌ **FAIL** (strand multiset 48,583 vs 48,049). Phase 1 open; residual now over-selected. |
| **Canonical closure run (v15)** | `workspace/runs/oracle_180709_E/canonical_full_v15` | ✅ Writer **COMPLETED** (2026-07-14) seeded from `v10` Energy/Vertices after post-watershed finalization fix. Edges ADR 0012 ✅ **PASS** (`adr0012_evaluated: true`, ownership 99.999863%; Python 69,500 vs MATLAB 69,500; trace failures 0). Network ADR 0012 ❌ **FAIL** by one strand (Python 48,048 vs MATLAB 48,049). Audit prior to `v16`. |
| **Canonical closure run (v16)** | `workspace/runs/oracle_180709_E/canonical_full_v16` | ✅ Writer **COMPLETED** (2026-07-15). Edges ADR 0012 ✅ **PASS** evaluated (`exact_proof_edges.json`: 69,500 / 69,500; ownership 99.999863%; trace failures 0). Network ADR 0012 ❌ **FAIL** (`exact_proof_network.json`: strand multiset 48,048 / 48,049). **Current claim surface for residual; Phase 1 OPEN.** |

Evidence template: [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md)

### Watershed iteration log (crop overlap KPI → 80% gate)

| Date | Fix / change | Overlap (MATLAB pairs) | Notes |
|------|--------------|------------------------|-------|
| 2026-07-06 | Baseline `crop_M_exact_v3` (seed pop-rank fix in `FrontierQueue.__init__`) | **57.89%** (8,979 / 15,511) | Production path via `generate_watershed_candidates`. |
| 2026-07-07 | `FrontierQueue.push` seed_idx tie-break | **57.89%** (no change) | Earlier **62.25%** was a false signal from gap probe calling the engine without `mpv[[2,0,1]]` permute — probe fixed. |
| 2026-07-07 | List-based `available_locations` queue (reverted) | **11.56%** | Regressed; heap retained. |
| 2026-07-07 | `FrontierQueue` peek-not-pop + deferred clear (MATLAB `available_locations(end)`) | **57.89%** (no change) | Semantically correct; KPI unchanged on crop. |
| 2026-07-07 | **SortedFrontier** (faithful `available_locations` port) + insert tail-drop fix + MATLAB `min` strel argmin | **57.89%** (8,979 / 15,511); **17,253** candidates | Golden trace: **13,706 / 13,706** vertex `iteration_start` pops match MATLAB (1-based offset); first divergence at iteration **13,707** (post-vertex frontier). Harness: `scripts/watershed_frontier_diff.py`. Default backend: `watershed_frontier_backend=sorted` (`heap` fallback). |
| 2026-07-07 | Parity proof plan Phase 0 baseline reconfirm (`pip install -e .`, gap probe) | **57.89%** (8,979 / 15,511); **17,253** candidates | No code change; golden trace artifacts cleared — regenerating harness before post-vertex fix loop. |
| 2026-07-07 | `mpv_matlab` forward-LUT + suppression parity + tiebreak argmin + diff harness `mpv[[2,0,1]]` | **97.31%** (15,094 / 15,511); **19,283** candidates | Crop edges rerun. Golden trace: vertex pops match through iter **13,760**; diverges iter **13,761**. **80% gate cleared** → `canonical_full_v6` writer launched. |
| 2026-07-07 | `-Inf` strel argmin parity (`_matlab_watershed_min_candidate_energies`) | **97.31%** (unchanged) | Honors MATLAB `min` NaN-ignore / `-Inf`-minimum semantics; trace still diverges iter **13,761** (frontier order). Full oracle ownership map written. |
| 2026-07-13 | Preserve vertex-center `energy_temp` `-Inf` sentinel after vertex pop (`VoxelClaimMap.restore_vertex_energy`) | **99.88%** (15,492 / 15,511); **19,292** candidates | No-writer generation gap **417 → 19**; first `iteration_start` divergence **13,761 → 13,833**. Crop final selected edges **15,046** vs MATLAB 15,511 (strict gap 465). `canonical_full_v7` launched. |
| 2026-07-13 | Align finalization voxel spacing from MATLAB `[Y,X,Z]` params to Python `[Z,Y,X]` traces | **100%** (15,511 / 15,511); **19,226** candidates | Crop final selected edges **15,728** vs MATLAB 15,511; final MATLAB overlap **15,361** (missing 150, extra 367). `canonical_full_v10`: Edges ✅, Network ❌ with over-selection. |
| 2026-07-13 | Re-test restoring popped vertex origins in shared `energy_temp_flat` | **96.29%** (14,936 / 15,511); **19,120** live candidates | **Rejected.** Target trace showed a local stale-origin `-Inf` at iter 22,421, but restoring the shared priority map regressed live generation (missing 575, extras 4,184). Keep baseline sentinel behavior and use action tracing for the next predecessor-state hypothesis. |
| 2026-07-13 | Clamp near-zero directional alignment before `-Inf` multiplication | **100%** (15,511 / 15,511); **19,226** candidates | Fixes the iter **22,421** target join reset and moves full-trace first `iteration_start` divergence **23,005 → 25,495**. Refreshed crop final overlap **15,362** (missing 149, extra 366), only +1 vs prior checkpoint; no full launch. |
| 2026-07-13 | Use MATLAB LUT `unit_vectors` for frontier directional alignment | **100%** (15,511 / 15,511); **19,225** candidates | Fixes both the iter **22,421** exact-zero skip and the iter **25,495** tiny-positive forward case. `watershed_frontier_diff.py` now reports **match** against the crop MATLAB trace. Refreshed crop final overlap remains **15,362** (missing 149, extra 365); next loop is final cleanup, not generation or full launch. |
| 2026-07-14 | Match MATLAB `clean_edge_pairs` double-precision metric row order | **100%** (15,511 / 15,511); **19,225** candidates | Comparator found and fixed 6 adjacent row swaps from float32 metric ties. MATLAB comparator now shows **0** row-index mismatches for `clean_edge_pairs`, degree pruning, and cycle pruning. Final overlap unchanged (**15,362**, missing 149, extra 365). |
| 2026-07-14 | Test MATLAB `get_edges_V300` read/write chunk eligibility as extra-candidate filter | **Rejected**; drops **0 / 19,225** crop candidates | Crop edge chunk lattice is single-chunk under MATLAB's `max_voxels_per_node = 1e8`, so chunk emission windows are not the missing suppression rule. Next loop is degree/cycle displacement by surviving extra candidates. |
| 2026-07-14 | Quantify extra-candidate displacement in final cleanup | Degree loss **103** MATLAB pairs; cycle loss **32** MATLAB pairs | Degree: **99/103** lost pairs have incident surviving extras, **97/103** have better-metric incident extras. Cycle: **32/32** have earlier and better incident extras. This confirms cleanup is faithfully pruning an over-strong extra-candidate surface. |
| 2026-07-14 | Bounded golden-trace regression after stale writer cleanup | **bounded_match** through iteration **30,000** | Killed two stale `watershed_frontier_diff.py --regenerate-python` processes that were interleaving JSONL writes. Patched bounded comparison so `--stop-after-iteration` compares only iteration-bearing rows. Current bounded trace passes well beyond the retired 13,761 split. |
| 2026-07-14 | Test boundary-based suppression as extra-candidate explanation | **Rejected** as production rule | Geometry-only endpoint boundary filter damages overlap badly (threshold 1: overlap **14,984**, missing **527**). Oracle-aware zero-degree-boundary upper bound is only modest (best tested: overlap **15,377**, missing **134**, extras **238**), so boundary-adjacent zero-degree endpoints are not the root cause. |
| 2026-07-14 | Match MATLAB post-watershed finalization and cleanup surface | **100%** generation; final **15,510 / 15,511** overlap | MATLAB raw watershed candidates equal Python raw candidates (**19,225 / 19,225**). Python now mirrors `resample_vectors`, map-resampled size/energy, smooth/crop unsigned casts, and cleanup on resampled traces. Refreshed crop final edges are **15,511** vs MATLAB **15,511** with one pair swap (`[4043, 6281]` vs `[4212, 6281]`). |
| 2026-07-15 | Crop re-selection + full residual localization | Crop final **15,511 / 15,511** (closed); full pair **69,499 / 69,500** | Crop one-pair swap closed under current Edge Selection. Full residual = degree-excess displacement by extra join `cand 46698` (ablation → 69,500/69,500). Claim root **`canonical_full_v16`**: Edges ✅ Network ❌ (one strand). **Live detail:** [ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk). |

**Probe command:** `.\.venv\Scripts\python.exe scripts/watershed_candidate_gap_probe.py --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 --oracle-root workspace/oracles/180709_E_crop_M_v2`

**80% milestone:** ✅ **cleared** (2026-07-07). **Current residual playbook:** [.claude/HANDOFF.md](../../../.claude/HANDOFF.md) § A (full join-displacement residual; crop = regression guard) → § B (successor full claim run after production fix). Do **not** re-open the retired crop one-pair loop as primary.

### 🔎 2026-07-04: Edge shortfall root cause (generation gap, not prune gap)

> **Historical diagnosis (2026-07-04).** Superseded as *current residual class* by ONE TRUTH (crop generation closed; full residual is join displacement after faithful selection). Kept for investigation trail only.

**Question:** why do Edges (and therefore Network) fail strict-field parity on both crop and full `180709_E`?

**Method (debug session, crop harness — reproduces the full-volume ~13% deficit in ~5 min):** read the completed `crop_M_exact` edge checkpoint diagnostics, reconstructed the selection funnel, cross-checked every step against the **active** (non-commented) MATLAB source in `external/Vectorization-Public/source/vectorize_V200.m`, then compared MATLAB's final edge pairs against Python's *candidate* set.

**Selection funnel is faithful — pruning is NOT the problem.**

```
17,227 candidates → −2,287 crop → −0 dedup → −879 degree(≤4) → −0 orphan → −506 cycles → 13,555 final
```

MATLAB's exact route runs the *same* active steps in the same order: `crop_edges_V200` (L3595) → `clean_edge_pairs` (L3620) → [`choose_edges_V200` commented out L3635] → `clean_edges_vertex_degree_excess(…,4)` (L3666) → `clean_edges_orphans` (L3674) → `clean_edges_cycles` (L3682). Python mirrors this (conflict painting off on the exact route). So the steps match.

**Decisive split** — of MATLAB's **15,511** final crop edges (vertex indices 0-based and aligned to Python; verified by exact-pair intersection):

| Category | Count | Share |
|---|---:|---:|
| present as a Python **candidate** | 8,785 | 56.6% |
| **generation gap** (never a Python candidate) | **6,726** | **43.4%** |
| prune gap (candidate but dropped in selection) | 916 | 5.9% |

**Not an isolated-vertex problem, not over-pruning — a *wiring* difference.** Python actually connects *more* vertices (13,258 vs 13,026) at *higher* mean candidate degree (2.63 vs 2.39); only 328 vertices MATLAB connects are left with zero Python candidates, and only 604 of the 6,726 missing edges touch such a vertex. Python pairs each vertex with **different neighbors** than MATLAB. The watershed diagnostic `frontier_origins_without_candidates: 2464` is a secondary symptom, not the driver.

**Conclusion:** the residual edge deficit is a **watershed candidate-generation adjacency divergence** (basin-meeting order / tie-break sensitivity on a near-bit-identical energy field), localized to `matlab_get_edges_by_watershed.py` / `matlab_watershed_heap.py` — *not* selection (`selection.py`/`cleanup.py`) and *not* Network. The ~57% candidate overlap here is the same phenomenon as the ~63.5% ADR 0012 ownership-map agreement. Network parity cannot close until the watershed generates MATLAB's adjacency set.

**Next-level hypotheses (for an instrumented watershed trace on the crop):** (H1) heap flooding tie-break order → basins meet different neighbors first; (H2) basin seeding/label assignment differs; (H3) at >2-basin meeting voxels a different neighbor pair is recorded; (H4) a per-vertex neighbor cap keeps a different subset; (H5) `[Y,X,Z]`/F-order remap mirrors adjacency for a subset.

**Evidence artifacts:** offline read-only diagnostics `workspace/scratch/edge_funnel_probe.py` (funnel diagnostics) and `workspace/scratch/edge_gap_split.py` (generation-vs-prune split + vertex-level characterization); per-stage proofs under `workspace/runs/oracle_180709_E/canonical_full_v4/03_Analysis/exact_proof_<stage>.json`.

### Current crop Energy evidence guard (2026-06-24)

- **Freshness command:** `slavv parity inspect-energy-evidence --run-root workspace/runs/oracle_180709_E/crop_M_exact`
- **Current result:** **valid** (checkpoint at `02_Output/python_results/checkpoints/checkpoint_energy.pkl`; parity `stage_metrics.energy` completed).
- **Proof oracle:** `workspace/oracles/180709_E_crop_M_v2` (`batch_260624-105705`).

### Latest crop Energy proof vs oracle v2 (2026-06-24)

- **MATLAB vectorization:** `batch_260624-105705` (~3h, lattice-6000) → promoted `180709_E_crop_M_v2`.
- **Proof:** `prove-exact --stage energy --oracle-root workspace/oracles/180709_E_crop_M_v2` → **FAIL** (`exact_proof_energy.json`).
- **`scale_indices`:** **0** mismatches (strict-zero vs v2).
- **`energy`:** **3,810,126** mismatches under strict `np.equal`; first `(0,0,0)` scale 90 agree; \|Δ\|≈1.1×10⁻¹⁴ (3 ULP).
- **ULP triage** (`workspace/scratch/energy_ulp_triage_v2.json`): on scale-agreeing mismatches — median **4 ULP**, p90 **13 ULP**, max \|Δ\| **1.99×10⁻¹¹**; **384,178** voxels bit-identical. Stored `best_energy.npy` matches single-octave Python replay on sampled mismatches (writer persistence ruled out).
- **Voxel probes vs v2:** `(0,0,0)` ≤8 ULP pass; `(40,83,116)` 47 ULP fail (stored==replay); `(12,0,0)` stored vs replay 8 ULP but both disagree with MATLAB.
- **Classification:** accumulated **NumPy vs MATLAB MKL** float drift at matching scales ([ADR 0010](../../adr/0010-random-component-parity-suite.md) documents ≥1 ULP IFFT floor; crop volume shows median 4 ULP). **No localized Python fix** — this is library-level FP non-associativity, not a logic bug.
- **✅ Resolved ([ADR 0011](../../adr/0011-energy-float-certification-policy.md) accepted 2026-06-24):** certification uses `np.allclose(rtol=1e-7, atol=1e-9)` on continuous float fields, strict on `scale_indices`/topology. `prove-exact --stage energy` now **PASSES** (exit 0, pass_rate 1.0). Pure ULP was rejected — it explodes near zero (36,074 false fails at 48 ULP, max 72,343 ULP, despite \|Δ\|≤2×10⁻¹¹). The generic comparator also applies `np.allclose` to other float fields (e.g. `lumen_radius_microns`, max \|Δ\|=7.1×10⁻¹⁵).
- **Advisory ULP gate** (`slavv parity prove-energy-ulp --max-ulps N`): strict `scale_indices`, bounded float ULP — **not** certification. Crop vs v2 @ `max_ulps=8`: **FAIL** (~755k voxels >8 ULP). @ `max_ulps=48`: **99.11%** pass rate (37,174 failures — denormal/near-zero energies with large bit-space ULP, \|Δ\| still ≤2×10⁻¹¹).
- **Downstream oracle v2:** `ensure-oracle-artifacts --stage all` passes (vertices 13,706; edges 15,511; network 10,722 strands).
- **Policy:** [ADR 0011](../../adr/0011-energy-float-certification-policy.md) **Accepted** — Option B refined to `np.allclose` (rtol=1e-7, atol=1e-9) on continuous floats; strict scales/topology.
- **Historical next step (superseded post-v6):** This old note predated ADR 0012 evaluated proofs. Current closure uses per-stage `prove-exact --stage edges` and `--stage network` with `adr0012_evaluated: true`; do not use `prove-exact-sequence` strict-field as the ship gate.

### Latest crop Energy proof (2026-06-22)

- **Writer:** `resume-exact-run --force-rerun-from energy --stop-after energy --skip-preflight --n-jobs 1`; `max_voxels_per_node_energy: 6000` (lattice `[3,3,2]`, 821 chunks).
- **Proof:** `prove-exact --stage energy` → **FAIL** (`03_Analysis/exact_proof_energy.json`).
- **First failure:** `energy.energy` @ `(0,0,0)` — scales agree (90); energy differs at ULP (~10⁻¹⁴) under strict `np.equal`.
- **Mismatch counts** (`exact_mismatch_energy.json`):
  - `energy`: **3,823,893** (max |Δ|≈26.4) — ~3,804,481 voxels have **matching scale** but fail bit-identical float compare (median |Δ|≈1.4×10⁻¹⁴).
  - `scale_indices`: **19,412** (0.46% of volume; max |Δ|=72) — first @ `(61,81,0)` MATLAB 44 vs Python 46.
  - `lumen_radius_microns`: **8** (machine epsilon).
- **Probe note:** One-voxel probe at `(12,0,0)` passes at `atol=1e-10` but fails strict `np.equal` on full volume — certification bar is bit-identical, not probe tolerance.
- **Adaptive probes:** `03_Analysis/energy_probe_requests.json` (3650 mismatch groups).
- **Next:** Triage scale-winner disagreements first; then float64 bit-identical path for scale-agreeing voxels. No downstream crop refresh.

### Fresh scale-winner triage (2026-06-24, post-writer)

Cross-octave Python replay on all **31** `scale_indices` mismatches (`workspace/scratch/fresh_scale_mismatch_triage.json`):

- **31/31** `cross_octave_reduction` — replayed Python winner matches **stored** `best_scale.npy` (not stale checkpoint drift).
- **0/31** replay matches MATLAB oracle winner.
- Scale delta histogram vs MATLAB: **14** at −1, **13** at +1 (remaining outliers at −5, −2, +7, +26).
- Example `(33,80,133)`: MATLAB 14, Python stored/replay **13**; octave-1 candidate wins replay (−20.15) over octave-5 scale 92 (−18.25).
- **Implication:** remaining scale gaps are live cross-octave winner math vs MATLAB, not writer persistence. Next fix surface: reduction/tie-break + invalid-octave candidate handling (probe shows `global_scale=-1`, energy `0.0` on octaves 3–4).

### MATLAB-backed cross-octave probe (top 3 voxels, 2026-06-24)

Batch replay: `workspace/scratch/cross_octave_top3/cross_octave_reduction.json` (R2019a, 15 per-octave probes).

| Voxel (Z,Y,X) | Oracle `scale_indices` | Python stored/replay | MATLAB live replay | Python↔MATLAB replay |
| --- | --- | --- | --- | --- |
| (33,80,133) | **14** | 13 | 13 | **agree** |
| (40,83,116) | **13** | 12 | 12 | **agree** |
| (33,83,131) | **13** | 12 | 12 | **agree** |

All three classify as **`matlab_oracle_state_path`**: promoted oracle plane is **exactly +1** vs fresh MATLAB batch replay on the same crop config; Python stored `best_scale.npy` matches live MATLAB replay on every sample.

**Revised implication:** the 31 `scale_indices` strict-zero failures may be dominated by **oracle vector indexing / promotion drift** (±1 vs live `get_energy_V202` replay), not Python cross-octave math. Next: audit oracle `scale_indices` plane convention vs `matlab_vector_loader` one-based shift and crop batch vintage before another Energy writer rerun.

### Oracle HDF5 scale-plane trace (top 3 voxels, 2026-06-24)

Source: `batch_260527-220010` HDF5 plane 0 (`get_energy_V202` writes `energy_chunk_scale_min + sum(octave_at_scales < current_octave)` — **1-based global** per `external/Vectorization-Public/source/get_energy_V202.m`).

| Voxel (Z,Y,X) | Raw HDF5 plane0 | After loader `−1` | Python stored | MATLAB live replay |
| --- | --- | --- | --- | --- |
| (33,80,133) | 15 | 14 | 13 | 13 |
| (40,83,116) | 14 | 13 | 12 | 12 |
| (33,83,131) | 14 | 13 | 12 | 12 |

**Loader is not double-subtracting:** `matlab_vector_loader` applies exactly one `one_based` shift. Identity: `raw_hdf5 − 2 == python_stored == matlab_live_replay` (0-based global); `raw_hdf5 − 1 == prove-exact oracle surface`.

**Conclusion:** promoted oracle `scale_indices` reflects the **May 27 full-volume MATLAB batch** (pre–lattice-6000 / pre–IFFT-fix crop writer), which is **+1 (0-based)** vs current Python and fresh MATLAB batch replay on the same voxels. Remediation path: **promote a fresh crop oracle** from a lattice-`6000` MATLAB run (or accept probe-surface proof) before chasing Python reduction code.

Historical crop Energy evidence (2026-06-21, superseded writer state):

- **Coarse-source / interpolation-mesh contract**: MATLAB `get_energy_V202` local ranges use `1 + floor(offset/rf) : 1 + ceil((offset + write_count - 1)/rf)` on the **padded FFT grid** (`fourier_transform_V2` output), not the strided read shape. Python had been clamping to `original_chunk.shape`, dropping one padded row on crop octaves 3–5 (e.g. octave 4 chunk 0: requested `(27,27,14)` vs retained `(26,26,13)`). Fixed via `_matlab_coarse_local_slices` in `matlab_get_energy_v202_chunked.py`.
- **Resolution-factor rounding (authoritative)**: MATLAB `get_energy_V202` line 116 uses `round(worst_resolution_to_downsample ./ resolutions_at_octave)` (positive half-up). Not `floor(L/(v*2.5))`.
- **Manual source audit (2026-06-22)**: MATLAB V202 and the Python exact route agree on octave consolidation, chunk geometry, symmetric even FFT padding, padded-grid local ranges, interpolation mesh construction, negative-only Energy handling, and first-winner min projection. V200's active principal-Energy code clips a positive third component and then uses an **unweighted** sum; the `[1.5, 1, 0.5]` weighted expression is commented out. Do not claim magnitude-descending eigenvalue sorting: MATLAB calls `eig`, while Python calls `eigh`; their ordering remains an empirical crop-probe and strict-proof check.
- **Coarse-range invariant**: Python now raises if a MATLAB-requested local range exceeds the padded FFT extent rather than silently shortening the interpolation source. The crop boundary regression covers the MATLAB-requested `(27,27,14)` support from raw `(26,26,13)` input.
- **Probe artifacts**: Python `workspace/scratch/crop_coarse_slice_probe_python.json`; MATLAB companion `workspace/scratch/matlab/probe_coarse_slice.m`.
- **Regression coverage**: `tests/unit/pipeline/energy/test_hessian_downsample.py::{test_exact_crop_coarse_slice_retains_padded_fft_support_not_strided_read,test_exact_crop_coarse_slice_octave4_chunk0_matches_matlab_local_ranges}`.
- **Prior proof (1M-chunk run, 2026-06-21)**: `prove-exact --stage energy` **FAIL** — 4,010,103 voxel diffs, max |Δ|≈135.4. First scale mismatch: `(12,0,0)` scale 54 / −13.52 (MATLAB) vs 61 / −16.45 (Python), octave `rf=[2,5,5]`.
- **Chunk-lattice root cause (2026-06-21)**: `max_voxels_per_node_energy` was `1_000_000` (lattice `[1,1,1]`, 1 chunk) vs MATLAB oracle `6_000` (lattice `[3,3,2]`, 18 chunks). Run-local `validated_params.json` restored to `6000`. With `6000`, one-voxel probe at `(12,0,0)` matches oracle: scale 54, energy −13.52067537392248.
- **Probe artifacts (voxel)**: `workspace/scratch/crop_voxel_12_0_0_probe_python.json`, `workspace/scratch/crop_voxel_12_0_0_probe_matlab.json`; helper `slavv_python/pipeline/energy/parity_energy_voxel_probe.py`.
- **Regression coverage (voxel + lattice)**: `tests/unit/pipeline/energy/test_voxel_probe.py` (3 tests, incl. lattice `[3,3,2]` and oracle match at `(12,0,0)`).
- **Ruled out for crop**: octave `unique(...,'last')` consolidation (scale groupings identical); coarse-slice truncation alone (fix did not move winner on 1M-chunk run).
- **Stale broadcast failure resolved**: earlier `(64,27,8)` into `(65,27,8)` crash came from unbounded slices; padded-bound slices stay inside the FFT grid on `(64,256,256)`.

### 🟡 2026-06-17: Energy Parity Discoveries (Crop Harness)

- **Eigenvalue Ordering**: The active V200 source calls `eig` and applies its special third-component clip in returned order; it does not sort eigenvalues by magnitude. Python uses `np.linalg.eigh` in the same returned-component role. Ordering is not certified by source inspection alone and remains covered by the crop one-voxel probe and strict Energy proof.
- **Resolution Factors**: MATLAB `get_energy_V202` uses `round(1/2.5 ./ resolutions_at_octave)` (positive half-up). Python now uses `floor(x+0.5)` in `matlab_energy_filter_v200.py`.
- **Validation Whitelist**: Identified that `validate_parameters` in `validation.py` was stripping `comparison_exact_network` and other orchestration keys during the `RunContext.prepare` phase, causing the pipeline to fall back to the standard "Paper" route. Whitelisted these keys to ensure the "Innovation" path is correctly triggered.
- **FFT Symmetry**: Verified that `_ifftn_matlab_symmetric` manual enforcement matches `np.fft.ifftn().real` and correctly handles Fortran-order raveling for conjugate-pair matching.

### Random component parity (2026-06-22)

Seeded white-noise differential suite ([ADR 0010](../../adr/0010-random-component-parity-suite.md), [PARITY_RANDOM_COMPONENT_SUITE.md](../workflow/PARITY_RANDOM_COMPONENT_SUITE.md)):

- **Structural gate (green):** 128 linspace contexts; 16 lattice/boundary `interp3` queries per case; Energy `padded_shape_yxz`, sample coordinates, and `valid`.
- **IFFT floor:** With a **byte-identical** MATLAB complex spectrum loaded in Python, `_ifftn_matlab_symmetric` vs MATLAB `ifftn(...,'symmetric')` differs by **1 ULP** at sample voxels — NumPy vs MKL FFT, not the symmetry mask.
- **Matching kernel:** `scipy.special.jv` vs MATLAB `besselj` drifts without the promoted `matlab_random_matching_reference.json` fixture on the Python runner.
- **Hessian floats:** Reported as advisory `hessian_diagnostics` (max ULP per case); they do **not** gate CI. Crop/canonical `prove-exact` remains the strict Energy certification surface.

*Status*: Incorporated into the current worktree, but not yet certified. See [Latest crop Energy proof (2026-06-22)](#latest-crop-energy-proof-2026-06-22).

**Implementation hardening:** Spec at [random-component-parity-hardening-spec.md](../../investigations/random-component-parity-hardening/random-component-parity-hardening-spec.md) (Phase 0 complete: spec landed + baseline captured + unit tests green). Future changes to the suite (decomposition, models, separation of structural gate from advisory) should follow that spec. Baseline artifacts in `workspace/scratch/random_component_baseline/`.

**Champion edges baseline (informal, not cert bar):** `workspace/runs/oracle_180709_E/validation_strel_fix_output_v29`

### Cold-start protocol

If resuming exact parity work from a fresh thread:

1. Read **[ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk)** (pass/fail, claim root, residual). Do **not** use the session diary or mid-file historical notes as status.
2. Read **[.claude/HANDOFF.md](../../../.claude/HANDOFF.md)** for commands only.
3. `slavv jobs list` — no concurrent writer on the target `--dest-run-root`.
4. `slavv parity ensure-oracle-artifacts --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage all --no-repair` (and same for `180709_E_full_v2` before canonical work).
5. **Residual loop (primary):** full-volume Candidate Set **join displacement** at degree-excess (see ONE TRUTH ablation). Crop frontier/generation/re-selection are **regression guards** (closed). Prefer `scripts/edge_selection_funnel_probe.py` on the claim root; keep crop `watershed_frontier_diff.py` / `watershed_candidate_gap_probe.py` as guards.
6. **Claim surface:** name and counts only in ONE TRUTH (currently `canonical_full_v16` lineage until ONE TRUTH moves). Historical audits `v6`…`v15` are not the open residual surface.
7. Harness **fail loud** if ADR 0012 cannot evaluate (maps missing) — not a valid closure attempt.
8. Capture evidence per [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md). Re-synthesize HANDOFF if ONE TRUTH changes.

Use `--monitor` on long reruns ([PARITY_JOB_MONITORING.md](../workflow/PARITY_JOB_MONITORING.md)).

Scratch diagnostics: prefer promoted `scripts/*` probes; historical `workspace/scratch/edge_gap_split.py`, `workspace/scratch/matlab_edge_instr/`.

### Operator commands

See **[.claude/HANDOFF.md](../../../.claude/HANDOFF.md)** for the current command block (full residual → successor full claim run). Legacy commands below are **audit-only**.

```powershell
# --- Legacy audit (2026-07-04 checkpoints; do not use for Phase 1 closure) ---
slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M_v2

slavv monitor --run-dir workspace/runs/oracle_180709_E/canonical_full_v4
```

---

## 📚 Compound learnings (parity-related)

Curated index of solved problems under `docs/solutions/` (from `/ce-compound`). Search all solutions via YAML frontmatter (`module`, `tags`, `problem_type`); see [docs/solutions/README.md](../../solutions/README.md).

| Topic | Doc |
|-------|-----|
| MATLAB energy HDF5 + `promote-oracle` | [matlab-v200-energy-hdf5-oracle-loader.md](../../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md) |
| Detached exact-run jobs | [detached-exact-run-jobs.md](../../solutions/parity/detached-exact-run-jobs.md) |
| Run/proof evidence template | [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md) |
| Sparse Meshgrid Memory Optimization | [sparse-meshgrid-memory-optimization.md](../../solutions/parity/sparse-meshgrid-memory-optimization.md) |
| MATLAB Stride Phase Lead | [matlab-stride-phase-lead.md](../../solutions/parity/matlab-stride-phase-lead.md) |
| Vertex NMS structuring element (float radii) | [vertex-structuring-element-float-radius.md](../../solutions/parity/vertex-structuring-element-float-radius.md) |
| Edge watershed faithfulness (seeds=2, no conflict painting) | [edge-watershed-matlab-faithfulness.md](../../solutions/parity/edge-watershed-matlab-faithfulness.md) |
| Curated vertices rank-ramp energies | [curated-vertices-rank-ramp-energies.md](../../solutions/integration-issues/curated-vertices-rank-ramp-energies.md) |

_Add rows here when a new compound doc is parity-relevant; do not duplicate full write-ups in this file._

---

## 🏆 June 2026 Memory Breakthrough (Canonical Scale-up)

A second major architectural breakthrough was achieved in June 2026, resolving persistent **ArrayMemoryError** blocks that prevented Phase 1 certification of the full 512x512x64 canonical volume.

### The Solution: Incremental Best-Scale Engine
- **4D Array Elimination**: Refactored `matlab_get_energy_v202_chunked.py` to discard the large per-chunk 4D energy stack. The engine now updates the `best_energy` and `best_scale_index` volumes incrementally within the multi-scale loop. Peak memory usage dropped from **~300 MiB/thread to ~10 MiB/thread**.
- **Kernel Pre-computation**: Optimized the Hessian backend to pre-compute scale-independent derivative kernels (9 complex/double volumes per chunk) once. This eliminated redundant allocations that were fragmenting the heap.
- **Explicit GC Control**: Integrated `gc.collect()` and explicit `del` of large DFT products to ensure immediate reclamation of working memory.
- **Outcome**: Enabled stable multi-scale processing of the full canonical volume on hardware with limited physical RAM (e.g. 16GB), unblocking full-volume Energy certification on `180709_E`.

### Bit-Perfect Mathematical Refinements
The memory-safe engine simultaneously absorbed two final mathematical refinements discovered during crop-harness isolation:
- **MATLAB `linspace` Roundoff**: Replaced standard arithmetic meshes with MATLAB-accurate `linspace` endpoints. This preserved tiny fractional drifts (e.g. $10^{-16}$) that were causing interpolation boundary flips in the MATLAB `interp3` engine.
- **Raw Intensity Preservation**: Forced the exact-route pipeline to skip all normalization and clipping steps, ensuring bit-perfect parity with the MATLAB TIFF/HDF5 source values.

---

## 🏆 Historical high-water mark breakthrough (May 2026)

A major architectural breakthrough was achieved in May 2026, dramatically narrowing the discrepancy gap in edge candidate generation.

This section is historical context only. See [Active Phase 1 operations](#-active-phase-1-operations) and [Active blockers](#-active-blockers) for current proof status.

### The Solution: Parameter Alignment & NaN Stability
- **Parameter Alignment (v29)**: Discovered that the MATLAB oracle was generated with `edge_number_tolerance = 4`, while Python was hardcoded to 2. Aligning this parameter allowed high-degree vertices (hubs) to initiate sufficient exploratory traces.
- **NaN Stability**: Fixed a floating-point instability where multiplying `-Inf` (vertex priority) by `0.0` (directional suppression factor) created `NaNs`, leading to incorrect seed selection in subsequent iterations.
- **Precision Alignment (May 22)**: Implemented bit-accurate tie-breaking using exact equality (`==`) and Fortran-order linear index priority. Removed all remaining `float32` casts in the expansion frontier.
- **Tightened Filtering**: Implemented hard distance cutoffs ($d/R > 3.0$) and aligned edge influence sigmas to exactly $2/3$.
- **Outcome**: Successfully reached the **88.7%** match rate milestone (1062/1197 pairs). This remains an informal edge baseline, not the Phase 1 certification bar.

### Historical mathematical impact
| Metric | Previous Baseline (v28) | High-Water Mark (v29) |
| :--- | :--- | :--- |
| **Matched MATLAB Pairs** | 958 | 1062 |
| **Total Match Rate** | 80.0% | 88.7% |
| Missing Pairs | 239 | 135 |
| Over-generated Pairs | 263 | 371 |

---

## ⚖️ Exact Parameter Fairness Gate

To guarantee a fair mathematical race, all exact-route experiments must maintain structural lock-step between Python and MATLAB configuration inputs. This is validated via the **Parameter Diffusion Matrix**.

Every compliant proof run maintains three persistent JSON manifests under `01_Params/`:

1. **`shared_params.json`**: The authoritative overlap of settings that must exist in both MATLAB and Python.
2. **`python_derived_params.json`**: Internal Python-only pipeline management levers.
3. **`param_diff.json`**: The cryptographic hash bridge that proves zero illicit divergence occurred between the split configuration states.

### Locked Mathematical Constants
The audit system mandates these exact value bindings (derived from source-hardcoded MATLAB constants):
- `step_size_per_origin_radius = 1.0`
- `max_edge_energy = 0.0`
- `distance_tolerance_per_origin_radius = 3.0`
- `edge_number_tolerance = 4` (Corrected from 2)

---

## 🛠️ Verified Infrastructure Fixes

The core codebase has absorbed the following permanent fixes, ensuring structural stability:

*   ✅ **Watershed Orientation Realignment**: Standardized the Edges stage on internal [Y, X, Z] orientation with Fortran contiguity. Input volumes are now explicitly transposed before watershed processing, and result maps are re-mapped to physical [Z, Y, X] for artifact persistence (2026-06-13). **⚠️ CORRECTION (2026-06-25):** the *explicit pre-transpose in `generate_watershed_candidates`* described here was a bug — the engine already reorients internally, so this produced a DOUBLE transpose. Removed in commit `e9dcc141`; the engine's internal single transpose is the sole, correct realignment. See [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md).
*   ✅ **Vertex Painting Rounding**: Replaced Python's built-in `round()` (round-to-even) with bit-perfect round-half-up logic in the painting and candidate filtering loops, preventing selection divergence at .5 coordinate boundaries (2026-06-13).
*   ✅ **Mesh Offset Alignment**: Fixed a 3-pixel coordinate shift in energy interpolation by removing saturated subtraction from chunk offsets and updating `_matlab_zero_based_linspace` to handle explicit local starts (2026-06-13).
*   ✅ **Exact-route intensity scale**: Skip min-max normalization and ignore `intensity_limits` clipping when `comparison_exact_network=True`, preserving the raw TIFF/HDF5 values used by the MATLAB crop oracle (2026-06-02).
*   ✅ **Energy downsample stride phase**: `_downsample_volume` uses MATLAB `get_starts_and_counts_V200` last-chunk alignment `start = (size-1) mod rf` per axis (whole-volume single chunk). Verified analytically (rf=9 → start 3; rf=20 → start 15) and empirically (origin energy −20.3757) (2026-05-29).
*   ✅ **HDF5 scale-index base**: `matlab_vector_loader` reads MATLAB global scale subscripts as 1-based (`one_based=True`), converting e.g. 91→90 for Python 0-based indexing (2026-05-29).
*   ✅ **Energy axis permutation**: `energy_axis_permutation` param permutes `microns_per_voxel` and `pixels_per_sigma_PSF` to the working image axis order so per-axis resolution factors map to the correct dimensions; added to `EXACT_ALLOWED_ORCHESTRATION_PARAMETER_KEYS` (2026-05-29).
*   ✅ **Upsample/`interp3` coordinate consistency**: exact-route chunk mesh uses MATLAB `get_energy_V202` local ranges and `mod(offset, rf)/rf` coordinates for each writing chunk (2026-06-02).
*   ✅ **MATLAB `interp3` invalid-neighbor semantics**: exact-route interpolation now propagates positive-weight `Inf` coarse-energy neighbors instead of finite-only reweighting, matching MATLAB's false-candidate suppression at downsampled octave boundaries (2026-06-02).
*   ✅ **MATLAB `linspace` mesh roundoff**: exact-route chunk meshes use MATLAB-style `linspace` endpoints rather than arithmetic `arange / rf`; this preserves tiny fractional drift such as `9.000000000000002`, which changes `interp3` `Inf` propagation at octave boundaries (2026-06-02).
*   ✅ **Double-Precision Energy Alignment**: Forced all core watershed maps (`energy_map_temp`, `vertex_energies`) and neighborhood penalty calculations to `float64`. This prevents precision-induced tie-breaking divergences where `float32` would collapse distinct energy values into identical bits, causing different seed selections than MATLAB's `double`.
*   ✅ **Bit-Accurate Tie-Breaking**: Replaced `np.isclose` with exact bitwise equality and added linear index priority to the frontier priority queue, matching MATLAB's hub vertex exploration behavior.
*   ✅ **Hard Distance Cutoff**: Implemented the MATLAB-exact $d/R > 3.0$ expansion cutoff in the watershed loop.
*   ✅ **Edge Influence Alignment**: Updated default `sigma_per_influence_edges` to $2/3$, aligning with MATLAB's conflict painting regions.
*   ✅ **Stable Frontier Splicing**: Verified and anchored the `available_locations` insertion logic to mirror MATLAB's `find(..., 'last')` and `find(..., 'first')` behavior, ensuring stable FIFO/LIFO handling for identical energy seeds.
*   ✅ **Backtracking Pointer Correction**: Fixed reverse-index logic, allowing trace recovery back to the origin vertex.
*   ✅ **Stable Discovery Sorting**: Forces deterministic processing orders matching MATLAB's explicit energy quality sorting.
*   ✅ **Trace Order Randomization**: Anchored native shuffling to a stable, reproducible seeded RNG generator.
*   ✅ **Distance Normalization (r/R)**: Corrected physical energy penalties to scale relatively to the vessel's radius ($R$).
*   ✅ **Strel Offset Alignment**: Realigned watershed structuring element (strel) offsets and loops to match the (Z, X, Y) universe layout, fixing major distance-penalty errors.
*   ✅ **Filtering Logic Reordering**: Realigned the cleanup sequence (Crop first $\rightarrow$ Pair Cleanup second) to protect valid pairs from phantom blocking.

## 2026-06-14 Update: Systemic float64 Enforcement & Spatial Alignment

*   **Systemic Precision Alignment**: Identified and replaced all `float32` and `np.float32` casts with `float64` across Energy, Vertices, and Edges stages. This resolves precision-induced divergences where rounding (e.g., `np.floor(pos + 0.5)`) or normalization would deviate from MATLAB's `double`.
*   **Bessel Sum Chunking**: Implemented a chunked computation loop for `jv` sums in `_matching_kernel_dft` to keep peak memory footprint minimal during kernel generation, preventing `ArrayMemoryError` on canonical volumes.
*   **Preprocessing Parity**: Fixed `preprocess_image` to respect `comparison_exact_network=True` by using `float64` and skipping min-max normalization, ensuring raw TIFF/HDF5 values are preserved for the Hessian engine.
*   **Spatial Shift Resolution**: Discovered a (0, 15, 15) pixel shift in the energy map. Root cause: MATLAB's "Last Chunk Alignment" rule in `get_starts_and_counts_V200` shifts the reading start to align with the final pixel. For a single-chunk volume (like the crop volume at coarse octaves), this results in a 15-pixel lead in the coarse grid. Reintroduced `sat_sub` in Python's `matlab_get_energy_v202_chunked.py` to correctly replicate this shifting behavior.

---

## 🚀 Active blockers

> Live residual detail and counts: [ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk) only.

1. **Full Edge Set residual (generation join displacement)** — extra watershed join displaces the oracle pair at degree-excess under equal post-resample max (see ONE TRUTH ablation). Cleanup is faithful. **Production fix = join/emission**, not cleanup reorder.
2. **Phase 1 ship gate = Network ADR 0012 multiset** — Edges evaluated PASS on claim root; Network FAIL until Edge Set multiset matches (MATLAB-edge isolation exact). Do not reopen Network as an independent port.
3. **Crop / frontier / cleanup** — regression guards only (closed on re-selection; generation gap 0; golden trace match; cleanup comparator green).

**Superseded guidance:** “100% parity”, “>95% match”, “block on 80% crop overlap”, “crop one-pair swap is the open loop”, or strict-field fallback as closure verdict. Only **evaluated** ADR 0012 on **both** Edges and Network count; see [ADR 0012 post-v6 addendum](../../adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-12-post-v6-residual--network-is-the-open-ship-gate).

---

## Session diary (historical — not live status)

The dated banners below are a chronological investigation log. **Do not read them as current status.** Current status is [ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk) only.

> 🟢 **2026-07-15 (full residual localized + ablated) — crop closed; full one-pair is generation-extra displacement at degree-excess; Phase 1 OPEN.**  
> **Crop:** re-selection **15,511 / 15,511** pair overlap (closed).  
> **Full funnel:** loss only at **degree-excess**; equal post-resample max metric `-4.870152991855598` (shared bottleneck sample); Python keeps earlier row `cand 46698` `(26444,38584)` and drops `cand 56786` `(34897,38584)`.  
> **Cleanup is faithful:** `compare_clean_edge_pairs_matlab.py` on full Python surface → **0** mismatches (pairs/degree/cycle). Do **not** add cleanup secondary-key hacks (breaks MATLAB≡Python on same surface; endpoint-descending already rejected).  
> **Falsifiable claim:** Python Candidate Set contains **extra** watershed join `(26444→38584)` (origin **26444**, not in oracle final) that shares post-resample max energy with oracle pair `(34897→38584)` and **outranks** it under stable length→metric→index order, so degree-excess removes the oracle pair.  
> **Ablation proof:** drop only candidate index **46698**, re-run `select_and_finalize_edge_set` → undirected pair overlap **69,500 / 69,500** (only_py=0, only_mat=0). Production fix = stop emitting that join (or match MATLAB join/emission so it never displaces), **not** a cleanup reorder.  
> **v16** evaluated Edges PASS / Network FAIL still the on-disk claim until generation fix + re-proof.

> 🟡 **2026-07-15 (canonical `v16` full audit) — Edges evaluated green; Network still FAIL by one strand; Phase 1 OPEN.** `canonical_full_v16` completed Edges→Network. **Evidence:** `03_Analysis/exact_proof_edges.json` **`passed: true`** (ADR 0012 evaluated: ownership **99.999863%**, trace failures **0**, connections **69,500 / 69,500**). **`exact_proof_network.json` `passed: false`** — strand endpoint-pair multiset mismatch (Python **48,048** vs MATLAB **48,049**). Do **not** treat approximate strand-count % as ADR 0012 Network pass; the bar is order-independent **multiset equality**. Residual is the full one-pair edge-set swap class (see banner above). Energy and Vertices remain certified.

> 🟢 **2026-07-14 (MATLAB post-watershed finalization parity) — crop final strict count is now essentially closed: one pair swap remains.** Scratch MATLAB instrumentation in `workspace/scratch/matlab_edge_instr/run_edges_standalone.m` now dumps post-watershed finalization stages. It proved Python raw candidates already match MATLAB raw watershed candidates exactly (**19,225 / 19,225**); the prior **3,714** "extras" are MATLAB raw candidates that `vectorize_V200` removes later, not Python over-generation. First divergence was `crop_edges_V200`: Python was cropping raw persisted traces, while MATLAB first runs `resample_vectors`, re-samples energy/size from the maps, smooths, then applies `crop_edges_V200` using MATLAB unsigned casts (`uint64`/`uint16`/`uint8`, nearest integer for positive values). **Fix:** `prefilter_edge_indices_for_cleanup_matlab_style()` now performs the MATLAB resample/map-resample path, `_matlab_crop_edges_v200()` mirrors unsigned casts, and `_choose_edges_matlab_style()` carries the resampled traces/energies/metrics into cleanup. Comparator result: MATLAB `clean_edge_pairs`, degree pruning, and cycle pruning all have **0** row-index mismatches on the resampled post-crop surface. Refreshed `crop_M_exact_v3` Edges: candidates remain **15,511 / 15,511** with **19,225** raw candidates; final selected edges are **15,511** vs MATLAB **15,511**, with **15,510 / 15,511** overlap (**1** missing, **1** extra: Python keeps `[4043, 6281]` while MATLAB keeps `[4212, 6281]`). Crop `prove-exact --stage edges` reports strict counts equal but **ADR 0012 not evaluated** because the proof harness does not discover the refreshed ownership map artifact from this resume path; do not use that fallback proof as closure evidence. Next: triage the single equal-metric degree-pruning pair swap, then decide whether the crop movement justifies a fresh full Edges→Network audit root.

> 🟡 **2026-07-14 (canonical `v15` full audit) — Edges are exact-count/evaluated green; Network is one strand short.** `canonical_full_v15` was seeded from `canonical_full_v10` certified Energy/Vertices and reran Edges→Network with current post-watershed finalization code. **Edges ADR 0012 PASS evaluated:** Python **69,500** vs MATLAB **69,500**, ownership-map agreement **99.999863%** (`5,843,205 / 5,843,213` MATLAB-claimed voxels), trace failures **0** over **69,499** compared traces. **Network ADR 0012 FAIL:** Python **48,048** vs MATLAB **48,049** strands, a one-strand residual. Edge-pair multiset delta is one swap: MATLAB has `(34897, 38584)`, Python has `(26444, 38584)`. Network endpoint delta: MATLAB has `(34897, 55337)` and `(26444, 41666)`, Python has `(41666, 55337)`. A trial endpoint-descending tie-break in `prepare_candidate_indices_for_cleanup()` fixed the known full swap pattern locally but regressed the crop from one pair swap to **7 missing / 9 extra**, so it was reverted. Next target is a narrower equal-metric degree-pruning tie analysis around the specific crop/full residuals, not another broad finalization rewrite.

> 🟡 **2026-07-14 (cleanup row-order comparator + displacement quantified) — MATLAB cleanup parity is now proven on the Python candidate surface, but final counts do not move.** Added `scripts/compare_clean_edge_pairs_matlab.py` to export the crop post-crop candidate list and call MATLAB `clean_edge_pairs`, `clean_edges_vertex_degree_excess`, and `clean_edges_cycles`. Initial comparator found six adjacent `clean_edge_pairs` row swaps caused by sorting on persisted float32 metrics; **fix:** `prepare_candidate_indices_for_cleanup()` now sorts by double-precision `max(edge_energies)` with MATLAB's NaN→`-1000` rule. Comparator after the patch: `clean_edge_pairs`, degree pruning, and cycle pruning all have **0** row-index mismatches. Crop candidate generation remains **15,511/15,511** with **3,714** extras, and final selected edges remain **15,727** with overlap **15,362** (missing **149**, extra **365**). A MATLAB `get_edges_V300` read/write chunk-eligibility diagnostic was added to `scripts/edge_selection_funnel_probe.py`; on the crop, MATLAB uses a single edge chunk and the filter drops **0/19,225** candidates, so chunk emission windows do **not** explain the extras. New aggregate displacement output shows degree pruning loses **103** MATLAB pairs, **99** with incident surviving extras and **97** with better-metric incident extras; cycle pruning loses **32** more, all **32** with earlier and better incident extras. Extra-candidate origins are diffuse (top candidate-extra origins only emit 2 extras each), while final extras concentrate more at boundary-adjacent / MATLAB-zero-degree vertices. A pure geometry boundary cutoff was rejected: dropping candidates touching vertices within 1 voxel of a boundary worsens final overlap to **14,984** and missing pairs to **527**. An oracle-aware zero-degree-boundary upper bound only reaches **15,377** overlap / **134** missing / **238** extras, so boundary zero-degree vertices are a symptom, not a sufficient root cause. Bounded `watershed_frontier_diff.py` regeneration matches MATLAB iteration/seed rows through iteration **30,000**, well past the retired 13,761 split. Conclusion: cleanup row order, chunk eligibility, broad boundary filtering, and the old frontier split are regression guards; remaining residual is over-strong extra candidates displacing MATLAB pairs during faithful degree/cycle cleanup.

> 🟢 **2026-07-13 (LUT unit-vector frontier parity) — crop watershed generation now matches the MATLAB golden trace; final selection remains open.** The broad epsilon clamp was too aggressive: the next split at iter **25,495** showed MATLAB keeps a scale-29 tiny-positive LUT dot (`2.78e-17`) selectable while still skipping an exact-zero orthogonal dot at iter **22,421**. **Fix:** `_matlab_frontier_adjusted_neighbor_energies()` now accepts neighbor `unit_vectors` from the MATLAB strel LUT instead of recomputing/renormalizing from offsets; negative dots still clamp to zero, exact zero still yields MATLAB-style `-Inf * 0 -> NaN` skip. No-writer `scripts/watershed_frontier_diff.py` now reports **`status: match`** against the crop MATLAB trace. Refreshed `crop_M_exact_v3`: candidate generation **15,511/15,511**, **19,225** candidates, **3,714** extras; final selection remains Python **15,727** vs MATLAB **15,511**, overlap **15,362** (missing **149**, extra **365**). Do **not** launch a successor full run from this alone; the active loop moves to candidate-to-final cleanup balance.

> ⚠️ **2026-07-13 (crop-axis finalization fix + `v10` full attempt) — crop strict gap nearly closed locally, but full Network still open and now over-selected.** The remaining crop funnel loss was dominated by applying MATLAB-order `microns_per_voxel` (`[Y, X, Z]`) directly to Python edge traces stored in internal `[Z, Y, X]` order during pre-clean smoothing/crop finalization. **Fix:** `prefilter_edge_indices_for_cleanup_matlab_style()` and `finalize_edges_matlab_style()` now align voxel spacing to trace order (`mpv[[2,0,1]]`) before `_matlab_smooth_edges_v2()` / `_matlab_crop_edges_v200()`. Refreshed `crop_M_exact_v3`: candidate generation remains complete (**15,511/15,511**, gap **0**); final selected edges moved to **15,728** with MATLAB-pair overlap **15,361/15,511** (missing **150**, extra **367**) versus prior **15,009** with overlap **14,542** (missing **969**). Full `canonical_full_v10` completed Edges→Network from the seeded v8 lineage. **Edges ADR 0012 PASS evaluated:** Python **70,247** vs MATLAB **69,500**, ownership-map agreement **99.9867%**. **Network ADR 0012 FAIL:** Python **48,583** vs MATLAB **48,049** strands. Phase 1 remains open; the residual has flipped from under-generation to over-selection, so the next loop is selection/cleanup balance on crop (final extra/missing pair diagnosis), not another generation or Network rewrite.

> 🟡 **2026-07-13 (frontier-action localization + orthogonal `-Inf` clamp) — the iter 23,005 split is fixed locally, but crop final movement is only +1 pair.** Added compact target-action tracing to Python and MATLAB (`frontier_action` rows for push / join reset / discard) plus target rows in MATLAB `strel_state`; `scripts/watershed_frontier_diff.py` can now run target traces without frontier snapshots and can stop after a requested iteration. Target history showed Python's later bad pop location (`2844114` zero-based) was pushed by both implementations at iter **19,247**; MATLAB removed it during a join reset at iter **22,421**, while Python's active baseline did not. Root cause: an exactly orthogonal direction pair (`pointer 22` vs `pointer 36`) landed at tiny positive dot product (`~3.8e-17`) in Python, keeping a vertex-origin `-Inf` selectable; MATLAB produced `NaN`/ignored it. **Fix:** clamp near-zero directional alignment (`<= eps`) to zero before multiplication. No-writer trace now matches the iter **22,421** join reset and reaches MATLAB's iter **23,005** pop; first `iteration_start` divergence moves to **25,495**. Live crop generation remains **15,511/15,511** with **19,226** candidates. Refreshed `crop_M_exact_v3` final remains Python **15,728** vs MATLAB **15,511**, overlap **15,362** (missing **149**, extra **366**) — only +1 retained MATLAB pair vs the prior refresh, so do **not** launch a successor full run from this alone. A trial patch that restored popped vertex origins in shared `energy_temp_flat` was rejected (live overlap **14,936/15,511**, missing **575**, extras **4,184**).

> 🟡 **2026-07-13 (claim-state diagnostic hardening) — strel-state comparison now exists; current split is upstream of strel claim.** `JsonExecutionTracer` now records true **pre-claim** and **post-claim** `vertex_index`, `pointer`, `d_over_r`, and `size` values separately for `strel_state` rows. Before this fix, fields named `*_before_claim` for pointer/distance/size were read after `claim_unowned_strel` mutated the maps. MATLAB `get_edges_by_watershed.m` now has an opt-in `SLAVV_WATERSHED_TRACE_PATH` hook for equivalent strel-state rows; scratch Edges-only run wrote `workspace/scratch/matlab_edge_dump/frontier_trace_state_iter23005.jsonl`. `scripts/compare_watershed_strel_state.py` shows iter **13,761** strel state **matches** Python modulo MATLAB one-based indexing. At iter **23,005**, MATLAB and Python already pop different current locations (MATLAB `2598494` one-based / Python `2844114` zero-based), so there is no same-strel adjusted-energy discrepancy to patch at that iteration. No-writer probes reconfirm candidate overlap **15,511/15,511**, generation gap **0**, extra candidates **3,715**. Next localization should trace predecessor queue/claim history for the two popped locations, not alter strel arithmetic.

> ⚠️ **2026-07-13 (queue insertion fix + `v8` audit attempt) — crop candidate generation now complete; full Network still open.** After the `-Inf` vertex-sentinel fix, the next crop split was MATLAB preserving an uncleared current tail when a primary seed inserted a better new location. Python popped before computing insertion. **Fix:** `_matlab_global_watershed_insert_available_location()` now computes MATLAB `location_idx` before tail clearing; join reset also removes only the first duplicate occurrence per MATLAB `find(..., 1, 'first')`. Crop no-writer first divergence moved **13,833 → 23,005** and live candidate coverage reached **15,511/15,511** (generation gap **19 → 0**). Refreshed `crop_M_exact_v3` checkpoint: Python candidates **19,226**, overlap **15,511**, final selected **15,009** vs MATLAB **15,511** (strict gap **502**; selection/crop pruning now the local residual). Full `canonical_full_v8` completed Edges→Network from `v7` lineage. **Edges ADR 0012 PASS evaluated:** Python **66,057** vs MATLAB **69,500** (gap **3,443**). **Network ADR 0012 FAIL:** Python **45,254** vs MATLAB **48,049** strands (gap **2,795**). Because `v8` regressed full strict connection and Network counts relative to `v7`, keep `canonical_full_v7` as the better full baseline (**66,224** connections; **45,417** strands) while using the crop zero-generation result to drive the next selection/funnel investigation. Phase 1 remains open; do not rewrite Network.

> ⚠️ **2026-07-08 (session) — Funnel localization + crop truncation fix (VALIDATED).** New offline probe `scripts/edge_selection_funnel_probe.py` replays the Python edge-selection funnel on the crop and measures overlap with the MATLAB oracle pair set after each stage. **Result: the dominant divergence is the CROP step, not generation (mostly closed) and not selection (faithful).** Of the 15,094 MATLAB pairs that ARE Python candidates, crop drops **1,284** (8.5%); degree-excess 128; cycles 51; generation gap (never a candidate) 417. **Fix:** MATLAB `crop_edges_V200` computes `uint16(radius/microns)` and `uint16(space)` — **truncation (floor)** — whereas Python `_matlab_crop_edges_V200` used `np.rint` (round). Rounding yields a larger radius than MATLAB's floor, so Python over-flags boundary points and over-crops. Changed space + radius to `np.floor` in `slavv_python/pipeline/edges/finalize.py:_matlab_crop_edges_V200`. **Validated via `prove-exact --stage edges` on crop:** crop final edges improved **14,403 → 14,922** (MATLAB 15,511), gap **1,108 → 589 (3.8%)**; MATLAB overlap 13,631 → 14,114 (+483 pairs). 61 edge unit tests green. The remaining 589 crop gap (777 crop-drops + 417 generation + degree/cycle) is a **symptom of residual trace-shape (generation) divergence** — the smoothing core (`_matlab_smooth_edges_v2`) was cross-checked against `smooth_edges_V2.m` and is faithful. Next: apply the same fix on `canonical_full_v6` (edges rerun) and attack the frontier-order generation divergence (iter 13,761) for full Phase 1 closure.

> ⚠️ **2026-07-08 (closure-grade rerun, ADR 0012 evaluated) — Edges ✅ PASS; Network ❌ FAIL; Phase 1 OPEN.** Closure-grade `canonical_full_v6` rerun (edges→network, `--include-debug-maps`, `parity_include_debug_maps=true`) completed. **Edges ADR 0012 PASS (evaluated):** ownership-map agreement **96.02%** (≥60%), trace tolerance NOT met (54,589/62,126 trace failures) but ownership bar carries the stage; Python **65,436** vs MATLAB **69,500** connections (gap 4,064 vs pre-fix 6,198 — 34% reduction). **Network ADR 0012 FAIL:** strand endpoint-pair multiset mismatch — Python **44,595** vs MATLAB **48,049** strands (gap 3,454 vs pre-fix 4,006). Network failure is **entirely downstream of the residual 4,064-edge gap** (no independent network bug; with identical MATLAB edges, Python network reproduces exact topology). **Remaining gap is generation-level** (watershed frontier-order divergence, golden trace iter 13,761) — the crop-truncation fix was necessary but insufficient to close Phase 1.

> ⚠️ **2026-07-09 (ownership-map diff — claiming divergence characterized).** Generated the Python crop `vertex_index_map` (`crop_M_exact_v3/99_Metadata/python_vertex_index_map.npy`, `(64,256,256)`) and diffed vs the MATLAB `watershed_ownership_map.mat` (oracle `180709_E_crop_M_v2`, `batch_260624-105705`). Aligns at permutation `(0,2,1)` (MATLAB `(Z,Y,X)` → `(Z,X,Y)`) with **96.26% exact-label agreement** (matches the ADR 0012 ownership bar). Of 4,194,304 voxels: both-claimed 1,581,746; **Python-only claimed 120,009**; MATLAB-only claimed 10,591; claimed-by-different-vertex 26,110. **Asymmetry:** Python claims ~11× more voxels than MATLAB leaves unclaimed, concentrated in a few vertices (top Python-only: vertex **80** = 18,196 voxels, **111** = 10,082, then 4874/985/1396/1114…). First diverging voxel in flat(F) scan: idx 18259 → `(z=0,y=71,x=83)`, Python = vertex **111**, MATLAB = unclaimed (and 111 is a top divergence vertex). This confirms the divergence is a **claiming-state difference** (not frontier pop-order), concentrated in specific vertices' regions — consistent with the trace finding (strel argmin picks a different neighbor at iter 13,724). The flat-scan "first" is memory-order, not flooding-order; the true first-divergent *claiming write* needs replay/instrumentation of `claim_unowned_strel`. The MATLAB golden trace EXISTS at `workspace/scratch/matlab_edge_dump/frontier_trace.jsonl` (crop `[256,256,64]`, 13,706 vertices) and pairs with `workspace/scratch/watershed_frontier_trace.jsonl`. Ran `scripts/watershed_frontier_diff.py` against `crop_M_exact_v3`. **Findings:** (a) The `+1` offset on `current_linear`/`selected_linear` is benign MATLAB **1-based vs Python 0-based** indexing (both are `(Y,X,Z)` frame; verified by decoding against the crop energy array — energies match to 1e-15). (b) The popped `current_location` is **identical (modulo +1) for all 13,760 iterations**; the first genuine split is the **strel `argmin` at iteration 13,761** (MATLAB trace) / the join at trace-idx 2544. (c) **Decisive detail:** at iteration 13,724 BOTH pop the same physical voxel `(y=204,x=249,z=51)`, but MATLAB's strel `argmin` selects a **−Inf vertex neighbor `(209,251,49)`** while Python selects a **finite neighbor `(201,252,49)` (−172.5)**. So the adjusted energies *inside the strel* differ → the divergence is a **claiming-state difference** (`d_over_r`/`size_map`/`vertex-assignment` written at that voxel during an earlier strel reveal), NOT a frontier pop-order tie-break. (d) The **equal-energy frontier tie-break is CONFIRMED CORRECT** (matches MATLAB insertion/reset/join + `argmin_with_linear_index_tiebreak`), so it is **NOT the cause**. (e) Re-verified the full flooding core vs `get_edges_by_watershed.m` and every structural component matches (energy penalties, argmin, frontier, join, `claim_unowned_strel`, `scale_label_clipped` no-op, `energy_tolerance=1.0`→`adjusted<0`). **Conclusion:** the residual edge gap (65,436 vs 69,500) is a **claiming-state divergence that surfaces as a different strel argmin**, not float round-off and not the frontier tie-break. Pinpointing the exact first claiming write that diverges requires comparing the full `vertex_index_map`/`d_over_r_map`/`size_map` state at iter ~13,724 between MATLAB and Python — needs either the MATLAB map dump or instrumenting `claim_unowned_strel`/`_matlab_frontier_adjusted_neighbor_energies` to record the strel adjusted energies + `vertices_of_current_strel` at the diverging iteration. Crop overlap 97.31%; edges ADR 0012 PASS (96% ownership). Post-v6 clarification: the residual connection gap is **not** an Edges ownership ship-gate failure, but it **does** block Phase 1 through the full-volume Network ADR 0012 multiset failure.

> ✅ **2026-07-06 — Phase 1 closure policy (operator synthesis).** **Ship gate:** evaluated ADR 0012 per-stage `prove-exact` on full `180709_E` (`canonical_full_v6` after 80% crop milestone). **Stretch:** overlap KPI on `crop_M_exact_v3`. See [ADR 0012 addendum](../../adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-06-phase-1-closure-bar-vs-strict-field-stretch).

> ⚠️ **2026-07-04 note — canonical full `180709_E` (`canonical_full_v4`) sequence ran; Energy ✅ + Vertices ✅ CERTIFIED on the full volume; Edges ⛔ + Network ⛔ FAIL (strict-field sequence only).** Historical `v4` checkpoints are **stale** for closure (pre–PR #103). Per-stage `prove-exact` results (`03_Analysis/exact_proof_<stage>.json`):
> - **Energy PASS** — 0 scale mismatches / **16,777,216** voxels (ADR 0011 `np.allclose`; max \|Δ\|≈2.36×10⁻¹¹).
> - **Vertices PASS** — positions/scales exact; energies within tolerance.
> - **Edges FAIL** — `edges.connections` shape mismatch: Python **60,213** vs MATLAB **69,500** (~13.4% short).
> - **Network FAIL (downstream of edges)** — strand endpoint-pair multiset mismatch: Python **39,623** vs MATLAB **48,049** strands; bifurcations **20,063** vs **25,371**. Every network field is a length/shape mismatch tracking the edge deficit — no independent network bug.
>
> **Root cause localized (2026-07-04 debug session, crop harness):** the edge shortfall is a **watershed candidate-*generation* divergence, not a selection/pruning bug.** See [§ Edge shortfall root cause](#-2026-07-04-edge-shortfall-root-cause-generation-gap-not-prune-gap) below. The crop reproduces the same ~13% deficit (Python 13,555 vs MATLAB 15,511) in ~5 min and was used for the split.

> ⚠️ **2026-07-02 note**: `prove-exact-sequence` (strict-field comparator) ran on crop and produced **FAIL at edges**: Python 13,555 connections vs MATLAB 15,511 (12.6% short). Energy ✅ PASS, Vertices ✅ PASS, Network ⛔ BLOCKED (downstream of edges FAIL). Evidence: `workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/exact_proof.json`.
>
> The 2026-07-01 "ALL FOUR STAGES CERTIFIED" entry below refers to **per-stage `prove-exact --stage <s>` runs using ADR 0012 spatial bars** (ownership-map ≥ 60% threshold for edges). These remain valid. The `prove-exact-sequence` strict-field comparator fails edges due to accepted watershed order-sensitivity (see [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md)). Task 9 of the `matlab-python-parity` spec requires ADR 0012 spatial-bar proof mode — run individual `prove-exact --stage <s>` for each stage, not `prove-exact-sequence`.

**[2026-07-01]** oracle v2; **ALL FOUR STAGES CERTIFIED** (crop) under per-stage `prove-exact` with ADR 0011/0012 gates; **Canonical full energy CERTIFIED**: the octave-3/4 divergence was the Python coarse→fine **upsample mesh not bit-matching MATLAB `linspace`** at coarse-cell boundaries; bit-exact `linspace` port fixed it — full-volume `prove-exact --stage energy` passes with **0 scale mismatches across all 16,777,216 voxels** — see [canonical-energy-high-octave-divergence](../../solutions/parity/canonical-energy-high-octave-divergence.md)

**Authoritative status log** for exact-parity alignment with MATLAB. **Live operational status** (active runs, proof failures, blockers) lives here—not in [TODO.md](../../TODO.md). Tasks and checkboxes: TODO only.

**Spec:** [phase-1-exact-route-spec.md](../../plans/phase-1-exact-route-spec.md)

---
