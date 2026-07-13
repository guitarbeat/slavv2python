# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-07-13

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

## 📊 Executive status (stage model)

Phase 1 exit criterion ([ADR 0012 post-v6 addendum](../../adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-12-post-v6-residual--network-is-the-open-ship-gate)): **evaluated** ADR 0012 on full `180709_E` for **both** Edges and Network. As of `canonical_full_v7`, Edges ✅ PASS (ownership 99.89%); Network ❌ FAIL (downstream of residual generation gap). Strict-field connection equality is stretch signal only once Network multisets pass. Operator brief: [.claude/HANDOFF.md](../../../.claude/HANDOFF.md).

**Proposal figure:** quantitative summary of these metrics (crop overlap trajectory, edge-pair recovery waterfall, full-volume counts / certification table) —
[figures/matlab_python_parity_journey.pdf](../../../figures/matlab_python_parity_journey.pdf)
([figures/README.md](../../../figures/README.md)). Regenerate the script when the numbers below move.

| Stage | Harness / prior work | Phase 1 certification bar |
| :--- | :--- | :--- |
| **Energy** | Native Hessian path exact-compatible | 🟢 `prove-exact --stage energy` vs **`180709_E_crop_M_v2`** **PASS** (ADR 0011 `np.allclose` gate, rtol=1e-7/atol=1e-9). `scale_indices` **0**; `energy` max \|Δ\|=1.99×10⁻¹¹; `lumen_radius_microns` max \|Δ\|=7.1×10⁻¹⁵. Cross-library float drift is bounded, not a logic difference. Strict `np.equal` available via `--strict-floats`. |
| **Vertices** | Verified on prior surfaces | 🟢 **PASS** vs `180709_E_crop_M_v2` (`prove-exact --stage vertices` exit 0): positions + scales match MATLAB **exactly** (13,706 = 13,706; 0 missing/extra) after the SE fix (`ellipsoid_offsets` ports MATLAB `construct_structuring_element.m` float-radius membership); `energies` certify under the ADR 0011 `np.allclose` policy after the loader recovers true energies from the raw `vertices.mat` (curated artifact stored a rank ramp). |
| **Edges** | Watershed parity fixes (2026-07-07), crop-truncation fix (2026-07-08), vertex `-Inf` sentinel fix (2026-07-13) | 🟢 **ADR 0012 PASS** on full `180709_E_full_v2` (`canonical_full_v7`, 2026-07-13): `adr0012_evaluated: true`, ownership-map **99.89%** (≥60% threshold). Python **66,224** vs MATLAB **69,500** (gap 3,276 vs v6 gap 4,064). Strict-field FAIL remains residual generation, not selection. |
| **Network** | End-to-end pipeline runs | ⛔ **ADR 0012 FAIL** on full `180709_E_full_v2` (`canonical_full_v7`, 2026-07-13): strand endpoint-pair multiset mismatch — Python **45,417** vs MATLAB **48,049** strands (gap 2,632 vs v6 gap 3,454). **Entirely downstream of residual edge gap** (no independent network bug; isolated network with MATLAB edges reproduces exact topology). |

---

## 🚦 Active Phase 1 operations

> **Closure policy (2026-07-13 post-v7):** Phase 1 closes when a fresh full-volume run (next successor root, e.g. **`canonical_full_v8`**) passes **evaluated** ADR 0012 for **Edges and Network**. **`canonical_full_v6`** and **`canonical_full_v7`** are audit closure attempts: Edges ✅ Network ❌. Crop overlap ≥80% is **cleared**; current crop generation gap is **19**. See [HANDOFF](../../../.claude/HANDOFF.md).

| Track | Run / artifact | Status |
|-------|----------------|--------|
| **Crop harness oracle** | `workspace/oracles/180709_E_crop_M_v2` | ✅ Fresh MATLAB batch `batch_260624-105705` (lattice-6000). Use v2 for all new proofs. |
| **Oracle artifact readiness** | `180709_E_crop_M_v2`, `180709_E_full_v2` | ✅ `ensure-oracle-artifacts --stage all` passes. ✅ Full oracle **`watershed_ownership_map.mat`** present (`batch_260626-125646/data/`). |
| **Crop harness run (audit)** | `workspace/runs/oracle_180709_E/crop_M_exact` | ✅ Per-stage ADR 0012 certified (2026-07-01). Stale for stretch KPI (pre–PR #103). |
| **Crop stretch run** | `workspace/runs/oracle_180709_E/crop_M_exact_v3` | ✅ Edges rerun (2026-07-13 after `-Inf` sentinel fix). **Overlap 99.88%** (15,492 / 15,511); **19,292** candidates; generation gap **19**. Final selected edges **15,046** vs MATLAB 15,511 (strict gap 465; final MATLAB overlap 14,514). Golden trace first `iteration_start` divergence now **13,833** (was 13,761). |
| **Canonical full oracle** | `workspace/oracles/180709_E_full_v2` | ✅ Batch `batch_260626-125646`; energy `(64,512,512)`. ✅ `watershed_ownership_map.mat` (2026-07-07 MATLAB harness). |
| **Canonical full run (audit)** | `workspace/runs/oracle_180709_E/canonical_full_v4` | ✅ Energy + Vertices **CERTIFIED** (2026-07-04). Stale edges/network (pre–PR #103). |
| **Canonical closure run (audit)** | `workspace/runs/oracle_180709_E/canonical_full_v5` | ✅ Writer succeeded (2026-07-06, ~2.1h). ⛔ Proof invalid ADR 0012 (`adr0012_evaluated: false`); strict-field edges 60,287 vs 69,500. Preserve as audit. |
| **Canonical closure run (v6)** | `workspace/runs/oracle_180709_E/canonical_full_v6` | ✅ Writer **COMPLETED** (2026-07-08, edges ~91min + network ~4min). Edges ADR 0012 ✅ **PASS** (`adr0012_evaluated: true`, ownership 96.02%). Network ADR 0012 ❌ **FAIL** (strand multiset 44,595 vs 48,049; downstream of edge gap). Phase 1 open. |
| **Canonical closure run (v7)** | `workspace/runs/oracle_180709_E/canonical_full_v7` | ✅ Writer **COMPLETED** (2026-07-13) seeded from `v6` Energy/Vertices; `parity_include_debug_maps=true`. Edges ADR 0012 ✅ **PASS** (`adr0012_evaluated: true`, ownership 99.89%; Python 66,224 vs MATLAB 69,500). Network ADR 0012 ❌ **FAIL** (strand multiset 45,417 vs 48,049). Phase 1 open, but residual improved. |

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

**Probe command:** `.\.venv\Scripts\python.exe scripts/watershed_candidate_gap_probe.py --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 --oracle-root workspace/oracles/180709_E_crop_M_v2`

**80% milestone:** ✅ **cleared** (2026-07-07). **Post-v6 residual playbook:** [.claude/HANDOFF.md](../../../.claude/HANDOFF.md) § A (generation loop) → § B (`v7` when crop residual moves).

### 🔎 2026-07-04: Edge shortfall root cause (generation gap, not prune gap)

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

If resuming exact parity work from a fresh thread, start with **[.claude/HANDOFF.md](../../../.claude/HANDOFF.md)** (post-v6 operator synthesis). Summary:

1. `slavv jobs list` — no concurrent writer on the target `--dest-run-root`.
2. `slavv parity ensure-oracle-artifacts --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage all --no-repair` (and same for `180709_E_full_v2` before canonical work).
3. **Residual loop (primary):** latest crop first-diverge **23,005**, live/checkpoint generation gap **0**, final strict gap **502** after the 2026-07-13 queue insertion fix. Measure with `scripts/watershed_frontier_diff.py`, `scripts/edge_selection_funnel_probe.py`, `scripts/watershed_candidate_gap_probe.py`.
4. **Current full-volume refresh:** **`canonical_full_v8`** completed from `v7` lineage with debug maps. Edges ADR 0012 still passed, but full strict connection and Network strand counts regressed versus `v7`; keep `canonical_full_v7` as the better full baseline and continue crop funnel/selection residual work before another successor root.
5. Harness **fail loud** if ADR 0012 cannot evaluate (maps missing) — not a valid closure attempt. `v6` already has evaluated Edges PASS.
6. Capture evidence per [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md). Re-synthesize HANDOFF if this findings top banner changes.

Use `--monitor` on long reruns ([PARITY_JOB_MONITORING.md](../workflow/PARITY_JOB_MONITORING.md)).

Scratch diagnostics: prefer promoted `scripts/*` probes; historical `workspace/scratch/edge_gap_split.py`, `workspace/scratch/matlab_edge_instr/`.

### Operator commands

See **[.claude/HANDOFF.md](../../../.claude/HANDOFF.md)** for the current command block (residual generation → `v7` closure). Legacy commands below are **audit-only**.

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

1. **Crop generation gap cleared; crop funnel/selection residual is now primary** — Crop-edge `uint16` floor truncation, `-Inf` vertex sentinel preservation, one-occurrence join reset, and MATLAB-order insertion-before-tail-clear are fixed. Latest crop checkpoint candidates cover **15,511/15,511** MATLAB pairs (generation gap **0**), but final selected pairs are **15,009** / 15,511 (strict gap **502**). Fix surface shifts to the candidate → crop → degree/cycle funnel; do not reopen Network as an independent port.
2. **Phase 1 ship gate = Network ADR 0012** — Edges ADR 0012 ✅ PASS evaluated on `canonical_full_v7` and `canonical_full_v8`. Network ❌ FAIL (`v7`: 45,417 / 48,049 strands; `v8`: 45,254 / 48,049), **entirely downstream of the edge gap** (MATLAB-edge isolation is exact). Keep `v7` as the best full baseline; launch a successor full root only after crop funnel metrics move.
3. **~~ADR 0012 measurement gap~~ (cleared)** — Full oracle ownership map + Python `--include-debug-maps` on `v6+` enabled evaluated Edges proof. Keep maps on any successor writer.

**Superseded guidance:** “>95% match”, “block on 80% crop overlap”, or strict-field fallback as closure verdict. Only **evaluated** ADR 0012 on **both** Edges and Network count; see [ADR 0012 post-v6 addendum](../../adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-12-post-v6-residual--network-is-the-open-ship-gate).

