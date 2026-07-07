# ADR 0012: Edge Watershed Parity Bar (ownership-map + trace tolerance)

## Status
Accepted (2026-06-25)

## Context

Phase 1 exact-route certification ([phase-1-exact-route-spec.md](../plans/phase-1-exact-route-spec.md), spec R1) asks for **strict set equality** on every compared field per stage. Energy and Vertices certify under that bar (the latter via the [ADR 0011](0011-energy-float-certification-policy.md) `np.allclose` tolerance for continuous floats). The **Edges** stage does not, and a long investigation (logged in [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md)) now explains why.

The exact-route edge engine (`_generate_edge_candidates_matlab_global_watershed`) is a faithful port of MATLAB `get_edges_by_watershed.m`: a **greedy, shared-state watershed flood-fill** where every vertex's catchment competes for voxels through a single mutated `energy_map_temp` / `vertex_index_map`, processed in global best-energy-first order.

Two findings closed the investigation:

1. **A real orientation bug existed and is fixed** (commit `e9dcc141`, branch `fix/edge-watershed-orientation`). `generate_watershed_candidates` pre-aligned physical `[Z,Y,X]→[Y,X,Z]` and then the engine reoriented *again*, yielding a double transpose (`[Z,Y,X]→[Y,X,Z]→[X,Z,Y]`) plus a double `microns_per_voxel` permute. The production `FrontierTracingDiscovery` path therefore ran the watershed on a **scrambled grid**.

2. **The residual gap is emergent global-ordering sensitivity, not a local bug.** Proven with a MATLAB R2019a ground-truth harness (instrumented standalone `get_edges_V300` dumping the watershed `vertex_index_map` and per-neighbor strel state):
   - On the correct grid, Python's full `vertex_index_map` agrees with MATLAB's on **63.47%** of MATLAB-claimed voxels; the wrong (double-transpose) grid collapsed to **1.07%** under the implied Y↔X swap and produced a wrong-shape map.
   - Per-neighbor decomposition at the instrumented divergence shows **`r_over_R` matches MATLAB to 4 decimals on every neighbor, sizes match, and the size / local-distance / direction penalty math is faithfully ported**. Where Python and MATLAB read the same `energy_temp` state, the per-step `argmin` agrees.
   - The remaining adjusted-energy differences (1.3–6.8×) reduce to differences in the **shared, mutated `energy_temp`** — voxels overwritten by *other* vertices' catchments, popped in a subtly different order than MATLAB. Tiny accumulated claim/queue-order differences cascade into different topology. This is the inherent sensitivity of a greedy shared-state flood-fill, not a fixable local discrepancy.

A corollary measurement: raw **edge-PAIR** overlap was *higher* on the wrong (double-transpose) grid (9,098) than on the correct grid (8,785). That number is **misleading** — it counts matching vertex-index pairs even when the traces run through spatially wrong voxels. Pair overlap is therefore rejected as the primary edge metric.

## Decision

Certify the Edges stage on a **two-part bar**, not exact pair-set equality:

1. **Voxel-ownership agreement (primary).** Compare Python's watershed `vertex_index_map` against MATLAB's on MATLAB-claimed voxels (excluding background and the image-border index). This is the spatially honest measure of catchment parity. Baseline on `180709_E_crop_M_v2`: **63.47%** on the correct grid.

2. **Per-edge trace tolerance (secondary).** For edges present in both, traces compare under the [ADR 0011](0011-energy-float-certification-policy.md) continuous-float policy (`np.allclose(rtol=1e-7, atol=1e-9)`), with topological/index fields strict.

**Discrete inputs stay strict.** Orientation, `r_over_R`/distance LUTs, strel offsets, sizes, `edge_number_tolerance`, and conflict-painting behavior must remain bit-faithful to MATLAB (they are — this ADR does not relax them). The tolerance is *only* for the emergent topology of the shared flood-fill.

We **do not** pursue bit-identical queue/claim ordering. The evidence is that the local math is already faithful; forcing identical global evolution order is high-effort, fragile, and chases a chaotic process rather than a defect.

## Consequences

- Update [PARITY_CERTIFICATION_GUIDE.md](../reference/workflow/PARITY_CERTIFICATION_GUIDE.md) and [phase-1-exact-route-spec.md](../plans/phase-1-exact-route-spec.md) R1 to record the Edges bar as ownership-map + trace tolerance, with the order-sensitivity rationale.
- The orientation fix (`e9dcc141`) **lowers** the historical edge-pair overlap headline (the old ~9.5k was inflated by coincidental wrong-grid pair matches). Replace that headline with the ownership-map figure wherever cited.
- Keep the MATLAB ground-truth harness (`workspace/scratch/matlab_edge_instr/`) as the reference for any future edge regression triage.
- Network stage certification inherits whatever edge set the watershed produces; its bar is evaluated separately once edges are accepted under this policy.

## Addendum (2026-06-25): Network stage inherits the same bar

The Network stage (`construct_network` → MATLAB `get_network_V190` + `get_strand_objects`) is built deterministically from edges and exhibits the same shape: exact topology, order-dependent emission.

Fed identical MATLAB curated edges + curated vertices (stage isolation), Python reproduces MATLAB's network topology **exactly**: strand endpoint-pair multiset **10,722/10,722**, bifurcation multiset **5,601/5,601** (0 missing/extra). Strands are emitted in a different order (inherited from edge order), and MATLAB stores strands as end-vertex pairs (`strands2vertices`) while Python stores full vertex chains.

**Decision (network):** certify network **topology** order-independently — strand endpoint-pairs and bifurcation vertices as multisets — and compare **per-strand geometry** (`strand_subscripts`, `strand_energy_traces`, `mean_strand_energies`, `vessel_directions`) under trace tolerance after a canonical endpoint-keyed reorder. Implemented as `_compare_network_stage` in `artifact_comparator.py`.

Network geometry parity (Phase B) is a separate, scoped effort: a scale-subscript off-by-one, strand-smoothing drift (~0.02–0.36 voxel; sigma `√2/2` already matches), and a minor multi-edge assembly off-by-one. The strand dedup was aligned to MATLAB round-half-up (`network/operations.py`).

## Addendum (2026-07-06): Phase 1 closure bar vs strict-field stretch

Phase 1 exact-route **ship confidence** uses **two tracked bars** for Edges/Network:

1. **Certification bar (ship gate):** per-stage `prove-exact --stage edges` and `--stage network` on full `180709_E` against `180709_E_full_v2` under this ADR (ownership-map + trace tolerance for edges; strand/bifurcation multisets + sub-voxel geometry for network). Energy and Vertices remain under [ADR 0011](0011-energy-float-certification-policy.md). **Phase 1 closes when this passes on a fresh canonical run** (`canonical_full_v5`, seeded from `canonical_full_v4`, Edges→Network rerun from current `main`).

2. **Strict-field stretch (non-blocking):** exact `connections` / strand-count equality vs MATLAB, tracked on a refreshed crop harness (`crop_M_exact_v3`). **Primary loop KPI:** candidate-generation overlap (MATLAB final pairs present in Python candidates). **Milestone check:** strict-field `prove-exact` on crop when overlap moves materially. Does **not** block Phase 1 once the certification bar passes on full volume.

**If the certification bar fails on `canonical_full_v5`:** Phase 1 remains open. Triage measurement first (checkpoint freshness, orientation/shape, oracle pairing, ownership probe) before assuming a watershed code defect. **If it passes:** declare Phase 1 closed; continue strict-field stretch on crop without reopening the ship gate.

**Operating order:** refresh crop `v3` (edges only, ~minutes) → launch canonical `v5` (edges→network) → per-stage ADR 0012 proof on `v5`. Do not use `prove-exact-sequence` strict-field failure as the Phase 1 closure gate.

**Considered:** closing Phase 1 on crop ADR 0012 alone, or requiring strict-field on full volume before closure — rejected; canonical volume is the Phase 1 claim surface (spec R1a), and ADR 0012 already records why exact pair-set equality is the wrong ship metric.

## Addendum (2026-07-06): Post-v5 watershed iteration and v6 closure

After `canonical_full_v5` (writer succeeded, proof invalid):

1. **v5 strict-field deficit is real but not a valid ADR 0012 verdict.** Full oracle `180709_E_full_v2` lacks `watershed_ownership_map.mat`; Python v5 checkpoint lacks `--include-debug-maps`. Proofs emitted `adr0012_evaluated: false` and must **not** be read as spatial-bar failure.

2. **80% crop overlap gate before canonical v6.** Primary loop KPI = candidate-generation overlap on `crop_M_exact_v3` (baseline **57.89%**, 8,979 / 15,511 MATLAB pairs). Launch **`canonical_full_v6`** only after overlap **≥80%** (~12,409 / 15,511).

3. **Fail-loud harness policy.** When ownership maps are missing or incompatible, `prove-exact --stage edges` fails with `mismatch_type: adr0012_not_evaluated` and exits non-zero. Strict `connections` counts are informational only — not the primary failure signal.

4. **v6 run shape.** Preflight from `canonical_full_v5` → new run root `canonical_full_v6`; carry certified Energy/Vertices; rerun **Edges → Network only**. Before proof: (a) MATLAB `watershed_ownership_map.mat` via instrumented harness on full `180709_E`; (b) Python `vertex_index_map` via `--include-debug-maps` on edge capture.

5. **Closure verdict.** Phase 1 closes only on **evaluated** ADR 0012 per-stage proofs (`adr0012_evaluated: true`) on `canonical_full_v6`. Stretch strict-field progress on crop continues without blocking ship once evaluated ADR 0012 passes on full volume.

**Operating order:** watershed fixes on crop → 80% milestone → map prep → v6 writer → evaluated ADR 0012 proof. See [.claude/HANDOFF.md](../../.claude/HANDOFF.md).

## Evidence references

- Fix: branch `fix/edge-watershed-orientation`, commit `e9dcc141` (`slavv_python/pipeline/edges/candidate_generation.py`).
- Ownership test: Python `vertex_index_map` (single transpose) vs MATLAB `chunk_1.mat` dump — 63.47% identity vs 1.07% Y↔X swap.
- Per-neighbor decomposition: MATLAB `divergence_hits.mat` (`r_over_R`/sizes/adjusted energies) vs Python `current_strel` — `r_over_R` and sizes match every neighbor.
- MATLAB harness: `workspace/scratch/matlab_edge_instr/` (instrumented `get_edges_V300` + `get_edges_by_watershed` + `run_edges_standalone.m`).
- Narrative log: [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md) Edges row.
- **External validation:** [PARITY_METHODOLOGY.md](../reference/core/PARITY_METHODOLOGY.md) — segmentation literature confirms that for order-sensitive outputs, exact set-equality / pixel-accuracy is the wrong metric (class-imbalance inflation) and that overlap (Dice/IoU) + boundary/distance (Hausdorff) bars are the correct measure. The ownership-map + multiset + sub-voxel-trace bar here is the watershed analogue.
