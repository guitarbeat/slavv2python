# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-07-14 (MATLAB post-watershed finalization parity; one crop pair swap remains)

This is the single successor brief for the current exact-route effort. Do not use
dated agent passovers, PID snapshots, or parallel-work checklists as current
status. When findings top-banner changes, re-synthesize this file the same day.

## Canonical records

| Need | Source of truth |
|---|---|
| Active work and checkboxes | [docs/TODO.md](../docs/TODO.md) |
| Verified run status, proof evidence, and blockers | [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) |
| Phase 1 requirements | [phase-1-exact-route-spec.md](../docs/plans/phase-1-exact-route-spec.md) |
| Edges/Network bar + closure policy | [ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md) (post-v6 addendum) |
| Run commands and evidence format | [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md), [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md) |
| Repository and parity guardrails | [AGENTS.md](../AGENTS.md) |

## Current decision point

> **Single canonical status source:** [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md).

### Snapshot (do not invent numbers — confirm in findings)

| Stage | Canonical surface | Status |
|-------|-------------------|--------|
| **Energy** | `canonical_full_v4` / `180709_E_full_v2` | ✅ CERTIFIED (ADR 0011) |
| **Vertices** | same | ✅ CERTIFIED (ADR 0011) |
| **Edges** | `canonical_full_v15` audit | ✅ ADR 0012 PASS evaluated (`v15`: 69,500 / 69,500; trace failures 0) |
| **Network** | `canonical_full_v15` audit | ❌ ADR 0012 FAIL (`v15`: 48,048 / 48,049 strands; short by 1) |

**Phase 1 is OPEN** solely because Network fails ADR 0012. Edges stage certification is done on `v6`. Network isolation with MATLAB edges reproduces exact topology — there is no independent network bug.

**Latest residual fix:** MATLAB standalone instrumentation proved Python raw candidates match MATLAB raw watershed candidates exactly (**19,225 / 19,225**). The active divergence was post-watershed finalization: MATLAB runs `resample_vectors`, re-samples energy/size from maps, smooths, then crops with unsigned integer casts before cleanup. Python now mirrors that path and carries resampled traces/energies/metrics into cleanup. Refreshed crop candidate generation remains **15,511/15,511** with **19,225** raw candidates; refreshed crop final selected edges are **15,511** vs MATLAB **15,511**, with **15,510** MATLAB pairs retained (missing **1**, extra **1**).

**Latest full result:** `canonical_full_v15` completed Edges→Network from `v10` Energy/Vertices with current post-watershed finalization code. Edges ADR 0012 passed evaluated with exact strict count (**69,500 / 69,500**), ownership agreement **99.999863%**, and **0** trace failures. Network ADR 0012 still fails by one strand (**48,048 / 48,049**). The full edge-pair delta is one swap: MATLAB has `(34897, 38584)`, Python has `(26444, 38584)`. Network endpoint delta: MATLAB has `(34897, 55337)` and `(26444, 41666)`, Python has `(41666, 55337)`.

**Latest cleanup comparator result:** `scripts/compare_clean_edge_pairs_matlab.py` now exports the resampled post-crop candidate list and calls MATLAB `clean_edge_pairs`, `clean_edges_vertex_degree_excess`, and `clean_edges_cycles`. Python matches MATLAB row indices exactly at all three cleanup stages on that surface. The sole remaining crop swap occurs in degree pruning: Python keeps `[4043, 6281]` while MATLAB keeps `[4212, 6281]`; both have the same resampled metric (`-15.137922715664892`) and share vertex `6281`. A broad endpoint-descending tie-break trial was rejected because it regressed crop to **7 missing / 9 extra**.

**Latest rejected hypothesis:** `scripts/edge_selection_funnel_probe.py --apply-matlab-chunk-eligibility` emulates the `get_edges_V300` reading/writing chunk emission rule. On the crop, MATLAB's edge chunk lattice is a single chunk under `max_voxels_per_node = 1e8`, and the diagnostic drops **0 / 19,225** candidates. Chunk read/write windows do **not** explain raw candidate extras, which are now known to be MATLAB raw candidates removed later by finalization/cleanup.

**Latest displacement result:** `scripts/edge_selection_funnel_probe.py` now reports a single final displacement. Degree pruning loses **1** MATLAB pair (`[4212, 6281]`), displaced by incident surviving extra `[4043, 6281]` with the same resampled metric and earlier row order. Cycle pruning leaves final count equal at **15,511**.

**Latest extra-source result:** candidate extras are diffuse at generation time (top extra-producing origins emit only **2** extras each), while final extras concentrate more around boundary-adjacent / MATLAB-zero-degree vertices. A production-style geometry boundary cutoff is rejected: removing candidates touching vertices within 1 voxel of a boundary worsens overlap to **14,984** and missing pairs to **527**. An oracle-aware zero-degree-boundary upper bound improves only modestly to **15,377** overlap / **134** missing / **238** extras, so boundary-adjacent zero-degree vertices are a symptom, not the root cause.

**Latest bounded trace regression:** two stale `watershed_frontier_diff.py --regenerate-python` processes were writing interleaved JSONL and were stopped. The script now treats `--stop-after-iteration` as a bounded comparison over iteration-bearing rows. A fresh compact replay reports `bounded_match` through iteration **30,000**, well past the retired **13,761** split.

**Latest diagnostic hardening:** Python `strel_state` tracing now records true pre-claim and post-claim `vertex_index`, `pointer`, `d_over_r`, and `size` values separately. MATLAB `get_edges_by_watershed.m` has an opt-in `SLAVV_WATERSHED_TRACE_PATH` state trace hook, and the scratch Edges-only runner wrote `workspace/scratch/matlab_edge_dump/frontier_trace_state_iter23005.jsonl`. Comparison result: iter **13,761** strel state matches Python modulo MATLAB one-based indexing; iter **23,005** is already a different current-location pop (MATLAB `2598494` one-based / Python `2844114` zero-based), so the remaining split is upstream queue/claim history, not same-strel arithmetic.

**Latest localization result:** compact `frontier_action` target tracing records push / join-reset / discard history without dumping full frontier snapshots. The Python bad-pop location (`2844114` zero-based) was pushed by both implementations at iter **19,247**; MATLAB removed it during a join reset at iter **22,421** because the competing `-Inf` vertex-origin neighbor became `NaN` under an exact-zero orthogonal direction factor. A second split at iter **25,495** showed why an epsilon clamp was too broad: MATLAB selects a tiny-positive LUT dot (`2.78e-17`) at scale 29. The LUT-unit-vector fix preserves both cases. A trial patch that restored popped vertex origins in shared `energy_temp_flat` was **rejected**: live crop generation fell to **14,936/15,511** overlap (missing **575**, extras **4,184**) versus the current **15,511/15,511** baseline.

**Prior full result:** `canonical_full_v10` completed Edges→Network from `v8` lineage with `parity_include_debug_maps=true`. Edges ADR 0012 passed evaluated, but strict connection and Network counts over-shot MATLAB (`v10` edges **70,247** vs **69,500**; Network strands **48,583** vs **48,049**). Superseded by `v15` for current residual planning.

**Cleared milestones (do not re-gate on these):**

- Crop candidate overlap **100%** on the refreshed live/checkpoint candidate surface (80% gate cleared 2026-07-07; generation gap now 0).
- `canonical_full_v6` writer + ownership maps present; Edges evaluated ADR 0012 PASS.

## Strategy (improved post-v6)

### What the ship gate actually requires now

1. **Edges ADR 0012** — met on `canonical_full_v6`. Do not reopen unless a regression appears.
2. **Network ADR 0012** — still open. Multiset topology equality needs a **nearer** edge-connection set than the ownership bar alone guarantees. Closing Network = improving the residual edge-connection set, not rewriting Network. Latest full result `v10` overshoots: Edges **747** connections over, Network **534** strands over. This is closer in absolute count than `v7`, but still not topology-equal.
3. **Strict-field stretch** (exact 69,500 connections) remains non-blocking for messaging once Network ADR 0012 passes — but the practical path to Network pass is the same generation work.

### Primary loop KPI (replaces the old 80% crop-overlap gate)

| KPI | Surface | Role |
|-----|---------|------|
| **Golden-trace first-diverge iteration** | crop, `scripts/watershed_frontier_diff.py` | Latest: **match** through the crop trace; iter 22,421 and 25,495 splits are fixed |
| **Crop final connection gap** | crop prove-exact / funnel probe | Latest: Python **15,511** vs MATLAB **15,511**; overlap **15,510** (missing **1**, extra **1**) |
| **Full edge connection gap** | `canonical_full_v15` audit | Latest residual is exact count but one pair swap: **69,500** vs **69,500** |
| **Network strand multiset** | full `prove-exact --stage network` | **Only** Phase 1 ship remaining |

### Do not waste cycles on

- Re-proving Energy/Vertices on full volume without a regression.
- Re-litigating Edges ownership once evaluated PASS holds.
- Treating `prove-exact-sequence` strict-field failure as the ship gate.
- Probe KPIs without the production orientation contract (`mpv` permute) — false 62% signals already burned a day.
- New one-off scratch scripts when `scripts/watershed_frontier_diff.py`, `scripts/edge_selection_funnel_probe.py`, and `scripts/watershed_candidate_gap_probe.py` already cover the loop.
- Re-testing shared `energy_temp_flat` vertex-origin restoration unless a new probe explains why it can preserve the **15,511/15,511** crop generation baseline.
- Softening Network to “close enough” without an ADR — that would reopen ship confidence.

## Operating sequence

### A. Crop final-selection loop (primary — until Network can pass)

1. Work on the single crop pair swap first. Candidate generation covers all crop MATLAB pairs, the crop frontier trace matches MATLAB, and post-watershed finalization/cleanup now matches MATLAB row choices except for one equal-metric degree-pruning tie: Python keeps `[4043, 6281]` while MATLAB keeps `[4212, 6281]`. Do not re-open the shared vertex-origin restore patch; it regressed live crop generation.
2. After each fix, reinstall the local package and check that no parity writer is already active:
   ```powershell
   .\.venv\Scripts\pip.exe install -e .
   .\.venv\Scripts\slavv.exe jobs list
   .\.venv\Scripts\slavv.exe parity ensure-oracle-artifacts `
     --oracle-root workspace/oracles/180709_E_crop_M_v2 `
     --stage all `
     --no-repair
   ```
3. Run the **no-writer** probes first. These are the fast loop; they should move before spending time on a checkpoint refresh.
   ```powershell
   .\.venv\Scripts\python.exe scripts/watershed_frontier_diff.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2 `
     --regenerate-python

   .\.venv\Scripts\python.exe scripts/watershed_candidate_gap_probe.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2 `
     --trace-missing `
     --sample-size 20
   ```
   Interpretation:
   - `watershed_frontier_diff.py` is now a regression guard. Current crop baseline is **match**.
   - `watershed_candidate_gap_probe.py --trace-missing` is now a regression guard for candidate generation. Current crop baseline is **15,511/15,511** with **19,225** candidates and **3,714** extras.
4. Only when the no-writer probes improve materially, refresh crop Edges with the current code:
   ```powershell
   .\.venv\Scripts\slavv.exe parity resume-exact-run `
     --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2 `
     --force-rerun-from edges `
     --stop-after edges `
     --skip-preflight
   ```
5. After a crop checkpoint refresh, measure the final funnel and checkpoint candidate gap:
   ```powershell

   .\.venv\Scripts\python.exe scripts/edge_selection_funnel_probe.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2

   .\.venv\Scripts\python.exe scripts/watershed_candidate_gap_probe.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2
   ```
   Interpretation:
   - `edge_selection_funnel_probe.py` locates final MATLAB pair loss through candidates → crop → dedup/choose-best → degree/orphan/cycle. Current guidance after `v10`: generation is solved; crop/finalization is much closer but now admits extras, so diagnose final extra/missing pair balance before another full run.
   - The final target is not the retired 80% overlap gate. Track crop final missing/extra pair counts, candidate generation gap, and whether the full-volume Network multiset is likely to move.
6. Log KPI deltas in [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) watershed iteration table (and residual gap numbers).
7. Prefer **claiming-state** hypotheses (Python over-claim, strel neighbor choice) over inventing non-linear-index queue tie-breaks ([UNPRODUCTIVE_LOOPS](../docs/reference/core/UNPRODUCTIVE_LOOPS.md)).

Current status: this loop moved materially on 2026-07-14. Raw watershed candidates match MATLAB exactly, post-watershed finalization now follows MATLAB's resample/map-resample/smooth/crop path, and cleanup row choices match MATLAB on the resampled post-crop surface. The refreshed crop final set is **15,511** vs MATLAB **15,511**, with **15,510/15,511** overlap. The only local residual is one equal-metric degree-pruning swap (`[4043, 6281]` vs `[4212, 6281]`). `canonical_full_v10` remains the latest full audit and still has Network red, so the next decision is whether to fix the single crop swap before launching a successor full audit root.

### Primary code and test surfaces

| Area | Files / functions | Planning note |
|------|-------------------|---------------|
| Candidate extras | `slavv_python/pipeline/edges/matlab_get_edges_by_watershed.py`, `scripts/edge_selection_funnel_probe.py`, candidate diagnostics/probes | Active residual loop; candidate generation covers all MATLAB pairs but contributes **3,714** extras that displace MATLAB final pairs in faithful cleanup. Degree/cycle displacement is quantified; chunk eligibility is rejected on crop (**0** candidates dropped). |
| Finalization crop | `slavv_python/pipeline/edges/finalize.py` | Smaller residual surface; crop currently loses only 14 MATLAB pairs before degree/cycle cleanup. |
| Selection row order | `slavv_python/pipeline/edges/selection_payloads.py` (`prepare_candidate_indices_for_cleanup`) | Regression guard; Python now matches MATLAB `clean_edge_pairs` row order on crop after double-precision metric sort. |
| Cleanup pruning | `slavv_python/pipeline/edges/cleanup.py` (`remove_excess_vertex_degrees`, `break_graph_cycles`) | Regression guard; comparator shows zero row-index mismatches for degree and cycle pruning on the Python candidate surface. |
| Watershed loop | `slavv_python/pipeline/edges/matlab_get_edges_by_watershed.py` (`_generate_edge_candidates_matlab_global_watershed`, `_matlab_global_watershed_current_strel`) | Regression guard now that crop frontier trace matches. |
| Claiming state | `slavv_python/pipeline/edges/matlab_watershed_heap.py` (`VoxelClaimMap.claim_unowned_strel`) | Regression guard for Python over-claim / strel neighbor divergence. |
| Adjusted energies | `slavv_python/pipeline/edges/matlab_get_edges_v300_geometry.py` (`_matlab_frontier_adjusted_neighbor_energies`) | Latest fix surface; neighbor direction must come from MATLAB LUT `unit_vectors`. |
| Tie-break guardrail | `slavv_python/pipeline/edges/matlab_indexing.py` (`_matlab_watershed_min_candidate_energies`, `_argmin_with_linear_index_tiebreak`) | Current evidence says frontier tie-breaking is correct. |

Focused unit checks after code changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_global_watershed_comprehensive.py
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_global_watershed_anisotropic.py tests/unit/pipeline/test_frontier_math.py tests/unit/pipeline/test_fortran_tie_breaking.py
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_edges_comprehensive.py tests/unit/pipeline/test_partner_substitution_regression.py
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_watershed_tracing.py tests/unit/pipeline/test_float64_dtype_invariant.py
.\.venv\Scripts\python.exe -m pytest tests/integration/parity/test_parity_pre_gate_tier1.py
```

### B. Canonical closure refresh (after crop residual moves materially)

When crop gap / golden-trace diverge improves enough to justify a full edges rewrite:

**Current:** `canonical_full_v10` completed as an audit attempt and overshot the full edge/network counts. Next full-volume launch should be a successor root only after another crop/funnel residual improvement.

1. Prefer a **new successor** run root (for example `canonical_full_v11`) preflighted from the latest audit lineage — do not destroy `v6`, `v7`, `v8`, or `v10` audit records.
2. Rerun **edges → network only** with `--include-debug-maps` / `parity_include_debug_maps=true`.
3. Per-stage evaluated proofs:

```powershell
slavv jobs list

slavv parity launch-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v11 `
  --oracle-root workspace/oracles/180709_E_full_v2 `
  --dataset-root workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353 `
  --force-rerun-from edges `
  --stop-after network `
  --skip-foreground-probe `
  --monitor

# After writer completes:
slavv parity prove-exact --stage edges `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v11 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v11 `
  --oracle-root workspace/oracles/180709_E_full_v2

slavv parity prove-exact --stage network `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v11 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v11 `
  --oracle-root workspace/oracles/180709_E_full_v2
```

4. Phase 1 closes when **both** evaluated proofs pass. Record evidence per [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md).
5. If Edges still pass and Network still fails: continue generation loop — do not treat Network as a separate porting project.

### C. After Phase 1 closes

- Promote summary to `workspace/reports/` if warranted.
- Strict-field stretch optional on crop.
- Phase 2 optimization / paper-profile cert per [phase-2-optimization-spec.md](../docs/plans/phase-2-optimization-spec.md) and roadmap — do not start broad Phase 2 unwinding while Network is red.

## Audit runs (do not overwrite)

| Run | Role |
|-----|------|
| `crop_M_exact` | Pre–PR #103 crop audit |
| `crop_M_exact_v3` | Stretch + residual generation KPI loop |
| `canonical_full_v4` | Certified Energy/Vertices |
| `canonical_full_v5` | Fresh edges/network writer; invalid ADR 0012 (maps missing) |
| `canonical_full_v6` | First evaluated closure attempt: Edges ✅ Network ❌ |
| `canonical_full_v7` | Post `-Inf` sentinel fix closure attempt: Edges ✅ Network ❌, improved residual |
| `canonical_full_v8` | Post queue-insertion fix audit: crop generation ✅, full Edges ✅, Network ❌ and strict counts regressed vs `v7` |
| `canonical_full_v9` | Failed launch stub; no initialized provenance / no checkpoints. Do not use as evidence. |
| `canonical_full_v10` | Post crop-axis finalization audit: full Edges ✅, Network ❌; strict counts now over-select (Edges +747, Network +534). |

## Meta / process (agent hygiene)

| Shortcoming | Fix |
|-------------|-----|
| HANDOFF / TODO lag findings by days | Re-synthesize HANDOFF when findings top banner changes; TODO checkboxes only mirror ship tasks |
| Stale “80% gate” / “57.89% baseline” as current | Historical only; primary KPI is residual generation + Network multiset |
| Ownership-map missing → false closure narrative | Require `adr0012_evaluated: true` only |
| Diagnostic script sprawl | Promote probes into `scripts/` + solution notes; delete one-offs after use |
| Claiming “Network bug” from multiset fail | Always re-check: Network exact under MATLAB edges isolation |

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above. Pre-v6 “do not launch canonical until 80%” is **retired**.
