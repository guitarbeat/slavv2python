# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-07-13 (queue insertion fix; `canonical_full_v8` audit verdict)

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
| **Edges** | `canonical_full_v7` baseline; `canonical_full_v8` audit | ✅ ADR 0012 PASS evaluated (`v7`: 66,224 / 69,500; `v8`: 66,057 / 69,500) |
| **Network** | `canonical_full_v7` baseline; `canonical_full_v8` audit | ❌ ADR 0012 FAIL (`v7`: 45,417 / 48,049 strands; `v8`: 45,254 / 48,049) |

**Phase 1 is OPEN** solely because Network fails ADR 0012. Edges stage certification is done on `v6`. Network isolation with MATLAB edges reproduces exact topology — there is no independent network bug.

**Latest residual fix:** Python insertion cleared the current tail before computing MATLAB's `location_idx`. MATLAB computes insertion on the uncleared `available_locations`, so a better primary seed can append after and preserve the old current for a later pop. `_matlab_global_watershed_insert_available_location()` now matches that ordering; join reset also removes only the first duplicate occurrence. Crop no-writer first divergence moved **13,833 → 23,005** and candidate generation gap moved **19 → 0**. Refreshed crop final selected edges are **15,009** vs MATLAB **15,511** (strict gap **502**).

**Latest full result:** `canonical_full_v8` completed Edges→Network from `v7` lineage with `parity_include_debug_maps=true`. Edges ADR 0012 passed evaluated, but strict connection and Network counts regressed relative to `v7` (`v8` edge gap **3,443**, Network strand gap **2,795**; `v7` edge gap **3,276**, Network strand gap **2,632**). Keep `v7` as the better full baseline; use the crop zero-generation result to investigate the remaining crop/funnel selection loss. Do not rewrite Network.

**Cleared milestones (do not re-gate on these):**

- Crop candidate overlap **100%** on the refreshed live/checkpoint candidate surface (80% gate cleared 2026-07-07; generation gap now 0).
- `canonical_full_v6` writer + ownership maps present; Edges evaluated ADR 0012 PASS.

## Strategy (improved post-v6)

### What the ship gate actually requires now

1. **Edges ADR 0012** — met on `canonical_full_v6`. Do not reopen unless a regression appears.
2. **Network ADR 0012** — still open. Multiset topology equality needs a **nearer** edge-connection set than the ownership bar alone guarantees. Closing Network = improving the residual edge-connection set, not rewriting Network. Best full baseline remains `v7`: Edges **3,276** connections short; Network **2,632** strands short. `v8` is audit evidence that crop generation closure alone is insufficient.
3. **Strict-field stretch** (exact 69,500 connections) remains non-blocking for messaging once Network ADR 0012 passes — but the practical path to Network pass is the same generation work.

### Primary loop KPI (replaces the old 80% crop-overlap gate)

| KPI | Surface | Role |
|-----|---------|------|
| **Golden-trace first-diverge iteration** | crop, `scripts/watershed_frontier_diff.py` | Latest: **23,005** (was 13,833) |
| **Crop final connection gap** | crop prove-exact / funnel probe | Latest strict gap: **502** (15,009 vs MATLAB 15,511; was 465 after sentinel fix) |
| **Full edge connection gap** | `canonical_full_v7` baseline / `canonical_full_v8` audit | Best residual **3,276** on `v7`; `v8` regressed to **3,443** |
| **Network strand multiset** | full `prove-exact --stage network` | **Only** Phase 1 ship remaining |

### Do not waste cycles on

- Re-proving Energy/Vertices on full volume without a regression.
- Re-litigating Edges ownership once evaluated PASS holds.
- Treating `prove-exact-sequence` strict-field failure as the ship gate.
- Probe KPIs without the production orientation contract (`mpv` permute) — false 62% signals already burned a day.
- New one-off scratch scripts when `scripts/watershed_frontier_diff.py`, `scripts/edge_selection_funnel_probe.py`, and `scripts/watershed_candidate_gap_probe.py` already cover the loop.
- Softening Network to “close enough” without an ADR — that would reopen ship confidence.

## Operating sequence

### A. Residual generation loop (primary — until Network can pass)

1. Work on current `main` in `matlab_get_edges_by_watershed.py` / `matlab_watershed_heap.py` / related strel/claim paths. Focus: **claim_unowned_strel / strel argmin** at the first diverge iteration — not pop-order of vertices (already matches through 13,760).
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
   - `watershed_frontier_diff.py` answers the golden-trace first divergence. Current known baseline is the strel/claiming-state split around iter **13,761**; progress means the first divergence moves later or disappears. `--oracle-root` is accepted for operator consistency; the run root remains the source surface for regeneration.
   - `watershed_candidate_gap_probe.py --trace-missing` answers whether live regenerated watershed candidates cover more MATLAB final edge pairs. Watch `generation_gap_count`, `missing_after_live_trace`, `live_trace_pair_count`, `join_events_in_trace`, and `join_skipped_events_in_trace`.
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
   - `edge_selection_funnel_probe.py` locates final MATLAB pair loss through candidates → crop → dedup/choose-best → degree/orphan/cycle. Current guidance: selection is faithful; the remaining issue is generation/claiming-state.
   - The final target is not the retired 80% overlap gate. Track crop final connection gap (**589** baseline), candidate generation gap, and whether the full-volume Network multiset is likely to move.
6. Log KPI deltas in [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) watershed iteration table (and residual gap numbers).
7. Prefer **claiming-state** hypotheses (Python over-claim, strel neighbor choice) over inventing non-linear-index queue tie-breaks ([UNPRODUCTIVE_LOOPS](../docs/reference/core/UNPRODUCTIVE_LOOPS.md)).

Current status: this loop moved materially on 2026-07-13; `canonical_full_v8` proved crop generation closure alone does not close full Network and regressed strict full counts. Keep `v7` as best full baseline; next work is crop funnel/selection loss after generation.

### Primary code and test surfaces

| Area | Files / functions | Planning note |
|------|-------------------|---------------|
| Watershed loop | `slavv_python/pipeline/edges/matlab_get_edges_by_watershed.py` (`_generate_edge_candidates_matlab_global_watershed`, `_matlab_global_watershed_current_strel`) | Main residual loop and strel argmin site. |
| Claiming state | `slavv_python/pipeline/edges/matlab_watershed_heap.py` (`VoxelClaimMap.claim_unowned_strel`) | Highest-probability fix surface for Python over-claim / strel neighbor divergence. |
| Adjusted energies | `slavv_python/pipeline/edges/matlab_get_edges_v300_geometry.py` (`_matlab_frontier_adjusted_neighbor_energies`) | Formula is believed faithful; use to instrument state, not invent new math. |
| Tie-break guardrail | `slavv_python/pipeline/edges/matlab_indexing.py` (`_matlab_watershed_min_candidate_energies`, `_argmin_with_linear_index_tiebreak`) | Current evidence says frontier tie-breaking is correct. |
| Selection/funnel | `slavv_python/pipeline/edges/finalize.py`, `selection.py` | Validation surface after generation changes; crop truncation floor fix lives here. |

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

**Current:** `canonical_full_v8` completed as an audit attempt and did not improve the full baseline. Next full-volume launch should be a successor root (for example `canonical_full_v9`) only after another crop/funnel residual improvement.

1. Prefer a **new successor** run root (for example `canonical_full_v9`) preflighted from the latest audit lineage — do not destroy `v6`, `v7`, or `v8` audit records.
2. Rerun **edges → network only** with `--include-debug-maps` / `parity_include_debug_maps=true`.
3. Per-stage evaluated proofs:

```powershell
slavv jobs list

slavv parity launch-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v9 `
  --oracle-root workspace/oracles/180709_E_full_v2 `
  --dataset-root workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353 `
  --force-rerun-from edges `
  --stop-after network `
  --skip-foreground-probe `
  --monitor

# After writer completes:
slavv parity prove-exact --stage edges `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v9 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v9 `
  --oracle-root workspace/oracles/180709_E_full_v2

slavv parity prove-exact --stage network `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v9 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v9 `
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
