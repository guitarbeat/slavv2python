# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-07-06 (post-v5 grill)

This is the single successor brief for the current exact-route effort. Do not use
dated agent passovers, PID snapshots, or parallel-work checklists as current
status.

## Canonical records

| Need | Source of truth |
|---|---|
| Active work and checkboxes | [docs/TODO.md](../docs/TODO.md) |
| Verified run status, proof evidence, and blockers | [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) |
| Phase 1 requirements | [phase-1-exact-route-spec.md](../docs/plans/phase-1-exact-route-spec.md) |
| Edges/Network bar + closure policy | [ADR 0012 addendum](../docs/adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-06-post-v5-watershed-iteration-and-v6-closure) |
| Run commands and evidence format | [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md), [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md) |
| Repository and parity guardrails | [AGENTS.md](../AGENTS.md) |

## Current decision point

> **Single canonical status source:** [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) holds authoritative per-stage status.

**Dual bar (unchanged):**

- **Ship gate:** evaluated ADR 0012 per-stage `prove-exact` on **full** `180709_E` (`canonical_full_v6` after milestone).
- **Stretch (non-blocking):** strict-field + candidate-overlap KPI on **crop** (`crop_M_exact_v3`).

**Certified (do not rerun):** Energy + Vertices on full `180709_E` (`canonical_full_v4` / `180709_E_full_v2`).

**v5 outcome (2026-07-06):** Writer **succeeded** (~2.1h, edges→network). Closure proof **not valid** — `adr0012_evaluated: false` (ownership maps missing on full oracle + Python checkpoint). Strict-field edge deficit (60,287 vs 69,500) confirms watershed generation gap still blocks closure.

**Canonical rerun gate:** crop overlap **≥80%** on `crop_M_exact_v3` (baseline **57.89%**, 8,979 / 15,511 MATLAB pairs) before launching `canonical_full_v6`.

## Operating sequence

### A. Watershed iteration (primary loop — until 80% crop overlap)

1. Fix candidate *generation* on current `main`: `matlab_get_edges_by_watershed.py` / `matlab_watershed_heap.py`.
2. After each merged fix: rerun **edges only** on crop (`crop_M_exact_v3` in-place or new `v4` crop root if preserving audit). Use **`.venv\Scripts\slavv.exe`** after **`pip install -e .`** so the writer picks up local code (system `slavv` may lag).
3. Log overlap KPI:
   ```powershell
   .\.venv\Scripts\python.exe scripts/watershed_candidate_gap_probe.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2
   ```
4. Optional diagnostics: `scripts/watershed_frontier_diff.py`, `scripts/watershed_candidate_gap_probe.py`, `workspace/scratch/matlab_edge_instr/` (golden trace).
5. **Do not** launch a full canonical writer until overlap ≥ **80%**.

**2026-07-07:** Production default is `SortedFrontier` (`watershed_frontier_backend=sorted`). Crop overlap KPI still **57.89%**; golden trace matches MATLAB for all **13,706** vertex pops; divergence begins post-vertex (iteration 13,707).

### B. v6 closure (after 80% milestone)

**Map prep (required before ADR 0012 proof):**

1. Run instrumented MATLAB harness on full `180709_E` → `watershed_ownership_map.mat` under batch `data/`.
2. Promote/patch oracle `180709_E_full_v2` (document batch suffix if new promotion).
3. Preflight `canonical_full_v5` → **`canonical_full_v6`**; rerun **edges → network** with Python `--include-debug-maps` on edge capture.

**Writer:**

```powershell
slavv jobs list
slavv parity launch-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v6 `
  --oracle-root workspace/oracles/180709_E_full_v2 `
  --dataset-root workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353 `
  --force-rerun-from edges `
  --stop-after network `
  --skip-foreground-probe `
  --monitor
```

**Closure proof (ship gate — fail loud if maps still missing):**

```powershell
slavv parity prove-exact --stage edges `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v6 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v6 `
  --oracle-root workspace/oracles/180709_E_full_v2

slavv parity prove-exact --stage network `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v6 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v6 `
  --oracle-root workspace/oracles/180709_E_full_v2
```

6. If ADR 0012 passes on both → **Phase 1 closed**. Record evidence per [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md).

### Audit runs (do not overwrite)

| Run | Role |
|-----|------|
| `crop_M_exact` | Pre–PR #103 crop audit |
| `crop_M_exact_v3` | Stretch baseline + iteration KPI |
| `canonical_full_v4` | Certified Energy/Vertices |
| `canonical_full_v5` | Fresh edges/network writer (2026-07-06); invalid ADR 0012 proof |

## Retired coordination material

The one-off `overnight_phase1_runs`, crop-rerun worker briefs, and
`TODO_GPT55_PARALLEL_WORK.md` were consolidated here and into the canonical
records above.
