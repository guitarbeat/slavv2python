# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-08-14 (Phase 1 CLOSED on `canonical_full_v18`; ADR 0013 ranking fix verified)

This is the operator brief for the current exact-route effort. Do not use
dated agent passovers, PID snapshots, or parallel-work checklists as current
status. When findings [ONE TRUTH](../docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) changes, re-synthesize this file the same day.

## Canonical records

| Need | Source of truth |
|---|---|
| **Live pass/fail, residual claim, proof paths** | [EXACT_PROOF_FINDINGS.md — ONE TRUTH](../docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) |
| Active work and checkboxes | [docs/TODO.md](../docs/TODO.md) |
| Phase 1 requirements | [phase-1-exact-route-spec.md](../docs/plans/phase-1-exact-route-spec.md) |
| Edges/Network bar + closure policy | [ADR 0012](../docs/adr/0012-edge-watershed-parity-bar.md) (post-v6 addendum) |
| Claimed Trace Energy provenance | [ADR 0013](../docs/adr/0013-claimed-energy-trace-provenance.md) |
| Run commands and evidence format | [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md), [PARITY_RUN_EVIDENCE.md](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md) |
| Doc authority map | [docs/README.md](../docs/README.md#documentation-authority-map-one-concept--one-home) |
| Repository and parity guardrails | [AGENTS.md](../AGENTS.md) |

## Current decision point

> **Single status home:** [ONE TRUTH](../docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk). Do not invent or restate live counts here.

### Snapshot (no frozen KPIs)

- **Phase 1 is CLOSED** on the claim root named in ONE TRUTH. Energy ✅, Vertices ✅, Edges ADR 0012 evaluated ✅, Network ADR 0012 evaluated ✅.
- Former Network one-strand fail was Edge Selection Ranking Residual; Claimed Trace Energy bake ([ADR 0013](../docs/adr/0013-claimed-energy-trace-provenance.md)) fixed Edge Set ranking and Network followed.
- **Crop = regression guard.** Preserve historical `v16` in place; do not overwrite. **Do not claim `v17`.**
- Cite proofs with `slavv parity inspect-proof --path <json> --require-evaluated`. Do not read the [findings diary](../docs/investigations/exact-proof-findings-diary/README.md) as status.

### Do not

- Re-open Phase 1 on historical `v16` Network FAIL (residual record only).
- Treat MATLAB finals (`edges_*.mat`) as raw watershed emission, or `canonical_full_v17` as a healthy writer.
- Treat approximate strand-count % as Network pass without evaluated multiset proof.
- Re-gate on retired 80% crop overlap or crop one-pair swap as the open loop.
- Rewrite Network; reopen join-rule / tie-scan as the ship-gate change; add endpoint tertiary sort keys.

## Strategy

### Ship gate

1. **Edges ADR 0012** — met on claim root (see ONE TRUTH).
2. **Network ADR 0012** — met on the same claim root (see ONE TRUTH).
3. **Strict-field stretch** — exact connections / order remain optional non-blocking follow-up on crop.

### True zero-tolerance stretch (operator notes)

Labeled **stretch** — does **not** reopen Phase 1 CLOSED / ONE TRUTH.

- Compare with `slavv parity prove-exact … --strict-floats` (default allclose ≠ stretch success).
- Crop Energy unlock first (`crop_M_exact_v3` lineage / `180709_E_crop_M_v2`); full `180709_E` only after a matching unlock field set.
- Engine float path: isolated Python 3.7 + `matlab.engine` (R2019a). Repo `.venv` is 3.12. Set `STRETCH_PY37_PYTHON`, then:
  ```powershell
  slavv parity resume-exact-run `
    --dest-run-root workspace\runs\oracle_180709_E\<new_crop_stretch_root> `
    --oracle-root workspace\oracles\180709_E_crop_M_v2 `
    --force-rerun-from energy --stop-after energy --n-jobs 1 `
    --energy-float-backend matlab_engine
  slavv parity prove-exact --stage energy --strict-floats ...
  ```
- Never overwrite `canonical_full_v18`; use a new stretch dest root.
- Status helpers: `slavv_python.analytics.parity.proof.stretch` (`gate_full_stretch_entry`, unlock + taxonomy).
- Plan: [docs/plans/2026-08-14-004-feat-true-zero-tolerance-parity-stretch-plan.md](../docs/plans/2026-08-14-004-feat-true-zero-tolerance-parity-stretch-plan.md).

### Primary loop KPI

| KPI | Surface | Role |
|-----|---------|------|
| Crop generation / frontier / re-selection | crop harness | Regression guards (closed) |
| Full Edge Set / Network | claim root evaluated proofs | Phase 1 closed — regression only |

Live numbers: ONE TRUTH only.

## Operating sequence

### A. Full residual (primary)

1. Work the full residual at the hub named in ONE TRUTH. Raw pair sets already match; the discriminator is claimed `energy_map` `sort_edges` vs original-field traces. Keep crop re-selection as a regression guard. Do not re-open join-emission, tie-scan, shared vertex-origin restore, or broad endpoint-descending cleanup reorder.
2. After each fix, reinstall and check no parity writer is active:
   ```powershell
   .\.venv\Scripts\pip.exe install -e .
   .\.venv\Scripts\slavv.exe jobs list
   .\.venv\Scripts\slavv.exe parity ensure-oracle-artifacts `
     --oracle-root workspace/oracles/180709_E_crop_M_v2 `
     --stage all `
     --no-repair
   ```
3. Prefer no-writer probes first (fast loop):
   ```powershell
   .\.venv\Scripts\python.exe scripts/watershed_frontier_diff.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2 `
     --regenerate-python

   .\.venv\Scripts\python.exe scripts/watershed_candidate_gap_probe.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2

   .\.venv\Scripts\slavv.exe parity inspect-proof `
     --path workspace/runs/oracle_180709_E/canonical_full_v16/03_Analysis/exact_proof_network.json `
     --require-evaluated
   ```
   Interpretation: crop frontier **match** and generation **closed** are regression guards. Full residual lives on the claim root Candidate Set / join emission.
4. Full-surface funnel / cleanup comparators when diagnosing degree-excess displacement:
   ```powershell
   .\.venv\Scripts\python.exe scripts/edge_selection_funnel_probe.py `
     --run-dir workspace/runs/oracle_180709_E/canonical_full_v16 `
     --oracle-root workspace/oracles/180709_E_full_v2

   .\.venv\Scripts\python.exe scripts/compare_clean_edge_pairs_matlab.py `
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2
   ```
5. Crop Edges refresh only when crop guards regress:
   ```powershell
   .\.venv\Scripts\slavv.exe parity resume-exact-run `
     --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact_v3 `
     --oracle-root workspace/oracles/180709_E_crop_M_v2 `
     --force-rerun-from edges `
     --stop-after edges `
     --skip-preflight
   ```

### B. Successor full claim run (when residual production fix lands)

1. Prefer a **new successor** run root preflighted from the latest claim/audit Energy/Vertices lineage — do **not** destroy `v6`…`v16` audit records.
2. Rerun **edges → network only** with `--include-debug-maps` / `parity_include_debug_maps=true`.
3. Example shape (replace run dir with the new root; use current claim lineage as seed):

On Windows do **not** use `slavv parity launch-exact-run` (child dies on its own lease). Do **not** block on `slavv jobs list` (it hangs). Check `writer_lease.json` + process liveness. Full command and seed rules: [detached-exact-run-jobs.md](../docs/solutions/parity/detached-exact-run-jobs.md) addendum.

```powershell
# After writer completes:
slavv parity prove-exact --stage edges `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v18 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v18 `
  --oracle-root workspace/oracles/180709_E_full_v2

slavv parity prove-exact --stage network `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v18 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v18 `
  --oracle-root workspace/oracles/180709_E_full_v2
```

4. Phase 1 closes when **both** evaluated proofs pass. Update ONE TRUTH + TODO + this handoff + figure series same session. Record evidence per [PARITY_RUN_EVIDENCE](../docs/reference/workflow/PARITY_RUN_EVIDENCE.md).
5. If Edges still pass and Network still fails: continue Edge Set residual loop — do not treat Network as a separate porting project.

### C. After Phase 1 closes (current)

- Promote summary to `workspace/reports/` if warranted.
- Strict-field stretch optional on crop.
- Phase 2 optimization / paper-profile cert per [phase-2-optimization-spec.md](../docs/plans/phase-2-optimization-spec.md) and roadmap.
- Cite claim proofs:
  ```powershell
  slavv parity inspect-proof --path workspace/runs/oracle_180709_E/canonical_full_v18/03_Analysis/exact_proof_edges.json --require-evaluated
  slavv parity inspect-proof --path workspace/runs/oracle_180709_E/canonical_full_v18/03_Analysis/exact_proof_network.json --require-evaluated
  ```

### D. Cold start

1. Read [ONE TRUTH](../docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).
2. No concurrent writer: read `writer_lease.json` and test the PID. Do not block on `slavv jobs list`.
3. Open checkboxes in [TODO.md](../docs/TODO.md).
4. Do not treat [PI_UPDATE.md](../docs/PI_UPDATE.md), investigation archives, or findings **session diary** as live status.

## Primary code and test surfaces

| Area | Files / functions | Planning note |
|------|-------------------|---------------|
| Claimed Trace Energy bake | `matlab_get_edges_by_watershed.py` assemble (`claim_map.energy_map`) | ADR 0013 — regression guard |
| Finalization | `pipeline/edges/finalize.py` | Crop guard; resample/map-resample/smooth/crop path |
| Selection row order | `selection_payloads.py` (`prepare_candidate_indices_for_cleanup`) | Regression guard (double-precision metric sort) |
| Cleanup pruning | `cleanup.py` | Regression guard; MATLAB comparator green on same surface |
| Adjusted energies | `matlab_get_edges_v300_geometry.py` | LUT `unit_vectors` direction |
| Tie-break guardrail | `matlab_indexing.py` | Frontier linear-index tie-break |

Focused unit checks after code changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_global_watershed_comprehensive.py
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_global_watershed_anisotropic.py tests/unit/pipeline/test_frontier_math.py tests/unit/pipeline/test_fortran_tie_breaking.py
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_edges_comprehensive.py tests/unit/pipeline/test_partner_substitution_regression.py
.\.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_watershed_tracing.py tests/unit/pipeline/test_float64_dtype_invariant.py
.\.venv\Scripts\python.exe -m pytest tests/integration/parity/test_parity_pre_gate_tier1.py
```

## Audit runs (do not overwrite)

Historical claim/audit roots (`crop_M_exact*`, `canonical_full_v4`…`v16`) stay on disk. Live claim surface name is only in ONE TRUTH — do not freeze a run ID here as “current” without re-checking findings.

## Meta / process

| Shortcoming | Fix |
|-------------|-----|
| HANDOFF / TODO lag findings | Re-synthesize HANDOFF when ONE TRUTH moves; TODO = checkboxes only |
| Second status tables in prose | Authority map: ONE TRUTH wins |
| Ownership-map missing → false closure | Require `adr0012_evaluated: true` |
| Claiming “Network bug” from multiset fail | Re-check MATLAB-edge isolation first |

## Anti-patterns

See [UNPRODUCTIVE_LOOPS.md](../docs/reference/core/UNPRODUCTIVE_LOOPS.md). Short list: stale gates, probe orientation (`mpv`), Network rewrite, cleanup secondary keys, inventing KPIs outside findings, reading session diary as current status.
