# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-08-16 (Phase 1 CLOSED; stretch `blocked_float_path`; stale open-ship-gate operator loop removed)

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

Labeled **stretch** — does **not** reopen Phase 1 CLOSED / ONE TRUTH. Rules and live KPIs: [findings stretch subsection](../docs/reference/core/EXACT_PROOF_FINDINGS.md#true-zero-tolerance-stretch-separate-from-phase-1) and dest `workspace/runs/oracle_180709_E/crop_M_stretch_engine_v2/stretch_status.json` (`blocked_float_path`). Do **not** relaunch that dest.

```powershell
slavv parity inspect-proof --path workspace\runs\oracle_180709_E\crop_M_stretch_engine_v2\03_Analysis\exact_proof_energy.json
```

U5/U6 stay gated without a stretch unlock token. Never overwrite `canonical_full_v18` or `crop_M_exact_v3`. One production chunk (ZYX `(13, 0, 0)`, scale 43, octave 2, chunk 0) named helper/oracle (re-run == v2, both ≠ oracle; packaging OK). Lattice/params: rf-matched lattices identical (821=821); octave-index 75 vs 726 is `unique()` labeling; residual is helper body vs original MATLAB chunk math. Not unlock. Scratch: `workspace/scratch/stretch_one_production_chunk.json`, `workspace/scratch/stretch_lattice_params_isolation.json`.

### Primary loop KPI

| KPI | Surface | Role |
|-----|---------|------|
| Crop generation / frontier / re-selection | crop harness | Regression guards (closed) |
| Full Edge Set / Network | claim root evaluated proofs | Phase 1 closed — regression only |

Live numbers: ONE TRUTH only.

## Operating sequence

### A. Stretch Energy isolation (current)

Phase 1 ranking residual is **closed** (Claimed Trace Energy / ADR 0013 on the claim root in ONE TRUTH). Do not re-open join-emission, tie-scan, or Network rewrite as the ship loop.

Current operator loop is [true zero-tolerance stretch](../docs/reference/core/EXACT_PROOF_FINDINGS.md#true-zero-tolerance-stretch-separate-from-phase-1): crop Energy `--strict-floats` is `blocked_float_path`. Inspect the existing v2 proof; do not relaunch v2.

```powershell
.\.venv\Scripts\pip.exe install -e .
slavv parity inspect-proof --path workspace\runs\oracle_180709_E\crop_M_stretch_engine_v2\03_Analysis\exact_proof_energy.json
```

Crop regression guards (read-only; do not overwrite `crop_M_exact_v3`):

```powershell
.\.venv\Scripts\python.exe scripts/watershed_frontier_diff.py `
  --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
  --oracle-root workspace/oracles/180709_E_crop_M_v2 `
  --regenerate-python
```

Closed ranking history: [Former residual (closed on v18)](../docs/reference/core/EXACT_PROOF_FINDINGS.md#former-residual-closed-on-v18). Funnel / cleanup comparators remain available as regression probes, not as a Phase 1 reopen.

### B. Successor full claim run (closed)

Phase 1 already closed on `canonical_full_v18`. Do **not** destroy `v6`…`v16` audit records. Cite evaluated proofs (section C). If a later parity-sensitive fix needs a new claim root, preflight from the certified Energy/Vertices lineage and rerun **edges → network only** — never overwrite `canonical_full_v18`.

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
