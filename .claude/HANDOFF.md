# Phase 1 parity handoff and synthesis

**Last synthesized:** 2026-08-18 (Phase 1 CLOSED + frozen baseline; workspace volumes lightened; stretch `blocked_float_path`)

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
- **Frozen baseline:** [phase1-baseline-freeze.json](../docs/reference/core/phase1-baseline-freeze.json). Do not overwrite freeze JSON `do_not_overwrite` / `LIVE_DEST_NAMES` (`canonical_full_v18`, `crop_M_exact_v3`, `crop_M_stretch_engine_v2`). Writer blocklist `PROTECTED_DEST_NAMES` also includes historical `canonical_full_v16`.
- Former Network one-strand fail was Edge Selection Ranking Residual; Claimed Trace Energy bake ([ADR 0013](../docs/adr/0013-claimed-energy-trace-provenance.md)) fixed Edge Set ranking and Network followed.
- **Crop = regression guard.** Historical `v16` proofs are archived (volume removed 2026-08-18). **Do not claim `v17`.**
- Cite proofs with `slavv parity inspect-proof --path <json> --require-evaluated`. Do not read the [findings diary](../docs/investigations/exact-proof-findings-diary/README.md) as status.

### Do not

- Re-open Phase 1 on historical `v16` Network FAIL (residual record only; proofs in `workspace/reports/phase1_volume_archive/canonical_full_v16/`).
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

**In short:** Phase 1 already shipped. Stretch is the extra “every Energy number identical bits” goal. The crop is about 90% exact; leftover last-digit diffs are **not** 100%. Tiny photos matching does not solve the crop. Do **not** rerun the crop Energy writer.

Labeled **stretch** — does **not** reopen Phase 1 CLOSED / ONE TRUTH. Live KPIs: [findings stretch subsection](../docs/reference/core/EXACT_PROOF_FINDINGS.md#true-zero-tolerance-stretch-separate-from-phase-1) and dest `stretch_status.json`. Readable diagnosis: [crop-energy-stretch-float-isolation.md](../docs/solutions/parity/crop-energy-stretch-float-isolation.md). Do **not** relaunch `crop_M_stretch_engine_v2`. U5/U6 stay gated without a stretch unlock token. Never overwrite freeze JSON `do_not_overwrite` (`LIVE_DEST_NAMES`). Writer blocklist is `PROTECTED_DEST_NAMES` (those three plus `canonical_full_v16`).

Inspect command is in [Operating sequence A](#a-stretch-energy-isolation-current).

### Primary loop KPI

| KPI | Surface | Role |
|-----|---------|------|
| Crop generation / frontier / re-selection | crop harness | Regression guards (closed) |
| Full Edge Set / Network | claim root evaluated proofs | Phase 1 closed — regression only |

Live numbers: ONE TRUTH only.

## Operating sequence

### A. Stretch Energy isolation (current)

Phase 1 ranking residual is **closed** (Claimed Trace Energy / ADR 0013 on the claim root in ONE TRUTH). Do not re-open join-emission, tie-scan, or Network rewrite as the ship loop.

Current operator loop is [true zero-tolerance stretch](../docs/reference/core/EXACT_PROOF_FINDINGS.md#true-zero-tolerance-stretch-separate-from-phase-1): crop Energy last-digit leftover is still open (`blocked_float_path`). Inspect the existing v2 proof; do not relaunch v2. Cheap next probe (not a writer): two tiles on a tiny volume — see the isolation note.

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

Phase 1 already closed on `canonical_full_v18`. Do **not** resurrect `v4`…`v16` volumes. Cite evaluated proofs (section C). If a later parity-sensitive fix needs a new claim root, preflight from the certified Energy/Vertices checkpoints on `v18` and rerun **edges → network only** — never overwrite `canonical_full_v18`.

### C. After Phase 1 closes (current)

- Frozen hash bridge: [phase1-baseline-freeze.json](../docs/reference/core/phase1-baseline-freeze.json).
- Phase 2 profiling baseline (read-only): [phase2-profiling-baseline.json](../docs/reference/core/phase2-profiling-baseline.json). Energy/Vertices elapsed 0 = carried lineage; measured dest bottleneck is Edges. Energy `--n-jobs auto` is implemented (opt-in; dest default stays 1) — do not reimplement. Next performance slice: Edges/Network profiling on an authorized writer. **Fortran-order unwind still needs an explicit Phase 2 ADR** before production code changes. Paper-profile volume/oracle TBD.
- Strict-field stretch optional on crop (gated on Energy unlock).
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
| Stretch leftover isolation | `pipeline/energy/stretch_helper_body_isolation.py` | Crop last-digit leftover; not unlock |
| Phase 2 profiling baseline | `analytics/performance/phase2_baseline.py` | Read-only frozen dest timings; not unwind |
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

Live dests: `canonical_full_v18`, `crop_M_exact_v3`, `crop_M_stretch_engine_v2`. Historical `v4`…`v16` / `crop_M_exact` / stretch v1 volumes were removed 2026-08-18; proof JSON lives under `workspace/reports/phase1_volume_archive/`. Live claim surface name is only in ONE TRUTH — do not freeze a run ID here as “current” without re-checking findings.

## Meta / process

| Shortcoming | Fix |
|-------------|-----|
| HANDOFF / TODO lag findings | Re-synthesize HANDOFF when ONE TRUTH moves; TODO = checkboxes only |
| Second status tables in prose | Authority map: ONE TRUTH wins |
| Ownership-map missing → false closure | Require `adr0012_evaluated: true` |
| Claiming “Network bug” from multiset fail | Re-check MATLAB-edge isolation first |

## Anti-patterns

See [UNPRODUCTIVE_LOOPS.md](../docs/reference/core/UNPRODUCTIVE_LOOPS.md). Short list: stale gates, probe orientation (`mpv`), Network rewrite, cleanup secondary keys, inventing KPIs outside findings, reading session diary as current status.
