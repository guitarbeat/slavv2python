# Phase 1 residual experiment analysis

[Up: Reference Docs](../README.md) · [Live status](../core/EXACT_PROOF_FINDINGS.md) · [Operator handoff](../../../.claude/HANDOFF.md) · [Figures](../../../figures/README.md)

This document applies the Experiment Analysis template to the current Phase 1
residual. It is a maintained planning aid, not the live status log. Update live
run results and blocker status in [EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md);
update commands in [.claude/HANDOFF.md](../../../.claude/HANDOFF.md).

## Experiment question

What remaining watershed-generation behavior prevents full-volume Network ADR
0012 from passing after Energy, Vertices, and Edges are already green?

## Hypothesis

The open Phase 1 failure is not an independent Network defect. It is caused by a
residual Edges candidate-generation / claiming-state divergence in the global
watershed. Specifically, Python over-claims or assigns different strel neighbors
before the first golden-trace divergence, so the emitted connection set remains
short enough that Network strand endpoint-pair multisets fail.

## Methodology

Use the crop harness as the fast iteration surface and the full canonical volume
as the claim surface.

Crop iteration loop:

1. Regenerate and diff the Python watershed frontier trace against the MATLAB
   golden trace.
2. Measure live candidate coverage against the crop MATLAB oracle.
3. Refresh the crop Edges checkpoint only after no-writer probes improve.
4. Use the selection funnel probe to confirm whether residual losses still arise
   before finalization.

Canonical closure loop:

1. Preserve audit roots (`canonical_full_v4`, `v5`, `v6`).
2. After crop movement is material, create a fresh full-volume root
   (`canonical_full_v7` preferred).
3. Rerun Edges -> Network with debug maps.
4. Run evaluated per-stage `prove-exact --stage edges` and
   `prove-exact --stage network` against `180709_E_full_v2`.

## Current results

Status snapshot, from the current findings and handoff:

| Surface | Metric | Current value | Interpretation |
|---|---:|---:|---|
| Crop `crop_M_exact_v3` | Candidate overlap | 15,094 / 15,511 (97.31%) | Retired 80% gate is cleared. |
| Crop `crop_M_exact_v3` | Final edge gap | 589 | Fast feedback target after claim-state changes. |
| Crop golden trace | First split | ~13,761 | Main localization target: strel argmin / claiming-state divergence. |
| Full `canonical_full_v6` | Edge connections | 65,436 / 69,500 | Edges ownership passes, but strict connection gap remains. |
| Full `canonical_full_v6` | Edges ADR 0012 | PASS, ownership 96.02% | Edges stage is green under the accepted bar. |
| Full `canonical_full_v6` | Network strands | 44,595 / 48,049 | Open ship gate; downstream of residual edge gap. |

## Interpretation

The evidence narrows the active fix surface:

- Energy and Vertices should not be rerun without regression evidence.
- Edges ownership-map certification is done on `canonical_full_v6`; do not
  reopen it unless an evaluated proof regresses.
- Network fails because the watershed emits a connection set that is still too
  short. Stage isolation with MATLAB edges reproduces Network topology exactly.
- Frontier pop order is not the leading hypothesis: current evidence says pops
  match through the known divergence.
- The next useful work is on claiming-state and strel-neighbor selection around
  `claim_unowned_strel`, `d_over_r_map`, `size_map`, pointer values, and adjusted
  strel energies.

## Limitations

- Crop movement is necessary but not sufficient; Phase 1 closes only on the full
  canonical volume.
- Strict-field edge-pair equality is a useful residual signal, not the Edges
  ship gate.
- `prove-exact-sequence` strict-field failure is diagnostic only for
  Edges/Network. Use evaluated per-stage ADR 0012 proofs for closure.
- Figure constants are currently manual. Until they are read from a shared data
  file, every figure update must be cross-checked against
  [EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md).

## Next steps

1. Add or refine targeted instrumentation for the first divergent strel:
   current linear index, strel linear indices, vertices of current strel,
   adjusted energies, pointer values, `d_over_r`, and size labels.
2. Compare the instrumented Python state against the MATLAB golden trace/state
   at the first split.
3. Patch the smallest confirmed claiming-state discrepancy.
4. Run the crop no-writer probes from the handoff.
5. Refresh crop Edges only after no-writer probes move.
6. Update findings and figures only when the tracked metrics move.
7. Launch `canonical_full_v7` only after crop movement justifies full-volume
   runtime.

## Figure updates tied to this experiment

The figures should support engineering decisions:

- The existing parity journey figure should mark the 80% crop overlap gate as
  retired and make Network the visible open ship gate.
- Add a residual-iteration figure or panel that tracks:
  first-diverge iteration, crop final gap, full edge gap, and full Network strand
  gap.
- Prefer a checked-in metrics file (for example
  `figures/parity_metrics_current.json`) so figure constants are not copied by
  hand across scripts and captions.

## Done criteria

The experiment is complete when a fresh full-volume run passes:

```powershell
.\.venv\Scripts\slavv.exe parity prove-exact --stage edges `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v7 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v7 `
  --oracle-root workspace/oracles/180709_E_full_v2

.\.venv\Scripts\slavv.exe parity prove-exact --stage network `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v7 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v7 `
  --oracle-root workspace/oracles/180709_E_full_v2
```

Both proofs must be evaluated ADR 0012 proofs. Edges-only pass is not closure.
