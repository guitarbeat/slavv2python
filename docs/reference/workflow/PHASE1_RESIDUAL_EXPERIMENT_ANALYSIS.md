# Phase 1 residual experiment analysis

[Up: Reference Docs](../README.md) · [Live status](../core/EXACT_PROOF_FINDINGS.md) · [Operator handoff](../../../.claude/HANDOFF.md) · [Figures](../../../figures/README.md)

This document applies the Experiment Analysis template to the current Phase 1
residual. It is a maintained planning aid, not the live status log. Update live
run results and blocker status in [EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md);
update commands in [.claude/HANDOFF.md](../../../.claude/HANDOFF.md).

## Experiment question

What remaining candidate-surface behavior prevents full-volume Network ADR 0012
from passing after Energy, Vertices, and Edges are already green?

## Hypothesis

The open Phase 1 failure is not an independent Network defect. Crop watershed
frontier generation now matches the MATLAB golden trace and candidate coverage
is complete. MATLAB cleanup row ordering, degree pruning, and cycle pruning also
match exactly on the Python candidate surface. The remaining problem is that the
Python candidate surface contains extras that enter faithful cleanup and displace
MATLAB final pairs. That residual edge-set balance is enough for Network strand
endpoint-pair multisets to fail on the canonical volume.

## Methodology

Use the crop harness as the fast iteration surface and the full canonical volume
as the claim surface.

Crop iteration loop:

1. Use the selection funnel probe to locate where extra candidates survive and
   displace MATLAB final pairs through crop -> degree/orphan/cycle cleanup.
2. Keep the MATLAB cleanup comparator, Python watershed frontier trace, and candidate-gap probes as
   regression guards.
3. Refresh the crop Edges checkpoint only after no-writer funnel evidence
   improves.

Canonical closure loop:

1. Preserve audit roots (`canonical_full_v4`, `v5`, `v6`, `v7`, `v8`, `v10`).
2. After crop movement is material, create a fresh successor full-volume root.
3. Rerun Edges -> Network with debug maps.
4. Run evaluated per-stage `prove-exact --stage edges` and
   `prove-exact --stage network` against `180709_E_full_v2`.

## Current results

Status snapshot, from the current findings and handoff:

| Surface | Metric | Current value | Interpretation |
|---|---:|---:|---|
| Crop `crop_M_exact_v3` | Candidate overlap | 15,511 / 15,511 (100%) | Retired 80% gate is cleared; generation gap is 0. |
| Crop `crop_M_exact_v3` | Candidate extras | 3,714 | Extra candidates feed final cleanup balance. |
| Crop `crop_M_exact_v3` | Final edge residual | 149 missing / 365 extra | Fast feedback target after candidate-extra changes. |
| Crop `crop_M_exact_v3` | MATLAB cleanup comparator | 0 row mismatches | `clean_edge_pairs`, degree pruning, and cycle pruning are regression guards. |
| Crop golden trace | First split | none observed | Frontier trace matches MATLAB end-to-end on the crop trace. |
| Full `canonical_full_v10` | Edge connections | 70,247 / 69,500 | Edges ownership passes, but strict connection set now over-selects. |
| Full `canonical_full_v10` | Edges ADR 0012 | PASS, ownership 99.9867% | Edges stage is green under the accepted bar. |
| Full `canonical_full_v10` | Network strands | 48,583 / 48,049 | Open ship gate; downstream of residual edge-set mismatch. |

## Interpretation

The evidence narrows the active fix surface:

- Energy and Vertices should not be rerun without regression evidence.
- Edges ownership-map certification is done on `canonical_full_v6`; do not
  reopen it unless an evaluated proof regresses.
- Network fails because the edge stage emits a connection set that is still not
  close enough. Stage isolation with MATLAB edges reproduces Network topology
  exactly.
- Frontier pop order is now a regression guard: the current crop trace matches.
- The next useful work is on candidate extras and the small crop-tail loss;
  cleanup implementation is now a regression guard.

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

1. Extend or run the funnel probe to explain which extra candidates displace the
   remaining 149 missing MATLAB final pairs.
2. Compare the affected candidate/crop behavior against active MATLAB
   `vectorize_V200.m` / `crop_edges_V200` / watershed candidate diagnostics.
3. Patch the smallest confirmed candidate-surface or crop-tail discrepancy.
4. Run the crop no-writer regression guards from the handoff, including the
   cleanup comparator.
5. Refresh crop Edges only after no-writer funnel evidence moves.
6. Update findings and figures only when the tracked metrics move.
7. Launch a successor canonical full root only after crop movement justifies
   full-volume runtime.

## Figure updates tied to this experiment

The figures should support engineering decisions:

- The existing parity journey figure should mark the 80% crop overlap gate as
  retired and make Network the visible open ship gate.
- Add a residual-iteration figure or panel that tracks:
  frontier-trace status, crop final missing/extra gap, full edge gap, and full
  Network strand gap.
- Prefer a checked-in metrics file (for example
  `figures/parity_metrics_current.json`) so figure constants are not copied by
  hand across scripts and captions.

## Done criteria

The experiment is complete when a fresh full-volume run passes:

```powershell
.\.venv\Scripts\slavv.exe parity prove-exact --stage edges `
  --source-run-root workspace/runs/oracle_180709_E/<successor_full_root> `
  --dest-run-root workspace/runs/oracle_180709_E/<successor_full_root> `
  --oracle-root workspace/oracles/180709_E_full_v2

.\.venv\Scripts\slavv.exe parity prove-exact --stage network `
  --source-run-root workspace/runs/oracle_180709_E/<successor_full_root> `
  --dest-run-root workspace/runs/oracle_180709_E/<successor_full_root> `
  --oracle-root workspace/oracles/180709_E_full_v2
```

Both proofs must be evaluated ADR 0012 proofs. Edges-only pass is not closure.
