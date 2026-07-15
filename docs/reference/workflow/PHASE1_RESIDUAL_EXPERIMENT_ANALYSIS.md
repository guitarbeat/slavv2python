# Phase 1 residual experiment analysis

[Up: Reference Docs](../README.md) · [Live status](../core/EXACT_PROOF_FINDINGS.md) · [Operator handoff](../../../.claude/HANDOFF.md) · [Figures](../../../figures/README.md)

This document applies the Experiment Analysis template to the current Phase 1
residual. It is a maintained planning aid, **not** the live status log. Update live
run results and blocker status in [EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md);
update commands in [.claude/HANDOFF.md](../../../.claude/HANDOFF.md).

## Experiment question

What remaining edge-set behavior prevents full-volume Network ADR 0012 multiset
equality after Energy, Vertices, and Edges ownership/count are already green?

## Hypothesis

The open Phase 1 failure is not an independent Network defect. Crop watershed
frontier generation matches the MATLAB golden trace and candidate coverage is
complete. MATLAB cleanup row ordering matches on the resampled post-crop surface.
The residual is one equal-metric degree-pruning pair swap (crop: MATLAB
`[4212, 6281]` vs Python `[4043, 6281]`), which becomes Network’s −1 strand on
full volume.

## Methodology

Use the crop harness as the fast iteration surface and the full canonical volume
as the claim surface.

Crop iteration loop:

1. Degree-pruning tie analysis around the shared hub vertex (equal resampled metric).
2. Keep cleanup comparator, golden-trace, and candidate-gap probes as regression guards.
3. Refresh crop Edges only after no-writer evidence improves.

Canonical closure loop:

1. Preserve audit roots (`canonical_full_v4` … `v16`).
2. After crop residual moves, create a fresh successor full-volume root if needed.
3. Rerun Edges → Network with debug maps when justified.
4. Run evaluated per-stage `prove-exact --stage edges` and `--stage network`.

## Current results

**Do not invent or re-table live KPIs here.** Single source of truth:

| Need | Read |
|------|------|
| Stage pass/fail, claim run root | [EXACT_PROOF_FINDINGS](../core/EXACT_PROOF_FINDINGS.md) executive status + active ops |
| Operator residual narrative | [.claude/HANDOFF.md](../../../.claude/HANDOFF.md) |
| Open checkboxes | [TODO.md](../../TODO.md) |

Hypothesis above is the *interpretation frame*; if findings banner or Network proof status moves, re-read findings before planning—not this section.

## Interpretation

- Energy and Vertices should not be rerun without regression evidence.
- Edges ownership/count green on the current claim surface → do not reopen unless proofs regress.
- Network fails because the edge multiset still differs by residual swap class — not a Network rewrite.
- Next useful work is equal-metric degree-pruning tie resolution on crop, then Network proof only if the connection set moves.

## Limitations

- Crop movement is necessary but not sufficient; Phase 1 closes only on full-volume Network multiset equality.
- Approximate strand-count % is **not** the Network ship gate (ADR 0012 = multiset equality).
- Figure KPIs are a publication snapshot: edit only
  [`figures/parity_campaign_series.py`](../../../figures/parity_campaign_series.py)
  and regenerate via `generate_parity_claim_figures.py` when findings move.

## Next steps

1. Triage the equal-metric degree-pruning swap (crop hub vertex 6281).
2. Keep regression guards green (trace match, cleanup comparator, generation gap 0).
3. Update findings + HANDOFF + figure series when residual moves.
4. Launch a successor canonical root only when crop residual justifies full runtime.

## Figure updates tied to this experiment

Canonical figure inventory and captions:
[figures/README.md](../../../figures/README.md).

Claim series data (single edit surface for figure constants):
[`figures/parity_campaign_series.py`](../../../figures/parity_campaign_series.py).

Figures should keep Network’s open multiset residual visible; do not paint
“all gates green” while `exact_proof_network.json` is `passed: false`.

## Done criteria

The experiment is complete when a fresh full-volume run passes:

```powershell
.\.venv\Scripts\slavv.exe parity prove-exact --stage edges `
  --dest-run-root workspace\runs\oracle_180709_E\<canonical_run> `
  --oracle-root workspace\oracles\180709_E_full_v2
.\.venv\Scripts\slavv.exe parity prove-exact --stage network `
  --dest-run-root workspace\runs\oracle_180709_E\<canonical_run> `
  --oracle-root workspace\oracles\180709_E_full_v2
```

with both proofs `passed: true` and Network ADR 0012 multiset equality.
