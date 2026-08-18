# ADR 0013: Claimed Energy Trace Provenance

## In short

When ranking candidate edges, use energies sampled from the **claimed** map
(after watershed writes), not from the original Energy photo. That ranking
mismatch was the last Phase 1 leftover; it is closed. Do not “fix” it by
rewriting Network.

## Status
Accepted (2026-08-14)

## Context

MATLAB ranks watershed candidates during Edge Selection by sampling the **claimed/penalized** watershed energy volume (claim writes, then `sort_edges` on raw `max`, ascending). Python’s Candidate traces historically carried samples from the **original** Energy-stage field. That provenance mismatch **was** the Certification residual class ([Edge Selection Ranking Residual](../reference/core/EXACT_PROOF_FINDINGS.md#former-residual-closed-on-v18)); it is closed on the claim root in [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

Three production placements were considered for correcting provenance:

1. **Bake at Watershed Discovery finalize** — sample the Claimed Energy Map into Candidate `energy_traces` when candidates are materialized.
2. **Re-sample at Edge Selection** — keep original-field traces on disk; re-sample the claim map only when ranking.
3. **Hardcoded per-pair overrides** (diagnostic E4-style) — force known residual pairs’ energies without a general provenance rule.

## Decision

**Production provenance is bake-at-finalize (option 1).** Watershed Discovery finalize stores Claimed Trace Energy on Candidates. Edge Selection keeps MATLAB-shaped raw-max `sort_edges` semantics and does **not** become the owner of claim-map sampling. Selection-time re-sample and hardcoded pair overrides remain **diagnostic-only**.

Do **not** change the watershed discovery algorithm beyond sampling the Claimed Energy Map into traces at finalize. Do **not** treat a Network-stage rewrite as the default response if Network ADR 0012 still fails after this fix—deepen ranking/provenance first.

## Consequences

- Candidate Set energies used for Certification ranking must mean Claimed Trace Energy after the fix lands; original-field samples are no longer the ranking surface.
- Cheap Parity Ladder (unit → crop → full no-writer) must stay green before merging the production change; ladder/audit greens are not Phase 1 Closure.
- After the fix, open a **new** Claim Run Root for evaluated ADR 0012 Edges **and** Network; preserve the historical claim root in place. Diagnostic successor writers remain non-claim until those proofs pass.
- Glossary terms: Claimed Energy Map, Claimed Trace Energy, Edge Selection Ranking Residual (AGENTS.md / GLOSSARY.md). Live pass/fail and residual pairs stay only in ONE TRUTH.

## Considered Options

| Option | Why rejected for production |
| --- | --- |
| Selection-time re-sample | Surprising second source of truth; Candidate Set on disk would still disagree with MATLAB ranking inputs; easy to “fix” later by baking anyway. Acceptable as a probe. |
| Hardcoded pair overrides | Non-general; hides provenance; cannot be Certification standing. |
| Network rewrite | Confirmed isolation shows Network is deterministic given edges; wrong default for this residual. |
