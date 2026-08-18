# Phase 1 residual experiment analysis

[Up: Reference Docs](../README.md) · [Live status](../core/EXACT_PROOF_FINDINGS.md) · [Operator handoff](../../../.claude/HANDOFF.md) · [Figures](../../../figures/README.md) · [Tasks](../../TODO.md)

## In short

This is the write-up of a **closed** leftover: Python ranked the wrong extra
edge because it sampled the original Energy field instead of the claimed map.
That is history. Live pass/fail is [ONE TRUTH](../core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) (CLOSED). Do not treat this file as an open Network bug.

**Role:** maintained *experiment framing* for the closed Phase 1 ranking residual—not the
live status log and not the task list. Live pass/fail: [ONE TRUTH](../core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) (CLOSED).

| Need | Single home |
|------|-------------|
| Pass/fail, claim run root, live residual claim | [ONE TRUTH](../core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) |
| Commands / operating sequence | [.claude/HANDOFF.md](../../../.claude/HANDOFF.md) |
| Checkboxes | [TODO.md](../../TODO.md) |
| Domain terms (Edge Set, Candidate Set, Edge Selection) | [AGENTS.md § Domain Glossary](../../../AGENTS.md#domain-glossary) |
| Ship vs stretch bars | [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md) |
| Reusable analysis skeleton | [EXPERIMENT_ANALYSIS_TEMPLATE](EXPERIMENT_ANALYSIS_TEMPLATE.md) |
| Experiment data / artifact-class rules | `slavv_python.analytics.parity.experiments` · [parity-experiment-hygiene.md](../../solutions/best-practices/parity-experiment-hygiene.md) |

---

## Experiment question

What remaining **Edge Set** behavior prevents full-volume Network ADR 0012
**multiset** equality after Energy, Vertices, and Edges ownership/count are green?

## Hypothesis (interpretation frame)

See **[Former residual (closed on v18)](../core/EXACT_PROOF_FINDINGS.md#former-residual-closed-on-v18)**
for the mechanism. Network was not independently broken. Crop generation /
re-selection are regression-closed. Full-volume raw Candidate Sets already match;
the residual was ranking (`sort_edges` on claimed `energy_map` vs original-field
traces) under an equal post-resample max, not a new Edge Selection policy and not
a join-emission rewrite. **Live whether Phase 1 is closed:** [ONE TRUTH](../core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

**Do not freeze pair IDs, candidate indices, or strand counts here.** Those live
only in the findings banner.

## Methodology

- **Cheap loop first:** unit/synthetic → crop pair-set → no-writer re-selection →
  full writer only if the cheap layer cannot falsify the hypothesis. Rules:
  [parity-experiment-hygiene.md](../../solutions/best-practices/parity-experiment-hygiene.md).
- **Artifact class:** compare raw↔raw and final↔final only
  ([raw-vs-final-candidate-compare.md](../../solutions/parity/raw-vs-final-candidate-compare.md)).
- **Iteration surface:** crop harness (`crop_M_exact_v3` candidates; not unevaluated
  proof JSON).
- **Claim surface:** full `180709_E` claim run root in findings. Do not claim `v17`.
- **Probes (prefer these):** `slavv_python.analytics.parity.experiments`,
  `scripts/edge_selection_funnel_probe.py`,
  `scripts/compare_clean_edge_pairs_matlab.py`,
  `scripts/persist_crop_edges_selection.py` (`select_and_finalize_edge_set`),
  `scripts/watershed_candidate_gap_probe.py` (`coverage_of_finals_by_raw`, not equality),
  `tests/unit/pipeline/test_watershed_energy_map_sort_experiments.py`.
- **Anti-patterns:** [UNPRODUCTIVE_LOOPS](../core/UNPRODUCTIVE_LOOPS.md) §16–17;
  no join-rule / tie-scan ship-gate change; no endpoint-descending cleanup
  reorder; no Network rewrite.

## Results / next steps

Read **[ONE TRUTH](../core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk)** and **TODO open Phase 1 rows**.
When the residual moves, update ONE TRUTH + HANDOFF + TODO the same session;
refresh [figures/parity_campaign_series.py](../../../figures/parity_campaign_series.py)
only if publication KPIs change.

## Done criteria

Evaluated `prove-exact --stage edges` and `--stage network` both `passed: true`
on the full claim root (Network = order-independent multiset equality).
