# Phase 1 residual experiment analysis

[Up: Reference Docs](../README.md) · [Live status](../core/EXACT_PROOF_FINDINGS.md) · [Operator handoff](../../../.claude/HANDOFF.md) · [Figures](../../../figures/README.md) · [Tasks](../../TODO.md)

**Role:** maintained *experiment framing* for the open Phase 1 residual—not the
live status log and not the task list.

| Need | Single home |
|------|-------------|
| Pass/fail, claim run root, live residual claim | [EXACT_PROOF_FINDINGS](../core/EXACT_PROOF_FINDINGS.md) (banner + blockers) |
| Commands / operating sequence | [.claude/HANDOFF.md](../../../.claude/HANDOFF.md) |
| Checkboxes | [TODO.md](../../TODO.md) |
| Domain terms (Edge Set, Candidate Set, Edge Selection) | [AGENTS.md § Domain Glossary](../../../AGENTS.md#domain-glossary) |
| Ship vs stretch bars | [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md) |
| Reusable analysis skeleton | [EXPERIMENT_ANALYSIS_TEMPLATE](EXPERIMENT_ANALYSIS_TEMPLATE.md) |

---

## Experiment question

What remaining **Edge Set** behavior prevents full-volume Network ADR 0012
**multiset** equality after Energy, Vertices, and Edges ownership/count are green?

## Hypothesis (interpretation frame)

Network is not independently broken. Crop Watershed Discovery / frontier /
generation and crop Edge Selection pair multiset are regression-closed on
re-selection. Full-volume residual is a **Candidate Set** join that Edge Selection
faithfully prunes via degree-excess under equal post-resample max energy: an
**extra** join displaces the oracle pair. Cleanup Python≡MATLAB on the same
exported surface—so production work is **join emission / Candidate Set**, not a
new Edge Selection tie-break policy.

**Do not freeze pair IDs, candidate indices, or strand counts here.** Those live
only in the findings banner.

## Methodology

- **Iteration surface:** crop harness (regression guards: frontier match, generation
  coverage, re-selection pair multiset, cleanup comparator).
- **Claim surface:** full `180709_E` claim run root in findings.
- **Probes (prefer these):** `scripts/edge_selection_funnel_probe.py`,
  `scripts/compare_clean_edge_pairs_matlab.py`,
  `scripts/persist_crop_edges_selection.py` (`select_and_finalize_edge_set`),
  `scripts/watershed_frontier_diff.py` / `watershed_candidate_gap_probe.py`.
- **Anti-patterns:** [UNPRODUCTIVE_LOOPS](../core/UNPRODUCTIVE_LOOPS.md); no
  endpoint-descending cleanup reorder; no Network rewrite.

## Results / next steps

Read **findings top banner + active blockers** and **TODO open Phase 1 rows**.
When the residual moves, update findings + HANDOFF + TODO the same session;
refresh [figures/parity_campaign_series.py](../../../figures/parity_campaign_series.py)
only if publication KPIs change.

## Done criteria

Evaluated `prove-exact --stage edges` and `--stage network` both `passed: true`
on the full claim root (Network = order-independent multiset equality).
