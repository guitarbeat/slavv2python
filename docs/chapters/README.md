# Chapters

[Up: Documentation Index](../README.md)

This directory holds long-running investigation chapters for the imported-MATLAB
parity effort and related follow-on work.

Use this file when you want the chapter system itself, not the chapter content.
The documentation stack has three layers:

- chapter index: this file, which says which chapter is active and where each
  chapter lives
- chapter entry: the chapter README, which gives the current framing and main
  goal
- working docs: the plan and checklist files, which carry the active loop and
  checkoff surface

Use this file for:

- which chapter is active
- which chapters are historical
- where each chapter starts
- where each chapter closes
- how new chapters should be structured

## Chapter Lifecycle

Each chapter should have one clear role in the documentation system.

- active chapter:
  the current investigation entry point and working spec
- historical handoff chapter:
  a former active chapter that narrowed the problem and handed work forward
- closed chapter:
  a chapter whose primary question was answered and has a closeout document

Each chapter should expose:

- one stable entry document
- one short status line near the top
- a clear successor or closeout link
- the smallest set of working docs needed for that chapter

## Chapter Map

| Role | Chapter | Status | Entry | Best use |
| --- | --- | --- | --- | --- |
| Active chapter | `Neighborhood Claim Alignment` | Active | [README.md](neighborhood-claim-alignment/README.md) | Current parity investigation, live blockers, and next loop |
| Historical handoff | `Candidate Generation Handoff` | Historical handoff | [README.md](candidate-generation-handoff/README.md) | Why the problem stopped being a generic candidate-generation chase |
| Closed chapter | `Imported-MATLAB Parity Closeout` | Closed / historical hub | [parity_closeout.md](imported-matlab-parity-closeout/parity_closeout.md) | What Chapter 1 established and what it handed off |

## Recommended Read Order

1. [Neighborhood Claim Alignment](neighborhood-claim-alignment/README.md)
2. [Root backlog](../../TODO.md)
3. [Candidate Generation Handoff](candidate-generation-handoff/README.md)
4. [Imported-MATLAB Parity Closeout](imported-matlab-parity-closeout/parity_closeout.md)

## Current Evidence Roots

Use these paths when you want the freshest imported-MATLAB evidence surfaces
without hunting through older release-audit notes:

- live `edges` rerun:
  `slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial`
- stage-isolated `network` gate:
  `slavv_comparisons/experiments/live-parity/runs/20260418_network_gate_trial`

## Common Jump Targets

Use this table for the most common jump targets without scanning each chapter.

| Item | Path | Purpose |
| --- | --- | --- |
| Chapter entry | [neighborhood-claim-alignment/README.md](neighborhood-claim-alignment/README.md) | Current parity investigation status and next loop |
| Active chapter plan | [neighborhood-claim-alignment/INVESTIGATION_PLAN.md](neighborhood-claim-alignment/INVESTIGATION_PLAN.md) | Working hypotheses and active loop |
| Active chapter checklist | [neighborhood-claim-alignment/NEIGHBORHOOD_AUDIT_CHECKLIST.md](neighborhood-claim-alignment/NEIGHBORHOOD_AUDIT_CHECKLIST.md) | Checkoff surface for the neighborhood audit |
| Comparison layout smoothing spec archive | [neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md](neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md) | Consolidated spec, design, and completed execution log for comparison layout migration |
| Release verification note | [neighborhood-claim-alignment/release_verification_2026-04-14.md](neighborhood-claim-alignment/release_verification_2026-04-14.md) | Historical release-audit conclusions and evidence summary |
| Lock-contention incident note | [neighborhood-claim-alignment/file_lock_contention_analysis_2026-04-13.md](neighborhood-claim-alignment/file_lock_contention_analysis_2026-04-13.md) | Windows rerun incident handling and safe recovery |
| Canonical acceptance pointer | [../../slavv_comparisons/pointers/canonical_acceptance.txt](../../slavv_comparisons/pointers/canonical_acceptance.txt) | Authoritative pointer to the current acceptance run root |
| Latest completed pointer | [../../slavv_comparisons/pointers/latest_completed.txt](../../slavv_comparisons/pointers/latest_completed.txt) | Authoritative pointer to the latest completed managed run |
| Best saved-batch pointer | [../../slavv_comparisons/pointers/best_saved_batch.txt](../../slavv_comparisons/pointers/best_saved_batch.txt) | Authoritative pointer to the preferred reusable saved-batch run |

## Chapter Conventions

For consistency, chapter entry docs should include these sections when useful:

- chapter status
- why this chapter exists
- starting facts
- lessons from previous runs
- main goal
- scope
- first loop
- deliverables
- working docs
- core references

Historical chapter docs should also include:

- successor chapter
- what the chapter handed off
- what no longer belongs in that chapter
- the closeout report or archive that superseded the working entry

## Template

When opening a new chapter, start from:

- [CHAPTER_TEMPLATE.md](CHAPTER_TEMPLATE.md)
