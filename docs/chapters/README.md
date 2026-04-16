# Chapters

This directory holds long-running investigation chapters for the imported-MATLAB
parity effort and related follow-on work.

Use this file when you want the chapter system itself:

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

## Current Map

| Chapter | Status | Entry | Best use |
| --- | --- | --- | --- |
| `shared-neighborhood-claim-alignment` | Active | [README.md](shared-neighborhood-claim-alignment/README.md) | Current parity investigation, live blockers, and next loop |
| `shared-candidate-generation` | Historical handoff | [README.md](shared-candidate-generation/README.md) | Why the problem stopped being a generic candidate-generation chase |
| `imported-matlab-parity` | Closed / historical hub | [PARITY_REPORT_2026-04-09.md](imported-matlab-parity/PARITY_REPORT_2026-04-09.md) | What Chapter 1 established and what it handed off |

## Recommended Read Order

1. [Active chapter](shared-neighborhood-claim-alignment/README.md)
2. [Root backlog](../../TODO.md)
3. [Historical handoff chapter](shared-candidate-generation/README.md)
4. [Chapter 1 closeout](imported-matlab-parity/PARITY_REPORT_2026-04-09.md)

## Reference Shortcuts

Use this table for the most common jump targets without scanning each chapter.

| Item | Path | Purpose |
| --- | --- | --- |
| Active chapter entry | [shared-neighborhood-claim-alignment/README.md](shared-neighborhood-claim-alignment/README.md) | Current parity investigation status and next loop |
| Comparison layout smoothing spec archive | [shared-neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md](shared-neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md) | Consolidated spec, design, and completed execution log for comparison layout migration |
| Release verification note | [shared-neighborhood-claim-alignment/release_verification_2026-04-14.md](shared-neighborhood-claim-alignment/release_verification_2026-04-14.md) | Release-audit conclusions and evidence summary |
| Lock-contention incident note | [shared-neighborhood-claim-alignment/file_lock_contention_analysis_2026-04-13.md](shared-neighborhood-claim-alignment/file_lock_contention_analysis_2026-04-13.md) | Windows rerun incident handling and safe recovery |
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

## Template

When opening a new chapter, start from:

- [CHAPTER_TEMPLATE.md](CHAPTER_TEMPLATE.md)
