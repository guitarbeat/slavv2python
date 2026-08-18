# SLAVV Python Weekly Lab Update - Presenter Notes

Deck: `workspace/reports/slavv_weekly_lab_update_2026-06-14.pptx`

Narrative: MATLAB parity moved from open debugging to staged proof. Crop Energy is done. Full Phase 1 certification is still pending.

Avoid overclaiming: do not say Phase 1 is certified, complete, or ship-ready.

## Slide 1: Thesis

Open with the one sentence on the slide. Then state the boundary: crop energy passed, but Phase 1 is not finished.

## Slide 2: Claim boundary

Say this plainly: the project is no longer chasing a better percentage. It is trying to pass exact equality gates in order.

## Slide 3: Work completed

Keep this slide short. The point is not the number of commits; it is that the workflow now has clear surfaces for running, proving, monitoring, and documenting.

## Slide 4: Blockers removed

This is the technical slide. Keep the explanation conceptual: memory fixes let the run finish; numerical/indexing fixes make the comparison fair.

## Slide 5: Current status

This is the most important slide for status. Say: Crop Energy is done. Everything else on the certification sequence remains pending.

## Slide 6: Next steps

End with the bottom line on the slide. If there is time, ask for protected long-run compute time and quick review of the first failing proof field.

## Source anchors

- `docs/TODO.md`: Crop Energy Proof complete; downstream crop and canonical gates pending.
- `docs/reference/core/EXACT_PROOF_FINDINGS.md`: strict-zero bar, v29 88.7% diagnostic baseline, June 2026 float64/Bessel updates.
- `docs/plans/phase-1-exact-route-spec.md`: exact-route Phase 1 boundary and sequential certification requirement.
- `docs/solutions/parity/sparse-meshgrid-memory-optimization.md`: >400 MB per-chunk sparse meshgrid saving.
- `docs/solutions/parity/detached-exact-run-jobs.md`: durable parity job monitoring workflow.