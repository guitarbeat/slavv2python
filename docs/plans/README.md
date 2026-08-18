# Plans / specs

## In short

Specs here are contracts (what to build). Live pass/fail is
[ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md). Phase 1 already shipped.
Identical last digits is the stretch leftover, not a Phase 1 reopen.

Scoped initiatives: **requirements and implementation in one spec file** when work is active.

**Tasks:** [TODO.md](../TODO.md) (checkboxes only). **Parity status:** [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md).

## Active specs

| Spec | Status |
|------|--------|
| [phase-1-exact-route-spec.md](phase-1-exact-route-spec.md) | **Active requirements** — ship bar already met in [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk); this file is the contract, not live pass/fail; commands in [HANDOFF](../../.claude/HANDOFF.md) |
| [phase-1-to-phase-2-transition-spec.md](phase-1-to-phase-2-transition-spec.md) | **Complete** — Phase 1 CLOSED; baseline freeze recorded 2026-08-17 |
| [phase-2-optimization-spec.md](phase-2-optimization-spec.md) | **Ideation / Draft** — profiling baseline recorded; Fortran unwind still needs an explicit ADR |
| [random-component-parity-hardening-spec.md](../investigations/random-component-parity-hardening/random-component-parity-hardening-spec.md) | **Complete (archived)** — random-component suite hardening/refactor |
| [random-component-references-deepening-plan.md](random-component-references-deepening-plan.md) | **Draft** — deepen the References module (follow-up) |
| [2026-08-14-004-feat-true-zero-tolerance-parity-stretch-plan.md](2026-08-14-004-feat-true-zero-tolerance-parity-stretch-plan.md) | **Active stretch** — extra “identical last digits” bar including Energy; Phase 1 stays CLOSED |
| [2026-08-15-001-feat-zero-tolerance-stretch-experiments-plan.md](2026-08-15-001-feat-zero-tolerance-stretch-experiments-plan.md) | **E11–E20 portfolio** — crop Energy ~90% exact, leftover last-digit diffs; U5/U6 gated |

## Workflow

1. Explore in `docs/brainstorms/` when intent is unclear.
2. Promote to `docs/plans/<initiative>-spec.md` (Part 1 requirements, Part 2 implementation).
3. Merge durable context into the spec, then remove the brainstorm or dated draft.
4. Do not maintain separate brainstorm + plan files for the same initiative.

**Naming:** `<initiative>-spec.md` (e.g. `phase-1-exact-route-spec.md`). Do not add dated active-plan filenames for initiatives that already have a spec.
