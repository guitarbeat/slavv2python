# Plans / specs

Scoped initiatives: **requirements and implementation in one spec file** when work is active.

**Tasks:** [TODO.md](../TODO.md) (checkboxes only). **Parity status:** [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md).

## Active specs

| Spec | Status |
|------|--------|
| [phase-1-exact-route-spec.md](phase-1-exact-route-spec.md) | **Active** — Energy/Vertices/Edges evaluated green on full volume; **Network residual** is the open ship gate (see [HANDOFF](../../.claude/HANDOFF.md)) |
| [phase-1-to-phase-2-transition-spec.md](phase-1-to-phase-2-transition-spec.md) | **Draft** — starts only after full-volume Network ADR 0012 is green and the canonical baseline is frozen |
| [phase-2-optimization-spec.md](phase-2-optimization-spec.md) | **Ideation / Draft** — blocked on Phase 1 Network green; do not unwind emulation early |
| [random-component-parity-hardening-spec.md](../investigations/random-component-parity-hardening/random-component-parity-hardening-spec.md) | **Complete (archived)** — random-component suite hardening/refactor |
| [random-component-references-deepening-plan.md](random-component-references-deepening-plan.md) | **Draft** — deepen the References module (follow-up) |

## Workflow

1. Explore in `docs/brainstorms/` when intent is unclear.
2. Promote to `docs/plans/<initiative>-spec.md` (Part 1 requirements, Part 2 implementation).
3. Merge durable context into the spec, then remove the brainstorm or dated draft.
4. Do not maintain separate brainstorm + plan files for the same initiative.

**Naming:** `<initiative>-spec.md` (e.g. `phase-1-exact-route-spec.md`). Do not add dated active-plan filenames for initiatives that already have a spec.
