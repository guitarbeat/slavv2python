# Shared Neighborhood Claim Investigation Plan

Status: Active Chapter 3 plan
Updated: 2026-04-20

Use [README.md](README.md) for chapter framing and
[NEIGHBORHOOD_AUDIT_CHECKLIST.md](NEIGHBORHOOD_AUDIT_CHECKLIST.md) for the
working audit checklist.

Use this file when you want to know:

- which Python surfaces should change next
- what the current leading hypotheses are
- what counts as progress for the new chapter

## Rapid Recall

- Already solved:
  - exact vertex parity on the imported-MATLAB surface
  - exact stage-isolated `network` parity when exact MATLAB `edges` are
    imported
- Current live evidence roots:
  - `slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial`
  - `slavv_comparisons/experiments/live-parity/runs/20260418_network_gate_trial`
- Current live imported-MATLAB `edges` trial:
  - `vertices 1682/1682`
  - `edges 1555/1379`
  - `strands 774/682`
- Current stage-isolated `network` gate:
  - `vertices 1682/1682`
  - `edges 1379/1379`
  - `strands 682/682`
- Strongest current clue:
  - the live divergence mix now points first at partner choice, branch
    invalidation, and a smaller claim-ordering remainder around shared active
    neighborhoods
- Standing warning:
  - better candidate counts alone do not guarantee better final edge or strand
    parity

## Current Problem Statement

The parity target is now narrower than "make candidate generation better."

The docs and retained artifacts point to a neighborhood-scoped mismatch:

- missing MATLAB pairs often never enter the Python candidate manifest
- nearby alternate partners remain active and are sometimes chosen instead
- `terminal_frontier_hit` can exceed the number of valid frontier connections
  that survive into the candidate payload

That suggests the active problem surface is the candidate lifecycle around a
shared neighborhood:

1. admission
2. temporary ownership
3. invalidation or reassignment
4. manifest survival
5. chooser compatibility

## Active Code Surface

Primary surfaces:

- [edge_candidates.py](../../../source/slavv/core/edge_candidates.py)
  for frontier admission, ownership, pointer maps, watershed supplementation,
  and candidate manifest construction
- [edge_selection.py](../../../source/slavv/core/edge_selection.py)
  for parent/child invalidation, bifurcation-half choice, and chooser-facing
  cleanup before final edge selection
- [comparison.py](../../../source/slavv/parity/comparison.py)
  for parity orchestration and diagnostic artifact capture

Supporting surfaces:

- [metrics.py](../../../source/slavv/parity/metrics.py)
  for candidate-coverage and shared-vertex reporting
- [reporting.py](../../../source/slavv/parity/reporting.py)
  for human-readable parity summaries
- [test_frontier_tracing.py](../../../dev/tests/unit/core/test_frontier_tracing.py)
  and
  [test_edge_cases.py](../../../dev/tests/unit/core/test_edge_cases.py)
  for local tracer semantics

## Leading Hypotheses

### 1. Selective local invalidation is still wrong

Some branches may be explored and briefly counted as terminal hits, but then
invalidated or reassigned before they survive into the candidate manifest.

Why it remains live:

- `terminal_frontier_hit > valid frontier connection count`
- worst-case neighborhoods stay active rather than dead
- blanket full-path claim was too aggressive, which implies the fix is likely
  selective rather than global

### 2. MATLAB may allow temporary local over-budget states

Python may suppress a branch too early because it enforces local degree or
ownership limits before MATLAB would.

Why it remains live:

- the relaxed geodesic experiment showed some missing pairs can appear when
  constraints are loosened
- docs explicitly ask whether MATLAB allows temporary over-budget admission
  before later cleanup

### 3. Bifurcation-half choice can still change incident pairs

If Python identifies the same trace but chooses the wrong bifurcation point or
origin half, it can keep the wrong branch alive.

Why it remains live:

- this section is still open in the historical audit checklist
- the artifact pattern fits local branch survival drift as much as pure
  discovery drift

### 4. Shared-map behavior is still under-modeled

Python may still be missing some of the neighborhood-level semantics implied by
MATLAB `get_edges_by_watershed`, especially where multiple origins compete
around the same active area.

## Implementation Phases

### Phase 1: Build the right artifact

- Add a neighborhood-scoped artifact that records:
  - admitted candidate branches
  - invalidated branches
  - reassigned or overwritten branches
  - surviving manifest pairs
  - final chosen pairs
- Group the artifact by neighborhood seed or shared-vertex cluster rather than
  only by origin.

### Phase 2: Explain one real neighborhood end to end

- Start with `359`, `866`, and `1283`.
- For each neighborhood, answer:
  - which branches did Python explore?
  - which branches did MATLAB preserve?
  - where did the first divergence happen?

### Phase 3: Isolate the mismatch in a testable form

- Reduce one real divergence to a deterministic unit or regression test.
- Keep the isolated case close to the artifact pattern instead of inventing a
  synthetic shape that loses the shared-neighborhood behavior.

### Phase 4: Make one selective fix

- Prefer a local ownership, invalidation, or bifurcation-choice fix over a new
  global threshold.
- Do not repeat:
  - blanket full-path claim
  - broad source-preference pruning
  - untargeted threshold sweeps

### Phase 5: Revalidate in the right order

1. neighborhood-scoped regression test
2. imported-MATLAB `edges` rerun on the current live trial surface
3. stage-isolated `network` gate
4. fresh live MATLAB confirmation

## Non-Goals

- Do not treat generic downstream `network` construction as the active problem
  unless the stage-isolated gate regresses.
- Do not optimize for count agreement without explaining the local semantic
  mismatch.
- Do not promote a fix just because it helps origin `64` if it worsens the
  shared-neighborhood behavior elsewhere.

## Acceptance Criteria

This chapter succeeds when:

- we can explain at least one worst-case neighborhood in terms of a specific
  local semantic mismatch
- we have an isolated regression test for that mismatch
- a targeted Python change improves the imported-MATLAB `edges` rerun
  without causing a worse regression elsewhere
- the stage-isolated `network` gate remains exact

## Related Docs

- [README.md](README.md)
- [NEIGHBORHOOD_AUDIT_CHECKLIST.md](NEIGHBORHOOD_AUDIT_CHECKLIST.md)
- [Candidate Generation Handoff](../candidate-generation-handoff/README.md)
- [MATLAB Parity Audit Checklist](../imported-matlab-parity-closeout/MATLAB_PARITY_AUDIT_CHECKLIST.md)

