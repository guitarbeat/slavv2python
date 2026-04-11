# Shared Neighborhood Audit Checklist

This checklist turns the new chapter goal into a concrete local audit loop.

Use this file when:

- you are inspecting a worst-case shared neighborhood such as `64`, `359`,
  `866`, or `1283`
- you want to prove where a branch disappears before chooser cleanup
- you need a check-off surface rather than a narrative chapter summary

## Goal

- [ ] Explain one real neighborhood-level mismatch in terms of a specific local
      semantic drift.
- [ ] Reduce that mismatch to an isolated regression test.
- [ ] Land one targeted Python fix without regressing the stage-isolated
      `network` gate.

## Highest-Priority Questions

### 1. Candidate Lifecycle Before Manifest Emission

- [ ] Record which branches are admitted into the temporary candidate state.
- [ ] Record which branches are later invalidated or reassigned.
- [ ] Record which branches survive into the candidate manifest.
- [ ] Confirm whether the first divergence is before or after manifest
      construction.

Current working clue:

- `terminal_frontier_hit > valid frontier connection count` means some branch
  activity is being lost before the manifest, not only during final cleanup.

### 2. Shared-Neighborhood Claim Ordering

- [ ] For each shared neighborhood, identify competing origins.
- [ ] Compare which origin first claims the relevant branch or contact.
- [ ] Check whether Python suppresses a locally plausible branch earlier than
      MATLAB.
- [ ] Treat the neighborhood as the unit of analysis, not just the seed origin.

### 3. Temporary Over-Budget States

- [ ] Check whether Python enforces a local degree or ownership limit before
      MATLAB would.
- [ ] Look for branches that appear only in relaxed experiments.
- [ ] Decide whether the missing behavior is true discovery drift or premature
      suppression.

### 4. Bifurcation-Half Choice And Local Invalidation

- [ ] Compare bifurcation index identification at the shared neighborhood.
- [ ] Compare parent/child energy slices on the exact branch that disappears.
- [ ] Check whether half selection changes which incident pair survives.
- [ ] Keep `edge_selection.py` in scope, not just `edge_candidates.py`.

## Neighborhood Triage Order

### Vertex `359`

- [ ] Confirm why origin `359` only contributes `[359, 181]` from the frontier
      path.
- [ ] Compare its missing MATLAB pairs with neighboring-origin survivors.
- [ ] Decide whether the first mismatch is admission, ownership, or partner
      substitution.

### Vertex `866`

- [ ] Explain why `terminal_frontier_hit = 3` yields only one valid frontier
      connection.
- [ ] Compare missing partners `[1023]`, `[1203]`, and `[1348]` against the
      chosen alternatives.
- [ ] Decide whether this is early invalidation, local competition, or
      bifurcation-half drift.

### Vertex `1283`

- [ ] Explain why `[1283, 1134]`, `[1283, 768]`, and watershed `[1283, 1659]`
      survive while the missing MATLAB pairs do not.
- [ ] Check whether `1319` is lost because of partner substitution rather than
      geometric impossibility.
- [ ] Compare the local branch lifecycle against the MATLAB artifact.

### Vertex `64`

- [ ] Keep `64` as the main saved-batch under-covered seed.
- [ ] Treat it as part of the shared-neighborhood audit, not as a standalone
      special case.
- [ ] Use it to test whether a selective fix generalizes beyond the April 6
      shared-vertex cluster.

## Artifact Requirements

- [ ] Add or preserve a per-neighborhood artifact with:
      admitted branches, invalidated branches, reassigned branches, surviving
      candidate pairs, and final chosen pairs.
- [ ] Make the artifact easy to diff against MATLAB behavior by neighborhood.
- [ ] Keep the artifact under the staged run root instead of scattering ad-hoc
      files.

## Test Requirements

- [ ] New regression tests should live near the owning Python surface.
- [ ] Prefer a deterministic unit or focused regression test over a broad new
      end-to-end sweep.
- [ ] The test should capture one real branch-lifecycle mismatch, not just a
      count delta.

## Verification Order

- [ ] Targeted pytest for the new local regression
- [ ] `python -m pytest -m "unit or integration"`
- [ ] Saved-batch imported-MATLAB loop
- [ ] Stage-isolated `network` gate
- [ ] Fresh live MATLAB confirmation only after the cheaper gates improve

## Completion Criteria

- [ ] We can point to the exact local control-flow moment where one real MATLAB
      branch is lost or replaced in Python.
- [ ] We have a regression test for that behavior.
- [ ] The targeted fix improves parity without a larger downstream regression.
