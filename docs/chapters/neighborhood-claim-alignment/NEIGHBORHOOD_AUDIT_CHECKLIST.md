# Shared Neighborhood Audit Checklist

Status: Working checklist
Started: 2026-04-10

Use [README.md](README.md) for chapter framing and
[INVESTIGATION_PLAN.md](INVESTIGATION_PLAN.md) for the active loop.

This checklist turns the new chapter goal into a concrete local audit loop.

Use this file when:

- you are inspecting a worst-case shared neighborhood such as `64`, `359`,
  `866`, or `1283`
- you want to prove where a branch disappears before chooser cleanup
- you need a check-off surface rather than a narrative chapter summary

## Goal

- [ ] Explain one real neighborhood-level mismatch in terms of a specific local
      semantic drift.
- [x] Reduce that mismatch to an isolated regression test.
- [ ] Land one targeted Python fix without regressing the stage-isolated
      `network` gate.

Current progress (2026-04-18):

- Focused regressions now cover:
  - pre-manifest loss when `terminal_frontier_hit > valid frontier connection count`
  - partner substitution at a shared neighborhood in `edge_selection`
  - cleanup retention vs `final_cleanup_dropped` in the frontier lifecycle artifact
- The local frontier fix is in `source/slavv/core/_edge_candidates/frontier_trace.py`.
- A fresh live imported-MATLAB parity trial now exists at
  `slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial`.
- That live trial moved the repo out of the broken fallback-only under-emission state and into an over-emission regime (`1379/1555` edges), with divergence now led by partner choice and branch invalidation.
- A fresh stage-isolated network trial now exists at
  `slavv_comparisons/experiments/live-parity/runs/20260418_network_gate_trial`,
  and it reproduced exact MATLAB counts at the `network` stage.
- Saved-batch rerun is still pending.

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

- [~] Explain why `[1283, 1134]`, `[1283, 768]`, and watershed `[1283, 1659]`
      survive while the missing MATLAB pairs do not.
      Fresh live trial still shows `1283` as the top shared neighborhood with
      `missing/candidate/final = 4/3/2`.
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

- [x] New regression tests should live near the owning Python surface.
- [x] Prefer a deterministic unit or focused regression test over a broad new
      end-to-end sweep.
- [x] The test should capture one real branch-lifecycle mismatch, not just a
      count delta.

## Verification Order

- [x] Targeted pytest for the new local regression
- [x] `python -m pytest -m "unit or integration"`
- [ ] Saved-batch imported-MATLAB loop
- [x] Stage-isolated `network` gate
- [ ] Fresh live MATLAB confirmation only after the cheaper gates improve

## Completion Criteria

- [x] We can point to the exact local control-flow moment where one real MATLAB
      branch is lost or replaced in Python.
- [x] We have a regression test for that behavior.
- [ ] The targeted fix improves parity without a larger downstream regression.

## Fresh Live Trial Snapshot

- Run root: `slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial`
- Counts:
  - vertices `1682/1682`
  - edges `1379/1555`
  - strands `682/774`
- Top divergence mix:
  - partner choice `14`
  - branch invalidation `5`
  - claim ordering `4`
- Highest-severity examples:
  - `1283` remains the top branch-invalidation neighborhood
  - `8`, `17`, `20`, `35`, and `40` are now strong partner-choice probes

## Fresh Network-Gate Snapshot

- Run root: `slavv_comparisons/experiments/live-parity/runs/20260418_network_gate_trial`
- Counts:
  - vertices `1682/1682`
  - edges `1379/1379`
  - strands `682/682`
- Interpretation:
  - downstream `network` assembly still matches exactly when exact MATLAB
    edges are supplied
  - the active regression surface remains upstream in `edge_candidates.py` and
    `edge_selection.py`
