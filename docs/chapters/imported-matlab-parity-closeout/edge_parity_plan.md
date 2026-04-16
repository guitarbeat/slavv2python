# Edge Parity Plan

Status: Historical Chapter 1 plan
Updated: 2026-04-08

This file is retained as a historical implementation plan from Chapter 1.

For the active chapter, use
[Candidate Generation Handoff](../candidate-generation-handoff/README.md).

During Chapter 1, this file was the edge-generation source of truth for the
active parity implementation plan.

Use this file when you want to know:

- what the remaining technical blocker is
- which code surface should change next
- what acceptance looks like for edge convergence

Use [PARITY_HUB.md](PARITY_HUB.md) for quick re-entry,
[parity_decision_memo_2026-04-08.md](parity_decision_memo_2026-04-08.md)
for the current implementation decision, and
[parity_findings.md](parity_findings.md) for the longer evidence behind this
plan.

## Rapid Recall

- Already solved:
  - vertex parity on the imported-MATLAB surface
  - stage-isolated `network` parity when exact MATLAB `edges` are imported and
    Python reruns from `network`
- Still failing:
  - Python edge generation from imported MATLAB `energy` and `vertices`
- Current blocker:
  - upstream frontier candidate generation and local partner selection
  - plus a newly confirmed cleanup-path mismatch between Python and the active
    MATLAB V200 flow
  - not generic downstream graph assembly
- Standing downstream gate:

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --input data/slavv_test_volume.tif `
  --skip-matlab `
  --resume-latest `
  --python-parity-rerun-from network `
  --comparison-depth deep
```

## Current Problem Statement

On the imported-MATLAB parity surface:

- vertices are exact
- edges are still mismatched
- strands are close when Python reruns from `edges`
- strands are exact when exact MATLAB `edges` are imported and Python reruns
  from `network`

That means the remaining primary blocker is the edge-candidate path before
downstream graph assembly, but the cleanup path also needs to match the active
MATLAB V200 code surface before further frontier debugging is fully
trustworthy.

## Evidence Snapshot

Current evidence, consolidated:

- Imported-MATLAB Python reruns are repeatable on the current machine.
- The stage-isolated MATLAB-edges-to-Python-network probe passes exactly.
- The remaining live mismatch is still concentrated in extra Python frontier
  edges and missing MATLAB endpoint pairs.
- The strongest extra frontier edges often cluster around the same shared
  vertices as the missing MATLAB pairs.
- For the worst shared vertices inspected so far, some missing MATLAB pairs
  never enter the Python candidate pool at all.
- A direct April 8 MATLAB/Python code audit showed that Python cleanup still
  models `choose_edges_V200`-style conflict painting even though the active
  `vectorize_V200.m` path uses `clean_edge_pairs` followed by degree/orphan/
  cycle cleanup.

Detailed metrics and experiment history live in:

- [parity_decision_memo_2026-04-08.md](parity_decision_memo_2026-04-08.md)
- [parity_findings.md](parity_findings.md)
- [stage_isolated_network_parity_2026-04-07.md](../../../dev/reports/stage_isolated_network_parity_2026-04-07.md)

## Active Code Surface

The remaining parity work is centered on:

- [comparison.py](../../../source/slavv/parity/comparison.py)
  for parity-mode orchestration and replay surfaces
- [edge_candidates.py](../../../source/slavv/core/edge_candidates.py)
  for frontier candidate generation, ownership, and local partner choice
- [graph.py](../../../source/slavv/core/graph.py)
  only as a downstream gate, not as the current primary suspect

## Working Hypothesis

The remaining gap is now best modeled as two layered problems:

1. upstream frontier-discovery mismatch
2. downstream cleanup-path mismatch with the active MATLAB V200 flow

Current working model:

- watershed supplementation is noisy, but it is not the whole explanation
- the final extra set is still frontier-heavy
- some strong extra frontier edges appear to be local partner substitutions
  around shared vertices
- the highest-value fixes are likely:
  - first, cleanup-path alignment with active MATLAB V200
  - second, local claim ordering, terminal ownership, bifurcation handling, or
    related frontier semantics

## Implementation Phases

### Phase 1: Keep the downstream gate fixed

- Preserve the stage-isolated `network` probe as a standing regression gate.
- Do not treat `graph.py` as the active problem surface unless that gate
  regresses.
- Keep parity-mode network assembly enabled for all comparison-mode reruns.

### Phase 2: Align cleanup with active MATLAB V200

- Replace the current mixed cleanup path with the active MATLAB V200 cleanup
  chain:
  - `clean_edge_pairs`
  - `clean_edges_vertex_degree_excess`
  - `clean_edges_orphans`
  - `clean_edges_cycles`
- Re-run the imported-MATLAB parity loop after cleanup alignment.
- Treat the residual mismatch after that rerun as the true upstream gap.

### Phase 3: Localize the first edge divergence

- Continue using candidate-endpoint coverage as the first triage gate.
- Keep using shared-vertex diagnostics to identify where missing MATLAB pairs
  and extra frontier edges overlap.
- Focus on the worst shared vertices first rather than broad global changes.

### Phase 4: Tighten frontier tracing semantics

Compare Python against MATLAB in the areas most likely to suppress MATLAB-like
candidate pairs before cleanup:

- frontier voxel admission and overwrite
- terminal-hit ownership semantics
- parent/child and bifurcation claim ordering
- post-hit pruning
- local partner choice around shared neighborhoods

Primary audit companion:

- [MATLAB_PARITY_AUDIT_CHECKLIST.md](MATLAB_PARITY_AUDIT_CHECKLIST.md)

### Phase 5: Keep watershed changes selective

- Avoid blunt global threshold sweeps unless a diagnostic specifically points
  there.
- Preserve strand-critical structure if watershed filtering is tightened.
- Prefer evidence-driven, local changes over broad source-preference rules.

### Phase 6: Keep cleanup narrow after alignment

- Keep cleanup aligned to the active MATLAB code path rather than blending old
  and new MATLAB helpers.
- Do not use cleanup changes as the primary way to mask upstream candidate
  drift once the cleanup model is aligned.
- Add diagnostics first if cleanup ordering becomes a suspected blocker again.

## Non-Goals

- Do not rework native Python-from-TIFF behavior as the primary parity fix.
- Do not treat generic downstream graph assembly as the main problem unless the
  stage-isolated `network` gate regresses.
- Do not broaden heuristics just to force count agreement without matching
  local behavior.

## Acceptance Criteria

The edge parity work is successful when the canonical imported-MATLAB parity
surface produces:

- exact vertices
- exact edges
- exact strands
- stable rerun-to-rerun counts and hashes
- continued exact strand parity in the stage-isolated
  MATLAB-edges-to-Python-network probe

## Verification

Run the cheaper checks first:

1. default imported-MATLAB edge loop
2. candidate-endpoint coverage and shared-vertex diagnostics
3. stage-isolated `network` gate
4. fresh full MATLAB confirmation only when the milestone is worth it

## Related Docs

- [PARITY_HUB.md](PARITY_HUB.md)
- [parity_decision_memo_2026-04-08.md](parity_decision_memo_2026-04-08.md)
- [parity_findings.md](parity_findings.md)
- [MATLAB_PARITY_AUDIT_CHECKLIST.md](MATLAB_PARITY_AUDIT_CHECKLIST.md)
- [dev/reports/stage_isolated_network_parity_2026-04-07.md](../../../dev/reports/stage_isolated_network_parity_2026-04-07.md)

