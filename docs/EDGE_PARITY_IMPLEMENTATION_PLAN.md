# Edge Parity Implementation Plan

Status: In Progress
Updated: 2026-04-08

This file is the edge-generation source of truth for parity implementation
work.

Use this file when you want to know:

- what the remaining technical blocker is
- which code surface should change next
- what acceptance looks like for edge convergence

Use [PARITY_HUB.md](PARITY_HUB.md) for quick re-entry and
[PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md) for the
evidence behind this plan.

## Rapid Recall

- Already solved:
  - vertex parity on the imported-MATLAB surface
  - stage-isolated `network` parity when exact MATLAB `edges` are imported and
    Python reruns from `network`
- Still failing:
  - Python edge generation from imported MATLAB `energy` and `vertices`
- Current blocker:
  - upstream frontier candidate generation and local partner selection
  - not generic downstream graph assembly
- Standing downstream gate:

```powershell
python workspace/scripts/cli/compare_matlab_python.py `
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
downstream graph assembly.

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

Detailed metrics and experiment history live in:

- [PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md)
- [stage_isolated_network_parity_2026-04-07.md](../workspace/reports/stage_isolated_network_parity_2026-04-07.md)

## Active Code Surface

The remaining parity work is centered on:

- [comparison.py](../source/slavv/evaluation/comparison.py)
  for parity-mode orchestration and replay surfaces
- [tracing.py](../source/slavv/core/tracing.py)
  for frontier candidate generation, ownership, and local partner choice
- [graph.py](../source/slavv/core/graph.py)
  only as a downstream gate, not as the current primary suspect

## Working Hypothesis

The remaining gap is most likely caused by a frontier-discovery mismatch rather
than by generic cleanup or graph assembly.

Current working model:

- watershed supplementation is noisy, but it is not the whole explanation
- the final extra set is still frontier-heavy
- some strong extra frontier edges appear to be local partner substitutions
  around shared vertices
- the highest-value fixes are likely in local claim ordering, terminal
  ownership, bifurcation handling, or related frontier semantics

## Implementation Phases

### Phase 1: Keep the downstream gate fixed

- Preserve the stage-isolated `network` probe as a standing regression gate.
- Do not treat `graph.py` as the active problem surface unless that gate
  regresses.
- Keep parity-mode network assembly enabled for all comparison-mode reruns.

### Phase 2: Localize the first edge divergence

- Continue using candidate-endpoint coverage as the first triage gate.
- Keep using shared-vertex diagnostics to identify where missing MATLAB pairs
  and extra frontier edges overlap.
- Focus on the worst shared vertices first rather than broad global changes.

### Phase 3: Tighten frontier tracing semantics

Compare Python against MATLAB in the areas most likely to suppress MATLAB-like
candidate pairs before cleanup:

- frontier voxel admission and overwrite
- terminal-hit ownership semantics
- parent/child and bifurcation claim ordering
- post-hit pruning
- local partner choice around shared neighborhoods

Primary audit companion:

- [MATLAB_PARITY_AUDIT_CHECKLIST.md](MATLAB_PARITY_AUDIT_CHECKLIST.md)

### Phase 4: Keep watershed changes selective

- Avoid blunt global threshold sweeps unless a diagnostic specifically points
  there.
- Preserve strand-critical structure if watershed filtering is tightened.
- Prefer evidence-driven, local changes over broad source-preference rules.

### Phase 5: Use cleanup as a downstream safety net

- Keep `_choose_edges_matlab_style()` focused on dedupe and pruning.
- Do not use cleanup changes as the primary way to mask upstream candidate
  drift.
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
- [PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md)
- [MATLAB_PARITY_AUDIT_CHECKLIST.md](MATLAB_PARITY_AUDIT_CHECKLIST.md)
- [workspace/reports/stage_isolated_network_parity_2026-04-07.md](../workspace/reports/stage_isolated_network_parity_2026-04-07.md)
