# Candidate Generation Handoff

[Up: Chapter Index](../README.md)

Status: Historical Chapter 2 handoff

Successor:

- [Neighborhood Claim Alignment](../neighborhood-claim-alignment/README.md)

Use [chapters/README.md](../README.md) for chapter-system navigation and
[TODO.md](../../../TODO.md) for the current root-level parity backlog.

This file is no longer the active spec entry point.

This chapter was opened on April 10, 2026.

This chapter continues the parity work after the imported-MATLAB chapter
closed. The remaining gap is narrow enough to treat as a focused
candidate-generation problem instead of a general parity chase.

It successfully reframed the parity gap from a general candidate-generation
problem into a narrower shared-neighborhood claim and ownership problem.

## Why This Chapter Exists

- Vertex parity is already stable on the retained imported-MATLAB surface.
- The best saved-batch result is close enough to isolate candidate discovery and
  ownership behavior.
- The previous chapter closed with a stable handoff and a clearer MATLAB source
  path through `get_edges_V300 -> get_edges_by_watershed`.

## Current Starting Facts

- Current best retained saved-batch result is `vertices 110/110`,
  `edges 94/93`, `strands 49/54`.
- Origin `64` remains the clearest next under-covered case.
- Retained geodesic widening to `k=10` improved the saved-batch result without
  breaking the broader suite.
- Relaxed geodesic and origin-owned pruning experiments were informative but
  rejected.

## What This Chapter Handed Off

- the parity gap was narrower than a general candidate-generation chase
- shared neighborhoods were a better unit of analysis than global edge counts
- local claim ordering and invalidation stayed live suspects even when
  candidate coverage improved
- later work needed to split frontier rejection, partner substitution, and
  cleanup loss into separate defect classes instead of one generic bucket

## Main Goal

Align Python candidate generation more closely with active MATLAB shared-state
and watershed behavior without regressing the already-stable imported-MATLAB
vertex path.

## Working Questions

1. Why does origin `64` remain under-covered in the retained candidate set?
2. Does MATLAB temporarily allow over-budget candidate admission before later
   cleanup in a way Python still does not?
3. Which parts of `get_edges_by_watershed` shared-map behavior are still not
   represented in Python candidate discovery?
4. Is the remaining loss happening during candidate admission, claim ownership,
   or candidate conflict resolution?

## Current Parity Context

- The staged comparison run root is the canonical place to inspect output-root
  preflight decisions, MATLAB rerun semantics, and human-readable manifests.
- The MATLAB batch importer now loads real HDF5 energy sidecars into
  checkpoint-compatible `energy_data` payloads.
- Exact vertex parity is reached when the Python pipeline runs under imported
  MATLAB energy with `comparison_exact_network=True`.
- The parity-only frontier tracer simulates MATLAB's
  `edge_number_tolerance = 2` to limit edge over-generation on imported runs.
- Watershed supplementation in `source/slavv/core/edge_candidates.py` applies Phase 2
  gates (Frontier Reachability and Per-Origin Caps) to prevent redundant
  strands during parity runs.
- Edge cleanup applies MATLAB-shaped duplicate ordering, including deterministic
  shorter-trace tie-breaking.
- Exact edge and strand parity are still in progress.
- Latest live rerun on April 6, 2026: `1425` Python edges vs `1379` MATLAB
  edges, and `681` Python strands vs `682` MATLAB strands.
- Candidate-endpoint coverage is still the first triage signal, but better
  candidate counts alone do not guarantee better final edge or strand parity.
- Final exact edge and strand confirmation still requires a fresh live
  MATLAB-enabled comparison run on a healthy local output root.

## Scope

In scope:

- candidate-endpoint coverage
- shared-state and watershed join semantics
- origin ownership and claim-map behavior
- bounded parity-only experiments on the saved batch

Out of scope:

- re-solving vertex parity
- broad threshold sweeps without a targeted diagnostic reason
- downstream network assembly behavior that already passes when exact MATLAB
  edges are imported

## Suggested First Loop

Historical note:

This loop is preserved for context only. The active workspace now keeps
comparison runs under
`C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\experiments\...`,
and the active parity chapter uses fresh imported-MATLAB trial roots rather
than the old local `saved_batch_run` example below.

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --input C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\saved_batch_run\01_Input\synthetic_branch_volume.tif `
  --skip-matlab `
  --output-dir C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\next_chapter_run `
  --params C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\saved_batch_run\99_Metadata\comparison_params.normalized.json
```

Current active evidence roots:

- `slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial`
- `slavv_comparisons/experiments/live-parity/runs/20260418_network_gate_trial`

## Core References

- [MATLAB Translation Guide](../../reference/core/MATLAB_TRANSLATION_GUIDE.md)
- [MATLAB Mapping](../../reference/core/MATLAB_MAPPING.md)
- [Comparison Run Layout](../../reference/core/COMPARISON_LAYOUT.md)
- [Imported-MATLAB Parity Closeout](../imported-matlab-parity-closeout/parity_closeout.md)
- [Parity Findings](../imported-matlab-parity-closeout/parity_findings.md)


