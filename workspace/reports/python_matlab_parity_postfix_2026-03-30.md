# Python vs MATLAB Parity, Post-Fix

Date: 2026-03-30

## What I Ran

I ran the full parity workflow with the deterministic Python edge-padding fix in place:

- Output root: `D:\slavv_comparisons\20260330_parity_full_postfix`
- MATLAB batch folder: `D:\slavv_comparisons\20260330_parity_full_postfix\01_Input\matlab_results\batch_260330-152815`
- Python results: `D:\slavv_comparisons\20260330_parity_full_postfix\02_Output\python_results`
- Comparison report: `D:\slavv_comparisons\20260330_parity_full_postfix\03_Analysis\comparison_report.json`
- Summary: `D:\slavv_comparisons\20260330_parity_full_postfix\03_Analysis\summary.txt`

The Python side was rerun from imported MATLAB energy and vertices, so this is the right apples-to-apples parity check.

## Main Outcome

Vertices are now an exact match, but edges and strands are still not.

| Metric | MATLAB | Python | Delta |
| --- | ---: | ---: | ---: |
| Vertices | 1,682 | 1,682 | 0 |
| Edges | 1,379 | 1,560 | +181 |
| Strands | 682 | 820 | +138 |

The parity gate still fails because edges and strands are not exact.

## Timing

Runtime improved on the Python side relative to MATLAB:

- MATLAB: `12m 40.3s`
- Python: `4m 52.4s`
- Speedup: `2.60x`

## What Changed After the Fix

The deterministic edge-padding fix did its job on the repeatability problem:

- Python vertices remain exact against MATLAB.
- Python edge generation is now repeatable across runs.
- The remaining mismatch is semantic, not random-run noise.

The post-fix comparison report shows:

- `candidate_endpoint_pair_count`: `8,892`
- `matched_matlab_endpoint_pair_count`: `1,184`
- `missing_matlab_endpoint_pair_count`: `195`
- `python_count`: `1,560`

That is a better candidate coverage story than before, but still not parity-complete.

## Diagnostics Worth Noting

The comparison report points to a few important asymmetries:

- Python has `7,399` watershed join supplements in this full parity path.
- Python rejected `6,366` candidates via conflict checks.
- Python pruned `480` by degree, `3` as orphans, and `483` as cycles.
- The first parity mismatch is still the same basic family of endpoint pairing differences, not a new vertex issue.

## Interpretation

The fix removed the nondeterministic Python-only drift, which was the first blocker. What remains is a genuine algorithmic parity gap in edge selection and network construction.

In other words:

- we solved repeatability
- we have not yet solved exact MATLAB parity

That is still progress, because we can now compare stable Python behavior against MATLAB and focus on the actual semantic differences rather than random direction padding.

## Next Likely Debug Target

The next useful place to investigate is the edge candidate path after the MATLAB-imported vertices stage, especially:

- how the frontier traces are seeded
- why the Python full parity path is generating `9,549` candidates versus MATLAB’s `1,379` final edges
- why the watershed supplements are so large in the parity rerun

The parity gap now looks like a real model difference instead of a reproducibility bug.
