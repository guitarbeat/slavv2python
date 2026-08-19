---
title: MATLAB watershed env-var trace hooks (local, discarded 2026-08-18)
module: pipeline/edges
tags: [edges, watershed, matlab, tracing, debug]
problem_type: workflow
resolution_type: runbook
---

# MATLAB watershed env-var trace hooks

## In short

Local uncommitted edits in `external/Vectorization-Public/source/get_edges_by_watershed.m`
added opt-in JSONL tracing via environment variables. Those working-tree hooks
were discarded on 2026-08-18 so the submodule matches registered commit
`c570965`. Re-apply the patch below if you need the same MATLAB↔Python strel
trace again. Do not commit the hooks into Vectorization-Public.

## Problem
Frontier/strel splits need the same `strel_state` / `frontier_action` JSONL
rows from MATLAB that Python writes with `JsonExecutionTracer`. The vendored
script `external/Vectorization-Public/scripts/extract_watershed_trace.m` is a
stub, not production tracing.

## Evidence
- Historical crop traces: `workspace/scratch/matlab_edge_dump/` (JSONL labeled
  there; `.mat` dumps removed 2026-08-18).
- Python: `slavv_python/pipeline/edges/execution_tracing.py`,
  `scripts/edges/frontier_diff.py`,
  `scripts/edges/strel_state.py`.
- Diary: [EXACT_PROOF_FINDINGS.HISTORICAL.md](../../investigations/exact-proof-findings-diary/EXACT_PROOF_FINDINGS.HISTORICAL.md)
  (2026-07-13 claim-state diagnostic).

## Root Cause
The hooks lived only in the submodule working tree. `energy_filter_V200.m`
looked dirty from line-ending noise on two identical `Inf` lines — not a
method change.

## Solution
Opt-in env vars (unset = no file, no behavior change):

| Variable | Role |
|----------|------|
| `SLAVV_WATERSHED_TRACE_PATH` | JSONL output path; opens `'w'` |
| `SLAVV_WATERSHED_TRACE_ITERATIONS` | Comma/space list of iterations to emit `strel_state` |
| `SLAVV_WATERSHED_TRACE_TARGETS` | Linear indices for `frontier_action` / target strel rows |
| `SLAVV_WATERSHED_TRACE_SAMPLE_LIMIT` | Top-N adjusted-energy strel rows (default 12) |

Re-apply from the Vectorization-Public root (does not change `energy_filter_V200.m`):

```powershell
git -C external/Vectorization-Public apply -- `
  docs/solutions/parity/matlab-watershed-env-trace-hooks.diff
```

Leave the submodule dirty while tracing. Restore when done:

```powershell
git -C external/Vectorization-Public restore -- source/get_edges_by_watershed.m
```

## Verification
`git -C external/Vectorization-Public status` is clean at `c570965` after
restore. Patch file is the 2026-08-18 working-tree diff of
`source/get_edges_by_watershed.m` only.

## Follow-Up
Keep trace JSONL under `workspace/scratch/`, not live dests. Do not overwrite
`LIVE_DEST_NAMES`.
