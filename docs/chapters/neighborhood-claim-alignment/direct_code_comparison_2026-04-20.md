# Direct Code Comparison 2026-04-20

[Up: Neighborhood Claim Alignment](README.md)

Status: Active investigation note

Date:

- April 20, 2026

## Why This Note Exists

This note records the first direct side-by-side code comparison between the
active MATLAB `edges` path and the imported-MATLAB Python path.

The main goal was to answer a narrower question than the broader chapter:

- are we still chasing missing MATLAB logic, or is Python now running a
  meaningfully different discovery architecture upstream of chooser cleanup?

## Files Compared

MATLAB:

- `external/Vectorization-Public/source/get_edges_V300.m`
- `external/Vectorization-Public/source/get_edges_by_watershed.m`
- `external/Vectorization-Public/source/get_edges_for_vertex.m`
- `external/Vectorization-Public/source/choose_edges_V200.m`
- `external/Vectorization-Public/source/clean_edges_vertex_degree_excess.m`
- `external/Vectorization-Public/source/clean_edges_orphans.m`
- `external/Vectorization-Public/source/clean_edges_cycles.m`

Python:

- `source/slavv/core/_edge_candidates/frontier_trace.py`
- `source/slavv/core/_edge_candidates/frontier_resolution.py`
- `source/slavv/core/_edge_candidates/generate.py`
- `source/slavv/core/_edge_candidates/watershed_candidates.py`
- `source/slavv/core/_edge_candidates/watershed_support.py`
- `source/slavv/core/_edge_candidates/geodesic_salvage.py`
- `source/slavv/core/_edge_selection/conflict_painting.py`
- `source/slavv/core/_edge_selection/workflow.py`
- `source/slavv/core/_edge_selection/cleanup.py`
- `source/slavv/core/_edges/standard.py`
- `source/slavv/core/_edges/resumable.py`

## Direct Findings

### 1. MATLAB discovery is global shared-state, Python discovery is still per-origin

The active MATLAB path is one shared watershed/claim process.
`get_edges_by_watershed.m` maintains image-sized shared maps such as:

- `vertex_index_map`
- `pointer_map`
- `branch_order_map`
- `vertex_adjacency_matrix`

It then advances one global best-first loop over shared available locations.

The imported-MATLAB Python path does not yet do that.
`frontier_trace.py` still traces one origin at a time with per-origin local
state, and `generate.py` appends those local results into a later manifest.

Why this matters:

- claim ordering between neighboring vertices is different before cleanup
- one origin can reject or retain branches without seeing MATLAB's global
  shared ownership state
- later chooser cleanup is not enough to recover discovery-time differences

### 2. MATLAB integrates watershed joins into discovery, Python still adds them later

MATLAB creates edge joins during the shared live watershed growth itself and
traces both halves back through the active `pointer_map`.

Python currently traces frontier candidates first and then augments with
watershed-contact candidates later from label contacts and synthetic join
traces.

Why this matters:

- Python watershed supplements do not inherit MATLAB's live pointer ownership
  history
- partner substitution can look plausible in the manifest while still being the
  wrong survivor compared with MATLAB

### 3. MATLAB V300 uses a hard `2`-edge budget, while live Python parity runs were using `4`

`get_edges_V300.m` hardcodes `edge_number_tolerance = 2`.

The live imported-MATLAB evidence root on April 18, 2026 still had
`number_of_edges_per_vertex = 4` in
`slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial/99_Metadata/validated_params.json`.

Before the April 20, 2026 fix below, Python frontier generation and cleanup
were therefore allowed to run with a broader local degree budget than MATLAB.

Why this matters:

- it is a real present-day upstream mismatch
- it can change which sibling branches are admitted, retained, or pruned
- it can inflate local survivor sets before later cleanup

### 4. Duplicate suppression timing still differs

MATLAB suppresses duplicate vertex pairs during discovery through the live
`vertex_adjacency_matrix`.

Python still deduplicates much later through manifest and cleanup surfaces.

Why this matters:

- even if both systems eventually remove duplicates, the timing changes which
  branches are still considered valid siblings during discovery

## Immediate Fix Landed On April 20, 2026

The first fix from this comparison is now implemented:

- imported-MATLAB frontier workflows force MATLAB's effective edge budget of
  `2` through candidate generation, watershed supplementation, local geodesic
  salvage, and degree cleanup
- imported-MATLAB frontier workflows also force the stricter
  `remaining_origin_contacts` watershed-admission mode so post-hoc watershed
  contacts do not keep overfilling origins that already met the MATLAB budget
- the override is private to the imported-MATLAB parity workflow and does not
  change the public `number_of_edges_per_vertex` setting for native Python
  tracing

Implementation surfaces:

- `source/slavv/core/_edge_candidates/common.py`
- `source/slavv/core/_edge_candidates/generate.py`
- `source/slavv/core/_edge_candidates/geodesic_salvage.py`
- `source/slavv/core/_edge_selection/conflict_painting.py`
- `source/slavv/core/_edge_selection/workflow.py`
- `source/slavv/core/_edges/standard.py`
- `source/slavv/core/_edges/resumable.py`

Regression coverage added:

- `dev/tests/unit/core/test_matlab_parity_edge_budget.py`
- `dev/tests/integration/test_regression_edges.py`

## What This Fix Does Not Solve Yet

This budget fix removes one confirmed MATLAB-vs-Python mismatch, but it does
not close the larger architectural gap.

Still open:

- Python discovery is still per-origin instead of one global shared-state
  watershed claim loop
- watershed candidates are still born after frontier tracing instead of during
  shared live ownership
- duplicate suppression and some claim-order consequences still happen later in
  Python than in MATLAB

## Best Current Explanation For The Remaining Gap

The repo is not blocked because MATLAB code was never copied.
The stronger explanation is that the imported-MATLAB Python path still runs a
different upstream discovery architecture:

- isolated per-origin frontier tracing
- post-hoc watershed supplementation
- later duplicate suppression and degree cleanup

That architecture can still explain:

- branch invalidation drift
- partner substitution
- cleanup-time loss after the "wrong" parent survives discovery

## Next Work From This Note

1. Re-run the imported-MATLAB `edges` comparison root after the forced-budget
   fix and record whether the over-emission count improves.
2. Keep using hotspots such as `866` and `1283` as proof neighborhoods for
   shared-map claim ordering.
3. Decide whether the next implementation step should emulate MATLAB's shared
   live ownership more directly or replace the current per-origin discovery
   architecture for imported-MATLAB parity runs.
