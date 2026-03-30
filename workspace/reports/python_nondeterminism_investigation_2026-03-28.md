# Python Nondeterminism Investigation

Date: 2026-03-28

## Question

Why do fresh standalone Python runs produce stable vertices but unstable edges and strands?

## Working Hypotheses

1. The standalone comparison-mode Python run might already be using the MATLAB parity tracer, so the instability could be in the parity path.
2. The instability might come from checkpoint merge order, file ordering, or resumable edge-unit consolidation.
3. The instability might come from the standard Python edge tracer itself, especially from direction generation or tie-breaking.

## What I Checked

### Hypothesis 1: standalone Python uses the MATLAB parity tracer

Result: rejected.

Evidence:

- [comparison.py](C:/Users/alw4834/OneDrive%20-%20The%20University%20of%20Texas%20at%20Austin/Documents%201/GitHub/slavv2python/source/slavv/evaluation/comparison.py#L291) enables `comparison_exact_network` by default for comparison-mode Python runs.
- But [tracing.py](C:/Users/alw4834/OneDrive%20-%20The%20University%20of%20Texas%20at%20Austin/Documents%201/GitHub/slavv2python/source/slavv/core/tracing.py#L906) only enables the MATLAB frontier tracer when `energy_origin == "matlab_batch_hdf5"`.
- The saved native-Python checkpoint energy payload had `energy_origin = null`, so the standalone Python runs stayed on the standard tracer, not the MATLAB frontier tracer.

Conclusion:

- The repeatability failure is in the native Python edge path, not the parity-only MATLAB frontier path.

### Hypothesis 2: variability comes from checkpoint merge order or resumable state

Result: strongly disfavored.

Evidence:

- I reran `extract_edges(...)` three times directly on the exact same saved `checkpoint_energy.pkl` and `checkpoint_vertices.pkl` from `run_01`.
- Those replays produced different outputs even without rerunning energy or vertices:

| Replay | Candidate traced edges | Terminal edges | Final edges |
| --- | ---: | ---: | ---: |
| 1 | 698 | 10 | 9 |
| 2 | 718 | 13 | 11 |
| 3 | 738 | 14 | 12 |

Conclusion:

- The nondeterminism is already present inside edge extraction itself.
- It is not caused by cross-run checkpoint reuse, edge-unit file ordering, or final export formatting.

### Hypothesis 3: the standard Python edge tracer injects randomness

Result: supported very strongly.

Evidence from code:

- [tracing.py](C:/Users/alw4834/OneDrive%20-%20The%20University%20of%20Texas%20at%20Austin/Documents%201/GitHub/slavv2python/source/slavv/core/tracing.py#L1957) `estimate_vessel_directions(...)` returns only the local vessel axis and its reverse, so at most 2 directions.
- The active params request `number_of_edges_per_vertex = 4`.
- In [tracing.py](C:/Users/alw4834/OneDrive%20-%20The%20University%20of%20Texas%20at%20Austin/Documents%201/GitHub/slavv2python/source/slavv/core/tracing.py#L1482) and specifically [tracing.py](C:/Users/alw4834/OneDrive%20-%20The%20University%20of%20Texas%20at%20Austin/Documents%201/GitHub/slavv2python/source/slavv/core/tracing.py#L1525), the standard tracer pads any missing directions with `generate_edge_directions(...)`.
- [tracing.py](C:/Users/alw4834/OneDrive%20-%20The%20University%20of%20Texas%20at%20Austin/Documents%201/GitHub/slavv2python/source/slavv/core/tracing.py#L474) uses `np.random.default_rng(seed)` and defaults to `seed=None`, which means fresh entropy each call.

Evidence from run data:

- All 292 vertices in the saved run returned exactly 2 Hessian directions.
- That means all 292 vertices needed random padding to reach the requested 4 directions.

Summary:

- `292 / 292` vertices required random padding.
- So every standalone Python edge run injected fresh random directions at every vertex.

### Confirmation test: force deterministic padding

I monkeypatched `generate_edge_directions` to use a fixed seed on the same saved energy and vertices and reran `extract_edges(...)` three times.

With fixed seed `0`:

| Replay | Candidate traced edges | Final edges | Digest |
| --- | ---: | ---: | --- |
| 1 | 654 | 7 | `b87f1289f5856534911da3cef7803007908e499bf1451cb70989bb2c07824035` |
| 2 | 654 | 7 | `b87f1289f5856534911da3cef7803007908e499bf1451cb70989bb2c07824035` |
| 3 | 654 | 7 | `b87f1289f5856534911da3cef7803007908e499bf1451cb70989bb2c07824035` |

With fixed seed `1`:

| Replay | Candidate traced edges | Final edges | Digest |
| --- | ---: | ---: | --- |
| 1 | 840 | 9 | `958bebdcde783628bec11cc4891f0a973669370189b31904e0a86189d45952ab` |
| 2 | 840 | 9 | `958bebdcde783628bec11cc4891f0a973669370189b31904e0a86189d45952ab` |
| 3 | 840 | 9 | `958bebdcde783628bec11cc4891f0a973669370189b31904e0a86189d45952ab` |

Interpretation:

- Once the random padding is made deterministic, repeated edge extraction on fixed inputs becomes perfectly repeatable.
- Different fixed seeds produce different but internally consistent graphs.
- So the topology drift is causally tied to the unseeded random direction filler.

## What Changes Across Fresh Standalone Runs

Across the 3 full standalone Python runs:

- only 4 undirected endpoint pairs were common to all three runs:
  - `(71, 138)`
  - `(90, 176)`
  - `(122, 135)`
  - `(143, 207)`
- the union across the three runs contained 17 distinct undirected pairs
- run-specific extra undirected pairs were:
  - `run_01`: 5 extra pairs
  - `run_02`: 2 extra pairs
  - `run_03`: 7 extra pairs

That is consistent with a small stable core plus a larger seed-sensitive fringe created by random filler traces.

## Conclusion

The most likely root cause is:

- the standalone Python path is using the standard edge tracer
- the standard tracer always needs to pad Hessian directions from 2 up to 4 on this workload
- that padding uses unseeded random directions
- those random traces materially change candidate generation and final chosen graph topology

In short:

- the instability is real
- it is downstream of vertices
- it is reproducibly explained by unseeded random direction padding in the standard Python edge tracer

## Most Likely Next Fix Options

1. Make the fallback direction padding deterministic.
2. Replace random filler directions with a fixed low-discrepancy or hard-coded direction set.
3. Reduce or remove filler directions when Hessian only yields the principal axis pair.
4. Route standalone comparison-mode runs through a deterministic parity edge tracer when that is the intended behavior.

Option 1 is the smallest change if the goal is repeatability first.

## Status After Fix

I updated the edge tracer so the fallback direction padding is seeded deterministically per vertex, and the new regression test passes.

I also replayed the exact same saved Python `checkpoint_energy.pkl` and `checkpoint_vertices.pkl` three times after the change:

| Replay | Candidate traced edges | Final edges | Digest |
| --- | ---: | ---: | --- |
| 1 | 711 | 8 | `4174d958303b559964434ce7ccedf9e066b7be182e9b91fdc2722fc9effbbe10` |
| 2 | 711 | 8 | `4174d958303b559964434ce7ccedf9e066b7be182e9b91fdc2722fc9effbbe10` |
| 3 | 711 | 8 | `4174d958303b559964434ce7ccedf9e066b7be182e9b91fdc2722fc9effbbe10` |

That confirms the nondeterminism was addressed at the edge-padding layer.
