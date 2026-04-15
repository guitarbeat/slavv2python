# Python Nondeterminism Investigation

## What this file is for

This is the canonical repeatability baseline for the March 28 to March 30
investigation set. It combines the original nondeterminism diagnosis, the
Python and MATLAB standalone consistency checks, and the immediate post-fix
parity outcome.

## Read this when

- you want to know whether Python repeatability is still an open blocker
- you need the root cause of the old Python-only graph drift
- you want the standalone MATLAB and Python consistency baselines in one place
- you need to separate solved repeatability work from the still-open semantic
  parity gap

## Executive Summary

- On March 28, standalone MATLAB runs were repeatable at the parsed-data level:
  `1682` vertices, `1379` edges, and `682` strands across all three runs.
- On March 28, standalone Python runs were not repeatable after vertices:
  vertices stayed at `292`, but edges drifted across `9`, `6`, and `11`, and
  strands drifted across `7`, `5`, and `10`.
- The strongest root-cause finding was unseeded fallback direction padding in
  the standard Python edge tracer. On this workload, all `292 / 292` vertices
  needed padded directions.
- After the deterministic padding fix on March 30, standalone Python runs were
  repeatable: `292` vertices, `8` edges, `8` strands, and identical normalized
  digests across runs.
- The post-fix full parity run still failed at MATLAB `1379 / 682` versus
  Python `1560 / 820`, so repeatability is solved but semantic parity is not.

## Handling Classification

- Status: Handled
- Why: The nondeterminism investigation is complete and its root cause plus
  deterministic fix baseline are closed.
- Remaining semantic parity gaps are tracked in
  `dev/reports/unhandled/` and are intentionally out of scope for this report.

## Current Status

### Solved

- Standalone MATLAB output is stable enough on this workload to treat it as a
  reproducible reference surface.
- Standalone Python output is now repeatable on this workload after the
  deterministic edge-padding fix.
- Checkpoint merge order, file ordering, and final export formatting are no
  longer the best explanation for the old Python-only drift.

### Active concern

- The remaining imported-MATLAB parity gap is semantic, not random-run noise.
- Edge candidate generation and edge cleanup semantics remain the live parity
  problem surfaces.

### Superseded conclusions

- Python repeatability is no longer the primary blocker.
- Broad suspicion of resumable state or export-only drift should not lead the
  next debugging round.

## Evidence Timeline

### March 28: pre-fix Python instability

Three fresh standalone Python runs produced:

| Run | Vertices | Edges | Strands |
| --- | ---: | ---: | ---: |
| 1 | 292 | 9 | 7 |
| 2 | 292 | 6 | 5 |
| 3 | 292 | 11 | 10 |

The run layout stayed consistent, but graph-level artifacts did not. Vertex
positions and scales matched; edge connectivity, edge traces, and network
strands did not.

Direct replay on the exact same saved `checkpoint_energy.pkl` and
`checkpoint_vertices.pkl` also drifted:

| Replay | Candidate traced edges | Terminal edges | Final edges |
| --- | ---: | ---: | ---: |
| 1 | 698 | 10 | 9 |
| 2 | 718 | 13 | 11 |
| 3 | 738 | 14 | 12 |

That ruled out checkpoint merge order and cross-run directory reuse as the main
cause.

### Root cause: unseeded filler directions

The standard Python tracer was the real culprit:

- standalone comparison-mode Python runs did not use the MATLAB frontier tracer
- `estimate_vessel_directions(...)` returned only the axis pair on this
  workload
- the active parameters requested `number_of_edges_per_vertex = 4`
- `generate_edge_directions(...)` filled the missing directions with fresh
  entropy when `seed=None`

On the saved run, all `292` vertices needed padding. That meant every fresh
Python run injected new direction samples at every vertex before edge tracing.

Fixed-seed confirmation replays made the result perfectly repeatable, which
proved the topology drift was tied to the padding randomness.

### March 30: post-fix Python consistency

After the deterministic edge-padding fix, three fresh standalone Python runs
produced:

| Run | Vertices | Edges | Strands | Candidate edges |
| --- | ---: | ---: | ---: | ---: |
| 1 | 292 | 8 | 8 | 711 |
| 2 | 292 | 8 | 8 | 711 |
| 3 | 292 | 8 | 8 | 711 |

Normalized digests were identical across all three runs:

- vertex digest: `27f14bd67c5573124a493f203bd9db499ee21fd301f284775166d8d0e849084c`
- edge digest: `4174d958303b559964434ce7ccedf9e066b7be182e9b91fdc2722fc9effbbe10`
- network digest: `8e438e82efee45453fdf7ba6b493a758044bc45a79701eff95755ebb40afefd3`

One of the three runs completed via resume, but the final outputs still matched
exactly, which is a useful confidence check for the fixed path.

### March 28 MATLAB standalone baseline

Three fresh standalone MATLAB runs were stable at the parsed-data level:

| Run | Vertices | Edges | Strands |
| --- | ---: | ---: | ---: |
| 1 | 1682 | 1379 | 682 |
| 2 | 1682 | 1379 | 682 |
| 3 | 1682 | 1379 | 682 |

Parsed digests for vertices, edges, and strands matched across all three runs.
Small raw timestamped file differences existed, but they did not change the
normalized graph outputs.

### March 30 post-fix parity implication

The immediate post-fix full parity rerun still failed:

| Metric | MATLAB | Python | Delta |
| --- | ---: | ---: | ---: |
| Vertices | 1682 | 1682 | 0 |
| Edges | 1379 | 1560 | +181 |
| Strands | 682 | 820 | +138 |

That is the key framing point for today: the repeatability fix mattered, but it
did not solve semantic parity.

## What this means for parity work now

- MATLAB output is a stable reference on this workload.
- Python standalone behavior is now stable enough to compare meaningfully.
- Any remaining imported-MATLAB parity mismatch should be treated as a semantic
  algorithm mismatch, not as nondeterministic noise.
- Repeatability work should stay closed unless a new regression test or new run
  evidence reopens it.

## Recommended next actions

1. Use the post-fix Python path as the repeatable baseline for future parity
   investigations.
2. Focus current debugging on edge candidate generation and cleanup semantics,
   not on repeatability plumbing.
3. Keep the deterministic padding regression coverage in place so this class of
   issue stays closed.
4. Reopen nondeterminism investigation only if a new workload or regression
   test shows graph-level drift again.
