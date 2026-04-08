# Parity Decision Memo 2026-04-08

This is the shortest high-signal summary of the April 8, 2026 parity work.

Read this first.

Appendices:

- [Imported-MATLAB Parity Evidence 2026-04-08](parity_investigation_notes_2026-04-08.md)
- [MATLAB/Python Code Audit 2026-04-08](matlab_python_code_audit_2026-04-08.md)
- [Parity Report Index 2026-04-08](parity_report_index_2026-04-08.md)

## Decision

The parity problem is not just an upstream frontier issue anymore.

There are now two concrete problems on the imported-MATLAB parity surface:

1. the candidate pool is still wrong upstream
2. the Python cleanup path is modeling the wrong MATLAB cleanup surface

The cleanup mismatch should be fixed first so that the remaining gap reflects
the real frontier-generation divergence rather than a mixed cleanup model.

## Current Status

- Vertices: exact
- Network: exact when exact MATLAB `edges` are imported and Python reruns from
  `network`
- Edges: still failing
- Strands: still failing by a narrow margin

Fresh April 8 rerun:

- MATLAB edges `1379`
- Python edges `1425`
- MATLAB strands `682`
- Python strands `681`

## What Changed In The Diagnosis

Before this audit, the standing diagnosis emphasized upstream frontier
candidate generation and local partner choice.

That is still true, but the code audit adds a second concrete finding:

- the active MATLAB `vectorize_V200.m` edge-cleaning path uses
  `clean_edge_pairs`, `clean_edges_vertex_degree_excess`,
  `clean_edges_orphans`, and `clean_edges_cycles`
- the Python parity path currently mixes in `choose_edges_V200`-style conflict
  painting inside `_choose_edges_matlab_style()`

So Python is not only generating the wrong candidates in places; it is also
cleaning them with the wrong MATLAB model.

## Why This Matters

If the downstream cleanup model is wrong, frontier debugging is harder to trust
because the final edge set is being transformed by a path MATLAB does not use
in the active V200 flow.

That means:

- some apparent frontier failures may actually be cleanup-model failures
- parity work will be easier to interpret once cleanup matches the active
  MATLAB code path

## Best Next Step

Replace the current Python cleanup path with the active MATLAB V200 cleanup
chain:

1. `clean_edge_pairs`
2. `clean_edges_vertex_degree_excess`
3. `clean_edges_orphans`
4. `clean_edges_cycles`

Then rerun the imported-MATLAB parity loop and measure what mismatch remains.

Whatever remains after that is the real upstream frontier-generation gap.

## What Not To Do Next

- Do not keep tuning shared-vertex tracing behavior while the cleanup model is
  still mismatched.
- Do not treat cleanup alone as the full fix; the candidate pool is still wrong
  upstream.
- Do not broaden heuristic filtering just to force count agreement.

## Read Order

1. This file
2. [MATLAB/Python Code Audit 2026-04-08](matlab_python_code_audit_2026-04-08.md)
3. [Imported-MATLAB Parity Evidence 2026-04-08](parity_investigation_notes_2026-04-08.md)
