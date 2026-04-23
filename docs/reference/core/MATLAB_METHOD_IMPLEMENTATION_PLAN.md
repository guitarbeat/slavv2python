# MATLAB Method Implementation Plan

[Up: Reference Docs](../README.md)

This document defines what it means to say that Python "implements the paper's
method" for SLAVV and records the remaining work to make that statement
truthful.

## Purpose

- Resolve ambiguity between paper prose, released MATLAB source, and current
  Python status.
- Define the canonical source-of-truth hierarchy for parity work.
- Separate "source-level porting" from "artifact-proven implementation".
- Provide the concrete sequence required to fully implement the released SLAVV
  method in Python.

## Canonical Hierarchy

When these sources differ, use this order:

1. Released MATLAB source under `external/Vectorization-Public/source/`
2. Preserved MATLAB artifacts validated by `prove-exact`
3. The paper PDF at `docs/reference/papers/journal.pcbi.1009451.pdf`
4. Maintained Python docs such as `MATLAB_PARITY_MAPPING.md`

Implications:

- The released MATLAB code is the executable specification for parity work.
- The paper prose is explanatory context, not a higher-priority spec than the
  released MATLAB code.
- Current Python docs must never overrule either the MATLAB source or proof
  artifacts.

## Claim Boundaries

Use the following labels precisely:

- `Conceptually consistent with the paper`:
  the Python stage follows the same high-level idea, but it is not yet proven
  equal to the MATLAB implementation.
- `Source-aligned`:
  the Python stage appears line-by-line aligned with the released MATLAB source,
  but `prove-exact` is not yet green for that stage or route.
- `Artifact-proven exact`:
  the Python stage matches preserved MATLAB vectors exactly under
  `prove-exact`.
- `Full paper method implemented in Python`:
  Python reproduces the method end to end from raw image inputs, not only on the
  imported-MATLAB exact route.

Do not use `Exact` or `100%` for a stage unless the stage is artifact-proven.

## Current Scope Boundary

The current exact imported-MATLAB route is narrower than the full paper method.

- It only activates when `comparison_exact_network` is enabled.
- It only activates when `energy_origin == matlab_batch_hdf5`.
- It therefore reuses MATLAB-produced energy artifacts rather than proving that
  Python independently reproduces the paper's linear filtering step from raw
  input images.

See:

- `source/slavv/core/_edge_candidates/common.py`
- `source/slavv/core/_edges/standard.py`
- `docs/reference/core/EXACT_PROOF_FINDINGS.md`

## Current Stage Status

| Stage | Current status | Truthful claim today | Main blocker |
| --- | --- | --- | --- |
| Energy / size image generation | Not part of the current exact imported-MATLAB proof target | Not yet a full Python implementation of the paper's end-to-end method | The exact route currently reuses MATLAB energy artifacts |
| Vertex extraction | Source-aligned and artifact-proven on the imported-MATLAB exact route | Exact only on the imported-MATLAB exact route, not yet full-method Python from raw input | Depends on imported MATLAB energy inputs |
| Edge extraction | Source-aligned in many places, but not artifact-proven | Not yet exact | `prove-exact` still fails at `edges.connections` |
| Edge cleanup / bridge insertion | Source-aligned in many places, but not artifact-proven | Not yet exact | Downstream of unresolved edge mismatch |
| Network / strand assembly | Source-aligned in many places, but not artifact-proven | Not yet exact | Downstream of unresolved edge mismatch |

## What Must Be True Before We Claim Full Python Implementation

1. The imported-MATLAB exact route must pass `prove-exact --stage all`.
2. The parity docs must stop using `Exact` for stages that are only source-level
   ports.
3. Python must independently reproduce the paper's energy / size image generation
   from raw inputs, not only consume imported MATLAB energy artifacts.
4. That Python-native path must then be validated against the released MATLAB
   implementation and artifacts.

## Immediate Implementation Order

### Phase 1: Close The Imported-MATLAB Exact Route

This phase is about getting the current exact-route artifact proof green.

Primary work items:

1. Close the remaining `edges.connections` mismatch on the exact route.
2. Re-run `prove-exact` after every math-bearing edge change.
3. Keep `EXACT_PROOF_FINDINGS.md` current with the first failing field and the
   measured effect of each fix.

Acceptance gate:

- `vertices`, `edges`, and `network` all pass `prove-exact` on the imported-
  MATLAB exact route.

### Current File-Level Gap Checklist

These are the concrete code surfaces that still need work before Phase 1 is
done.

1. `source/slavv/core/_edge_candidates/global_watershed.py`
   Close the remaining candidate-generation gap against preserved MATLAB edge
   pairs. The current exact route still misses MATLAB-valid connections before
   chooser cleanup.
2. `source/slavv/core/_edge_selection/conflict_painting.py`
   Continue auditing conflict-painting acceptance order against released MATLAB
   `choose_edges_V200.m`. This is still one of the main places where MATLAB-
   valid pairs are lost after candidate generation.
3. `source/slavv/core/_edge_selection/cleanup.py`
   Re-check crop, degree, orphan, and cycle cleanup against the exact-route
   proof artifacts whenever `edges.connections` improves but is still not
   green.
4. `source/slavv/core/_edges/bridge_vertices.py`
   Keep the bridge path in sync with the exact-route proof surface. This stage
   is downstream of the unresolved edge mismatch and should not be called exact
   until the full edge artifact proof is green.
5. `source/slavv/io/matlab_exact_proof.py` and
   `dev/scripts/cli/parity_experiment.py`
   Preserve the proof harness as the acceptance gate for every exact-route
   change. Do not regress the staged artifact comparison contract while fixing
   edge math.

### Phase 2: Remove The Remaining "Imported MATLAB Only" Boundary

This phase is about implementing the paper's full method in Python rather than
only the imported-MATLAB parity route.

Primary work items:

1. Audit Python's native energy / size generation against the released MATLAB
   energy-filtering implementation and the paper's "Energy: Multi-scale linear
   filtering" section.
2. Define a proof surface for Python-native energy outputs.
3. Validate the native Python path from raw image to vectors against the released
   MATLAB implementation.

Acceptance gate:

- Python can produce energy, vertices, edges, and network outputs from raw image
  inputs without relying on imported MATLAB energy artifacts, and those outputs
  are artifact-proven against MATLAB.

## Documentation Rules

Apply these rules across parity docs:

- Use `source-aligned` when code appears ported but proof is still pending.
- Use `artifact-proven exact` only when `prove-exact` is green for that stage.
- Say `imported-MATLAB exact route` when the route still depends on preserved
  MATLAB energy.
- Do not describe the current route as the full paper method in Python while the
  energy stage is still outsourced to MATLAB artifacts.

## Related Docs

- `MATLAB_PARITY_MAPPING.md`: source-level stage map
- `EXACT_PROOF_FINDINGS.md`: current proof failures and measured fixes
- `../papers/journal.pcbi.1009451.pdf`: paper narrative and published methods
