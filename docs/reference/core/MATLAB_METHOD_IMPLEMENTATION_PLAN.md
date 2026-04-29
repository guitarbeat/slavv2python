# MATLAB Method Implementation Plan

[Up: Reference Docs](../README.md)

This document defines what it means to say that Python implements the released
SLAVV method and records the remaining work to make that statement truthful.

Use this file for claim boundaries, source-of-truth rules, and roadmap phases.
Use `EXACT_PROOF_FINDINGS.md` for live proof status and current v22 watershed
readouts.

## Purpose

- resolve ambiguity between paper prose, released MATLAB source, and current
  Python status
- define the canonical source-of-truth hierarchy for parity work
- separate source-level porting from artifact-proven implementation
- track the native-first transition from historical MATLAB-imported exact reruns
  to a canonical Python exact route

## Canonical Hierarchy

When these sources differ, use this order:

1. Released MATLAB source under `external/Vectorization-Public/source/`
2. Preserved MATLAB artifacts validated by `prove-exact`
3. The paper PDF at `docs/reference/papers/journal.pcbi.1009451.pdf`
4. Maintained Python docs such as `MATLAB_PARITY_MAPPING.md`

Implications:

- The released MATLAB code is the executable specification for parity work.
- Preserved MATLAB vectors are the oracle proof artifacts.
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
  but the maintained proof gate is not yet green for that stage.
- `Artifact-proven exact`:
  the Python stage matches preserved MATLAB vectors exactly under the maintained
  proof surface.
- `Full paper method implemented in Python`:
  Python reproduces the method end to end from raw image inputs without runtime
  dependence on imported MATLAB energy artifacts.

Do not use `exact` or `100%` for a stage unless that stage is artifact-proven.

## Current Exact-Route Boundary

The maintained exact route is native-first.

- It activates when `comparison_exact_network` is enabled.
- It accepts any exact-compatible energy provenance.
- The canonical provenance is `python_native_hessian`.
- `matlab_batch_hdf5` remains accepted only for historical replay,
  regression comparison, and oracle-backed diagnostics.
- Preserved MATLAB vectors remain the proof oracle for `prove-exact`.

The parity-facing orchestration surface for this work now lives under
`source/core/matlab_compat/`, which mirrors the released MATLAB stage and
function boundaries while delegating into the maintained modular Python code.

## Current Stage Status

| Stage | Truthful claim today | Main blocker |
| --- | --- | --- |
| Energy / size image generation | Python has a native exact-route energy implementation and no longer depends on imported MATLAB energy at runtime | Keep MATLAB-oracle fixture coverage broad and direct/resumable parity green |
| Vertex extraction | Source-aligned and exact-route ready on native energy | Downstream proof bookkeeping is still centered on edges and network |
| Edge extraction | Source-aligned on the native-first exact route | Candidate-generation and chooser proof are still red; see `EXACT_PROOF_FINDINGS.md` |
| Edge cleanup / bridge insertion | Source-aligned | Downstream of unresolved edge mismatch |
| Network / strand assembly | Source-aligned | Downstream of unresolved edge mismatch |

## What Must Be True Before We Claim Full Python Implementation

1. The native-first exact route must pass `prove-exact --stage all`.
2. Maintained docs must describe `python_native_hessian` as the canonical
   exact-compatible source surface and must not describe imported MATLAB energy
   as the active runtime dependency.
3. Native energy fixture coverage must remain green for projected energy,
   `scale_indices`, `energy_4d`, and key intermediates such as Laplacian and
   valid-mask surfaces.
4. Vertices, edges, and network must be artifact-proven on the native-first
   exact route, with preserved MATLAB vectors still serving as the oracle.

## Implementation Phases

### Phase 1: Native Energy Cutover

Status: complete enough to change the canonical route.

Completed work:

1. Native Hessian matched filtering now implements the maintained raw-image
   energy stage.
2. `python_native_hessian` is the canonical exact-compatible provenance.
3. The exact-route gate and proof tooling no longer require
   `matlab_batch_hdf5`.
4. `source/core/matlab_compat/` now provides MATLAB-shaped orchestration and
   function wrappers for audits and proof routing.

### Phase 2: Close Downstream Native Exact Parity

Status: active.

Primary work items:

1. close the remaining `edges.connections` mismatch on the native-first exact
   route
2. re-run `prove-exact` after every math-bearing edge or network change
3. keep `EXACT_PROOF_FINDINGS.md` current with the first failing field and the
   measured effect of each fix
4. continue using `source/core/matlab_compat/` as the parity-facing audit
   surface instead of ad hoc route descriptions

Acceptance gate:

- `vertices`, `edges`, and `network` all pass `prove-exact` on the native-first
  exact route

## Current File-Level Gap Checklist

These are the concrete code surfaces that still need work before downstream
native exact parity is done.

1. `source/core/_edge_candidates/global_watershed.py`
   Close the remaining candidate-generation gap against preserved MATLAB edge
   pairs. Candidate emission still appears to be the first major downstream
   mismatch surface.
2. `source/core/_edge_selection/conflict_painting.py`
   Replace sequential trace iteration with MATLAB-matching randomized trace
   order if the exact route is going to claim literal chooser parity.
3. `source/core/_edge_selection/cleanup.py`
   Re-check crop, degree, orphan, and cycle cleanup whenever
   `edges.connections` improves but remains red.
4. `source/core/_edges/bridge_vertices.py`
   Keep the bridge path in sync with the exact-route proof surface.
5. `source/core/graph.py`
   Audit strand ordering and network assembly only after the upstream edge proof
   surfaces are materially closer.
6. `dev/scripts/cli/parity_experiment.py` and `source/io/matlab_exact_proof.py`
   Preserve the proof harness as the acceptance gate for native-first exact
   reruns.

## Documentation Rules

Apply these rules across parity docs:

- Use `native-first exact route` for the current maintained route.
- Use `historical imported-MATLAB replay` only for the preserved-energy
  compatibility surface.
- Use `source-aligned` when code appears ported but proof is still pending.
- Use `artifact-proven exact` only when the maintained proof gate is green for
  that stage.
- Do not describe the current route as imported-MATLAB-only unless you are
  explicitly talking about the historical replay surface.

## Related Docs

- `MATLAB_PARITY_MAPPING.md`: source-level stage map and confirmed structural
  deviations
- `EXACT_PROOF_FINDINGS.md`: live proof status and v22 watershed readouts
- `ENERGY_METHODS.md`: maintained native energy backend surface
- `../papers/journal.pcbi.1009451.pdf`: paper narrative and published methods
