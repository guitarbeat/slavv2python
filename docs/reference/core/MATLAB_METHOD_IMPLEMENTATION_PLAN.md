# MATLAB Method Implementation Plan

[Up: Reference Docs](../README.md)

## In short

This file says what “Python implements the published SLAVV method” is allowed to
mean. Live pass/fail is [EXACT_PROOF_FINDINGS.md](EXACT_PROOF_FINDINGS.md), not
this plan. Public `slavv run` (**Paper Path**, `paper` profile) is separate from
the exact MATLAB proof track — and is **not** the 2021 publication. Phase 1
exact-route certification is already closed. Papers index:
[papers/README.md](../papers/README.md).

This document defines what it means to say that Python implements the released
SLAVV method and records the remaining work to make that statement truthful.

Use this file for claim boundaries, source-of-truth rules, and implementation phases.
Use `EXACT_PROOF_FINDINGS.md` for live proof status and current parity
blockers.

## Purpose

- resolve ambiguity between paper prose, released MATLAB source, and current
  Python status
- define the canonical source-of-truth hierarchy for parity work
- separate the public paper-complete Python product from artifact-proven exact
  MATLAB parity
- separate source-level porting from artifact-proven implementation
- track the native-first transition from historical MATLAB-imported exact reruns
  to a canonical Python exact route

## Canonical Hierarchy

When these sources differ, use this order:

1. Released MATLAB slavv_python under `external/Vectorization-Public/source/`
2. Preserved MATLAB artifacts validated by `prove-exact`
3. The published paper (DOI [10.1371/journal.pcbi.1009451](https://doi.org/10.1371/journal.pcbi.1009451); explanatory, not a higher-priority spec than the MATLAB source)
4. Maintained Python docs such as `MATLAB_PARITY_MAPPING.md`

Implications:

- The released MATLAB code is the executable specification for parity work.
- Preserved MATLAB vectors are the oracle proof artifacts and should be stored
  as standalone packages under the maintained `oracles/` experiment-root
  surface.
- The paper prose is explanatory context, not a higher-priority spec than the
  released MATLAB code.
- Current Python docs must never overrule either the MATLAB slavv_python or proof
  artifacts.

Annotated bibliography for this hierarchy and the other four common confusions:
[papers/README.md](../papers/README.md).

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
  users can run the maintained native Python TIFF-to-network workflow through
  the public surfaces, receive the authoritative JSON export, and analyze or
  visualize the result without MATLAB runtime dependencies.
- `Exact MATLAB parity complete`:
  the native-first exact route is artifact-proven at the maintained
  `prove-exact` gate through `network`.

Do not use `exact` or `100%` for a stage unless that stage is artifact-proven.

## Current Exact-Route Boundary

The maintained exact route is native-first.

- It activates when `comparison_exact_network` is enabled.
- It accepts any exact-compatible energy provenance.
- The canonical provenance is `python_native_hessian`.
- Preserved MATLAB vectors remain the proof oracle for `prove-exact`.

The parity-facing orchestration surface for this work now lives under
`slavv_python/pipeline/edges/watershed/matlab_*.py` modules, which mirror the released MATLAB function
boundaries while delegating into the maintained modular Python code.

## Current Stage Status

> **Authority Delegation**: Live per-stage parity status, proof paths, and residual definitions are exclusively maintained in [EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk](EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk). Phase 1 Exact Route parity is **CLOSED** on `canonical_full_v18` with all four stages certified (Energy ADR 0011, Vertices ADR 0011, Edges ADR 0012 evaluated, Network ADR 0012 evaluated).

## What Must Be True Before We Claim Full Paper Implementation

1. `slavv run` and the Streamlit processing page expose a first-class `paper`
   profile as the primary public workflow.
2. The public `paper` workflow must run the native Python TIFF-to-network
   pipeline without runtime dependence on imported MATLAB energy artifacts.
3. `network.json` must be the authoritative versioned export, and
   `slavv analyze` / `slavv plot` must consume it directly.
4. Maintained docs must clearly distinguish the paper-complete public workflow
   from the exact MATLAB proof track.
5. CLI, app, and automated integration coverage must stay green for the paper
   workflow on a maintained example dataset.

## What Must Be True Before We Claim Exact MATLAB Parity Complete

1. The native-first exact route must pass `prove-exact --stage all`.
2. Maintained docs must describe `python_native_hessian` as the canonical
   exact-compatible slavv_python surface and must not describe imported MATLAB energy
   as the active runtime dependency.
3. Native energy fixture coverage must remain green for projected energy,
   `scale_indices`, `energy_4d`, and key intermediates such as Laplacian and
   valid-mask surfaces.
4. Vertices, edges, and network must be artifact-proven on the native-first
   exact route, with preserved MATLAB vectors still serving as the oracle.

## Public Paper Workflow

The maintained public workflow is now paper-first:

- default public profile: `paper`
- alternate legacy-oriented profile: `matlab_compat`
- authoritative export: versioned `network.json`
- public acceptance gate: CLI + Streamlit app + automated tests

That public finish line is intentionally separate from the developer-only exact
proof tooling.

## Implementation Phases

### Phase 1: Native Energy Cutover

Status: complete enough to change the canonical route.

Completed work:

1. Native Hessian matched filtering now implements the maintained raw-image
   energy stage.
2. `python_native_hessian` is the canonical exact-compatible provenance.
3. The exact-route gate and proof tooling no longer accept imported MATLAB
   energy provenance.
4. `slavv_python/pipeline/edges/watershed/matlab_*.py` now provides MATLAB-shaped ports and
   function wrappers for audits and proof routing.

### Phase 2: Close Downstream Native Exact Parity

Status: **COMPLETE** (Architectural Alignment)

Completed work:
1. Implemented bit-accurate tie-breaking using exact equality and Fortran-order linear index priority.
2. Plugged precision leaks by enforcing `float64` across all watershed maps and suppression factors.
3. Tightened candidate filtering with hard $d/R$ cutoffs matching MATLAB's `get_edges_by_watershed`.
4. Fixed the double-transpose orientation bug in the watershed candidate path; edges now certify on the voxel ownership-map (~63.5%) + per-edge trace tolerance per [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md). (The historical **88.7%** v29 pair-match figure is deprecated — it was inflated by the wrong grid; pair-set equality is not the edge bar.)

### Phase 3: Certification and Release

Status: **COMPLETE** (Phase 1 Exact Route Certified on `canonical_full_v18`)

Completed work:
1. Passed sequential stage certification across Energy (ADR 0011), Vertices (ADR 0011), Edges (ADR 0012 evaluated), and Network (ADR 0012 evaluated) on full `180709_E` (`canonical_full_v18`).
2. Executed staged proofs sequentially, establishing the immutable baseline freeze in `phase1-baseline-freeze.json`.
3. Promoted the native Python engine to standard research deployment.

Acceptance gate:
- `energy`, `vertices`, `edges`, and `network` certified per ADR 0011 and ADR 0012 on the native-first exact route (`canonical_full_v18`).

## Resolved File-Level Gap Checklist

> [!TIP]
> The real-time task checklist and planning hub (plans, brainstorms, compound solutions) live in the [TODO.md Developer Dashboard](../../TODO.md). Live per-stage proof status is in [EXACT_PROOF_FINDINGS.md](EXACT_PROOF_FINDINGS.md).

These reference surfaces were resolved during Phase 1 parity certification (closed on `canonical_full_v18`):

1. `slavv_python/pipeline/edges/candidate_generation.py`
   Candidate-generation surface fully aligned and certified under ADR 0012 ownership-map and trace tolerances.
2. `slavv_python/pipeline/edges/selection.py`
   Claimed trace energy provenance baked at Watershed Discovery finalize (ADR 0013), matching MATLAB's claimed/penalized surface ranking.
3. `slavv_python/pipeline/edges/cleanup.py`
   Crop, degree, orphan, and cycle cleanup aligned and verified against MATLAB exact comparator.
4. `slavv_python/pipeline/edges/bridge_insertion.py`
   Bridge insertion synchronized with the exact-route proof surface and certified through Network assembly.
5. `slavv_python/pipeline/network/`
   Strand assembly, bifurcation multiset, and trace geometry certified under ADR 0012 on `canonical_full_v18`.
6. `slavv parity` and `slavv_python/analytics/parity/proofs.py`
   Proof harness maintained as the acceptance gate for native-first exact verification, with disposable trial runs under `workspace/runs/`, preserved MATLAB truth under `workspace/oracles/`, and promoted summaries under `workspace/reports/`.

## Documentation Rules

Apply these rules across parity docs:

- Use `native-first exact route` for the current maintained route.
- Use `source-aligned` when code appears ported but proof is still pending.
- Use `artifact-proven exact` only when the maintained proof gate is green for
  that stage.
- Do not describe the current route as imported-MATLAB-compatible.

## Related Docs

- `MATLAB_PARITY_MAPPING.md`: source-level stage map and confirmed structural
  deviations
- `EXACT_PROOF_FINDINGS.md`: live proof status and current parity blockers
- `ENERGY_METHODS.md`: maintained native energy backend surface
- `../workflow/PAPER_PROFILE.md`: public paper-first CLI/app workflow and JSON
  export contract
- [doi:10.1371/journal.pcbi.1009451](https://doi.org/10.1371/journal.pcbi.1009451): paper narrative and published methods
