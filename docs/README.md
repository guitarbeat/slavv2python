# Documentation

This directory now stays intentionally small at the top level:

- `README.md`
- `chapters/`
- `reference/`

Chapter-specific history lives under `chapters/`, and cross-chapter reference
material lives under `reference/`. Historical operational and parity writeups
now live directly inside the relevant chapter folders under `chapters/`.

Use [chapters/README.md](chapters/README.md) as the entry point for the
chapter system itself.

## Start Here

Follow this maintained reading path:

1. [Repository README](../README.md)
2. [Contributor workflow commands](../AGENTS.md)
3. [MATLAB Translation Guide](reference/MATLAB_TRANSLATION_GUIDE.md)
4. [MATLAB Mapping](reference/MATLAB_MAPPING.md)
5. [Energy Computation Methods](reference/ENERGY_METHODS.md)
6. [Comparison Run Layout](reference/COMPARISON_LAYOUT.md)
7. [Chapter index](chapters/README.md)
8. [Shared Neighborhood Claim Alignment](chapters/shared-neighborhood-claim-alignment/README.md)

## Chapter Status

### Closed Chapter

Name:

- Imported-MATLAB Parity Investigation

Closure document:

- [PARITY_REPORT_2026-04-09.md](chapters/imported-matlab-parity/PARITY_REPORT_2026-04-09.md)

What it closed:

- stable exact vertex parity on the saved imported-MATLAB surface
- a narrowed saved-batch edge gap from `94 vs 85` to `94 vs 93`
- a stronger diagnosis that the remaining problem is upstream in candidate generation
- a source-backed correction that the active public MATLAB path is `get_edges_V300 -> get_edges_by_watershed`

Historical support docs:

- [PARITY_HUB.md](chapters/imported-matlab-parity/PARITY_HUB.md)
- [EDGE_PARITY_IMPLEMENTATION_PLAN.md](chapters/imported-matlab-parity/EDGE_PARITY_IMPLEMENTATION_PLAN.md)
- [PARITY_FINDINGS_2026-03-27.md](chapters/imported-matlab-parity/PARITY_FINDINGS_2026-03-27.md)
- [MATLAB_PARITY_AUDIT_CHECKLIST.md](chapters/imported-matlab-parity/MATLAB_PARITY_AUDIT_CHECKLIST.md)

## Active Chapter

Name:

- Shared Neighborhood Claim Alignment

Started:

- April 10, 2026

Active chapter home:

- [Shared Neighborhood Claim Alignment](chapters/shared-neighborhood-claim-alignment/README.md)

Core references for the active chapter:

- [MATLAB Translation Guide](reference/MATLAB_TRANSLATION_GUIDE.md)
- [MATLAB Mapping](reference/MATLAB_MAPPING.md)
- [Comparison Run Layout](reference/COMPARISON_LAYOUT.md)
- [Shared Candidate Generation Alignment](chapters/shared-candidate-generation/README.md)
- [Imported-MATLAB Parity Report](chapters/imported-matlab-parity/PARITY_REPORT_2026-04-09.md)
- [Parity Workflow Completion Spec Archive](chapters/shared-neighborhood-claim-alignment/parity-workflow-completion-spec/tasks.md)
- [Comparison Layout Smoothing Spec Archive](chapters/shared-neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md)
- [Release Verification 2026-04-14](chapters/shared-neighborhood-claim-alignment/release_verification_2026-04-14.md)

Current live-run lessons:

- the broad over-emission regime from March 30, 2026 is no longer the active
  blocker
- the repo moved into an under-emission regime by April 1, 2026, and the
  remaining gap is now narrower and more local
- several older hotspots such as `64`, `359`, and `1283` now have much better
  candidate coverage than they did earlier in the investigation
- the worst live blockers now split into distinct classes rather than one
  generic "candidate generation gap":
  - frontier pre-manifest rejection around origins such as `1482` and `1666`
  - manifest partner substitution around origins such as `1654` and `866`
  - smaller true candidate-admission gaps such as `2305` and `2492`
  - final cleanup loss at origins such as `1283`
- repeated failures with the same terminal ids, including terminal `1009`,
  suggest that some missing neighborhoods share one branch-ownership or
  parent/child invalidation rule instead of being unrelated local accidents

## Historical Handoff Chapter

Name:

- Shared Candidate Generation Alignment

What it handed off:

- the problem was narrower than a generic candidate-generation gap
- shared neighborhoods around `64`, `359`, `866`, and `1283` became the most
  actionable artifact surfaces
- candidate counts alone were not enough; local claim ordering and invalidation
  stayed live suspects

Historical handoff home:

- [Shared Candidate Generation Alignment](chapters/shared-candidate-generation/README.md)

## Reference Shelf

These docs remain active references across chapters.

| File | Purpose |
| --- | --- |
| `reference/MATLAB_TRANSLATION_GUIDE.md` | Canonical MATLAB-to-Python semantics and override guide |
| `reference/MATLAB_MAPPING.md` | Maintained MATLAB-to-Python mapping reference |
| `reference/ENERGY_METHODS.md` | Supported energy backends, parameter interactions, and extension points |
| `reference/ZARR_ENERGY_STORAGE.md` | Optional Zarr-backed storage for resumable energy artifacts |
| `reference/NAPARI_CURATOR.md` | Experimental napari-based manual curation prototype |
| `reference/ADDING_EXTRACTION_ALGORITHMS.md` | Contributor guide for wiring new extraction modes into validation, CLI, pipeline, tests, and docs |
| `reference/COMPARISON_LAYOUT.md` | Canonical staged comparison-run layout |
| `reference/EXTERNAL_LIBRARY_SURVEY_2026-04-06.md` | External package survey and context |

## By Question

| Question | Best file |
| --- | --- |
| What closed the last spec? | [PARITY_REPORT_2026-04-09.md](chapters/imported-matlab-parity/PARITY_REPORT_2026-04-09.md) |
| What chapter is active right now? | [Shared Neighborhood Claim Alignment](chapters/shared-neighborhood-claim-alignment/README.md) |
| What is the next chapter trying to solve? | [Shared Neighborhood Claim Alignment](chapters/shared-neighborhood-claim-alignment/README.md) |
| What still blocks imported-MATLAB parity right now? | [TODO.md](../TODO.md) |
| What chapter narrowed the problem before the current one? | [Shared Candidate Generation Alignment](chapters/shared-candidate-generation/README.md) |
| What MATLAB-vs-Python translation rules matter here? | [MATLAB Translation Guide](reference/MATLAB_TRANSLATION_GUIDE.md) |
| How do large resumable energy arrays get stored? | [ZARR_ENERGY_STORAGE.md](reference/ZARR_ENERGY_STORAGE.md) |
| How does the experimental napari curator fit in? | [NAPARI_CURATOR.md](reference/NAPARI_CURATOR.md) |
| Where is the historical parity workflow context? | [PARITY_HUB.md](chapters/imported-matlab-parity/PARITY_HUB.md) |
| Where is the detailed MATLAB-to-Python map? | [MATLAB_MAPPING.md](reference/MATLAB_MAPPING.md) |
| How do staged comparison outputs work? | [COMPARISON_LAYOUT.md](reference/COMPARISON_LAYOUT.md) |

## Root-Level Docs

These docs stay at the repository root because they function as entry points or
project metadata:

| File | Purpose |
| --- | --- |
| `README.md` | Project overview, quick start, and common workflows |
| `CHANGELOG.md` | Notable recent development changes |
| `AGENTS.md` | Repository instructions for coding agents and the canonical developer-command reference |

## Historical Investigation Notes

Historical parity and release notes now live inside the chapters that use them.

| File | Purpose |
| --- | --- |
| `chapters/imported-matlab-parity/parity_decision_memo_2026-04-08.md` | Decision memo near the end of the Chapter 1 parity diagnosis set |
| `chapters/imported-matlab-parity/matlab_python_code_audit_2026-04-08.md` | Technical appendix for cleanup-path and candidate-coverage evidence |
| `chapters/imported-matlab-parity/python_nondeterminism_investigation_2026-03-28.md` | Repeatability baseline and deterministic-fix context |
| `chapters/shared-neighborhood-claim-alignment/release_verification_2026-04-14.md` | Canonical-data release audit, staged metadata verification, and timing snapshot |
| `chapters/shared-neighborhood-claim-alignment/file_lock_contention_analysis_2026-04-13.md` | April 13 operational incident analysis and recovery runbook |

