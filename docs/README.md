# Documentation

This directory now stays intentionally small at the top level:

- `README.md`
- `chapters/`
- `reference/`

Chapter-specific history lives under `chapters/`, and cross-chapter reference
material lives under `reference/`. Historical operational and parity writeups
now live directly inside the relevant chapter folders under `chapters/`.

Use [chapters/README.md](chapters/README.md) as the entry point for the
chapter system itself, and [reference/README.md](reference/README.md) as the
entry point for maintained cross-cutting reference docs.

## Start Here

Follow this maintained reading path:

1. [Repository README](../README.md)
2. [Contributor workflow commands](../AGENTS.md)
3. [Project Glossary](reference/GLOSSARY.md)
4. [MATLAB Translation Guide](reference/MATLAB_TRANSLATION_GUIDE.md)
5. [MATLAB Mapping](reference/MATLAB_MAPPING.md)
6. [Energy Computation Methods](reference/ENERGY_METHODS.md)
7. [Comparison Run Layout](reference/COMPARISON_LAYOUT.md)
8. [Chapter index](chapters/README.md)
9. [Neighborhood Claim Alignment](chapters/neighborhood-claim-alignment/README.md)

## New Contributor Guide

If you are new to the project:

- Read the [Glossary](reference/GLOSSARY.md) to understand project terminology.
- Follow the **Setup** instructions in the [Root README](../README.md).
- Use `slavv info` to verify your installation.
- Check the [MATLAB Translation Guide](reference/MATLAB_TRANSLATION_GUIDE.md) before touching any core algorithmic code.
- Read the [Active Chapter](chapters/neighborhood-claim-alignment/README.md) to see what is currently being worked on.

## Chapter Status

### Closed Chapter

Name:

- Imported-MATLAB Parity Closeout

Closure document:

- [parity_closeout.md](chapters/imported-matlab-parity-closeout/parity_closeout.md)

What it closed:

- stable exact vertex parity on the saved imported-MATLAB surface
- a narrowed saved-batch edge gap from `94 vs 85` to `94 vs 93`
- a stronger diagnosis that the remaining problem is upstream in candidate generation
- a source-backed correction that the active public MATLAB path is `get_edges_V300 -> get_edges_by_watershed`

Historical support docs:

- [PARITY_HUB.md](chapters/imported-matlab-parity-closeout/PARITY_HUB.md)
- [edge_parity_plan.md](chapters/imported-matlab-parity-closeout/edge_parity_plan.md)
- [parity_findings.md](chapters/imported-matlab-parity-closeout/parity_findings.md)
- [MATLAB_PARITY_AUDIT_CHECKLIST.md](chapters/imported-matlab-parity-closeout/MATLAB_PARITY_AUDIT_CHECKLIST.md)

### Active Chapter

Name:

- Neighborhood Claim Alignment

Started:

- April 10, 2026

Active chapter home:

- [Neighborhood Claim Alignment](chapters/neighborhood-claim-alignment/README.md)

Core references for the active chapter:

- [MATLAB Translation Guide](reference/MATLAB_TRANSLATION_GUIDE.md)
- [MATLAB Mapping](reference/MATLAB_MAPPING.md)
- [Comparison Run Layout](reference/COMPARISON_LAYOUT.md)
- [Candidate Generation Handoff](chapters/candidate-generation-handoff/README.md)
- [Imported-MATLAB Parity Closeout](chapters/imported-matlab-parity-closeout/parity_closeout.md)
- [Parity Workflow Completion Spec Archive](chapters/neighborhood-claim-alignment/parity-workflow-completion-spec/tasks.md)
- [Comparison Layout Smoothing Spec Archive](chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md)
- [Release Verification 2026-04-14](chapters/neighborhood-claim-alignment/release_verification_2026-04-14.md)
- [Working docs index](chapters/neighborhood-claim-alignment/working/README.md)
- [Archived notes index](chapters/neighborhood-claim-alignment/archive/README.md)

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

### Historical Handoff Chapter

Name:

- Candidate Generation Handoff

What it handed off:

- the problem was narrower than a generic candidate-generation gap
- shared neighborhoods around `64`, `359`, `866`, and `1283` became the most
  actionable artifact surfaces
- candidate counts alone were not enough; local claim ordering and invalidation
  stayed live suspects

Historical handoff home:

- [Candidate Generation Handoff](chapters/candidate-generation-handoff/README.md)

## Reference Shelf

These docs remain active references across chapters. Start with the folder
index at [reference/README.md](reference/README.md) for a grouped view of the
maintained material.

| File | Purpose |
| --- | --- |
| `reference/README.md` | Entry point for the maintained reference shelf |
| `reference/GLOSSARY.md` | Domain-specific and project-specific terms |
| `reference/MATLAB_TRANSLATION_GUIDE.md` | Canonical MATLAB-to-Python semantics and override guide |
| `reference/MATLAB_MAPPING.md` | Maintained MATLAB-to-Python mapping reference |
| `reference/ENERGY_METHODS.md` | Supported energy backends, parameter interactions, and extension points |
| `reference/SIMPLEITK_ENERGY_BACKEND.md` | Spacing-aware vesselness backend |
| `reference/CUPY_ENERGY_BACKEND.md` | GPU-accelerated energy backend |
| `reference/ZARR_ENERGY_STORAGE.md` | Optional Zarr-backed storage for resumable energy artifacts |
| `reference/NAPARI_CURATOR.md` | Experimental napari-based manual curation prototype |
| `reference/ADDING_EXTRACTION_ALGORITHMS.md` | Contributor guide for wiring new extraction modes into validation, CLI, pipeline, tests, and docs |
| `reference/COMPARISON_LAYOUT.md` | Canonical staged comparison-run layout |
| `reference/EXTERNAL_LIBRARY_SURVEY_2026-04-06.md` | Short external-package status note |

## By Question

| Question | Best file |
| --- | --- |
| What does this term mean? | [Glossary](reference/GLOSSARY.md) |
| What closed the last spec? | [Imported-MATLAB Parity Closeout](chapters/imported-matlab-parity-closeout/parity_closeout.md) |
| What chapter is active right now? | [Neighborhood Claim Alignment](chapters/neighborhood-claim-alignment/README.md) |
| What is the next chapter trying to solve? | [Neighborhood Claim Alignment](chapters/neighborhood-claim-alignment/README.md) |
| What still blocks imported-MATLAB parity right now? | [TODO.md](../TODO.md) |
| What chapter narrowed the problem before the current one? | [Candidate Generation Handoff](chapters/candidate-generation-handoff/README.md) |
| What MATLAB-vs-Python translation rules matter here? | [MATLAB Translation Guide](reference/MATLAB_TRANSLATION_GUIDE.md) |
| How do large resumable energy arrays get stored? | [ZARR_ENERGY_STORAGE.md](reference/ZARR_ENERGY_STORAGE.md) |
| How does the experimental napari curator fit in? | [NAPARI_CURATOR.md](reference/NAPARI_CURATOR.md) |
| Where is the historical parity workflow context? | [PARITY_HUB.md](chapters/imported-matlab-parity-closeout/PARITY_HUB.md) |
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
| `chapters/imported-matlab-parity-closeout/parity_decision_memo_2026-04-08.md` | Decision memo near the end of the Chapter 1 parity diagnosis set |
| `chapters/imported-matlab-parity-closeout/matlab_python_code_audit_2026-04-08.md` | Technical appendix for cleanup-path and candidate-coverage evidence |
| `chapters/imported-matlab-parity-closeout/python_nondeterminism_investigation_2026-03-28.md` | Repeatability baseline and deterministic-fix context |
| `chapters/neighborhood-claim-alignment/release_verification_2026-04-14.md` | Canonical-data release audit, staged metadata verification, and timing snapshot |
| `chapters/neighborhood-claim-alignment/file_lock_contention_analysis_2026-04-13.md` | April 13 operational incident analysis and recovery runbook |
