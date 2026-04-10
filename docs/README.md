# Documentation

This directory now stays intentionally small at the top level:

- `README.md`
- `chapters/`
- `reference/`

Chapter-specific history lives under `chapters/`, and cross-chapter reference
material lives under `reference/`.

## Start Here

1. [PARITY_REPORT_2026-04-09.md](chapters/imported-matlab-parity/PARITY_REPORT_2026-04-09.md)
2. [README.md](#active-chapter)
3. [MATLAB_MAPPING.md](reference/MATLAB_MAPPING.md)

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

- Shared Candidate Generation Alignment

Started:

- April 10, 2026

Why this is a new chapter:

- The previous spec ended with a stable handoff rather than an open-ended parity chase.
- Vertex parity is already stable.
- The remaining gap is narrow enough to treat as a focused candidate-generation chapter instead of a general parity chapter.

Current starting facts:

- current best retained saved-batch result is `vertices 110/110`, `edges 94/93`, `strands 49/54`
- origin `64` remains the clearest next target
- the retained geodesic widening to `k=10` improved the saved-batch result without breaking the broader suite
- the relaxed geodesic and origin-owned pruning experiments were informative but rejected

Main goal:

- align Python candidate generation more closely with active MATLAB shared-state / watershed behavior

Primary questions:

1. Why does origin `64` remain under-covered in the retained candidate set?
2. Does MATLAB temporarily allow over-budget candidate admission before later cleanup in a way Python still does not?
3. Which parts of `get_edges_by_watershed` shared-map behavior are still not represented in Python candidate discovery?
4. Is the remaining loss happening during candidate admission, claim ownership, or candidate conflict resolution?

In scope:

- candidate-endpoint coverage
- shared-state / watershed join semantics
- origin ownership and claim-map behavior
- bounded parity-only experiments on the saved batch

Out of scope:

- re-solving vertex parity
- broad threshold sweeps without a targeted diagnostic reason
- generic downstream network assembly work already known to pass when exact MATLAB edges are imported

Suggested first loop:

```powershell
python workspace/scripts/cli/compare_matlab_python.py `
  --input comparison_output_synthetic_final_20260409/00_InputFixtures/synthetic_branch_volume.tif `
  --skip-matlab `
  --output-dir comparison_output_next_chapter `
  --params comparison_output_synthetic_final_20260409_rerun/99_Metadata/comparison_params.normalized.json
```

Core references for the active chapter:

- [PARITY_REPORT_2026-04-09.md](chapters/imported-matlab-parity/PARITY_REPORT_2026-04-09.md)
- [MATLAB_MAPPING.md](reference/MATLAB_MAPPING.md)
- [COMPARISON_LAYOUT.md](reference/COMPARISON_LAYOUT.md)
- [PARITY_FINDINGS_2026-03-27.md](chapters/imported-matlab-parity/PARITY_FINDINGS_2026-03-27.md)

## Reference Shelf

These docs remain active references across chapters.

| File | Purpose |
| --- | --- |
| `reference/MATLAB_MAPPING.md` | Maintained MATLAB-to-Python mapping reference |
| `reference/COMPARISON_LAYOUT.md` | Canonical staged comparison-run layout |
| `reference/EXTERNAL_LIBRARY_SURVEY_2026-04-06.md` | External package survey and context |
| `reference/ARNIS_CROSS_PLATFORM_ARCHITECTURE_2026-04-08.md` | Architecture reference unrelated to the parity chapter split |

## By Question

| Question | Best file |
| --- | --- |
| What closed the last spec? | [PARITY_REPORT_2026-04-09.md](chapters/imported-matlab-parity/PARITY_REPORT_2026-04-09.md) |
| What chapter is active right now? | [README.md](#active-chapter) |
| What is the next chapter trying to solve? | [README.md](#active-chapter) |
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
| `BOTTLENECK_TODO.md` | Workflow notes and cross-cutting backlog |
| `AGENTS.md` | Repository instructions for coding agents |

## High-Value Reports Outside `docs/`

These are not chapter control docs, but they remain useful evidence.

| File | Purpose |
| --- | --- |
| `workspace/reports/stage_isolated_network_parity_2026-04-07.md` | Proof that exact MATLAB `edges` plus Python `network` can converge exactly |
| `workspace/reports/parity_decision_memo_2026-04-08.md` | Decision memo near the end of the closed parity chapter |
| `workspace/reports/python_matlab_parity_postfix_2026-03-30.md` | Historical parity checkpoint |
| `workspace/reports/python_standalone_consistency_postfix_2026-03-30.md` | Python repeatability context |
