# Parity Hub

Status: Historical Chapter 1 hub

Successor:

- [Neighborhood Claim Alignment](../neighborhood-claim-alignment/README.md)

Use [chapters/README.md](../README.md) for chapter-system navigation and
[TODO.md](../../../TODO.md) for the current root-level parity backlog.

This file is no longer the active spec entry point.

Use [README.md](../../README.md) for chapter status,
[Neighborhood Claim Alignment](../neighborhood-claim-alignment/README.md)
for the active chapter, and
[parity_closeout.md](parity_closeout.md) for the Chapter 1 closeout.

This was the fastest re-entry point during Chapter 1.

If you need the active starting point now, read
[Neighborhood Claim Alignment](../neighborhood-claim-alignment/README.md)
and the maintained
[MATLAB Translation Guide](../../reference/MATLAB_TRANSLATION_GUIDE.md).

## Historical Status At Chapter 1 Close

- Vertex parity: exact on the imported-MATLAB parity surface.
- Network parity: exact when Python is given exact MATLAB `edges` and reruns
  from `network` with parity-mode network assembly enabled.
- Edge parity: still the main blocker.
- Repeatability: imported-MATLAB Python reruns are repeatable on the current
  machine; the remaining gap is systematic, not stochastic.
- Current best diagnosis:
  - the candidate pool is still wrong upstream in edge generation
  - the Python cleanup path is also modeling the wrong MATLAB V200 cleanup
    surface
  - downstream generic network assembly is still not the primary blocker

## Default Loops

### 1. Default edge-convergence loop

Use when you are working on edge generation and want the normal imported-MATLAB
parity surface.

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --input data/slavv_test_volume.tif `
  --skip-matlab `
  --resume-latest `
  --python-parity-rerun-from edges
```

### 2. Stage-isolated network gate

Use when you want to prove that a regression is in `edges`, not in downstream
network assembly.

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --input data/slavv_test_volume.tif `
  --skip-matlab `
  --resume-latest `
  --python-parity-rerun-from network `
  --comparison-depth deep
```

Expected result on the current parity surface: exact vertices, exact edges, and
exact strands.

### 3. Analysis-only loop

Use when you want to compare existing artifacts without rerunning MATLAB or
Python.

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --standalone-matlab-dir <matlab_results_dir> `
  --standalone-python-dir <python_results_dir> `
  --python-result-source checkpoints-only `
  --comparison-depth shallow
```

## Read Order

1. [TODO.md](../../../TODO.md)
   Current workflow surface, what still blocks imported-MATLAB parity, and what
   the next iteration should optimize.
2. [parity_decision_memo_2026-04-08.md](parity_decision_memo_2026-04-08.md)
   Current short decision memo for the April 8 code-and-artifact audit.
3. [edge_parity_plan.md](edge_parity_plan.md)
  Current parity plan focused on the remaining edge-generation gap.
4. [parity_findings.md](parity_findings.md)
  Verified findings and the longer evidence behind the standing diagnosis.

## Which File Answers Which Question

| Question | Best file |
| --- | --- |
| What is true right now, quickly? | [Neighborhood Claim Alignment](../neighborhood-claim-alignment/README.md) |
| What is the current implementation decision from the April 8 audit? | [parity_decision_memo_2026-04-08.md](parity_decision_memo_2026-04-08.md) |
| What evidence supports the current diagnosis? | [parity_findings.md](parity_findings.md) |
| What should I run next? | [TODO.md](../../../TODO.md) |
| What is the current edge-specific plan? | [edge_parity_plan.md](edge_parity_plan.md) |
| Where should I audit MATLAB vs Python tracer semantics? | [MATLAB_PARITY_AUDIT_CHECKLIST.md](MATLAB_PARITY_AUDIT_CHECKLIST.md) |
| How do staged comparison run roots work? | [COMPARISON_LAYOUT.md](../../reference/COMPARISON_LAYOUT.md) |
| Where is the MATLAB-to-Python module map? | [MATLAB_MAPPING.md](../../reference/MATLAB_MAPPING.md) |
| What proved stage-isolated network parity? | [stage_isolated_network_parity_2026-04-07.md](../../../dev/reports/stage_isolated_network_parity_2026-04-07.md) |

## Historical Working Model

- Treat parity-mode `network` assembly as a standing downstream gate.
- Treat edge generation as the active upstream problem surface.
- Treat cleanup-path alignment with active MATLAB V200 as the current
  downstream parity prerequisite.
- Use candidate-endpoint coverage and shared-vertex diagnostics to localize the
  first divergence.
- Avoid broad global threshold sweeps unless a diagnostic points there
  specifically.

## High-Value Reports

- [stage_isolated_network_parity_2026-04-07.md](../../../dev/reports/stage_isolated_network_parity_2026-04-07.md)
  Proof that MATLAB `edges` plus Python `network` can already converge exactly.
- [python_matlab_parity_postfix_2026-03-30.md](../../../dev/reports/python_matlab_parity_postfix_2026-03-30.md)
  Earlier parity checkpoint with useful historical context.
- [python_standalone_consistency_postfix_2026-03-30.md](../../../dev/reports/python_standalone_consistency_postfix_2026-03-30.md)
  Repeatability context for Python-only reruns.

## Next Likely Move

If you are resuming implementation work rather than documentation work, use
[TODO.md](../../../TODO.md) first. The current highest-leverage target is still
`source/slavv/core/edge_candidates.py`, but only for the repeated frontier
parent/child invalidation pattern that now shows up across several missing
neighborhoods.

