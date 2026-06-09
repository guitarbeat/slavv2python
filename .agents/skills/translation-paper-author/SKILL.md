---
name: translation-paper-author
description: Synthesize lessons learned during the MATLAB-to-Python parity translation to author a comprehensive technical paper or internal engineering guide. Use when the user asks to extract lessons, write sections of the translation paper, or document the exact parity journey.
---

# Translation Paper Author

Author and expand the definitive guide on porting an untyped, implicitly-broadcasted MATLAB scientific pipeline to a strict, typed, production-grade Python application.

## Core Directives

- Append, refine, and structure the paper located at `docs/investigations/MATLAB_PYTHON_TRANSLATION_PAPER.md`.
- Group lessons by architectural themes rather than chronological sequence.
- Include concrete examples from the `slavv2python` codebase (e.g., how MATLAB's `find(..., 'last')` mapping impacted parity, the switch from `np.isclose` to exact bitwise equality for precision, or the differences between MATLAB's `linspace` and Python's `np.arange`).
- Always cite verified fixes and completed Parity Runs as the soil for the paper.

## Paper Structure

Ensure the paper contains the following structured sections:
1. **Introduction & Motivation:** The necessity of moving from a research script to a production pipeline.
2. **Phase 1: The Exact Parity Method:** The importance of strict 1:1 structural mapping and bitwise equality to establish trust.
3. **Floating Point & Numerical Nuances:** Examples like `float32` vs `float64` collapses, `NaN` propagation, and mesh generation roundoff differences.
4. **Architectural Translation:** Moving from global MATLAB workspace scope to encapsulated Python managers and Run State ledgers.
5. **Phase 2 Ideation:** How this exact-parity soil allows us to brainstorm future performance and parallelization improvements safely.

## Workflow

1. Read the current draft of `docs/investigations/MATLAB_PYTHON_TRANSLATION_PAPER.md`.
2. Review the latest entries in `docs/reference/core/EXACT_PROOF_FINDINGS.md` and `docs/solutions/`.
3. Extract any new, verified numerical or architectural lessons.
4. Add the lessons to the appropriate section of the paper, formulating them as engineering best practices for scientific translation.
5. Provide a brief summary of what was added.
