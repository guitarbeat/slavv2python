# Reference Docs

Use this folder for current, cross-cutting technical references.

Recommended reading order:

1. [MATLAB Method Implementation Plan](core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
2. [MATLAB Parity Mapping](core/MATLAB_PARITY_MAPPING.md)
3. [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md)
4. [Glossary](core/GLOSSARY.md)
5. [Energy Computation Methods](core/ENERGY_METHODS.md)
6. [Adding Extraction Algorithms](workflow/ADDING_EXTRACTION_ALGORITHMS.md)
7. [Papers](papers/README.md)
8. Optional backend docs under `backends/`

This shelf documents the maintained Python implementation and the current
parity rules for exact imported-MATLAB work. Treat the released MATLAB source as
the executable specification, `prove-exact` as the proof gate, and the paper as
explanatory context. The maintained parity runner remains
`dev/scripts/cli/parity_experiment.py`, which supports both rerun summaries and
exact artifact proof on reusable staged runs.
