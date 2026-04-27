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

## Active Bug Investigations

**v22 Global Watershed Pointer Corruption** (April 27, 2026):
- [V22 Summary](core/V22_SUMMARY.md) - Quick overview and fix instructions
- [V22 Investigation Handover](core/V22_INVESTIGATION_HANDOVER.md) - Complete investigation details
- [V22 Bug Fixes](core/V22_BUG_FIXES.md) - Detailed bug tracking and test results
- [V22 Blocking Bugs](core/V22_BLOCKING_BUGS.md) - Original bug report
- [V22 Pointer Corruption Investigation](core/V22_POINTER_CORRUPTION_INVESTIGATION.md) - Phase-by-phase log

**Status**: Root cause identified (scale clipping mismatch), ready for fix implementation.

---

This shelf documents the maintained Python implementation and the current
native-first exact-route rules. Treat the released MATLAB source as the
executable specification, preserved MATLAB vectors as the oracle artifacts,
`prove-exact` as the proof gate, and the paper as explanatory context.

The maintained parity runner remains `dev/scripts/cli/parity_experiment.py`.
Its canonical exact route now accepts native Hessian energy provenance
(`python_native_hessian`) while still allowing preserved MATLAB energy
provenance (`matlab_batch_hdf5`) for historical replay and regression checks.
