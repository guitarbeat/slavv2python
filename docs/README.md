# Documentation

The documentation tree is intentionally small:

- `reference/` for maintained technical references
- `chapters/` for lightweight historical notes when they are still useful

Start here:

1. [Repository README](../README.md)
2. [Agent and workflow guide](../AGENTS.md)
3. [Reference index](reference/README.md)
4. [Test placement guide](../dev/tests/README.md)

The old rich MATLAB/parity investigation harness has been retired.
The maintained developer replacement is `dev/scripts/cli/parity_experiment.py`,
which supports both rerunning the live Python pipeline against reusable staged
comparison roots and exact artifact proof against preserved MATLAB vectors.
The canonical implementation-status and claim-boundary doc for this work now
lives at `docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md`.
The maintained source map for exact imported-MATLAB parity work now lives at
`docs/reference/core/MATLAB_PARITY_MAPPING.md`.
Current exact-proof findings live at
`docs/reference/core/EXACT_PROOF_FINDINGS.md`.
