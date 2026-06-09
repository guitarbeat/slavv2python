# Documentation

The documentation tree has several explicit owners:

- `reference/` for maintained technical guidance and live parity status
- `TODO.md` for active task checkboxes
- `plans/` for active specs
- `adr/` for architecture decisions
- `investigations/` for intentionally archival narratives that still help explain the
  current Python codebase

Treat `investigations/` as historical context, not as an executable spec.

## Start Here

1. [Repository README](../README.md)
2. [Developer dashboard (tasks & planning hub)](TODO.md) — Active checkboxes, links to plans, brainstorms, and compound solutions
3. [Tutorial](TUTORIAL.md) — Get started with your first extraction.
4. [Agent and workflow guide](AGENTS.md)
5. [Contributing](CONTRIBUTING.md)
6. [Changelog](CHANGELOG.md)
7. [Reference index](reference/README.md)
8. [Investigation index](investigations/README.md)
9. [Test placement guide](../tests/README.md)

## Parity Closure Fast Path

When continuing exact MATLAB parity work:

1. [Exact Proof Findings](reference/core/EXACT_PROOF_FINDINGS.md) — live run truth, blockers, and accepted findings.
2. [Phase 1 exact-route spec](plans/phase-1-exact-route-spec.md) — certification intent and pass/fail loop.
3. [Parity Pre-Gate](reference/workflow/PARITY_PRE_GATE.md) — crop harness commands.
4. [Parity Certification Guide](reference/workflow/PARITY_CERTIFICATION_GUIDE.md) — canonical promotion and proof commands.
5. [Parity Job Monitoring](reference/workflow/PARITY_JOB_MONITORING.md) — automated tracking for long-running experiments.

## Planning, Specs, And Knowledge

| Folder | Purpose |
|--------|---------|
| [TODO.md](TODO.md) | **Hub** — what to do next; index of plans and solutions |
| [plans/](plans/) | Active specs (`*-spec.md`: requirements + implementation in one file) |
| [brainstorms/](brainstorms/) | Pre-spec ideas only; promote durable context into `plans/`, then remove the brainstorm |
| [solutions/](solutions/) | Documented fixes and runbooks (`/ce-compound`; YAML frontmatter for search) |
| [adr/](adr/) | Architecture decision records |

## Core Maintained References

- [MATLAB Method Implementation Plan](reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- [MATLAB Parity Mapping](reference/core/MATLAB_PARITY_MAPPING.md)
- [Exact Proof Findings](reference/core/EXACT_PROOF_FINDINGS.md)
- [Energy Computation Methods](reference/core/ENERGY_METHODS.md)
- [Paper Profile](reference/workflow/PAPER_PROFILE.md)
- [Python Naming Guide](reference/workflow/PYTHON_NAMING_GUIDE.md)

## Archive Entry Points

- [v22 Pointer Corruption Archive](investigations/v22-pointer-corruption/README.md)
- [Investigation index](investigations/README.md)
