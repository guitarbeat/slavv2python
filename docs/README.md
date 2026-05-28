# Documentation

The documentation tree has two jobs:

- `reference/` for maintained technical guidance
- `investigations/` for intentionally archival narratives that still help explain the
  current Python codebase

Only `reference/` is a maintained slavv_python of truth for current behavior, public
workflow, and parity status. Treat `investigations/` as historical context, not as an
executable spec.

## Start Here

1. [Repository README](../README.md)
2. [Developer dashboard (tasks & planning hub)](TODO.md) — Active checkboxes, links to plans, brainstorms, and compound solutions
3. [Tutorial](TUTORIAL.md) — Get started with your first extraction.
4. [Roadmap](ROADMAP.md) — Project status and milestones (less tactical than TODO.md)
5. [Agent and workflow guide](AGENTS.md)
6. [Contributing](CONTRIBUTING.md)
7. [Changelog](CHANGELOG.md)
8. [Reference index](reference/README.md)
9. [Investigation index](investigations/README.md)
10. [Test placement guide](../tests/README.md)

## Planning & knowledge (not reference specs)

| Folder | Purpose |
|--------|---------|
| [TODO.md](TODO.md) | **Hub** — what to do next; index of plans and solutions |
| [plans/](plans/) | Active specs (`*-spec.md`: requirements + implementation in one file) |
| [brainstorms/](brainstorms/) | Pre-spec ideas only; promote into `plans/` then stub the brainstorm |
| [solutions/](solutions/) | Documented fixes and runbooks (`/ce-compound`; YAML frontmatter for search) |
| [adr/](adr/) | Architecture decision records |

## Core Maintained References

- [MATLAB Method Implementation Plan](reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- [MATLAB Parity Mapping](reference/core/MATLAB_PARITY_MAPPING.md)
- [Exact Proof Findings](reference/core/EXACT_PROOF_FINDINGS.md)
- [Energy Computation Methods](reference/core/ENERGY_METHODS.md)
- [Paper Profile](reference/workflow/PAPER_PROFILE.md)
- [Python Naming Guide](reference/workflow/PYTHON_NAMING_GUIDE.md)
- [Parity Experiment Storage](reference/workflow/PARITY_EXPERIMENT_STORAGE.md)

## Archive Entry Points

- [v22 Pointer Corruption Archive](investigations/v22-pointer-corruption/README.md)
- [Translation Pair Analysis Archive](investigations/translation_pair_analysis/README.md)
