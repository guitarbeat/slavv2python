# Reference Docs

[Up: Documentation Index](../README.md) · [AI Agent Guide](../../AGENTS.md) · [Glossary](core/GLOSSARY.md) · [Quick Reference](../QUICK_REFERENCE.md)

Use this folder for current, maintained technical references. These docs outrank archival chapters and paper prose when they describe the current Python product surface.

**In short:** Phase 1 already shipped (close enough to MATLAB). Live pass/fail is [ONE TRUTH](core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk). Identical last digits is a separate leftover — [plain-English diagnosis](../solutions/parity/crop-energy-stretch-float-isolation.md).

**⭐ Start Here:**
- **Brand new to the repo?** [NEW_ENGINEER_START_HERE.md](core/NEW_ENGINEER_START_HERE.md) — pick Paper Path vs Exact Route before diving in (“paper” is also the 2021 publication)
- **New to parity work?** [EXACT_PROOF_FINDINGS.md](core/EXACT_PROOF_FINDINGS.md) → [PARITY_PRE_GATE.md](workflow/PARITY_PRE_GATE.md)
- **New to repository code?** [TECHNICAL_ARCHITECTURE.md](core/TECHNICAL_ARCHITECTURE.md) → [GLOSSARY.md](core/GLOSSARY.md)
- **Contributing code?** [PYTHON_NAMING_GUIDE.md](workflow/PYTHON_NAMING_GUIDE.md) → [tests/README.md](../../tests/README.md)
- **Need quick answers?** [QUICK_REFERENCE.md](../QUICK_REFERENCE.md) ⚡

---

## Core Docs

Read these first when working on the live implementation:

| Document | Purpose | Related Docs |
|----------|---------|--------------|
| [MATLAB Method Implementation Plan](core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) | Claim boundaries, source-of-truth hierarchy, and implementation phases | [MATLAB_PARITY_MAPPING.md](core/MATLAB_PARITY_MAPPING.md) |
| [MATLAB Parity Mapping](core/MATLAB_PARITY_MAPPING.md) | MATLAB-to-Python surface map and confirmed structural deviations | [MATLAB_METHOD_IMPLEMENTATION_PLAN.md](core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) |
| [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md) ⭐ | Live pass/fail (Phase 1 closed = ship, not 100%) plus stretch leftover | [PARITY_PRE_GATE.md](workflow/PARITY_PRE_GATE.md), [PARITY_CERTIFICATION_GUIDE.md](workflow/PARITY_CERTIFICATION_GUIDE.md), [phase1-baseline-freeze.json](core/phase1-baseline-freeze.json), [phase2-profiling-baseline.json](core/phase2-profiling-baseline.json) |
| [Global Watershed Implementation Notes](core/WATERSHED_IMPLEMENTATION_NOTES.md) | Technical design, shared state management, and parity details for the discovery algorithm | [MATLAB_PARITY_MAPPING.md](core/MATLAB_PARITY_MAPPING.md) |
| [Energy Computation Methods](core/ENERGY_METHODS.md) | Supported energy backends, projection modes, and extension points | [ZARR_ENERGY_STORAGE.md](backends/ZARR_ENERGY_STORAGE.md) |
| [Glossary](core/GLOSSARY.md) | Shared parity and pipeline terminology | [AGENTS.md § Domain Glossary](../../AGENTS.md#domain-glossary) |
| [Technical Architecture](core/TECHNICAL_ARCHITECTURE.md) | Engine design, component overview, and processing workflow | [PYTHON_NAMING_GUIDE.md](workflow/PYTHON_NAMING_GUIDE.md) |
| [New Engineer Start Here](core/NEW_ENGINEER_START_HERE.md) | Paper Path vs Exact Route (and vs the 2021 publication), first-week paths, traps | [TUTORIAL.md](../TUTORIAL.md), [papers/README.md](papers/README.md), [ONE TRUTH](core/EXACT_PROOF_FINDINGS.md) |
| [SLAVV Method Explained](core/SLAVV_METHOD_EXPLAINED.md) | Walkthrough of the 2021 publication (narrative; not Paper Path) | [papers/README.md](papers/README.md), [Research paper review](../research/slavv-original-paper-review.md) |
| [Parity Methodology](core/PARITY_METHODOLOGY.md) | Why the cert bars are tolerance-based (literature); 5-row ambiguity index | [papers/README.md](papers/README.md), [ADR 0011](../adr/0011-energy-float-certification-policy.md), [ADR 0012](../adr/0012-edge-watershed-parity-bar.md) |

## Workflow Docs

Operator guides and contributor references:

| Document | Purpose | Related Docs |
|----------|---------|--------------|
| [Parity Pre-Gate](workflow/PARITY_PRE_GATE.md) | Three-tier parity testing (synthetic → crop → canonical) | [EXACT_PROOF_FINDINGS.md](core/EXACT_PROOF_FINDINGS.md), [PARITY_CERTIFICATION_GUIDE.md](workflow/PARITY_CERTIFICATION_GUIDE.md) |
| [Random Component Parity Suite](workflow/PARITY_RANDOM_COMPONENT_SUITE.md) | Fast MATLAB/Python differential on seeded noise (linspace, interp3, Energy structure) | [ADR 0010](../adr/0010-random-component-parity-suite.md), [PARITY_PRE_GATE.md](workflow/PARITY_PRE_GATE.md) |
| [Parity Certification Guide](workflow/PARITY_CERTIFICATION_GUIDE.md) | Step-by-step instructions for running mathematical proofs against MATLAB oracles | [PARITY_PRE_GATE.md](workflow/PARITY_PRE_GATE.md), [PARITY_JOB_MONITORING.md](workflow/PARITY_JOB_MONITORING.md) |
| [Parity Job Monitoring](workflow/PARITY_JOB_MONITORING.md) | Automated tracking and notifications for long-running parity experiments | [PARITY_CERTIFICATION_GUIDE.md](workflow/PARITY_CERTIFICATION_GUIDE.md) |
| [Parity Run Evidence](workflow/PARITY_RUN_EVIDENCE.md) | Copy-paste template after writers and `prove-exact` attempts | [EXACT_PROOF_FINDINGS.md](core/EXACT_PROOF_FINDINGS.md), [PARITY_CERTIFICATION_GUIDE.md](workflow/PARITY_CERTIFICATION_GUIDE.md) |
| [Experiment Analysis Template](workflow/EXPERIMENT_ANALYSIS_TEMPLATE.md) | Reusable structure for hypothesis-driven experiment docs without duplicating live status | [Phase 1 Residual Experiment Analysis](workflow/PHASE1_RESIDUAL_EXPERIMENT_ANALYSIS.md) |
| [Phase 1 Residual Experiment Analysis](workflow/PHASE1_RESIDUAL_EXPERIMENT_ANALYSIS.md) | Historical framing of the closed Network leftover (not live status) | [EXACT_PROOF_FINDINGS.md](core/EXACT_PROOF_FINDINGS.md), [HANDOFF](../../.claude/HANDOFF.md), [figures](../../figures/README.md) |
| [Paper Profile](workflow/PAPER_PROFILE.md) | Public paper-first CLI/app workflow and authoritative JSON export contract | [PYTHON_NAMING_GUIDE.md](workflow/PYTHON_NAMING_GUIDE.md) |
| [Python Naming Guide](workflow/PYTHON_NAMING_GUIDE.md) | Preferred Python names, package groupings, and compatibility policy | [TECHNICAL_ARCHITECTURE.md](core/TECHNICAL_ARCHITECTURE.md) |
| [Performance Benchmarking Guide](workflow/PERFORMANCE_BENCHMARKING_GUIDE.md) | Methodology and tools for measuring and optimizing processing speed | — |
| [Production Release Guide](workflow/PRODUCTION_RELEASE_GUIDE.md) | Mandatory steps for stable research deployments and parity promotion | — |
| [Adding Extraction Algorithms](workflow/ADDING_EXTRACTION_ALGORITHMS.md) | Contributor guide for new algorithms | [PYTHON_NAMING_GUIDE.md](workflow/PYTHON_NAMING_GUIDE.md) |

## Adjacent Reference Docs

These stay separate because they describe distinct maintained surfaces rather
than alternate versions of the same content:

- [Zarr Energy Storage](backends/ZARR_ENERGY_STORAGE.md)
- [Napari Curator](backends/NAPARI_CURATOR.md)
- [Papers](papers/README.md) — annotated index of papers **and** “no paper / vendor docs” notes, grouped by the five common confusions (publication vs Paper Path, float bars, watershed metric, Fortran ties)

The `papers/` folder is that index, not the source of truth
for current Python behavior or exact MATLAB parity claims.

## Proposal / methods figures

- [figures/](../../figures/README.md) — all publication figures (claim charts +
  [research drafts](../../figures/research/)).

## Live Status And Historical Context

- [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md) is the maintained owner
  for live exact-parity proof status.
- For exact MATLAB parity closure, read in this order:
  [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md),
  [Parity Pre-Gate](workflow/PARITY_PRE_GATE.md),
  [Parity Certification Guide](workflow/PARITY_CERTIFICATION_GUIDE.md),
  [Parity Job Monitoring](workflow/PARITY_JOB_MONITORING.md).
- [v22 Pointer Corruption Archive](../investigations/v22-pointer-corruption/README.md)
  preserves the April 2026 investigation trail and archived Kiro planning.
