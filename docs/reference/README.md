# Reference Docs

Use this folder for current, maintained technical references.
These docs outrank archival chapters and paper prose when they describe the
current Python product surface.

## Core Docs

Read these first when working on the live implementation:

- [MATLAB Method Implementation Plan](core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
  Claim boundaries, source-of-truth hierarchy, and implementation phases.
- [MATLAB Parity Mapping](core/MATLAB_PARITY_MAPPING.md)
  MATLAB-to-Python surface map and confirmed structural deviations.
- [Global Watershed Implementation Notes](core/WATERSHED_IMPLEMENTATION_NOTES.md)
  Technical design, shared state management, and parity details for the discovery algorithm.
- [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md)
  Live proof status, current parity blockers, and historical breakthrough context.
- [Energy Computation Methods](core/ENERGY_METHODS.md)
  Supported energy backends, projection modes, and extension points.
- [Glossary](core/GLOSSARY.md)
  Shared parity and pipeline terminology.
- [Paper Profile](workflow/PAPER_PROFILE.md)
  Public paper-first CLI/app workflow and authoritative JSON export contract.
- [Python Naming Guide](workflow/PYTHON_NAMING_GUIDE.md)
  Preferred Python names, package groupings, and compatibility policy.
- [Technical Architecture](core/TECHNICAL_ARCHITECTURE.md)
  Engine design, component overview, and processing workflow.

- [Parity Certification Guide](workflow/PARITY_CERTIFICATION_GUIDE.md)
  Step-by-step instructions for running mathematical proofs against MATLAB oracles.
- [Performance Benchmarking Guide](workflow/PERFORMANCE_BENCHMARKING_GUIDE.md)
  Methodology and tools for measuring and optimizing processing speed.
- [Production Release Guide](workflow/PRODUCTION_RELEASE_GUIDE.md)
  Mandatory steps for stable research deployments and parity promotion.
- [Adding Extraction Algorithms](workflow/ADDING_EXTRACTION_ALGORITHMS.md)
- [Proactive Agent Guide](workflow/external/PROACTIVE_AGENT_GUIDE.md)
  Architecture and protocols for proactive AI agent behaviors.

## Adjacent Reference Docs

These stay separate because they describe distinct maintained surfaces rather
than alternate versions of the same content:

- [Zarr Energy Storage](backends/ZARR_ENERGY_STORAGE.md)
- [Napari Curator](backends/NAPARI_CURATOR.md)
- [Papers](papers/README.md)

The `papers/` folder is background reading, not the executable slavv_python of truth
for current Python behavior or exact MATLAB parity claims.

## Live Status And Historical Context

- [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md) is the maintained owner
  for live exact-parity proof status.
- For exact MATLAB parity closure, read in this order:
  [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md),
  [Parity Pre-Gate](workflow/PARITY_PRE_GATE.md),
  [Parity Certification Guide](workflow/PARITY_CERTIFICATION_GUIDE.md).
- [v22 Pointer Corruption Archive](../investigations/v22-pointer-corruption/README.md)
  preserves the April 2026 investigation trail and archived Kiro planning.
