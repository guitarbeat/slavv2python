# Reference Docs

Use this folder for current, maintained technical references.
These docs outrank archival chapters and paper prose when they describe the
current Python product surface.

## Core Docs

Read these first when working on the live implementation:

- [MATLAB Method Implementation Plan](core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
  Claim boundaries, source-of-truth hierarchy, and remaining roadmap.
- [MATLAB Parity Mapping](core/MATLAB_PARITY_MAPPING.md)
  MATLAB-to-Python surface map and confirmed structural deviations.
- [Global Watershed Implementation Notes](core/WATERSHED_IMPLEMENTATION_NOTES.md)
  Technical design, shared state management, and parity details for the discovery algorithm.
- [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md)
  Live proof status, current v22 watershed read, and breakthrough status.
- [Energy Computation Methods](core/ENERGY_METHODS.md)
  Supported energy backends, projection modes, and extension points.
- [Glossary](core/GLOSSARY.md)
  Shared parity and pipeline terminology.
- [Paper Profile](workflow/PAPER_PROFILE.md)
  Public paper-first CLI/app workflow and authoritative JSON export contract.
- [Python Naming Guide](workflow/PYTHON_NAMING_GUIDE.md)
  Preferred Python names, package groupings, and compatibility policy.

## Workflow Guides

- [Parity Implementation Guide](workflow/PARITY_IMPLEMENTATION_GUIDE.md)
  Technical instructions and best practices for MATLAB parity alignment.
- [Performance Benchmarking Guide](workflow/PERFORMANCE_BENCHMARKING_GUIDE.md)
- [Adding Extraction Algorithms](workflow/ADDING_EXTRACTION_ALGORITHMS.md)
- [Parity Experiment Storage](workflow/PARITY_EXPERIMENT_STORAGE.md)

## Adjacent Reference Docs

These stay separate because they describe distinct maintained surfaces rather
than alternate versions of the same content:

- [Zarr Energy Storage](backends/ZARR_ENERGY_STORAGE.md)
- [Napari Curator](backends/NAPARI_CURATOR.md)
- [Papers](papers/README.md)

The `papers/` folder is background reading, not the executable source of truth
for current Python behavior or exact MATLAB parity claims.

## Live Status And Historical Context

- [Exact Proof Findings](core/EXACT_PROOF_FINDINGS.md) is the maintained owner
  for live v22 and downstream proof status.
- [v22 Pointer Corruption Archive](../chapters/v22-pointer-corruption/README.md)
  preserves the April 2026 investigation trail and archived Kiro planning.
