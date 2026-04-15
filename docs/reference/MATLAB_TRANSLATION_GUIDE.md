# MATLAB Translation Guide

This document is the maintained reference for MATLAB-to-Python translation
semantics in `slavv2python`. Use it when changing parity-sensitive code,
reviewing translated logic, or deciding whether a difference is an acceptable
Python adaptation or a parity regression.

## What This Guide Covers

- semantic traps where MATLAB and NumPy/Python differ in ways that matter here
- project conventions for array shapes, indexing, transpose behavior, and
  network export formats
- intentional divergences and hand-maintained override surfaces
- a short checklist for parity-sensitive edits

For file-to-file correspondence, use
[MATLAB_MAPPING.md](./MATLAB_MAPPING.md). For run layout and staged outputs, use
[COMPARISON_LAYOUT.md](./COMPARISON_LAYOUT.md). For the active parity work, use
[the shared neighborhood claim alignment chapter](../chapters/shared-neighborhood-claim-alignment/README.md).

## Semantic Traps That Matter In This Repo

### Indexing

- MATLAB is 1-based; Python is 0-based.
- File formats or interoperability layers may still require 1-based numbering
  even when the in-memory Python representation is 0-based.
- Be explicit at boundaries: candidate IDs, edge ownership maps, VMV/CASX
  indices, and imported MATLAB metadata should document whether they are
  MATLAB-facing or Python-facing.

### Array Shapes And Axis Semantics

- MATLAB code often treats vectors as explicit row or column matrices, while
  NumPy has a separate 1-D array concept.
- In this repo, translated geometric calculations should prefer explicit array
  shapes over relying on implicit transpose behavior of 1-D arrays.
- When porting a MATLAB expression that depends on orientation, normalize the
  Python shape first and only then apply downstream math.

### Matrix Algebra vs Elementwise Math

- MATLAB's `*` is matrix multiplication and `.*` is elementwise multiplication.
- In Python, `@` is matrix multiplication and `*` is elementwise multiplication.
- Review translated formulas with this difference in mind, especially in
  geometry, registration, and energy-processing code.

### Transpose Behavior

- MATLAB `'` is conjugate transpose and `.'` is non-conjugating transpose.
- NumPy `.T` is a shape transpose only; it does not apply complex conjugation.
- If translated code is conceptually porting MATLAB `'`, use explicit
  conjugation where needed rather than assuming `.T` is equivalent.

### Copy vs View Semantics

- MATLAB indexing usually returns copied arrays. NumPy slicing often returns
  views into the same memory.
- When translated code mutates cropped or sliced arrays, verify whether the
  mutation is intentionally shared with the original data.
- For parity-sensitive cleanup and ownership bookkeeping, use explicit copies
  when the MATLAB behavior assumes isolation.

### Broadcasting And Logical Selection

- MATLAB's implicit expansion and logical indexing rules do not always line up
  with NumPy broadcasting.
- Prefer shape checks near translated boolean masks and per-origin filtering so
  accidental broadcasting does not silently change candidate counts.

## Project Conventions

### Shapes And Coordinates

- Treat translated geometry and tracing code as parity-oriented logic first and
  generic NumPy code second.
- Preserve explicit coordinate ordering in code and tests; do not rely on a
  transpose or reshape trick if an equivalent explicit expression is clearer.
- When a helper accepts traces, vertex positions, or connection arrays, keep
  the documented structure stable and cover any reshaping in tests.

### Indexing Boundaries

- Internal Python collections are 0-based unless a file format or imported
  MATLAB artifact explicitly requires otherwise.
- Exporters and interoperability layers may emit 1-based identifiers for MATLAB
  compatibility. Keep that conversion at the boundary instead of leaking it into
  core data structures.

### Export Formats

- `network.json` is the portable Python-facing export used by package tools and
  analysis commands.
- `vmv` and `casx` are interoperability exports and may preserve MATLAB-facing
  conventions such as 1-based indexing or format-specific metadata layout.
- Comparison runs should record authoritative orchestration state under
  `99_Metadata/`, not infer it later from ad-hoc output files.

## Intentional Divergence And Manual Override Registry

Use this section to track maintained behavior that is intentionally not a
literal line-by-line translation.

| Surface | Reason | Current policy |
| --- | --- | --- |
| `source/slavv/apps/parity_cli.py` | Python packages the MATLAB batch workflow as a CLI instead of a MATLAB script entrypoint | Preserve behavior and staged outputs; do not chase file-level symmetry |
| `source/slavv/parity/*` | Comparison orchestration and reporting are Python-native control surfaces | Keep the run contract stable and document semantics in reference docs |
| `source/slavv/visualization/interactive_curator.py` | Interactive curation is implemented in Python-native UI code | Match workflow intent rather than MATLAB UI implementation details |
| `source/slavv/io/network_io.py` and exporters | File format emission is boundary code, not direct MATLAB control flow | Prefer deterministic Python writers with explicit compatibility tests |

When adding a new intentional divergence, record:

1. the maintained Python surface
2. why literal MATLAB structure is not the goal there
3. what compatibility promise still matters

## Parity-Sensitive Edit Checklist

Before merging a change in translated or parity-facing code:

1. Confirm whether the code is core translated logic, a Python-native wrapper,
   or an interoperability boundary.
2. Check indexing assumptions at every MATLAB/Python boundary.
3. Check whether array orientation or transpose behavior changed during the
   edit.
4. Check whether slices or masks now mutate shared NumPy views.
5. Update or add the narrowest regression tests that prove the intended
   behavior.
6. If the change affects parity workflows, rerun the relevant parity or
   diagnostic tests and record any skipped live-MATLAB verification.

## External References

- [NumPy for MATLAB users](https://numpy.org/doc/2.0/user/numpy-for-matlab-users.html)
- [SMOP](https://github.com/victorlei/smop)
- [Motopy](https://github.com/falwat/motopy)
