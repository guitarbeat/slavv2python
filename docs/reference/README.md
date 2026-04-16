# Reference Docs

[Up: Documentation Index](../README.md)

This folder collects the maintained cross-cutting reference material for the
SLAVV Python port. Use this index when you want the stable docs organized by
topic instead of reading the files one by one.

## Reading Order

If you are new to the reference shelf, start here:

1. [Glossary](GLOSSARY.md)
2. [MATLAB Translation Guide](MATLAB_TRANSLATION_GUIDE.md)
3. [MATLAB Mapping](MATLAB_MAPPING.md)
4. [Energy Computation Methods](ENERGY_METHODS.md)
5. [Comparison Run Layout](COMPARISON_LAYOUT.md)

## Topic Indexes

- [Core Parity And Translation](core/README.md)
- [Optional Backends And Runtime Surfaces](backends/README.md)
- [Contributing And Context](workflow/README.md)

## Direct Files

- [Glossary](GLOSSARY.md) - project terms and workflow vocabulary
- [MATLAB Translation Guide](MATLAB_TRANSLATION_GUIDE.md) - semantics that matter when editing parity-sensitive code
- [MATLAB Mapping](MATLAB_MAPPING.md) - maintained MATLAB-to-Python correspondence
- [Energy Computation Methods](ENERGY_METHODS.md) - supported `energy_method` options and extension points
- [Comparison Run Layout](COMPARISON_LAYOUT.md) - canonical staged comparison-run contract
- [SimpleITK Energy Backend](SIMPLEITK_ENERGY_BACKEND.md) - spacing-aware vesselness backend
- [CuPy Energy Backend](CUPY_ENERGY_BACKEND.md) - GPU-accelerated energy backend
- [Zarr Energy Storage](ZARR_ENERGY_STORAGE.md) - resumable chunked storage for large energy artifacts
- [napari Curator Prototype](NAPARI_CURATOR.md) - experimental napari-based curation surface
- [Adding Extraction Algorithms](ADDING_EXTRACTION_ALGORITHMS.md) - checklist for wiring in new extraction modes
- [External Library Survey](EXTERNAL_LIBRARY_SURVEY_2026-04-06.md) - short status note on external packages already adopted or still open

## Notes

- These documents are intentionally maintained and should stay small, durable,
  and cross-cutting.
- Chapter-specific investigation notes live under `docs/chapters/` instead of
  here.