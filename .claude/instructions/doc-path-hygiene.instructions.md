---
applyTo: "{docs,*.md,.agents}/**/*.md"
description: "Use when editing markdown documentation. Prevents path drift by enforcing that all slavv_python/* references match the actual package layout."
---
# Documentation Path Hygiene

## Background

This repository has undergone a major package reorganization. Documentation path drift has been a recurring maintenance burden. These rules prevent it from reoccurring.

## Canonical Package Layout

When referencing package paths in documentation, use these actual paths:

| Surface | Correct Path | Common Mistake |
|:--------|:-------------|:---------------|
| Pipeline engine | `slavv_python/engine/` | ~~`slavv_python/core/pipeline.py`~~ |
| Energy stage | `slavv_python/pipeline/energy/` | ~~`slavv_python/processing/stages/energy/`~~ |
| Vertex stage | `slavv_python/pipeline/vertices/` | ~~`slavv_python/processing/stages/vertices/`~~ |
| Edge stage | `slavv_python/pipeline/edges/` | ~~`slavv_python/processing/stages/edges/`~~ |
| Network stage | `slavv_python/pipeline/network/` | ~~`slavv_python/processing/stages/network/`~~ |
| MATLAB parity ports | `slavv_python/pipeline/{energy,edges}/matlab_*.py` | ~~`matlab_algorithms/`~~, ~~`global_watershed.py`~~ |
| Parity diagnostics | `slavv_python/pipeline/energy/parity_*.py`, `tests/support/parity_probe_*.py` | ~~`voxel_probe.py`~~ |
| Analytics | `slavv_python/analytics/` | ~~`slavv_python/analysis/`~~ |
| Parity harness | `slavv_python/analytics/parity/` | ~~`slavv_python/analysis/parity/`~~ |
| Storage (I/O) | `slavv_python/storage/` | ~~`slavv_python/io/`~~ |
| CLI | `slavv_python/interface/cli/` | ~~`slavv_python/apps/cli/`~~ |
| Streamlit app | `slavv_python/interface/streamlit/` | ~~`slavv_python/apps/streamlit/`~~ |
| Run state | `slavv_python/engine/state/` | ~~`slavv_python/runtime/`~~ |

## MATLAB Parity Filenames

When citing parity-sensitive modules, use the current `matlab_*` / `parity_*` names
(see [PYTHON_NAMING_GUIDE.md](../../docs/reference/workflow/PYTHON_NAMING_GUIDE.md#matlab-parity-filename-convention)):

| MATLAB | Python |
| --- | --- |
| `get_energy_V202.m` | `matlab_get_energy_v202_chunked.py` |
| `energy_filter_V200.m` | `matlab_energy_filter_v200.py` |
| `get_edges_by_watershed.m` | `matlab_get_edges_by_watershed.py` |
| `get_edges_V300.m` | `matlab_get_edges_v300_frontier.py`, `matlab_get_edges_v300_geometry.py` |
| `calculate_linear_strel_range.m` | `matlab_calculate_linear_strel_range.py` |

## Python Import Paths

When writing import examples in docs:
```python
# Correct — uses top-level re-exports
from slavv_python import SlavvPipeline, load_tiff_volume

# Correct — uses actual module path
from slavv_python.engine import SlavvPipeline

# WRONG — these paths do not exist
from slavv_python.core import SlavvPipeline  # ← NO
from slavv_python.io import load_tiff_volume  # ← NO
```

## pyproject.toml Entrypoints

The correct entrypoint paths are:
```toml
slavv = "slavv_python.interface.cli:main"
slavv-app = "slavv_python.interface.streamlit_launcher:main"
```

## Rules
- Before adding a `slavv_python/*` path reference in any markdown file, verify the path exists.
- Never use the old `slavv_python/core/`, `slavv_python/apps/`, `slavv_python/analysis/`, `slavv_python/runtime/`, `slavv_python/io/`, or `slavv_python/processing/stages/` paths in maintained documentation.
- Pipeline tests live under `tests/unit/pipeline/` (owner-aligned), not `tests/unit/core/`.
- When in doubt, grep the live tree and [MATLAB_PARITY_MAPPING.md](../../docs/reference/core/MATLAB_PARITY_MAPPING.md) before adding a module path.