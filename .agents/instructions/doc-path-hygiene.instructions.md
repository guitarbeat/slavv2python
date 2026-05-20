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
| Energy stage | `slavv_python/processing/stages/energy/` | ~~`slavv_python/core/energy/`~~ |
| Vertex stage | `slavv_python/processing/stages/vertices/` | ~~`slavv_python/core/vertices/`~~ |
| Edge stage | `slavv_python/processing/stages/edges/` | ~~`slavv_python/core/edges/`~~ |
| Network stage | `slavv_python/processing/stages/network/` | ~~`slavv_python/core/network.py`~~ |
| MATLAB shims | `slavv_python/processing/stages/edges/matlab_algorithms/` | ~~`slavv_python/core/edges/matlab_algorithms/`~~ |
| Analytics | `slavv_python/analytics/` | ~~`slavv_python/analysis/`~~ |
| Parity harness | `slavv_python/analytics/parity/` | ~~`slavv_python/analysis/parity/`~~ |
| Storage (I/O) | `slavv_python/storage/` | ~~`slavv_python/io/`~~ |
| CLI | `slavv_python/interface/cli/` | ~~`slavv_python/apps/cli/`~~ |
| Streamlit app | `slavv_python/interface/streamlit/` | ~~`slavv_python/apps/streamlit/`~~ |
| Run state | `slavv_python/engine/state/` | ~~`slavv_python/runtime/`~~ |

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
- Never use the old `slavv_python/core/`, `slavv_python/apps/`, `slavv_python/analysis/`, `slavv_python/runtime/`, or `slavv_python/io/` paths in maintained documentation.
- Test directory names (`tests/unit/core/`, `tests/unit/apps/`, etc.) use a simplified owner convention and are NOT affected by this rule.
