# 3D Visualization and Export

This guide explains how to view SLAVV vascular network exports in Blender/VessMorphoVis.

## Overview

SLAVV exports multiple formats:

- **VMV (`.vmv`)**: For VessMorphoVis/Blender rendering.
- **CASX (`.casx`)**: XML-based vascular network format.
- **CSV**: Vertex/edge tabular output.
- **JSON**: Full hierarchical run output.

## Viewing in Blender (VessMorphoVis)

### Prerequisites

1. [Blender](https://www.blender.org/download/)
2. [VessMorphoVis plugin](https://github.com/BlueBrain/VessMorphoVis)

### Setup

1. Install Blender.
2. Download VessMorphoVis ZIP.
3. In Blender, open **Edit > Preferences > Add-ons**.
4. Click **Install...** and select the ZIP.
5. Enable the plugin.

### Load a Network

1. Run the SLAVV pipeline (for example via `workspace/scripts/cli/compare_matlab_python.py`).
2. Locate a generated VMV file (for example `comparisons/.../python_results/network.vmv`).
3. In Blender, open the VessMorphoVis panel and click **Load Morphology**.
4. Select your `network.vmv` file.

## Comparison Manifests

Each comparison run includes a `MANIFEST.md` file with:

- File inventory (paths and sizes)
- Summary statistics
- Run-specific viewing instructions

For interactive inspection, use `workspace/notebooks/04_Comparison_Dashboard.ipynb`.

## Export Formats

### VMV

ASCII text with parameters, vertices (`x, y, z, radius`), and strands (vertex index sequences).

### CASX

XML structured as `<Network><Vertices>...</Vertices><Edges>...</Edges></Network>`.

### CSV

- `*_vertices.csv`: `y_position`, `x_position`, `z_position`, `radius_microns`
- `*_edges.csv`: `start_vertex`, `end_vertex`
