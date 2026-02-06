# 3D Visualization and Export

This guide explains how to view vascular networks exported by SLAVV in 3D visualization tools like [Blender](https://www.blender.org) via [VessMorphoVis](https://github.com/BlueBrain/VessMorphoVis).

## Overview

SLAVV automatically exports networks in multiple formats for interoperability:
*   **VMV (`.vmv`)**: For VessMorphoVis/Blender (3D visualization, rendering).
*   **CASX (`.casx`)**: XML-based vascular network format (Linninger Lab).
*   **CSV**: Vertex and edge data for pandas/R dataframes.
*   **JSON**: Complete hierarchical results.

## Viewing in 3D (Blender/VessMorphoVis)

### Prerequisites
1.  **[Blender](https://www.blender.org/download/)** (Free, Open Source 3D Suite)
2.  **[VessMorphoVis Plugin](https://github.com/BlueBrain/VessMorphoVis)**

### Setup
1.  Install Blender.
2.  Download the VessMorphoVis plugin (ZIP file) from GitHub.
3.  Open Blender, go to **Edit > Preferences > Add-ons**.
4.  Click **Install...** and select the VessMorphoVis ZIP.
5.  Check the box to **Enable** the plugin.

### How to View a Network
1.  Run the SLAVV pipeline (e.g., via `compare_matlab_python.py`).
2.  Locate the `.vmv` file in the results directory (e.g., `comparisons/YYYYMMDD.../python_results/network.vmv`).
3.   - Copy the folder `external/blender_resources/VessMorphoVis/vessmorphovis` to the Blender addons folder.bar, press `N` to toggle).
4.  Click **Load Morphology**.
5.  Select your `network.vmv` file.
6.  The network will load as a 3D object. You can now:
    *   Change visualization mode (Radius, Strand ID).
    *   Adjust material settings.
    *   render high-resolution images or animations.

## Comparison Manifests

Each comparison run generates a `MANIFEST.md` file in its output directory. This file contains:
*   **File Inventory**: List of all generated files with paths and sizes.
*   **Summary Stats**: Quick look at vertices, edges, and performance.
*   **Instructions**: Copy-pasteable commands for viewing that specific run's results.

To list all available comparison runs:
```bash
python scripts/list_comparisons.py
```

## Export Formats

### VMV Format
ASCII text file with sections for parameters, vertices (x, y, z, radius), and strands (sequences of indices).

### CASX Format
XML file structured as `<Network><Vertices>...</Vertices><Edges>...</Edges></Network>`.

### CSV Export
Two files are generated:
*   `*_vertices.csv`: `y_position`, `x_position`, `z_position`, `radius_microns`.
*   `*_edges.csv`: `start_vertex`, `end_vertex` (indices).
