# Parity Deviations

This document records known differences between the Python port and the original MATLAB implementation where exact numerical or visual parity is not feasible. Each item includes a short rationale.

## Energy Field
- **PSF weighting**: The energy filters weight the point spread function using a simplified Gaussian approximation. MATLAB's `get_energy_V202.m` uses more detailed kernels.
- **Filter ratios**: Gaussian-to-ideal and spherical-to-annular ratios are exposed as parameters but default to values that approximate rather than exactly replicate MATLAB behavior.

## Vertex and Edge Extraction
- **Gradient descent**: Edge tracing uses floating point updates by default but can snap to voxel centers via the `discrete_tracing` option. MATLAB always steps on integer voxels, so minor path differences may remain.
- **Direction estimation**: Hessian-based vessel directions fall back to uniform orientations when eigenanalysis is unstable, unlike MATLAB's more aggressive smoothing.

## Visualization
- **Plotly rendering**: 2D/3D visualizations use Plotly instead of MATLAB's native graphics. Color maps and camera controls therefore differ.
- **Strand coloring**: Strand-based coloring in 3D uses the `Set3` palette rather than MATLAB's hardcoded color table.

These deviations are intentional trade-offs for clarity, performance, or library compatibility and do not affect overall algorithm validity.
