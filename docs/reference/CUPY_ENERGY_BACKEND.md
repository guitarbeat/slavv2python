# CuPy Energy Backend

This note describes the experimental `cupy_hessian` energy backend for the
Python SLAVV port.

## When To Use It

Use `cupy_hessian` when you have a CUDA-capable NVIDIA GPU and want to
accelerate the Gaussian/Hessian derivative work in the energy stage without
changing the rest of the pipeline.

This backend is a good fit for:

- exploratory performance work on NVIDIA-equipped machines
- large 3D volumes where Gaussian derivative filtering dominates runtime
- comparisons against the default `hessian` path while keeping the same
  high-level energy aggregation logic

It is not intended to become the default backend yet.

## Installation

Install a CuPy build that matches the target CUDA runtime. For example:

```powershell
pip install cupy-cuda12x
```

The generic `.[cupy]` extra is also declared in `pyproject.toml`, but in
practice most GPU machines should install the CUDA-matched CuPy package
explicitly.

## CLI Usage

```powershell
slavv run -i volume.tif -o slavv_output --energy-method cupy_hessian
```

## What Runs On GPU In V1

The first implementation accelerates:

- Gaussian derivative filtering
- Hessian-style derivative tensor preparation

The following still run on CPU in v1:

- eigendecomposition of the filtered Hessian samples
- final per-scale aggregation and resumable bookkeeping

## Parameters Honored In V1

The first implementation honors the same MATLAB-style Hessian controls as the
CPU Hessian path, including:

- `microns_per_voxel`
- `radius_of_smallest_vessel_in_microns`
- `radius_of_largest_vessel_in_microns`
- `scales_per_octave`
- `gaussian_to_ideal_ratio`
- `spherical_to_annular_ratio`
- `approximating_PSF`
- `energy_sign`
- `return_all_scales`

## How It Differs From Other Energy Methods

- `hessian`
  Default CPU baseline.
- `frangi`
  Scikit-image vesselness filtering.
- `sato`
  Scikit-image tubeness filtering.
- `simpleitk_objectness`
  Spacing-aware SimpleITK objectness backend.
- `cupy_hessian`
  GPU-accelerated derivative backend for NVIDIA/CUDA environments.

For now, `cupy_hessian` should be treated as an experimental performance path
rather than a new default.
