# External Library Survey For The Python Port

[Up: Documentation Index](../README.md)

Date: 2026-04-06

## Scope

This note looks at external Python libraries that could improve the "basic
plumbing" in the Python port: preprocessing, multi-scale vessel enhancement,
large-volume handling, watershed/label work, graph analysis, and interactive
curation.

I grounded the recommendations in the current repo instead of treating this as
a generic bioimage-processing list.

## Current Baseline In This Repo

The current Python stack already covers a lot:

- `scikit-image` is a declared dependency in `pyproject.toml` and is used in
  `source/slavv/core/energy.py`, `source/slavv/core/edge_primitives.py`, and
  `source/slavv/core/edge_candidates.py`.
- `scipy.ndimage` is used in `source/slavv/utils/preprocessing.py`,
  `source/slavv/core/edge_primitives.py`, and `source/slavv/core/edges.py`.
- `cKDTree` is used in `source/slavv/core/edge_primitives.py`,
  `source/slavv/core/edge_candidates.py`, `source/slavv/core/edges.py`,
  `source/slavv/parity/metrics.py`, and `source/slavv/analysis/geometry.py`.
- `networkx` is used in `source/slavv/analysis/geometry.py` and
  `source/slavv/io/network_io.py`.
- `pyvista` and `pyqtgraph` are used in
  `source/slavv/visualization/interactive_curator.py`.
- `plotly` and `streamlit` are used in the web-facing visualization code.
- `tifffile` already backs volume TIFF loading and writing in
  `source/slavv/io/tiff.py`.
- `pydicom` is already supported behind the `dicom` optional extra for DICOM
  import in `source/slavv/io/tiff.py`.

There is also a low-risk performance lever already sitting in the repo:

- `numba` is already listed as an optional dependency in `pyproject.toml`.
- `source/slavv/core/energy.py` has a Numba path stubbed out, but
  `_NUMBA_AVAILABLE = False` right now.
- `source/slavv/utils/system_info.py` already probes for CuPy and CUDA-capable
  devices, but CuPy is not yet a declared dependency.

That means the cleanest path is not necessarily "add lots of libraries." In a
few places, it is "finish the optional acceleration path you already started,"
then add one or two libraries where they clearly beat the current stack.

## Recommendations

### Best overall additions

| Library | Where it fits here | Recommendation | Why |
| --- | --- | --- | --- |
| `SimpleITK` / `ITK` | `source/slavv/core/energy.py`, `source/slavv/utils/preprocessing.py`, `source/slavv/io/tiff.py` | `High priority` | Best fit for 3D microscopy/medical-style volumes, anisotropic spacing, recursive Gaussian filtering, objectness/vesselness filters, and image I/O pipelines. This would complement, not replace, the existing `tifffile`/`pydicom` I/O path. |
| `CuPy` | `source/slavv/core/energy.py`, `source/slavv/core/edge_primitives.py`, `source/slavv/core/edge_candidates.py`, `source/slavv/utils/preprocessing.py` | `High priority if you have NVIDIA GPUs` | Easiest way to move NumPy/SciPy-style array math and `ndimage`-like work onto GPU. |
| `cuCIM` | `source/slavv/core/edge_candidates.py`, `source/slavv/core/edge_selection.py`, `source/slavv/utils/preprocessing.py`, future segmentation helpers | `High priority if you have NVIDIA GPUs` | Gives you GPU image operations in a scikit-image-shaped API, which matches this repo's current mental model better than rewriting everything around OpenCV. |
| `Dask` + `Zarr` | `source/slavv/core/energy.py`, `source/slavv/runtime/run_state.py`, checkpoint/artifact storage | `High priority for large datasets` | Strong fit for chunked, resumable, larger-than-memory arrays and persistent chunked storage. |
| `napari` | `source/slavv/visualization/interactive_curator.py` | `Medium-high priority` | Better ecosystem for image + points + labels curation than a hand-rolled Qt stack, especially if manual review grows. |

### Useful targeted additions

| Library | Where it fits here | Recommendation | Why |
| --- | --- | --- | --- |
| `connected-components-3d` (`cc3d`) | cleanup helpers, label analysis, adjacency extraction | `Medium` | Nice fast utility if you end up doing more 3D connected-component passes or region cleanup. |
| `OpenCV` | slice-wise preprocessing, thresholding, 2D morphology, possibly some export helpers | `Low-medium` | Good library, but not my first choice for the core 3D vessel pipeline in this repo. |
| `MONAI` | future learned segmentation or denoising experiments | `Conditional` | Great if you pivot toward model-based 3D segmentation/inference; overkill for the current deterministic/parity-focused port. |

## Library-by-library Notes

### 1. SimpleITK / ITK

This is the strongest single addition if your goal is to improve the core
volume-processing path without turning the port into a deep-learning project.

Why it matches this repo:

- Your pipeline is explicitly 3D and spacing-aware.
- `source/slavv/core/energy.py` already spends a lot of effort on Gaussian
  smoothing, Hessian-like structure measurements, PSF-aware scaling, and
  vesselness/objectness behavior.
- `source/slavv/utils/preprocessing.py` currently does only min-max scaling and
  a simple axial Gaussian background subtraction. That is a minimal baseline,
  not an advanced microscopy preprocessing stack.

Where I would try it first:

- Replace or benchmark against the current Hessian/objectness path in
  `source/slavv/core/energy.py`.
- Evaluate whether N4-style bias correction or better recursive Gaussian
  filtering improves `source/slavv/utils/preprocessing.py`.
- Consider it for more robust DICOM / medical-style image import workflows if
  the input surface grows.

What I like:

- Better native 3D story than OpenCV.
- Stronger fit for anisotropic volumes than generic CV libraries.
- Lets you stay in a deterministic algorithmic world, which is important for
  MATLAB parity work.

Main downside:

- Different array/image conventions than NumPy-first code, so integration is a
  little heavier than adding one more NumPy-adjacent helper package.

Verdict:

- If you only spike one new stack for the core algorithm, make it this one.

### 2. CuPy

CuPy is the most direct "speed up the math you already have" option.

Why it matches this repo:

- The heavy parts of `source/slavv/core/energy.py` are array-oriented:
  Gaussian filtering, Hessian-style operations, eigenvalue work, per-scale
  evaluation, and chunk aggregation.
- The tracing code also leans heavily on NumPy arrays and local numeric loops.

Where I would try it first:

- Multi-scale energy computation in `source/slavv/core/energy.py`.
- Preprocessing filters in `source/slavv/utils/preprocessing.py`.
- Any repeated volume-wise transforms that currently operate scale-by-scale on
  CPU.

Main downside:

- NVIDIA-only in practice.
- You still need a CPU/GPU handoff story for code paths that remain in SciPy,
  scikit-image, or Python loops.

Verdict:

- Very good if your dev box or production environment has CUDA-capable GPUs.
- Not worth building the repo around if GPU availability is uncertain.

### 3. cuCIM

cuCIM is especially interesting because this repo already thinks in
`scikit-image` terms.

Why it matches this repo:

- `source/slavv/core/edge_candidates.py` and `source/slavv/core/edges.py` already import `skimage.segmentation.watershed`.
- `source/slavv/core/energy.py` already imports `skimage.filters.frangi` and
  `sato` when available.
- cuCIM exposes a GPU image-processing surface that is much closer to
  scikit-image than OpenCV is.

Where I would try it first:

- Morphology/segmentation helpers.
- Cleanup passes around label maps and boundary extraction.
- Slice/volume transforms that currently bottleneck in pure CPU image ops.

Main downside:

- Same NVIDIA dependency story as CuPy.
- Coverage is strong in many image-processing areas, but you still need to
  verify exact functions against the specific operations you care about.

Verdict:

- If you go GPU, I would test `CuPy` and `cuCIM` together rather than trying
  OpenCV first.

### 4. Dask + Zarr

This pair solves a different problem: not "new algorithms," but "stop fighting
volume size and chunk persistence."

Why it matches this repo:

- `source/slavv/core/energy.py` already has explicit chunking logic.
- `source/slavv/core/edges.py` and `source/slavv/runtime/run_state.py` already
  have resumable/persisted execution patterns.
- Your repo has active comparison and checkpoint workflows, so chunked on-disk
  array storage is already part of the mental model.

Where I would try it first:

- Persistent energy volumes and multi-scale outputs.
- Checkpoint artifacts that are currently stored as ad hoc `.npy`, `.pkl`, or
  MATLAB-side HDF5 fragments.
- Big-volume workflows where you want chunked reads/writes and lazy execution.

What I like:

- Dask helps with chunk-aware computation.
- Zarr is a much cleaner fit than one-off temporary files when you want
  chunked, compressed N-dimensional storage.

Main downside:

- Adds orchestration complexity.
- Lazy execution can make debugging harder if you are actively doing
  MATLAB-parity debugging at the same time.

Verdict:

- Strong addition if dataset size or checkpoint volume is a real pain point.

### 5. napari

For curation and inspection, napari is more compelling than OpenCV.

Why it matches this repo:

- You already have an interactive curator in
  `source/slavv/visualization/interactive_curator.py`.
- The upstream MATLAB repo has rich GUI curation surfaces (`vertex_curator.m`
  and `edge_curator.m`).
- Manual review of points, labels, overlays, and regions is exactly napari's
  sweet spot.

Where I would try it first:

- Display the raw image as one layer.
- Display vertices as a points layer.
- Display selected edges or watershed regions as labels/shapes overlays.
- Use it as the next-generation manual QA tool rather than extending a custom
  Qt/PyVista editor forever.

Main downside:

- It is a UI investment, not an algorithmic speedup.

Verdict:

- Worth it if interactive review matters to you long-term.

### 6. OpenCV

OpenCV is the obvious library to ask about, but I would not make it the center
of this repo.

Where it could help:

- Fast 2D per-slice preprocessing.
- Thresholding and some standard morphology.
- General image utility work around import/export or camera-style cleanup.

Why I would not prioritize it for the core pipeline:

- This repo is fundamentally 3D and spacing-aware.
- The critical code is vessel enhancement, 3D tracing, watershed-adjacent
  region logic, and parity-sensitive volume math, not classic 2D CV tasks.
- My read of the official OpenCV docs is that OpenCV is much better positioned
  for 2D image-processing workflows than for the kind of 3D anisotropic volume
  pipeline this port is building. That is an inference from the docs and the
  repo architecture, not a claim that OpenCV is incapable in every case.

Verdict:

- Add it only if you specifically want a fast 2D preprocessing layer.
- I would choose `SimpleITK`, `CuPy`, `cuCIM`, or `Dask` ahead of it for the
  main algorithmic path.

### 7. connected-components-3d

This is not a platform library, but it is a good tactical helper.

Where it could help:

- Connected-component labeling of 3D masks.
- Post-processing after thresholding or label cleanup.
- Fast extraction of 3D connected regions where SciPy/skimage are good enough
  functionally but not ideal ergonomically or performance-wise.

Verdict:

- Nice utility if region-label logic grows.
- Not a first move if the main bottleneck is still energy computation.

### 8. MONAI

I would only bring this in if you intentionally move toward learned methods.

Where it could help:

- Learned denoising.
- Learned vessel segmentation.
- Sliding-window inference on large 3D volumes.

Why I would wait:

- The current repo is still very much a deterministic algorithm port with a
  MATLAB parity agenda.
- MONAI is excellent, but it changes the problem framing from "port and match"
  to "train or deploy better models."

Verdict:

- Keep it on the shelf for a future branch, not as the next core dependency.

## What I Would Do First

If the goal is practical improvement with the least chaos, this is the order I
would use:

1. Re-enable and benchmark the optional `numba` path already hinted at in
   `source/slavv/core/energy.py`.
2. Spike `SimpleITK` on one canonical volume and compare it against the current
   `source/slavv/core/energy.py` outputs.
3. If you have an NVIDIA GPU, prototype `CuPy` and `cuCIM` on the scale loop in
   `source/slavv/core/energy.py`.
4. If volume size or checkpoint churn hurts, move persistent intermediate arrays
   toward `Zarr`, with `Dask` only where lazy chunked computation truly helps.
5. If manual review matters, prototype a `napari`-based curator before spending
   more effort growing the custom Qt/PyVista surface.

For a parity-preserving port, I would treat `numba` and `SimpleITK` as the
first non-UI experiments, and only move to GPU-backed options if the deployment
environment actually has stable NVIDIA support.

## MATLAB Packages / Toolboxes Evidenced In The Upstream Repo

This is the cleanest read I could justify from the checked-in upstream sources.

| MATLAB package / toolbox | Evidence in repo | Notes |
| --- | --- | --- |
| Base MATLAB | `interp3` in `external/Vectorization-Public/source/get_energy_V202.m` and multiple visualization scripts; `isosurface` in `calculate_surface_area.m` and visualization scripts; `figure`, `patch`, `uigetfile`, `h5create`, `h5read`, `h5write` across the repo | The upstream code heavily relies on core MATLAB numerics, graphics, interpolation, file dialogs, and HDF5 APIs. |
| Parallel Computing Toolbox | `parfor` in `get_energy_V202.m`, `get_vertices_V200.m`, `gaussian_blur_in_chunks.m`, and `get_edges_V203.m` | Clear evidence of parallel chunk processing in the MATLAB implementation. |
| Image Processing Toolbox | `dicominfo` and `dicomread` in `external/Vectorization-Public/source/dicom2tif.m` | Clear DICOM import dependency. |
| MATLAB graphics / UI stack | `figure` and `uicontrol` all over `vertex_curator.m` and `edge_curator.m` | The upstream curation tools are MATLAB GUI-heavy. |
| Python bridge rather than a MATLAB ML toolbox | `vectorize_V200.m` shells out to `python3 MLDeployment.py`; upstream also ships `MLDeployment.py`, `MLLibrary.py`, and `MLTraining.py` | The ML-curation path appears to lean on external Python, not a directly evidenced MATLAB Statistics and Machine Learning Toolbox workflow. |

## MATLAB Notes Worth Calling Out

- I did **not** find strong direct evidence in the checked-in `.m` sources that
  the upstream pipeline fundamentally depends on the MATLAB Statistics and
  Machine Learning Toolbox.
- I also did **not** find strong evidence that the upstream pipeline is built
  around MATLAB's built-in `watershed` function. The watershed-named edge code
  in the upstream repo is largely custom logic in
  `get_edges_by_watershed*.m`, not just a thin wrapper around a toolbox call.

## Bottom Line

If you want better core image processing for this Python port, I would rank the
options like this:

1. `SimpleITK` / `ITK`
2. `CuPy` + `cuCIM` if you have NVIDIA GPUs
3. `Dask` + `Zarr` for large-volume infrastructure
4. `napari` for curation/UI
5. `OpenCV` only for targeted 2D preprocessing, not as the backbone

The highest-probability "worth it" move is a small `SimpleITK` benchmark plus a
real revisit of the disabled `numba` path that is already in the codebase.

## External Sources

- SimpleITK Objectness filter:
  [simpleitk.org/doxygen/v2_5/html/sitkObjectnessMeasureImageFilter_8h.html](https://simpleitk.org/doxygen/v2_5/html/sitkObjectnessMeasureImageFilter_8h.html)
- SimpleITK N4 bias correction example:
  [simpleitk.readthedocs.io/en/v2.4.0/link_N4BiasFieldCorrection_docs.html](https://simpleitk.readthedocs.io/en/v2.4.0/link_N4BiasFieldCorrection_docs.html)
- CuPy `ndimage` docs:
  [docs.cupy.dev/en/v8.6.0/reference/ndimage.html](https://docs.cupy.dev/en/v8.6.0/reference/ndimage.html)
- cuCIM docs:
  [docs.rapids.ai/api/cucim/stable](https://docs.rapids.ai/api/cucim/stable)
- napari points-layer docs:
  [napari.org/howtos/layers/points.html](https://napari.org/howtos/layers/points.html)
- Dask Array docs:
  [docs.dask.org/en/stable/array.html](https://docs.dask.org/en/stable/array.html)
- Zarr docs:
  [zarr.readthedocs.io/en/main/](https://zarr.readthedocs.io/en/main/)
- connected-components-3d on PyPI:
  [pypi.org/project/connected-components-3d/](https://pypi.org/project/connected-components-3d/)
- MONAI transforms docs:
  [docs.monai.io/en/1.1.0/transforms.html](https://docs.monai.io/en/1.1.0/transforms.html)
- MathWorks Parallel Computing Toolbox docs:
  [se.mathworks.com/help/parallel-computing/index.html](https://se.mathworks.com/help/parallel-computing/index.html)
- MathWorks `dicomread` docs:
  [mathworks.com/help/images/ref/dicomread.html](https://www.mathworks.com/help/images/ref/dicomread.html)
- MathWorks `isosurface` docs:
  [mathworks.com/help/matlab/ref/isosurface.html](https://www.mathworks.com/help/matlab/ref/isosurface.html)
- OpenCV morphology tutorial:
  [docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- OpenCV watershed tutorial:
  [docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)

