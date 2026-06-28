# Review: the original SLAVV methodology paper

**Citation (method of record for this port):**

> Mihelic SA, Sikora WA, Hassan AM, Williamson MR, Jones TA, Dunn AK.
> *Segmentation-Less, Automated, Vascular Vectorization.*
> **PLOS Computational Biology** 17(10): e1009451 (2021).
> doi:[10.1371/journal.pcbi.1009451](https://doi.org/10.1371/journal.pcbi.1009451)
> · Preprint: bioRxiv 2020.06.15.151076 · Code: `external/Vectorization-Public`
> (UTFOIL) · Data: doi:[10.18738/T8/NA08NU](https://doi.org/10.18738/T8/NA08NU)

> **Sourcing note:** content reviewed from the open-access PLOS HTML. The bioRxiv
> PDF blocks automated fetch (HTTP 403); the equations below are transcribed from
> the peer-reviewed PLOS version (equivalent to the preprint). **Equations are an
> automated transcription — verify each against the published PDF before quoting
> in our paper.**

---

## Core contribution

SLAVV extracts vascular networks **directly from raw grayscale 3D two-photon
microscopy images without segmentation** — no thresholding, no machine learning,
no training data. It replaces segmentation with (a) multiscale linear matched
filtering and (b) topologically-constrained vector extraction that **guarantees
shape and connectivity** of the output network. Open-source MATLAB; validated on
in-vivo mouse-brain vasculature and synthetic ground truth.

---

## The four stages (identical to this repo's pipeline)

1. **Energy** — multiscale matched filter detecting vessels idealized as
   spherical/annular objects; Fourier-domain implementation, PSF-convolved,
   sampled in octaves; projected to a 3D energy image (centerline likelihood) +
   size image (radius estimate).
2. **Vertices** — local energy minima → non-overlapping spheres (painted
   low-to-high energy); only negative-energy voxels retained.
3. **Edges** — watershed/min-energy tracing between vertex pairs via backpointer
   maps; hard constraints (below).
4. **Network** — bifurcations (≥3 edges), endpoints (1 edge), strands (maximal
   chains), then energy-weighted Gaussian strand smoothing.

---

## Equations (transcribed — verify before publication)

**Matched filter / energy formation**
- Matched vessel radius: `R² = σ² + r²` — `R` characteristic detected radius,
  `σ` Gaussian std, `r` ideal-kernel radius.
- Ideal spherical kernel: `K_S(ρ) = 1_{ρ < r}` (indicator over the lumen).
- Ideal annular kernel: `K_A(ρ) = δ(ρ − r)` (Dirac delta at the wall radius).
- Spherical fraction: `f_S = |K_S| / (|K_S| + |K_A|)` (L¹ norms; lumen vs wall
  weighting).
- Gaussian factor: `f_G = σ / (σ + r)` (noise-robustness vs size/position
  accuracy; tested 60%, 80%, 100%).
- Filter built by convolving a Laplacian-of-Gaussian (std `σ`) with the ideal
  kernel (radius `r`); a single combined filter per scale is formed and applied
  in the **Fourier domain**, then convolved with a **3D anisotropic Gaussian
  PSF**.

**Scale space**
- One **octave = doubling of vessel volume**; scales exponentially distributed.
- Example (Image 1): 16 octaves × 6 scales/octave = **96 discrete scales** → a 4D
  multiscale energy image.

**Projection 4D → 3D** (per voxel `x`, scale-`n` energy `E_n(x)`)
- Spherical (plasma) signal: energy-magnitude-weighted average over scales with
  `E_n(x) < 0`, weight `= |E_n(x)|`.
- Annular (endothelial) signal: `E(x) = E_n(x)` at `n = argmin_n E_n(x)`.
- Each second derivative is weighted by the Gaussian variance in its dimension to
  compensate for blur-induced derivative attenuation.

**Vertices**
- Center = local minimum of `E(x)`; radius `r_v = size(x_v)`; energy
  `E_v = E(x_v)`. Non-overlap: paint spheres low→high energy; later vertices may
  not overlap earlier ones.

**Edges**
- Watershed exploration follows the lowest-energy neighbor via a backpointer map;
  on reaching a terminal vertex, return the shortest path with the **lowest
  maximum energy**. **Hard constraints: ≤ 4 edges per vertex; trace length
  20–100× the seed vertex radius.** Edges accepted in order of increasing maximum
  energy; small 3-vertex cycles removed via adjacency-matrix recomputation.

**Strand smoothing**
- 1D Gaussian along each strand with `σ_smooth = r_vessel` (local radius),
  energy-weighted (favoring lower-energy/more-probable locations); smooths
  position, radius, and energy; special vertices held fixed.

**Metrics**
- Voxel accuracy `= (TP+TN)/(TP+TN+FP+FN)`; bulk stats on a cylinder idealization
  (volume fraction, surface-area density, length density, bifurcation density).

---

## Validation & results

- **Synthetic ground truth**: simulated 2PM images from a curated real
  vectorization, swept over **contrast-to-noise ratio CNR ∈ [1, ∞)**.
- **Peak voxel classification accuracy > 97%** across all CNR; volume/surface
  estimates robust at low CNR where intensity thresholding degrades (topological
  salt-and-pepper errors).
- Real data: in-vivo mouse brain, plasma (Texas Red dextran) + endothelial (GFP)
  labels; largest image 1.4×0.9×0.6 mm³, 1.6×10⁸ voxels.
- Bulk stats plausible vs literature (length density ~0.6 m/mm³; volume ~6%,
  higher than post-mortem, attributed to in-vivo perfusion).
- **MATLAB runtimes (10-core Xeon E5-2687 v3): ~140–360 s**, scaling ~linearly
  with volume; parallelized across image chunks (energy/vertices) and per vertex
  (edges).

---

## Limitations

**Stated by the authors:** bifurcation detection is noise-sensitive; `f_G` tuning
is image-quality-dependent; tiling artifacts force many manual vertex selections;
semi-automated curation still needs a human.

**Additional (reviewer's view):** the synthetic ground truth is generated from
the authors' own curated vectorization (somewhat self-referential); parameter
sensitivity beyond `f_G` is not fully characterized; MATLAB-only — which is
exactly the gap this port closes.

---

## Why this matters for the port (and our paper)

- **Performance baseline.** The paper's **~140–360 s** MATLAB runtimes are the
  number our optimized Python is measured against. Our **exact route trades this
  away for bit-exactness** (energy alone runs hours) — so frame the exact route as
  a *certification* tool, and target the paper-route/optimized Python back toward
  seconds-scale. The original paper has **no reproducibility/parity analysis** —
  that is our novel contribution.
- **Confirmed parity specs.** "≤ 4 edges per vertex" is the authoritative source
  for the degree-4 cap our edge-parity work found Python under-enforcing
  (Python degree 5 vs MATLAB 4). "Trace length 20–100× seed radius" maps to
  `max_edge_length_per_origin_radius`. (Distinct from the watershed
  `edge_number_tolerance = 2` hard-code.)
- **Validation template.** Our paper should mirror their evidence framework (CNR
  sweep, ROC, voxel accuracy, bulk network stats) **and add** the port-specific
  layer: ADR 0011/0012 parity bars, the ULP-vs-`allclose` argument, and the
  bit-exact parallelism speedup (see
  [post-parity-optimization-and-paper.md](post-parity-optimization-and-paper.md)
  and [figures/](figures/)).
- **Positioning.** Cite this as the method of record; position our work as *"a
  certified, optimized, production Python reimplementation of SLAVV."*

## Related manuscripts
- VessMorphoVis export target: Abdellah et al., *Bioinformatics* 2020 (PMC7355309).
- See the `external/Vectorization-Public` README for application manuscripts.
