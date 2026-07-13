# How the SLAVV Method Works

Original method from Mihelic et al. (2021): how a 3D two-photon volume becomes a network of vessel centerlines.

Paper: Mihelic SA, Sikora WA, Hassan AM, Williamson MR, Jones TA, Dunn AK. *Segmentation-Less, Automated, Vascular Vectorization.* PLOS Computational Biology 17(10): e1009451 (2021).  
doi:[10.1371/journal.pcbi.1009451](https://doi.org/10.1371/journal.pcbi.1009451) · PDF: [docs/reference/papers/journal.pcbi.1009451.pdf](docs/reference/papers/journal.pcbi.1009451.pdf)

---

The SLAVV method never builds a binary vessel mask and never trains a classifier. Most vessel pipelines segment first, then skeletonize. SLAVV scores how much each location looks like a vessel center at many sizes, then places points and paths on that score field under geometric rules so the output already has shape and connectivity.

SLAVV stands for Segmentation-Less, Automated, Vascular Vectorization.

Start with a three-dimensional stack of two-photon fluorescence images of vasculature. End with a vectorized network: centerline positions, local radii, and how those centerlines connect. Four stages, in order:

1. Energy — multiscale filtering that scores vessel centers and estimates radius  
2. Vertices — seed points at the strongest centers  
3. Edges — centerline traces between seeds  
4. Network — longer vessel segments (strands), branches, and light smoothing  

```text
3D two-photon stack
   → Energy     (score + size maps)
   → Vertices   (seeds)
   → Edges      (traces between seeds)
   → Network    (strands + junctions)
```

---

## Why not just use intensity?

Two-photon vessel images are uneven and noisy. Bright voxels are not automatically vessel, and vessels span a wide range of calibers. The label may sit in the lumen (plasma dye) or on the wall (endothelial label). The microscope blurs everything—the point-spread function (PSF), usually worse along depth than in-plane.

So the first product is not a mask. It is an energy field. More negative energy means the filter thinks you are near a vessel center; near-zero or positive means it does not. Each location also gets a size: the vessel radius that best fits the local image.

The rest is geometry on that field. Seeds go on deep local minima. Paths follow low-energy corridors. The network stage turns those paths into strands and bifurcations. If the energy does not support a path, the constraints should kill it.

---

## 1. Energy

The raw volume is filtered at many vessel sizes (a 4D space × scale array), then collapsed to two 3D maps: energy and size.

After prep/normalization, the method steps through a ladder of vessel radii. One octave is a doubling of vessel volume (radius cubed). Several scales sit inside each octave. The paper’s Image 1 example used 16 octaves × 6 scales/octave = 96 scales; the important part is that capillaries and large vessels get different filters.

At each scale the filter is matched to an idealized vessel: roughly a solid ball of radius \(r\) for lumen-filled vessels, or a thin shell at radius \(r\) for wall-labeled vessels. Those ideals are mixed with Gaussian / Laplacian-of-Gaussian blur of width \(\sigma\), applied in the Fourier domain, with a 3D anisotropic Gaussian PSF model.

From the paper:

\[
R^{2} = \sigma^{2} + r^{2}
\qquad
f_{G} = \frac{\sigma}{\sigma + r}
\]

\(R\) is the effective radius the filter responds to. \(f_G\) balances smooth (noise-tolerant) vs sharp (better localized); the authors tried about 0.6, 0.8, and 1.0. Ideal kernels:

\[
K_{S}(\rho) = \mathbf{1}_{\{\rho < r\}}
\qquad
K_{A}(\rho) = \delta(\rho - r)
\]

\(\rho\) is distance from the kernel center. \(K_S\) is the solid lumen model; \(K_A\) is the wall shell. Software knobs for the mix include `spherical_to_annular_ratio` and `gaussian_to_ideal_ratio`.

The score at each scale uses second derivatives. Where the Laplacian is negative (local bright spot), the Hessian is diagonalized for principal curvatures. Energy favors large negative principal curvatures, each weighted by a symmetry term from the gradient—bright, fairly symmetric vessel centers, not random speckles.

Projection collapses scale to one energy and one size per voxel:

- Minimum projection (MATLAB description; this repo’s `energy_projection_mode=matlab`): most negative energy across scales and its scale index.  
- Paper-style projection (this repo’s `paper` profile): best scale for the annular case; magnitude-weighted average of scale indices over negative energies for the spherical case; blend those estimates; sample the 4D stack at the nearest scale.

Done: a 3D score field and a 3D size field. Centerlines sit in the valleys; the size map says how wide the vessel should be.

---

## 2. Vertices

Vertices are the seeds. Continuous energy is not a graph; you need discrete anchors.

Candidates are local minima of energy—deep, high-contrast spots (centers, often near bifurcations). In the released V200-style path, energy is usually already projected to 3D with a scale map, so the search is spatial local minima, each carrying a radius. Only locations that pass the energy gate are kept (negative energy under the usual sign; upper bound defaults to 0).

Sort strongest first (more negative first). Accept greedily with painting: for each candidate, check a volume sized from its radius (sometimes slightly dilated). If free, accept and paint that volume; if anything is already painted, reject. Strongest seeds claim space first, so you do not get a cloud of overlapping centers on one bright spot.

Rule of thumb: at least one vertex per eventual strand (segment between junctions). Too few seeds miss connections; too many make the edge stage fight itself.

Each kept vertex has position, radius (or scale), and energy.

---

## 3. Edges

Edges are centerline paths between seeds: a sequence of points along the vessel, each with position, local radius, and energy, connecting two vertices (or a vertex and a bridge point).

From each origin seed the algorithm walks the 3D energy image, preferring lower energy and requiring motion away from the origin—discrete propagation on the energy field, not a snake on raw intensity.

The paper and MATLAB edge code describe this in watershed / min-energy terms. Seeds claim low-energy neighborhoods. A frontier of candidate voxels is ordered by energy. Backpointers record where each claim came from. When another vertex is reached, walk pointers back to rebuild the path. Path quality is the maximum energy on the route (the worst point):

\[
\mathrm{score}(\text{path}) = \max_{x \in \text{path}} E(x)
\]

Lower max-energy paths are accepted first. A good centerline stays in the valley the whole way; one high-energy step usually means you left the vessel.

Hard limits: about four edges per vertex (`number_of_edges_per_vertex`, default 4). Trace length is bounded relative to seed radius—on the order of tens of radii in the paper (~20–100×); software uses `max_edge_length_per_origin_radius` (MATLAB docs often 30; this port’s public profiles often 60). After candidates exist, the pipeline crops, cleans bad pairs, and removes excess degree, orphans, and small cycles (including 3-vertex loops the paper mentions), depending on workflow.

If topology needs a junction that was never an energy minimum, the released code can insert bridge vertices after edge selection (`add_vertices_to_edges`). Those are structural, not seed minima.

For faithful watershed behavior: distances are often in units of local radius (\(r/R\)); large jumps, abrupt size changes, and sharp turns can be penalized when picking the next seed; the shared frontier is still sorted on original energies, not the penalized ones.

In this Python repo the same stage has two implementations. Tracing Discovery is the public paper-path walk. Watershed Discovery is the MATLAB-shaped global watershed on the exact-parity route. Both look for low-energy corridors between seeds. Neither is the skimage label-adjacency watershed used in some experimental helpers.

---

## 4. Network

Edges are still short pieces. The network stage builds what people analyze.

Accepted edges define an undirected graph. Degree one is an endpoint (tip or volume boundary); two is a waypoint on a single vessel; three or more is a bifurcation.

A strand is a maximal chain of edges from junction/endpoint to junction/endpoint through any degree-2 vertices. Strands are usually longer than single edges. Along a strand you still have position, radius, and energy samples. The set of strands is the minimal set of 1D objects that connect all bifurcations and endpoints given the adjacency.

Voxel traces are a bit stair-stepped. A 1D Gaussian along each strand, width on the order of local vessel radius (`sigma_strand_smoothing` defaults to 1 in MATLAB, meaning one radius), weights lower-energy samples more. Position, radius, and energy can all be smoothed; true junctions and endpoints stay fixed.

Output is a vector network—strands, bifurcations, endpoints, and optional stats (length density, volume fraction, branching counts, etc.). This package writes versioned `network.json`. Original MATLAB tools also write VMV and CASX.

---

## Summary

| Stage    | Input             | Output                          |
| -------- | ----------------- | ------------------------------- |
| Energy   | Raw volume        | Energy map + size map           |
| Vertices | Energy (+ size)   | Seed points                     |
| Edges    | Energy + vertices | Traces and connections          |
| Network  | Vertices + edges  | Strands, junctions, smoothed geometry |

Vertices and edges still carry energy values, so the raw result is a ranked family of possible structures. Thresholding trades sensitivity for specificity. The original software also shipped interactive curators when automatic choices were not enough.

---

## Results and limits (from the paper)

On synthetic data swept over contrast-to-noise ratio, voxel classification accuracy peaked above 97%, and bulk geometry held up better at low CNR than intensity thresholding. On real in-vivo mouse brain (plasma and endothelial labels), including volumes around \(1.6 \times 10^8\) voxels, network statistics were in a plausible literature range (e.g. length density ~0.6 m/mm³, volume fraction ~6%). MATLAB runtimes on a 10-core Xeon of that era were about 140–360 s and grew roughly with volume.

Known weaknesses: bifurcations are noisier than mid-vessel seeds; the Gaussian-vs-ideal mix needs to match image quality; large tiled volumes can need manual vertex fixes; the method assumes the matched filter can see the vessels.

---

## Terms

- Energy — vessel-center score; more negative is stronger evidence  
- Scale / octave — size ladder; an octave doubles vessel volume  
- Vertex / seed — discrete vessel point from a local energy minimum  
- Bridge vertex — junction added for topology, not found as a minimum  
- Edge — centerline trace between vertices  
- Strand — vessel segment between bifurcations/endpoints (one or more edges)  
- Bifurcation — branch point (degree ≥ 3)  
- PSF — microscope blur  
- Vectorization — vessels as curves and a graph, not a voxel mask  

---

## Links in this repository

- Run: [README.md](README.md), [docs/TUTORIAL.md](docs/TUTORIAL.md)  
- Paper notes: [docs/research/slavv-original-paper-review.md](docs/research/slavv-original-paper-review.md)  
- MATLAB README / sources: [external/Vectorization-Public/README.md](external/Vectorization-Public/README.md), `external/Vectorization-Public/source/`  
- Energy projection options: [docs/reference/core/ENERGY_METHODS.md](docs/reference/core/ENERGY_METHODS.md)  
- Glossary: [docs/reference/core/GLOSSARY.md](docs/reference/core/GLOSSARY.md)  

```powershell
slavv run -i volume.tif -o slavv_output --export json
```

Default profile `paper` runs Energy → Vertices → Edges → Network. Where paper prose and released MATLAB disagree on a detail, this project treats `external/Vectorization-Public/source/` as the executable reference.
