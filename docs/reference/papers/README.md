# Papers

[Up: Reference Docs](../README.md)

## In short

This folder is an **index of papers and vendor docs this repo actually uses**,
grouped by the five questions people mix up. It is not live pass/fail and not
the executable spec.

**Two meanings of “paper”:** Mihelic et al. 2021 is the **publication**.
**Paper Path** is the public `paper` profile (`slavv run`). There is no external
paper for that split. When a sentence says “paper”, it must say which one.

No PDF is stored in this repo. Live Python-vs-MATLAB status is
[ONE TRUTH](../core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).
Phase 1 is closed.

---

## 1. Paper prose vs released MATLAB vs Python

**Question:** If the article, the `.m` files, and Python disagree, which wins?

**Answer:** MATLAB source is the executable spec. The article is narrative.

| Rank | Source | Role |
|------|--------|------|
| 1 | `external/Vectorization-Public/` | Executable spec for parity |
| 2 | Preserved MATLAB oracles (`workspace/oracles/`) | Proof artifacts (`prove-exact`) |
| 3 | Mihelic et al. 2021 (and preprint) | Explanatory prose — not a higher-priority spec |
| 4 | Maintained Python docs | Must not overrule 1–2 |

Full hierarchy: [MATLAB_METHOD_IMPLEMENTATION_PLAN.md](../core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md#canonical-hierarchy).
Narrative walkthrough: [SLAVV_METHOD_EXPLAINED.md](../core/SLAVV_METHOD_EXPLAINED.md).

| Citation | Why we use it | PDF in repo? |
|----------|---------------|--------------|
| Mihelic et al., *Segmentation-Less, Automated, Vascular Vectorization*, PLOS Computational Biology 17(10): e1009451 (2021). [doi:10.1371/journal.pcbi.1009451](https://doi.org/10.1371/journal.pcbi.1009451) | Method publication | No |
| Preprint: bioRxiv 2020.06.15.151076 | Earlier public draft of the same method | No |
| Data: [doi:10.18738/T8/NA08NU](https://doi.org/10.18738/T8/NA08NU) | Accompanying data deposit | No |

---

## 2. “Paper” (publication) vs Paper Path (pipeline profile)

**Question:** Is “paper” the 2021 article or `slavv run --profile paper`?

**Answer:** Both words exist. They are not the same thing.

| Phrase | Means | Not |
|--------|-------|-----|
| **Paper (publication)** | Mihelic et al. 2021 (section 1) | The CLI profile |
| **Paper Path** | Public `paper` profile: Tracing Discovery, `float32`, C-order `[Z, Y, X]` | The 2021 article |
| **Exact Route** | MATLAB-faithful cert path: Watershed Discovery, `float64`, F-order `[Y, X, Z]` | The tutorial default |

**No external paper** for this split. It is a repo product/certification decision
([ADR 0005](../../adr/0005-edge-discovery-strategy-seam.md),
[PAPER_PROFILE.md](../workflow/PAPER_PROFILE.md),
[AGENTS.md § Paper Path](../../../AGENTS.md#paper-path)).

Do not rename the `paper` profile. Say “publication” or “Paper Path” on first use.

---

## 3. Why MATLAB and Python do not match every last float bit

**Question:** Why isn’t Energy bit-identical, and why is that still certified?

**Answer:** Different math libraries; floating-point summation is non-associative.
Ship bar is stated tolerance (`np.allclose`), not last-digit identity. Identical
last digits is stretch, not Phase 1.

Repo decision: [ADR 0011](../../adr/0011-energy-float-certification-policy.md).
Citation home (do not duplicate here): [PARITY_METHODOLOGY.md](../core/PARITY_METHODOLOGY.md).
Extra notes: [post-parity-optimization-and-paper.md](../../research/post-parity-optimization-and-paper.md).

| Citation | Why we use it | PDF in repo? |
|----------|---------------|--------------|
| Lyon et al., *The CARFAC v2 Cochlear Model in Matlab, NumPy, and JAX*, [arXiv:2404.17490](https://arxiv.org/abs/2404.17490) | Golden-master MATLAB→NumPy port; real tolerance figures | No |
| Demmel et al., [EECS-2015-229](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-229.pdf) | Parallel reproducibility / non-associative sums | No (tech report URL) |
| SC’24 / [arXiv:2408.05148](https://arxiv.org/abs/2408.05148) | Run-to-run floating-point variability | No |
| [ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/) | Non-associativity; reproducible BLAS | No (project site) |
| Ahrens, Demmel, Nguyen, *Algorithms for Efficient Reproducible Floating Point Summation*, ACM TOMS. [doi:10.1145/3389360](https://doi.org/10.1145/3389360) | Binned-number reproducible accumulators | No |
| Intel oneMKL [Conditional Numerical Reproducibility](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-2/get-started-with-conditional-num-reproducibility.html) | Vendor: CNR does not transfer across hardware | No (vendor docs) |
| Cornea, *ULPs and Relative Error*, IEEE ARITH-24. [PDF](https://www.acsel-lab.com/arithmetic/arith24/data/1965a090.pdf) | ULP vs relative error (why pure-ULP gates fail near zero) | No |
| NumPy [`assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html) | The actual ship-gate criterion | No (vendor docs) |

---

## 4. Why exact edge-pair / pixel equality is the wrong watershed bar

**Question:** Why don’t we require the same edge-pair list (or pixel-perfect catchments)?

**Answer:** Raw pair overlap is the analogue of **pixel-accuracy** (easy to inflate).
Ownership-map agreement plus strand/junction bags is the analogue of **Dice/IoU
plus spatial tolerance**. That is [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md).
Phase 1 closed on that bar.

Citation home: [PARITY_METHODOLOGY.md §4](../core/PARITY_METHODOLOGY.md).

| Citation | Why we use it | PDF in repo? |
|----------|---------------|--------------|
| Müller et al., *Towards a guideline for evaluation metrics in medical image segmentation*, BMC Res Notes. [PMC9208116](https://pmc.ncbi.nlm.nih.gov/articles/PMC9208116/) | Pixel-accuracy discouraged; Dice/IoU + distance metrics | No |
| Beucher & Lantuéjoul, *Use of watersheds in contour detection*, Int. Workshop on Image Processing, Rennes, 1979 | Historical watershed algorithm class (workshop; **no DOI**) | No |
| Meyer, *Topographic distance and watershed lines*, Signal Processing 38(1):113–125 (1994). [doi:10.1016/0165-1684(94)90060-4](https://doi.org/10.1016/0165-1684(94)90060-4) | Later watershed formulation (optional background) | No |

The 1979/1994 papers describe the **algorithm class**. They do not set SLAVV’s
certification bar — ADR 0012 does.

---

## 5. Tie-breaking: Fortran linear index, not FIFO

**Question:** When two voxels have the same energy, who wins?

**Answer:** MATLAB `min` / `sort` / `find` ties break on the **lowest Fortran
(column-major) linear index**, not FIFO/LIFO.

**No peer-reviewed SLAVV paper.** This is MATLAB/NumPy language semantics plus
repo ports.

| Citation | Why we use it | PDF in repo? |
|----------|---------------|--------------|
| MathWorks, [column-major vs row-major](https://www.mathworks.com/help/coder/ug/what-are-column-major-and-row-major-representation-1.html) | MATLAB memory layout | No (vendor docs) |
| NumPy, [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) | C-order vs F-order for MATLAB readers | No (vendor docs) |

Repo: glossary **Lowest Linear Index Priority**;
[WATERSHED_IMPLEMENTATION_NOTES.md](../core/WATERSHED_IMPLEMENTATION_NOTES.md);
[UNPRODUCTIVE_LOOPS.md §8](../core/UNPRODUCTIVE_LOOPS.md).

---

## Related reading in this repo

- [PARITY_METHODOLOGY.md](../core/PARITY_METHODOLOGY.md) — literature home for items 3–5
- [NEW_ENGINEER_START_HERE.md](../core/NEW_ENGINEER_START_HERE.md) — Paper Path vs Exact Route on day one
- [slavv-original-paper-review.md](../../research/slavv-original-paper-review.md) — notes on Mihelic 2021
