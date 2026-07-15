# Parity Methodology — external validation for SLAVV's MATLAB↔Python certification

**Purpose.** This document records the established, literature-backed methodology for
verifying a scientific/numerical port from MATLAB to Python (NumPy/SciPy), and shows
how SLAVV's certification bars ([ADR 0011](../../adr/0011-energy-float-certification-policy.md),
[ADR 0012](../../adr/0012-edge-watershed-parity-bar.md)) align with it. It exists so the
project's tolerance-based parity gates rest on a citable foundation, not just internal
preference.

All claims below were sourced from primary/authoritative references (NumPy, MathWorks,
Intel docs; peer-reviewed IEEE/ACM/BMC papers; original-author preprints) and
adversarially fact-checked. Sources are listed at the end.

> **TL;DR.** Bit-exact MATLAB↔Python parity is generally *unachievable* (floating-point
> non-associativity + BLAS/FFT/ISA non-determinism). The established practice is
> **golden-master/oracle testing with tolerance gates**, and for **order-sensitive
> algorithms** (region-growing/watershed) **exact output-set equality is the wrong
> metric** — spatial/topological tolerance bars are required instead. SLAVV's ADR 0011
> (`np.allclose`) and ADR 0012 (ownership-map / topology multisets + sub-voxel trace
> tolerance) are direct instances of this practice.

**Figure (proposal appendix):** quantitative summary of the port —
candidate-pair overlap trajectory, edge-pair recovery waterfall, full-volume
counts and certification bars — at
[figures/README.md](../../../figures/README.md)
([README](../../../figures/README.md)). Live numbers: [EXACT_PROOF_FINDINGS.md](EXACT_PROOF_FINDINGS.md).

---

## 1. Golden-master / oracle testing is the dominant method

The best-documented public MATLAB→Python numerical port is Google's **CARFAC v2**
cochlear model (MATLAB → NumPy → JAX). Its approach is SLAVV's approach: the MATLAB
reference emits "golden data" that the Python tests consume, with tests written
*concurrently* with the transliteration to certify correctness and keep the
implementations synchronized.

- Capture reference outputs from the original (here: preserved MATLAB oracle vectors,
  `.mat`/HDF5).
- Assert the port reproduces them **within tolerance**, not bit-for-bit.
- Treat the reference as the source of truth; any undocumented deviation is a bug.

## 2. Bit-exactness is unachievable — certify by tolerance

- **Floating-point summation is non-associative**, so results depend on summation
  order. Parallel scheduling, reduction-tree shape, BLAS/LAPACK/MKL ISA code paths,
  thread count, and memory alignment all perturb low-order bits.
- This run-to-run variability is *measurable* (≈1e-15 for 100 elements to ≈4e-13 for
  1M elements; compounding to ~20% over iterative solvers) and can **mask real errors**
  in threshold tests — so tolerance must be set deliberately, not guessed.
- Even **Intel MKL Conditional Numerical Reproducibility** guarantees identical results
  only under *fixed* executable + thread count + ISA + alignment, and **does not transfer
  across hardware**.
- Real-world tolerance precedent: CARFAC matched ~**1e-6** (≈4 sig figs) in double
  precision; only **1e-3–1e-6** in single precision. (SLAVV's measured energy drift is
  ~**2e-11**, far tighter — see EXACT_PROOF_FINDINGS.)

**SLAVV mapping:** [ADR 0011](../../adr/0011-energy-float-certification-policy.md) adopts
exactly this — discrete/topological fields strict; continuous floats within
`np.allclose`.

## 3. The gate: `assert_allclose`, and `allclose` vs ULP

- Use `numpy.testing.assert_allclose` with the asymmetric criterion
  `|actual − desired| ≤ atol + rtol·|desired|`.
- **Gotcha:** `assert_allclose` defaults `atol=0` (purely relative → fails near zero),
  while `np.allclose` defaults `atol=1e-8`. Always set a non-zero `atol` for near-zero
  comparisons. (SLAVV uses `rtol=1e-7, atol=1e-9`.)
- **ULP vs relative error** are inter-convertible (a 1-ULP tolerance ≈ relative bound
  `2^-(N-1)` for an N-bit significand), but conversion is lossy unless precision is
  known — which is *why pure-ULP gates explode near zero*. ADR 0011 rejected pure ULP
  for this reason; this is the documented, expected failure mode.

## 4. Order-sensitive algorithms: exact set-equality is the wrong metric

For greedy region-growing / watershed-type outputs:

- **Exact-match / pixel-accuracy is discouraged** — class imbalance makes it spuriously
  high (>90% even for a degenerate all-background prediction).
- Use **overlap metrics (Dice/IoU)** — Dice primary because it ignores true negatives —
  **plus boundary/distance metrics (Average Hausdorff Distance)** as a spatial
  tolerance bar. (Caveat: AHD has ranking-bias when used to *rank methods*; it is fine
  as a pass/fail tolerance bar.)

**SLAVV mapping:** [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md) is the
watershed analogue: edges certify on **voxel ownership-map agreement** (a spatial
overlap measure on the catchment partition) and network on **strand endpoint-pair +
bifurcation multisets** (topology) + **sub-voxel trace tolerance** (a boundary/distance
bar). Raw edge-pair overlap is explicitly rejected — the direct analogue of "don't use
pixel-accuracy."

## 5. Convention pitfalls (all confirmed against primary docs)

| Pitfall | MATLAB | NumPy/Python | SLAVV touchpoint |
|---|---|---|---|
| **Indexing** | 1-based (`a(1)`, `a(end)`) | 0-based (`a[0]`, `a[-1]`) | scale/connection/vertex index normalization |
| **Memory order** | column-major (Fortran) | row-major (C); use `order='F'` for scan-order reshapes | `[Y,X,Z]` F-order watershed grid; the fixed double-transpose bug |
| **Rounding** | `round` = half-away-from-zero | `np.round` = half-to-**even** (banker's); use `floor(x+0.5)` to match | strand dedup fix in `network/operations.py` |

## 6. Where SLAVV is ahead of the published guidance

The research found **no** authoritative source documenting three things SLAVV's harness
already does:

1. **Stage-isolation certification** — feeding each stage the prior stage's *reference*
   (MATLAB) output to localize divergence (SLAVV: curated-input stage isolation).
2. **`.mat` reference-fixture capture/versioning** — including v7.3/HDF5 handling
   (SLAVV: `scipy.io.loadmat` + `h5py` for reversed-axis v7.3 oracles).
3. **Cross-language parity CI orchestration** (SLAVV: `prove-exact` / `prove-exact-sequence`
   gates + the `/prove-parity` skill).

These remain open questions in the literature; SLAVV's methodology is well-founded and,
on certification mechanics, somewhat ahead of published practice.

---

## Sources

| # | Source | Type | Used for |
|---|---|---|---|
| 1 | Lyon et al., *The CARFAC v2 Cochlear Model in Matlab, NumPy, and JAX*, [arXiv:2404.17490](https://arxiv.org/abs/2404.17490) | primary (original authors) | Golden-master case study; real tolerance figures |
| 2 | ReproBLAS, [bebop.cs.berkeley.edu/reproblas](https://bebop.cs.berkeley.edu/reproblas/) | primary | FP non-associativity; non-determinism sources |
| 3 | Demmel et al., [EECS-2015-229](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-229.pdf) | primary | Parallel reproducibility challenge |
| 4 | SC'24 Workshops, [arXiv:2408.05148](https://arxiv.org/html/2408.05148v3) | peer-reviewed | FPNA run-to-run variability magnitudes |
| 5 | Intel oneMKL, [Conditional Numerical Reproducibility](https://intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-2/get-started-with-conditional-num-reproducibility.html) | primary (vendor) | CNR conditions/limits |
| 6 | NumPy, [`assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html) + [testing guide](https://numpy.org/doc/stable/reference/testing.html) | primary | Tolerance criterion + defaults gotcha |
| 7 | Cornea (Intel), *ULPs and Relative Error*, IEEE ARITH-24, [PDF](https://www.acsel-lab.com/arithmetic/arith24/data/1965a090.pdf) | primary | ULP↔relative-error conversion |
| 8 | Müller et al., *Evaluation metrics in medical image segmentation*, [PMC9208116](https://pmc.ncbi.nlm.nih.gov/articles/PMC9208116/) | peer-reviewed | Dice/IoU/Hausdorff; accuracy discouraged |
| 9 | NumPy, [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) | primary | Indexing + memory-order conventions |
| 10 | MathWorks, [column-major vs row-major](https://www.mathworks.com/help/coder/ug/what-are-column-major-and-row-major-representation-1.html) | primary | Memory layout |
| 11 | NumPy, [`numpy.round`](https://numpy.org/doc/stable/reference/generated/numpy.round.html) | primary | Banker's rounding |

*Compiled 2026-06-25 from a fact-checked deep-research pass (5 search angles, 25 sources,
25 claims verified 3-of-3). See [ADR 0011](../../adr/0011-energy-float-certification-policy.md)
and [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md) for the SLAVV decisions this validates.*
