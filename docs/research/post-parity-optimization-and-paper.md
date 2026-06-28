# Post-Parity Optimization & the Translation Paper

**Generated:** 2026-06-27 via the deep-research workflow (fan-out web search →
source fetch → 3-vote adversarial verification → synthesis).
**Scope:** how to go from a *certified bit-exact* MATLAB→Python baseline (SLAVV
3D vessel extraction: FFT/Hessian energy → vertices → edges → network) to an
*optimized, publishable* Python pipeline — mapped onto roadmap
[Phase 2 (performance & scale)](../ROADMAP.md) and Phase 3 (productization).
**Method stats:** 3 search angles · 15 sources fetched · 67 claims extracted ·
25 verified · 20 confirmed / 5 refuted · 14 findings after synthesis.

> Confidence labels and source URLs are from adversarial verification (a claim
> needed ≥2/3 verifier votes to survive). The **Refuted claims** section lists
> what failed verification — do not treat those as established.

> **Data-backed draft figures** generated from real run artifacts are in
> [figures/](figures/) — the ULP histogram + parity composition (the
> certification-policy argument) and the bit-exact ~4.1× speedup. Regenerate via
> `scripts/make_report_figures.py`.

---

## Executive summary

After certifying a bit-exact baseline, the path to optimized, publishable Python
rests on three pillars:

1. **Methodology** — lock the reference behind a differential / characterization
   ("golden-master") test suite that runs on *every* change; make the baseline
   reproducible first (seed + disable stochastic behavior); and decide
   **per field** whether an optimization may change results: bitwise-identical
   for discrete/topological fields, tolerance-bounded (`np.allclose`) for
   continuous floats. This is exactly the project's ADR 0011/0012 split,
   corroborated by external practice.
2. **Performance** — the central reproducibility hazard is **floating-point
   non-associativity**: parallel/out-of-order reductions, FMA, AVX/SIMD,
   `-O2/-O3`, GPU `atomicAdd`, and `fastmath` all reorder additions and perturb
   bits. Parallelism does **not** break bit-exactness *if reduction order is
   fixed* — which is precisely why this project's joblib chunk parallelism stays
   bit-exact (the merge is applied in deterministic `c_idx` order). Classify every
   technique as bit-preserving or bit-perturbing before adopting it.
3. **Publication** — JORS (Journal of Open Research Software) is a verified
   fit-for-purpose, peer-reviewed venue with a track for full-length papers on
   developing/evaluating research software; report speedup/scaling **alongside**
   explicit parity and reproducibility evidence.

---

## Part 1 — Methodology: optimize without breaking the reference

**Lock the reference behind a differential/characterization test suite** run on
every change, with per-comparison tolerances tuned to float precision *(high
confidence).* The CARFAC cochlear-model port (Google Research) built a
cross-language suite *simultaneously* with the code to keep MATLAB, NumPy, and
JAX synchronized — printing "golden data" and adjusting per-test tolerances as
needed. This is the exact optimize-against-a-golden-baseline loop Phase 2 needs.
[arXiv:2404.17490]

**Make the baseline reproducible before optimizing** *(high).* "Stochastic
behavior must be disabled and the seed … made explicit" — a non-deterministic
reference cannot be diff-tested. Seeding is **necessary but not sufficient**:
GPU atomics, FP non-associativity, and BLAS thread order can still reintroduce
nondeterminism. [arXiv:2502.00902]

**Decide per field when results may change** *(high).* Bitwise-identical results
are "essential for debugging and trusting" deterministic computation (→ apply to
discrete fields), and this genuinely *conflicts* with the "stated precision"
(tolerance) norm used for continuous floats. The project must choose per field —
which it already does. [arXiv:2402.07530, arXiv:2404.17490]

**Set tolerances above the nondeterministic-noise floor** *(high).* Parallel
reductions can vary run-to-run by more than scientific tolerances (a
million-element sum varied ~1.25e-12, exceeding CP2K's 1e-14 tolerance), which
can *mask* correctness-test failures. `np.allclose` bounds must sit above this
floor or the suite throws false failures. [arXiv:2408.05148]

**The optimize–measure–verify loop (recommended):**
1. Profile to find the true hotspot (don't guess) — see Part 2 tooling.
2. Classify the candidate optimization as bit-preserving or bit-perturbing.
3. Apply it in isolation on a branch.
4. Run the golden-master suite: strict equality on discrete fields, `allclose`
   (tolerance ≥ noise floor) on continuous fields.
5. Benchmark speedup/scaling; keep only if it passes **both** gates.
6. Record the result (this repo: `docs/solutions/` + the parity findings).

---

## Part 2 — Performance techniques, classified by bit-impact

The audit checklist for "where could this perturb bits?" (ReproBLAS taxonomy,
all rooted in FP non-associativity): **data partitioning across processors,
variable processor count, instruction selection (SSE vs AVX), memory alignment,
reduction-tree scheduling, input data ordering** *(high).*
[bebop.cs.berkeley.edu/reproblas]

### Bit-PRESERVING (safe for the certified path)
- **Parallelism with a fixed reduction order** *(high).* Non-reproducibility
  comes from *out-of-order* execution, **not parallelism per se** — fix the order
  and threaded chunking is bit-exact. (This is why this project's joblib energy
  parallelism certified bit-identical: the per-chunk min-merge runs in
  deterministic `c_idx` order.) [reproblas; arXiv:2402.07530; arXiv:2408.05148]
- **ReproBLAS-style binned "reproducible accumulator"** *(high).* Order-*independent*
  bitwise-reproducible summation via a 6-word accumulator, one read pass + one
  parallel reduction, independent of processor count/scheduling, compatible with
  tiling — at ~4× single-core / ~1.2× large-parallel cost (conditional on
  IEEE-754 binary, round-to-nearest, bounded summand counts). Proof that you *can*
  have both parallelism and bit-exactness on reductions.
  [dl.acm.org/10.1145/3389360; reproblas]
- **Numba `@njit` (nopython) and Cython** *(high)* — large speedups on CPU-bound
  numeric loops (Numba docs show ~37× on one loop example) **with `fastmath`
  off**. Best first step for the loop-heavy non-FFT stages (watershed, network).
  *(Magnitude is from single benchmarks — validate on our own stages.)*
  [numba performance-tips; arXiv:2203.08263]

### Bit-PERTURBING (only where tolerance-bounded agreement is acceptable)
- **Out-of-order / dynamically-scheduled parallel reductions** *(high)* — the
  core hazard. [reproblas; arXiv:2402.07530; arXiv:2408.05148]
- **`-O2/-O3`, FMA, AVX/SIMD vectorization, out-of-order execution** *(high)* —
  each can cost reproducibility, even run-to-run on supercomputers; pin/control
  for a bit-exact build (relevant to Numba/Cython compile flags and BLAS paths).
  [arXiv:2402.07530]
- **Numba `fastmath`** *(high)* — permits FP reassociation to vectorize
  reductions; trades accuracy (1 ULP → 4 ULP) for ~2× speed. Keep OFF on
  bit-exact paths. [numba performance-tips]
- **GPU `atomicAdd`-only reductions** *(high)* — non-deterministic (runtime-dependent
  order) **and** up to 4 orders of magnitude slower than the fastest reductions.
  Naive CuPy/cuFFT porting risks losing *both* bit-exactness and speed; needs a
  deterministic-reduction design. [arXiv:2408.05148]

### FFT backends (re-verify after any swap)
- **pyFFTW exposes a `scipy.fft`-compatible drop-in** *(high)* — minimal-code
  backend swap for the FFT-heavy energy stage. **But** it is *not* guaranteed
  byte-identical in corner cases, and backend choice is a known reproducibility
  variable — re-run the parity gate after swapping. [pyfftw docs]
- Open question: do scipy.fft / pyFFTW / mkl_fft preserve bit-exactness against
  the MATLAB FFTW oracle, or only tolerance-bounded agreement? (see Open
  questions).

### Profiling
- **Scalene** *(high)* — line- and function-level CPU/GPU/memory profiling that
  **separates Python time from native/library (NumPy/FFT) time and system/I/O
  time** — ideal for finding whether a hotspot is interpreted Python vs native vs
  I/O in an FFT-heavy 3D pipeline. **Windows caveat:** CPU/GPU only (no memory
  profiling), GPU is NVIDIA-only — relevant to this project's win32 environment.
  [github.com/plasma-umass/scalene]

---

## Part 3 — Writing the paper / engineering report

- **Venue: JORS (Journal of Open Research Software)** *(high)* — peer-reviewed,
  with two relevant tracks: **software metapapers** (reusable research software)
  and **full-length research papers** on developing/maintaining/evaluating
  open-source research software. The port + optimization + parity work fits the
  full-length track; the released package fits the metapaper track. Note: only
  JORS was verified here — broaden venue options before submission (see Open
  questions). [openresearchsoftware.metajnl.com]
- **What to report:** speedup and strong/weak scaling curves **alongside**
  explicit parity/accuracy tables and a reproducibility statement (environment
  pinning, seeds, tolerance definitions). The project's ADR 0011/0012 parity
  bars, the random-component suite, and the n_jobs bit-exactness A/B are
  directly citable evidence.
- **Framing:** treat the translation as an *opportunity to optimize*, not a 1:1
  port — and present the certified baseline as the foundation that makes
  optimization trustworthy.

---

## Mapping to the roadmap

- **Phase 2 (performance & scale):** adopt the optimize–measure–verify loop;
  the bit-impact matrix tells you which techniques are safe for the certified
  path (fixed-order parallelism ✓, Numba `@njit` ✓, ReproBLAS for any
  order-sensitive reduction ✓) vs which require a tolerance decision (`fastmath`,
  GPU atomics, aggressive compiler flags, FFT-backend swaps). Profile with
  Scalene before optimizing.
- **Phase 3 (breadth & productization):** GPU acceleration needs a
  deterministic-reduction design, not naive `atomicAdd`; the paper/metapaper is a
  Phase 3 deliverable; broaden venue research first.

---

## Caveats (from the verification pass)

- Library specifics (Scalene Windows memory limitation, pyFFTW corner cases,
  Numba flag semantics) drift across releases — re-check docs at write-up time.
- Numba ~37× and the Pereira/Garcia speedups come from **single benchmarks** —
  they validate *direction, not magnitude* for our specific stages. Benchmark our
  own FFT/Hessian/watershed/network code.
- ReproBLAS guarantees are **conditional** (IEEE-754 binary, round-to-nearest,
  bounded summand counts) and apply to summation/BLAS-style reductions, not
  arbitrary FFT pipelines — feasibility, not a turnkey whole-pipeline solution.
- Several reproducibility sources are HPC/ML (CP2K, GraphSAGE), not 3D image
  processing — mechanisms transfer, magnitudes do not.
- Venue section is single-venue (JORS only); broaden before submission.

## Refuted claims (failed adversarial verification — do NOT assert)

- ❌ "float32 vs float64 is *the* primary driver of port parity tolerance" (0-3).
- ❌ "NumPy matches MATLAB to ~1e-6 while JAX only 1e-6–1e-3, proving ports
  converge to tolerance not bit-exactness" (0-3).
- ❌ "FPNA non-determinism provably *compounds* with operation/epoch count" (1-2).
- ❌ "CPython/PyPy are *unsuitable* for parallel CPU-bound numerical work" (1-2).
- ❌ "Numba `parallel=True`/`prange` gives a specific ~5–6× over NumPy/@njit" (1-2).

## Open questions

1. Do scipy.fft / pyFFTW / mkl_fft preserve **bit-exactness** against the MATLAB
   FFTW oracle for the energy stage, or only tolerance-bounded agreement?
2. Beyond JORS — SoftwareX, JOSS, Computer Physics Communications, IEEE CiSE,
   PeerJ CS — which fit a port+optimization+parity paper, and what
   benchmarking/reproducibility evidence do they expect?
3. Can a ReproBLAS-style binned accumulator (or another deterministic reduction)
   be integrated into the watershed/frontier and network reductions to keep them
   bit-exact under joblib parallelism, and at what real cost?
4. What is the accepted best-practice structure + required reporting set for
   research-software *port* papers, from accepted exemplars?

## Sources

Verified, by angle and quality (claim counts in brackets):

- [arXiv:2404.17490 — CARFAC v2 in MATLAB/NumPy/JAX](https://arxiv.org/pdf/2404.17490) — primary [5]
- [arXiv:2402.07530 — Reproducibility survey (HPC focus)](https://arxiv.org/pdf/2402.07530) — primary [5]
- [arXiv:2408.05148 — FP non-associativity / parallel reductions (ORNL SC'24)](https://arxiv.org/html/2408.05148v1) — primary [5]
- [arXiv:2502.00902 — ML reproducibility practices (2025)](https://arxiv.org/html/2502.00902v2) — primary [5]
- [ReproBLAS — reproducible BLAS (UC Berkeley, Demmel/Nguyen)](https://bebop.cs.berkeley.edu/reproblas/) — primary [5]
- [ACM TOMS 10.1145/3389360 — reproducible accumulator (binned numbers)](https://dl.acm.org/doi/10.1145/3389360) — primary [5]
- [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html) — primary [5]
- [arXiv:2203.08263 — Numba/Cython vs CPython/PyPy (CACIC 2021)](https://arxiv.org/pdf/2203.08263) — primary [3]
- [pyFFTW scipy_fft interface](https://pyfftw.readthedocs.io/en/latest/source/pyfftw/interfaces/scipy_fft.html) — primary [4]
- [Scalene profiler](https://github.com/plasma-umass/scalene) — primary [5]
- [JORS — Journal of Open Research Software](https://openresearchsoftware.metajnl.com/) — primary [5]
- [Neuroinformatics — Why/How to translate MATLAB→Python](https://neuroinformatics.dev/blog/matlab_to_python.html) — blog [5]
- [Concise guide to reproducible MATLAB (Sheffield RSE)](https://rse.shef.ac.uk/blog/2022-05-05-concise-guide-to-reproducible-matlab/) — secondary [5]
- [2D FFT performance: NumPy/pyFFTW/cuFFT](https://www.johnaparker.com/blog/2020-10-14-fft_2d_performance/) — blog [5]
- [Intel Conditional Numerical Reproducibility (CNR)](https://www.intel.com/content/www/us/en/developer/archive/training/conditional-numerical-reproducibility-cnr.html) — flagged unreliable by verifier [0]
