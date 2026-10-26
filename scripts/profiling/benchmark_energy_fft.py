"""Benchmark Energy FFT multi-threading and pre-indexed symmetric IFFT acceleration.

Measures wall-clock time and memory efficiency of:
1. Bare np.fft vs scipy.fft 3D forward/inverse FFT.
2. Original un-cached vs pre-indexed _ifftn_matlab_symmetric.
3. End-to-end single-octave chunked energy computation.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.fft

from slavv_python.pipeline.energy import matlab_energy_filter_v200 as native_hessian
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    compute_exact_parity_energy_single_octave,
)


def benchmark_bare_ffts(shape: tuple[int, int, int] = (128, 128, 64), iterations: int = 10) -> dict[str, float]:
    """Benchmark raw FFT operations."""
    rng = np.random.default_rng(42)
    arr_real = rng.standard_normal(shape, dtype=np.float64)
    arr_complex = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    # 1. Forward FFT: np.fft vs scipy.fft
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = np.fft.fftn(arr_real)
    np_fftn_ms = (time.perf_counter() - t0) / iterations * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scipy.fft.fftn(arr_real, workers=1)
    sp_fftn_1_ms = (time.perf_counter() - t0) / iterations * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scipy.fft.fftn(arr_real, workers=-1)
    sp_fftn_all_ms = (time.perf_counter() - t0) / iterations * 1000

    # 2. Inverse FFT: np.fft vs scipy.fft
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = np.fft.ifftn(arr_complex).real
    np_ifftn_ms = (time.perf_counter() - t0) / iterations * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scipy.fft.ifftn(arr_complex, workers=1).real
    sp_ifftn_1_ms = (time.perf_counter() - t0) / iterations * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = scipy.fft.ifftn(arr_complex, workers=-1).real
    sp_ifftn_all_ms = (time.perf_counter() - t0) / iterations * 1000

    # 3. Accelerated _ifftn_matlab_symmetric
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = native_hessian._ifftn_matlab_symmetric(arr_complex, workers=1)
    sym_ifftn_1_ms = (time.perf_counter() - t0) / iterations * 1000

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = native_hessian._ifftn_matlab_symmetric(arr_complex, workers=-1)
    sym_ifftn_all_ms = (time.perf_counter() - t0) / iterations * 1000

    return {
        "np_fftn_ms": np_fftn_ms,
        "sp_fftn_1_ms": sp_fftn_1_ms,
        "sp_fftn_all_ms": sp_fftn_all_ms,
        "np_ifftn_ms": np_ifftn_ms,
        "sp_ifftn_1_ms": sp_ifftn_1_ms,
        "sp_ifftn_all_ms": sp_ifftn_all_ms,
        "sym_ifftn_1_ms": sym_ifftn_1_ms,
        "sym_ifftn_all_ms": sym_ifftn_all_ms,
    }


def benchmark_octave_execution(n_jobs: int = 1) -> float:
    """Benchmark end-to-end Octave 1 execution on a representative volume."""
    rng = np.random.default_rng(123)
    volume_shape = (64, 128, 128)  # Z, Y, X
    synthetic_image = (rng.standard_normal(volume_shape) * 100 + 500).astype(np.float64)

    params = {
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 8.0,
        "scales_per_octave": 3.0,
        "microns_per_voxel": [1.0, 1.0, 2.0],
        "pixels_per_sigma_PSF": [1.0, 1.0, 1.0],
        "max_voxels": 262144,
        "energy_projection_mode": "matlab",
        "energy_sign": -1.0,
        "approximating_PSF": True,
        "gaussian_to_ideal_ratio": 1.0,
        "spherical_to_annular_ratio": 0.5,
    }
    config = _prepare_energy_config(synthetic_image, params)

    energy_3d = np.full(volume_shape, np.inf, dtype=np.float64)
    scale_indices = np.full(volume_shape, -1, dtype=np.int16)

    t0 = time.perf_counter()
    _ = compute_exact_parity_energy_single_octave(
        image=synthetic_image,
        config=config,
        current_octave=1,
        energy_3d=energy_3d,
        scale_indices=scale_indices,
        completed_chunk_units=0,
        total_chunk_units=1,
        n_jobs=n_jobs,
    )
    elapsed = time.perf_counter() - t0
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Energy FFT multi-threading acceleration.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of FFT benchmark iterations")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of worker jobs for octave benchmark")
    args = parser.parse_args()

    print("=================================================================")
    print("  PHASE 2 ENERGY FFT ACCELERATION BENCHMARK")
    print("=================================================================")
    print(f"Shape: (128, 128, 64) | Iterations: {args.iterations}")
    print("-----------------------------------------------------------------")

    stats = benchmark_bare_ffts(iterations=args.iterations)
    print("Forward FFT:")
    print(f"  NumPy fftn (single):        {stats['np_fftn_ms']:6.2f} ms")
    print(f"  SciPy fftn (1 worker):      {stats['sp_fftn_1_ms']:6.2f} ms ({stats['np_fftn_ms']/stats['sp_fftn_1_ms']:4.2f}x speedup)")
    print(f"  SciPy fftn (all cores):     {stats['sp_fftn_all_ms']:6.2f} ms ({stats['np_fftn_ms']/stats['sp_fftn_all_ms']:4.2f}x speedup)")
    print()
    print("Inverse FFT:")
    print(f"  NumPy ifftn (single):       {stats['np_ifftn_ms']:6.2f} ms")
    print(f"  SciPy ifftn (1 worker):     {stats['sp_ifftn_1_ms']:6.2f} ms ({stats['np_ifftn_ms']/stats['sp_ifftn_1_ms']:4.2f}x speedup)")
    print(f"  SciPy ifftn (all cores):    {stats['sp_ifftn_all_ms']:6.2f} ms ({stats['np_ifftn_ms']/stats['sp_ifftn_all_ms']:4.2f}x speedup)")
    print()
    print("MATLAB Symmetric Inverse FFT (_ifftn_matlab_symmetric):")
    print(f"  Pre-indexed (1 worker):     {stats['sym_ifftn_1_ms']:6.2f} ms")
    print(f"  Pre-indexed (all cores):    {stats['sym_ifftn_all_ms']:6.2f} ms ({stats['sym_ifftn_1_ms']/stats['sym_ifftn_all_ms']:4.2f}x speedup)")
    print("-----------------------------------------------------------------")

    print(f"Running Octave 1 End-to-End Execution (n_jobs={args.n_jobs})...")
    octave_time = benchmark_octave_execution(n_jobs=args.n_jobs)
    print(f"  Octave 1 elapsed time:      {octave_time:.2f} s")
    print("=================================================================")


if __name__ == "__main__":
    main()
