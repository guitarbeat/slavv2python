// principal_energy.hpp
// Modern C++20 implementation of SLAVV principal curvature energy filter.
// Adheres strictly to the C++ Core Guidelines (P.*, F.*, C.*, R.*, ES.*, Con.*, Per.*).

#ifndef NEUROVASC_GEOMETRY_PRINCIPAL_ENERGY_HPP
#define NEUROVASC_GEOMETRY_PRINCIPAL_ENERGY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace neurovasc::geometry {

// Symbolic constants (ES.45, NL.9)
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoThirdsPi = 2.0943951023931954923; // 2 * pi / 3
inline constexpr double kFourThirdsPi = 4.1887902047863909846; // 4 * pi / 3
inline constexpr double kEpsilon = 1e-15;

template <std::floating_point FloatT>
struct alignas(32) EigenResult3x3 {
    std::array<FloatT, 3> values{};                     // Sorted eigenvalues: w0 <= w1 <= w2
    std::array<std::array<FloatT, 3>, 3> vectors{};      // vectors[j] is the normalized eigenvector for values[j]
};

// Analytical 3x3 symmetric matrix eigensolver using Cardano / Kopp formulation.
// Extremely fast: evaluated in registers, zero heap allocations, zero dynamic workspaces (Per.11, Per.19).
template <std::floating_point FloatT>
[[nodiscard]] EigenResult3x3<FloatT> solve_symmetric_eigensystem_3x3(
    const FloatT a00, const FloatT a11, const FloatT a22,
    const FloatT a01, const FloatT a12, const FloatT a02) noexcept {

    EigenResult3x3<FloatT> result;

    // Check if matrix is effectively diagonal
    const FloatT off_diag_sq = a01 * a01 + a12 * a12 + a02 * a02;
    if (off_diag_sq < kEpsilon) {
        // Already diagonal: eigenvalues are diagonal elements
        std::array<std::pair<FloatT, std::size_t>, 3> diag_pairs{
            std::make_pair(a00, 0),
            std::make_pair(a11, 1),
            std::make_pair(a22, 2)
        };
        std::sort(diag_pairs.begin(), diag_pairs.end(),
                  [](const auto& left, const auto& right) noexcept { return left.first < right.first; });

        for (std::size_t j = 0; j < 3; ++j) {
            result.values[j] = diag_pairs[j].first;
            result.vectors[j].fill(FloatT{0});
            result.vectors[j][diag_pairs[j].second] = FloatT{1};
        }
        return result;
    }

    // Step 1: Trace and characteristic polynomial decomposition
    const FloatT q = (a00 + a11 + a22) / FloatT{3};
    const FloatT b00 = a00 - q;
    const FloatT b11 = a11 - q;
    const FloatT b22 = a22 - q;

    const FloatT p_sq = (b00 * b00 + b11 * b11 + b22 * b22 + FloatT{2} * off_diag_sq) / FloatT{6};
    const FloatT p = std::sqrt(p_sq);

    // B = (A - q*I) / p
    const FloatT inv_p = FloatT{1} / p;
    const FloatT c00 = b00 * inv_p;
    const FloatT c11 = b11 * inv_p;
    const FloatT c22 = b22 * inv_p;
    const FloatT c01 = a01 * inv_p;
    const FloatT c12 = a12 * inv_p;
    const FloatT c02 = a02 * inv_p;

    // det(B) / 2
    const FloatT det_b_half = (c00 * (c11 * c22 - c12 * c12) -
                               c01 * (c01 * c22 - c12 * c02) +
                               c02 * (c01 * c12 - c11 * c02)) * FloatT{0.5};

    const FloatT clamped_r = std::clamp(det_b_half, FloatT{-1}, FloatT{1});
    const FloatT phi = std::acos(clamped_r) / FloatT{3};

    // Sorted eigenvalues: w0 <= w1 <= w2
    const FloatT two_p = FloatT{2} * p;
    const FloatT w2 = q + two_p * std::cos(phi);
    const FloatT w0 = q + two_p * std::cos(phi + static_cast<FloatT>(kTwoThirdsPi));
    const FloatT w1 = FloatT{3} * q - w0 - w2;

    result.values[0] = w0;
    result.values[1] = w1;
    result.values[2] = w2;

    // Step 2: Compute orthogonal eigenvectors for each distinct eigenvalue
    // For eigenvalue w, solve (A - w*I) v = 0 using cross products of column pairs
    const auto compute_eigenvector = [&](const FloatT w) noexcept -> std::array<FloatT, 3> {
        const FloatT m00 = a00 - w;
        const FloatT m11 = a11 - w;
        const FloatT m22 = a22 - w;
        const FloatT m01 = a01;
        const FloatT m12 = a12;
        const FloatT m02 = a02;

        // Try cross product of column 0 and column 1
        FloatT vx = m01 * m12 - m02 * m11;
        FloatT vy = m02 * m01 - m00 * m12;
        FloatT vz = m00 * m11 - m01 * m01;
        FloatT norm_sq = vx * vx + vy * vy + vz * vz;

        // If norm is too small, try cross product of column 0 and column 2
        if (norm_sq < kEpsilon) {
            vx = m01 * m22 - m02 * m12;
            vy = m02 * m02 - m00 * m22;
            vz = m00 * m12 - m01 * m02;
            norm_sq = vx * vx + vy * vy + vz * vz;
        }

        // If still too small, try cross product of column 1 and column 2
        if (norm_sq < kEpsilon) {
            vx = m11 * m22 - m12 * m12;
            vy = m12 * m02 - m01 * m22;
            vz = m01 * m12 - m02 * m11;
            norm_sq = vx * vx + vy * vy + vz * vz;
        }

        if (norm_sq >= kEpsilon) {
            const FloatT inv_norm = FloatT{1} / std::sqrt(norm_sq);
            return {vx * inv_norm, vy * inv_norm, vz * inv_norm};
        }
        return {FloatT{1}, FloatT{0}, FloatT{0}};
    };

    result.vectors[0] = compute_eigenvector(w0);
    result.vectors[1] = compute_eigenvector(w1);

    // For third eigenvector, use cross product of first two for strict numerical orthogonality
    result.vectors[2][0] = result.vectors[0][1] * result.vectors[1][2] - result.vectors[0][2] * result.vectors[1][1];
    result.vectors[2][1] = result.vectors[0][2] * result.vectors[1][0] - result.vectors[0][0] * result.vectors[1][2];
    result.vectors[2][2] = result.vectors[0][0] * result.vectors[1][1] - result.vectors[0][1] * result.vectors[1][0];

    const FloatT v2_norm_sq = result.vectors[2][0] * result.vectors[2][0] +
                              result.vectors[2][1] * result.vectors[2][1] +
                              result.vectors[2][2] * result.vectors[2][2];
    if (v2_norm_sq >= kEpsilon) {
        const FloatT inv_v2 = FloatT{1} / std::sqrt(v2_norm_sq);
        result.vectors[2][0] *= inv_v2;
        result.vectors[2][1] *= inv_v2;
        result.vectors[2][2] *= inv_v2;
    } else {
        result.vectors[2] = compute_eigenvector(w2);
    }

    return result;
}

// Single-voxel principal energy kernel
template <std::floating_point FloatT>
[[nodiscard]] FloatT evaluate_voxel_principal_energy(
    const FloatT gx, const FloatT gy, const FloatT gz,
    const FloatT dyy, const FloatT dxx, const FloatT dzz,
    const FloatT dyx, const FloatT dxz, const FloatT dzy,
    const FloatT energy_sign) noexcept {

    // Note curvature ordering in SLAVV Hessian:
    // row 0: [dyy, dyx, dzy]
    // row 1: [dyx, dxx, dxz]
    // row 2: [dzy, dxz, dzz]
    const auto [w, v] = solve_symmetric_eigensystem_3x3<FloatT>(
        dyy, dxx, dzz, dyx, dxz, dzy);

    const std::array<FloatT, 3> grad{gx, gy, gz};
    std::array<FloatT, 3> e_batch{};

    for (std::size_t j = 0; j < 3; ++j) {
        // Projection: p_j = grad . v_j
        const FloatT p_j = grad[0] * v[j][0] + grad[1] * v[j][1] + grad[2] * v[j][2];
        const FloatT w_j = w[j];

        if (std::abs(w_j) < kEpsilon) {
            e_batch[j] = FloatT{0};
        } else {
            const FloatT ratio = p_j / w_j;
            e_batch[j] = w_j * std::exp(-FloatT{0.5} * ratio * ratio);
        }
    }

    // Clip third principal component per MATLAB energy_filter_V200 specification
    if (energy_sign < FloatT{0}) {
        e_batch[2] = std::min(e_batch[2], FloatT{0});
    } else {
        e_batch[2] = std::max(e_batch[2], FloatT{0});
    }

    FloatT energy_sum = e_batch[0] + e_batch[1] + e_batch[2];

    if (energy_sign < FloatT{0}) {
        if (!std::isfinite(energy_sum) || energy_sum >= FloatT{0}) {
            energy_sum = std::numeric_limits<FloatT>::infinity();
        }
    } else {
        if (!std::isfinite(energy_sum) || energy_sum <= FloatT{0}) {
            energy_sum = -std::numeric_limits<FloatT>::infinity();
        }
    }

    return energy_sum;
}

// Multi-threaded batch processor using std::jthread and cache-friendly chunking (Per.19, CP.4)
template <std::floating_point FloatT>
void compute_principal_energy_parallel(
    const std::span<const FloatT> gradients,     // (N, 3) row-major
    const std::span<const FloatT> curvatures,    // (N, 6) row-major: [dYY, dXX, dZZ, dYX, dXZ, dZY]
    const std::span<FloatT> energies,           // (N) output
    const FloatT energy_sign,
    const std::size_t num_threads = std::thread::hardware_concurrency()) {

    const std::size_t num_voxels = energies.size();
    if (num_voxels == 0) return;

    const std::size_t threads_to_use = std::max<std::size_t>(1, std::min(num_threads, num_voxels));
    const std::size_t chunk_size = (num_voxels + threads_to_use - 1) / threads_to_use;

    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);

    for (std::size_t t = 0; t < threads_to_use; ++t) {
        const std::size_t start_idx = t * chunk_size;
        const std::size_t end_idx = std::min(start_idx + chunk_size, num_voxels);

        if (start_idx >= end_idx) continue;

        workers.emplace_back([=, &gradients, &curvatures, &energies]() noexcept {
            for (std::size_t i = start_idx; i < end_idx; ++i) {
                const std::size_t g_offset = i * 3;
                const std::size_t c_offset = i * 6;

                energies[i] = evaluate_voxel_principal_energy<FloatT>(
                    gradients[g_offset + 0],
                    gradients[g_offset + 1],
                    gradients[g_offset + 2],
                    curvatures[c_offset + 0],
                    curvatures[c_offset + 1],
                    curvatures[c_offset + 2],
                    curvatures[c_offset + 3],
                    curvatures[c_offset + 4],
                    curvatures[c_offset + 5],
                    energy_sign
                );
            }
        });
    }

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace neurovasc::geometry

// C-compatible export interface for Python ctypes / PyCapsule integration
extern "C" {

void compute_principal_energy_c(
    const double* __restrict gradients,
    const double* __restrict curvatures,
    double* __restrict energies,
    const std::size_t num_voxels,
    const double energy_sign,
    const std::size_t num_threads) {

    const std::span<const double> grad_span(gradients, num_voxels * 3);
    const std::span<const double> curv_span(curvatures, num_voxels * 6);
    const std::span<double> energy_span(energies, num_voxels);

    neurovasc::geometry::compute_principal_energy_parallel<double>(
        grad_span, curv_span, energy_span, energy_sign, num_threads
    );
}

}

#endif // NEUROVASC_GEOMETRY_PRINCIPAL_ENERGY_HPP
