// principal_energy.cpp
// Implementation file compiling the C++20 principal energy kernel.

#include "principal_energy.hpp"

// Explicit instantiation to ensure symbol emission
namespace neurovasc::geometry {
template void compute_principal_energy_parallel<double>(
    const std::span<const double>,
    const std::span<const double>,
    const std::span<double>,
    const double,
    const std::size_t
);
}
