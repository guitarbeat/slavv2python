# ADR-0001: Use Python over MATLAB for Data Processing

**Status:** Accepted  
**Date:** 2026-01-27  
**Decision Makers:** SLAVV Development Team

---

## Context

The SLAVV vascular vectorization pipeline was originally implemented in MATLAB (`vectorize_V200`). We needed to evaluate whether to:
1. Continue maintaining the MATLAB implementation
2. Migrate to a Python implementation
3. Maintain both in parallel

Key considerations:
- Processing speed for 3D microscopy volumes (50-500M voxels)
- Development velocity and maintainability
- Dependency management and deployment
- Cross-platform compatibility

---

## Decision

**We will use Python as the primary implementation** for all SLAVV data processing modules. The MATLAB implementation will be retained in `legacy/` for reference and validation purposes only.

---

## Consequences

### Performance (Measured)

| Stage | MATLAB R2019a | Python | Speedup |
|-------|---------------|--------|---------|
| Energy Calculation | 59 min | 7.5 min | **7.9x** |
| Vertex Extraction | 30 sec | 20 sec | 1.5x |
| Edge Tracing | 97 sec | 20 sec | 4.9x |
| Network Construction | 7 sec | 5 sec | 1.4x |
| **Total** | **62.9 min** | **8.3 min** | **7.6x** |

*Test volume: 222×512×512 (58M voxels). MATLAB used 4 parallel workers; Python was single-threaded.*

### Benefits
- **7.6x faster** overall processing time
- NumPy/SciPy optimizations outperform MATLAB loops
- Easier dependency management (pip/conda vs MATLAB licenses)
- Better cross-platform support
- Modern type hints and testing infrastructure

### Trade-offs
- Must migrate remaining MATLAB-only features
- Loss of some MATLAB-specific optimizations
- Learning curve for MATLAB-experienced contributors

### Migration Path
1. MATLAB code remains in `legacy/Vectorization-Public/`
2. Python implementation in `source/slavv/`
3. Comparison framework validates equivalence
4. Gradual feature migration as needed

---

## References

- [Post-mortem: MATLAB vs Python Debugging](../archive/2026-01-27_post_mortem.md)
- [Comparison Framework Status](../archive/2026-01-27_comparison_framework.md)
