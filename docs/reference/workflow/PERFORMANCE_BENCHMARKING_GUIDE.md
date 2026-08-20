# SLAVV Performance Benchmarking Guide

[Up: Reference Docs](../README.md)

## In short

How to time the pipeline without silently breaking MATLAB match. Speed work is
Phase 2. Do not unwind Fortran-order tie-breaks without an explicit Phase 2 ADR.

This guide defines the methodology and tools for measuring and optimizing the performance of the SLAVV Python implementation.

---

## 📊 Performance Metrics

Use these core metrics when evaluating changes to the pipeline:

1.  **Throughput (Candidates/sec)**: The rate of edge candidate generation during the watershed discovery phase.
2.  **Peak Memory (MB)**: The maximum resident set size (RSS) during a full run, critical for large microscopy volumes.
3.  **Initialization Latency (sec)**: Time to build spatial maps and LUTs before discovery begins.
4.  **Tracing Efficiency (ms/point)**: Average time to trace one point along a geodesic path.

---

## 🛠️ Profiling Tools

### 1. CPU Profiling (cProfile)
Identify bottlenecks in the watershed loop or energy calculations.
```powershell
python -m cProfile -s cumulative -m slavv_python.analytics.parity launch-exact-run ... > profile.txt
```

### 2. Memory Profiling
Track memory consumption line-by-line or at specific checkpoints.

#### Manual Checkpoints
Use the internal helper to record RSS memory at key points in the algorithm:
```python
from slavv_python.utils.profiling import get_process_memory_usage

print(f"Memory before expansion: {get_process_memory_usage():.2f} MB")
# ... expansion logic ...
print(f"Memory after expansion: {get_process_memory_usage():.2f} MB")
```

#### Line-by-Line (External Tool)
If `memory_profiler` is installed in your local environment, you can use the `@profile` decorator:

### 3. Execution Heartbeats
The watershed discovery emits progress heartbeats every 512 iterations or 5 seconds. Use these to monitor live performance during long runs.

---

## ⚠️ Profiling & Resume Gotchas

When resuming an `exact-route` pipeline for targeted profiling (e.g., picking up from cached Energy/Vertices checkpoints), you may encounter two primary roadblocks:

1. **Missing Dataset Tiffs**: Older canonical runs (like `v18`) often have their raw `.tif` files removed from `00_Refs` to save disk space. If `resume-exact-run` fails with a missing dataset error, you must manually provide the paths to the original dataset and oracle using `--dataset-root` and `--oracle-root`.
2. **Parameter Fingerprint Blocks**: SLAVV's `RunContext` enforces strict parameter fingerprinting to prevent semantic drift. If you attempt to resume a pipeline and hit a `RuntimeError: Resume blocked because the parameters fingerprint changed`, you can temporarily bypass this by:
   - Modifying `slavv_python/engine/state/run_ledger.py` -> `ensure_resume_allowed`
   - Commenting out `raise RuntimeError(message)` and returning early.
   - **CRITICAL**: Always revert this bypass immediately after launching your profiling run to maintain repository integrity.

---

## 🚀 Optimization Workflow

### 1. Establish Baseline
Run a parity experiment on the `180709_E` dataset and record the metrics from the summary report.

### 2. Isolate Component
Use unit tests in `tests/unit/pipeline/test_global_watershed_comprehensive.py` with larger synthetic volumes (e.g., 64x64x64) to profile specific functions.

### 3. Optimize with Parity Guard
Ensure that optimizations do not change functional semantics.
- **Rule**: If an optimization changes the resulting `pointer_map` or `edge_pairs` by even one voxel/pair, it must be demoted from the exact route.
- **Preferred**: Use Numba-compatible array operations and maintain F-contiguity to improve speed without altering logic.

### 4. Verify Improvement
Rerun the baseline experiment and compare metrics. Document the delta in the pull request.

---

## 📚 Reference Baselines (May 2026) (unverified; provenance not captured)

*Baseline measurements taken on standard developer workstation (32GB RAM, i7 CPU).*

- **Dataset**: `180709_E` (center crop)
- **Throughput**: ~85 candidates/sec
- **Peak Memory**: ~1.2 GB
- **Total Edge Discovery**: ~12.5 seconds
