# SLAVV Performance Benchmarking Guide

[Up: Reference Docs](../README.md)

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
python -m cProfile -s cumulative dev/scripts/cli/parity_experiment.py ... > profile.txt
```

### 2. Memory Profiling
Track memory consumption line-by-line or at specific checkpoints.

#### Manual Checkpoints
Use the internal helper to record RSS memory at key points in the algorithm:
```python
from source.utils.profiling import get_process_memory_usage

print(f"Memory before expansion: {get_process_memory_usage():.2f} MB")
# ... expansion logic ...
print(f"Memory after expansion: {get_process_memory_usage():.2f} MB")
```

#### Line-by-Line (External Tool)
If `memory_profiler` is installed in your local environment, you can use the `@profile` decorator:

### 3. Execution Heartbeats
The watershed discovery emits progress heartbeats every 512 iterations or 5 seconds. Use these to monitor live performance during long runs.

---

## 🚀 Optimization Workflow

### 1. Establish Baseline
Run a parity experiment on the `180709_E` dataset and record the metrics from the summary report.

### 2. Isolate Component
Use unit tests in `dev/tests/unit/core/test_global_watershed_comprehensive.py` with larger synthetic volumes (e.g., 64x64x64) to profile specific functions.

### 3. Optimize with Parity Guard
Ensure that optimizations do not change functional semantics.
- **Rule**: If an optimization changes the resulting `pointer_map` or `edge_pairs` by even one voxel/pair, it must be demoted from the exact route.
- **Preferred**: Use Numba-compatible array operations and maintain F-contiguity to improve speed without altering logic.

### 4. Verify Improvement
Rerun the baseline experiment and compare metrics. Document the delta in the pull request.

---

## 📚 Reference Baselines (May 2026)

*Baseline measurements taken on standard developer workstation (32GB RAM, i7 CPU).*

- **Dataset**: `180709_E` (center crop)
- **Throughput**: ~85 candidates/sec
- **Peak Memory**: ~1.2 GB
- **Total Edge Discovery**: ~12.5 seconds
