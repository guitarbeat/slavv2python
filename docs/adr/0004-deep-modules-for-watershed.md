# ADR 0004: Deep Modules for Watershed Discovery

**Date**: 2026-05-26
**Status**: Accepted

## Context

The `_generate_edge_candidates_matlab_global_watershed` function implemented the frontier expansion logic as a 200+ line `while` loop that directly mutated 8 flat `numpy` arrays (`vertex_index_map`, `pointer_map`, `energy_map_temp`, etc.) and a native Python list (`available_locations`). 

This architecture was extremely **shallow** and had poor **locality**:
1. The rules for LIFO/FIFO tie-breaking and Linear Index priority were implemented as inline binary searches within the loop.
2. The rules for voxel claiming were implemented as inline boolean array masking.
3. This tangling caused hours of friction when attempting to track down a remaining 11% divergence in exact bit-parity with the MATLAB oracle, because it was impossible to isolate priority bugs from state-mutation bugs.

## Decision

We will deepen the watershed architecture by extracting two highly-focused modules:

1. **`FrontierQueue` (Priority Seam)**: A module solely responsible for enforcing MATLAB's exact queue priority rules (Energy -> Linear Index -> LIFO/FIFO). It encapsulates the binary search and list mutation.
2. **`VoxelClaimMap` (State Seam)**: A module that encapsulates the 8 flat numpy arrays and provides an atomic `claim_unowned_strel` interface. 

The Orchestration loop will be refactored to act as a pure coordinator between these modules and the `ExpansionPolicy`.

## Consequences

*   **Positive (Leverage)**: Callers (the orchestrator) no longer need to know how tie-breaking works or which of the 8 arrays need updating. They express intent (`queue.push()`, `map.claim_voxels()`).
*   **Positive (Locality/Testability)**: The LIFO/FIFO rules and the claim overwriting logic can be unit-tested in isolation, accelerating the final closure of the parity gap.
*   **Negative (Overhead)**: We introduce object-oriented method call overhead into a tight numerical loop. If performance degrades unacceptably, we will use Cython or Numba for the deep modules, rather than reverting to flat arrays.
