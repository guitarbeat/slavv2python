# Progress - Vertices Parity Explorer

Last visited: 2026-06-08T22:45:00Z

- **Setup**: Verified environment and read `SCOPE.md`.
- **Discovery**: 
  - Ran parity experiments and tracked down the execution pipeline.
  - Investigated the "Lowest Linear Index Priority" rule.
  - Mapped Python (`detection.py`, `results.py`) to MATLAB (`get_vertices_V200.m`, `vectorize_V200.m`).
  - Found that MATLAB relies on chunk iteration order (Y-fastest via `ind2sub`) and stable sorting for inter-chunk tie-breaking.
  - Found that Python's `iter_overlapping_chunks` uses Z-fastest iteration, breaking exact parity for identical-energy vertices across chunks.
- **Reporting**: Drafted fix strategy and evidence in `handoff.md`.
- **Handoff**: Sent final message to main agent.
