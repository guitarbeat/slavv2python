# Journal Submission Cover Letter

**Date:** October 26, 2026  
**To:** Editorial Board, *Journal of Open Research Software* (JORS) / *IEEE Computing in Science & Engineering* (CiSE)  
**From:** The SLAVV Translation & Development Team  
**Subject:** Submission of Original Research / Software Metapaper: *"MATLAB to Python Scientific Translation: A Study in Exact Parity"*

Dear Editors,

We are pleased to submit our manuscript titled **"MATLAB to Python Scientific Translation: A Study in Exact Parity"** for consideration as an original research article / software metapaper in your journal.

### Background & Research Significance
Modern computational biomedicine relies heavily on legacy scientific codebases written in proprietary environments such as MATLAB. In high-stakes applications—such as 3D vascular network vectorization from multi-scale optical microscopy scans—subtle numerical divergences in floating-point math, boundary conditions, or tie-breaking heuristics can alter extracted vascular graph topologies, directly impacting downstream hemodynamic simulations and clinical morphometry.

Migrating these pipelines to open-source languages (Python) typically results in "approximate" conversions that discard ground-truth mathematical equivalence. In this work, we present the design, mathematical framework, and empirical verification of **SLAVV2Python**, achieving certified **1:1 structural and topological exact parity** against legacy MATLAB oracles across large volumetric datasets ($512 \times 512 \times 64$ voxels, over 16.8 million voxels).

### Key Contributions & Innovations
1. **Formal Discrete-Continuous Certification Policy (ADR 0011 / ADR 0012):** We formalize an actionable verification framework that enforces zero-tolerance discrete invariance (exact vertex and edge multisets) while bounding continuous float divergences under calibrated relative tolerances ($10^{-7}$).
2. **Reverse Engineering of Closed-Source Built-Ins:** Using the REA framework and ARM64 disassembly, we provide the first instruction-level analysis of closed MATLAB built-in binaries (`libmwimagescore`, `libmwmorphop_flat`, `libmwimfilter`), uncovering the exact micro-architectural invariants (Meyer bucket priority queues, column-major linear tie-breaks, 1D structuring element decompositions) required for exact reproduction.
3. **Ten Parity-Preserving Performance Innovations:** We present ten architectural innovations—including online 4D scale accumulation ($30\times$ RAM reduction), batched $3 \times 3$ LAPACK eigensolvers ($>20\times$ speedup), and JIT-compiled geometric penalties—that reduce full-volume pipeline runtime from over 50 hours to under 3 hours on standard workstation hardware.
4. **Cross-Version & Cross-Architecture Generalization:** We demonstrate verified portability across major platform and runtime transitions, proving identical topological output between MATLAB R2019a on x86_64 Windows and MATLAB R2024b on ARM64 Apple Silicon.

### Open Science & Software Availability
The software, verification test suites, benchmark datasets, and ground-truth architecture specifications are fully open-source and version-controlled under the MIT License. Complete replication instructions and automated test suites (`pytest`, `ruff`) accompany this submission.

We confirm that this manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. All authors have approved the submission.

Thank you for your consideration of our work.

Sincerely,

The SLAVV Translation Team  
*NeuroVASC Project*  
