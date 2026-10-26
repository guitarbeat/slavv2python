# Manuscript Submission Metadata & Author Checklist

**Title:** MATLAB to Python Scientific Translation: A Study in Exact Parity  
**Target Journals:** 
1. *Journal of Open Research Software* (JORS) (Primary)
2. *SoftwareX* (Elsevier) (Alternative)
3. *IEEE Computing in Science & Engineering* (CiSE) (Alternative)

---

## 1. Article Classification & Metadata

- **Article Type:** Original Research Paper / Software Metapaper
- **Primary Subject Area:** Scientific Software Engineering, Computational Biology, Biomedical Image Analysis
- **Keywords:** Scientific software translation; Exact numerical parity; Topological invariants; 3D vascular vectorization; MATLAB to Python; High-performance computing; Binary reverse engineering

---

## 2. Abstract (Structured for Submission)

**Background:** Translating legacy scientific computing pipelines from proprietary environments (MATLAB) to open-source ecosystems (Python) often encounters subtle arithmetic and architectural divergences that compound nonlinearly in multi-stage pipelines. In volumetric biomedical image vectorization, these divergences alter graph topology and clinical morphometry.

**Methods:** We design and implement the SLAVV translation framework, enforcing a formal discrete-continuous boundary certification policy (ADR 0011 / ADR 0012). We decompile closed-source MATLAB built-in libraries (`libmwimagescore`, `libmwmorphop_flat`, `libmwimfilter`) using binary disassembly to uncover low-level tie-breaking and floating-point accumulation rules. Ten parity-preserving performance innovations are engineered, including online 4D tensor elimination, batched LAPACK Hessian eigensolvers, and JIT-compiled watershed geometric propagation.

**Results:** On a full-scale optical scan ($512 \times 512 \times 64$ voxels), the Python pipeline achieves certified 1:1 structural exact parity: zero scale mismatches across 16.8 million voxels, 100% exact array equality on 69,500 vascular connections, and exact multiset isomorphism on 25,371 bifurcation nodes. Execution time on standard workstations decreases from $>50$ hours serial to $<3$ hours with parallel chunking ($17.1\times$ speedup on Apple Silicon), replicating identically across MATLAB R2019a/x86_64 and R2024b/ARM64.

**Conclusions:** Certified exact parity provides an empirical methodology for migrating legacy scientific codebases without sacrificing mathematical fidelity or computational scalability.

---

## 3. Software & Repository Availability

- **Repository:** `https://github.com/guitarbeat/slavv2python`
- **Programming Language:** Python 3.10–3.13 (PyPI package: `slavv-python`)
- **Core Dependencies:** `numpy`, `scipy`, `numba`, `joblib`, `tifffile`, `h5py`, `matplotlib`
- **License:** MIT License
- **Operating Systems Supported:** macOS (ARM64 & x86_64), Linux (x86_64), Windows (x86_64)
- **Continuous Integration & Testing:** Automated `pytest` test suites (unit, integration, and parity verification probes)

---

## 4. Suggested Peer Reviewers

1. **Dr. Stéfan van der Walt** (UC Berkeley / scikit-image Core Team)
   - *Expertise:* Scientific Python ecosystem, image processing algorithms, numerical reproducibility across programming languages.
2. **Dr. Ralf Gommers** (Quansight / SciPy Core Team)
   - *Expertise:* Scientific computing architecture, BLAS/LAPACK performance, array API standards, array-order transitions.
3. **Dr. Alejandro F. Frangi** (University of Manchester / Turing Institute)
   - *Expertise:* Multi-scale vessel enhancement filtering, biomedical image vectorization, vascular network topology.
4. **Dr. Richard F. Lyon** (Google Research / CARFAC Port Project)
   - *Expertise:* Golden-master translation methodology, bit-exact and certified numerical porting between MATLAB and Python.
