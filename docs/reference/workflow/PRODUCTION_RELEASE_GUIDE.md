# Production Release Guide

[Up: Reference Docs](../README.md)

This guide outlines the mandatory steps for promoting the SLAVV Python engine to a stable research release.

---

## 🛠️ Release Prerequisites

Before a version can be tagged or deployed, the following gates must be green:

### 1. Mathematical Parity (The Exact Proof)
The native-first exact route must pass the maintained parity certification.
- **Goal**: sequential **per-stage** `prove-exact --stage <s>` passing for energy, vertices, edges, and network against the defined certification oracle — strict zero missing/extra for energy/vertices ([ADR 0011](../../adr/0011-energy-float-certification-policy.md)); **evaluated** ADR 0012 ownership-map / multiset + sub-voxel bars for edges/network ([ADR 0012](../../adr/0012-edge-watershed-parity-bar.md)). Do not treat `prove-exact-sequence` strict-field failure as the release ship gate.
- **Audit**: per-stage `exact_proof_<stage>.json` with `adr0012_evaluated: true` for edges/network where applicable; promote durable summaries to `workspace/reports/`.

### 2. Quality Gate (CI/CD)
The GitHub Actions `regression-gate.yml` must pass for the `main` branch.
- **Linter**: `ruff` (zero errors).
- **Type Checker**: `mypy` (zero errors).
- **Tests**: `pytest` (100% pass rate, including synthetic integration tests).

### 3. Documentation Alignment
- [CHANGELOG.md](../../CHANGELOG.md) must be updated with all notable changes since the last version.
- `TODO.md`, `EXACT_PROOF_FINDINGS.md`, and the active spec must reflect the current state.
- `TUTORIAL.md` must be verified against the live CLI and app interface.

---

## 🚀 Release Process

### Step 1: Version Bump
Update the version number in `pyproject.toml`. Follow [Semantic Versioning](https://semver.org/).
```toml
[project]
name = "slavv_python"
version = "0.1.0" # Bump this
```

### Step 2: Final Quality Sweep
Run the full local suite to ensure no last-minute regressions.
```powershell
python -m ruff check slavv_python tests --fix
python -m mypy
python -m pytest tests/unit tests/integration/test_paper_profile_ci.py
```

### Step 3: Certification Promotion
Promote the latest parity experiment report to the authoritative certification record.
```powershell
# Example: Promotes trial v32 to the baseline proof summary
Copy-Item workspace/runs/v32_final/03_Analysis/exact_proof.json workspace/reports/CERTIFICATION_V0.1.0.json
```

### Step 4: Tag and Commit
Tag the release in the git repository.
```powershell
git add pyproject.toml docs/CHANGELOG.md workspace/reports/
git commit -m "Release v0.1.0: Final bit-accurate parity baseline"
git tag -a v0.1.0 -m "Vascular vectorization engine stable beta"
```

---

## 📦 Deployment

### Standalone Installation
Users can install the release directly from the repository:
```powershell
pip install git+https://github.com/UTFOIL/slavv2python.git@v0.1.0
```

### Research Deployment
For local research servers, use the `slavv-app` (Streamlit) for shared access to the vectorization dashboard.
