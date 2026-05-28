# Production Release Guide

[Up: Reference Docs](../README.md)

This guide outlines the mandatory steps for promoting the SLAVV Python engine to a stable research release.

---

## 🛠️ Release Prerequisites

Before a version can be tagged or deployed, the following gates must be green:

### 1. Mathematical Parity (The Exact Proof)
The native-first exact route must pass the maintained parity certification.
- **Goal**: >95% match rate on `edges.connections` against the `180709_E` oracle.
- **Audit**: `slavv prove-exact --stage all` must produce a compliant report in `workspace/reports/`.

### 2. Quality Gate (CI/CD)
The GitHub Actions `regression-gate.yml` must pass for the `main` branch.
- **Linter**: `ruff` (zero errors).
- **Type Checker**: `mypy` (zero errors).
- **Tests**: `pytest` (100% pass rate, including synthetic integration tests).

### 3. Documentation Alignment
- [CHANGELOG.md](../../CHANGELOG.md) must be updated with all notable changes since the last version.
- `ROADMAP.md` priorities must reflect the current state.
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
# Example: Promotes trial v32 to the baseline report
Copy-Item workspace/runs/v32_final/report.json workspace/reports/CERTIFICATION_V0.1.0.json
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
pip install git+https://github.com/google/slavv2python.git@v0.1.0
```

### Research Deployment
For local research servers, use the `slavv-app` (Streamlit) for shared access to the vectorization dashboard.
