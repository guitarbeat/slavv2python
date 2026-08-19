# Quick Reference Card

[Up: Documentation Index](README.md)

## In short (plain English)

Phase 1 already matches MATLAB closely enough to **ship**. Live numbers live only in [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

**100% / stretch** is a harder extra goal: every Energy number identical bits, not just “close.” On the real crop, about 90% of voxels match exactly; leftover diffs are last digits (`1e-10`). Tiny cut-outs that match as their own photo do **not** solve the crop (full crop uses overlapping tiles with extra border).

Do **not** rerun the crop Energy writer. Readable leftover: [crop-energy-stretch-float-isolation.md](solutions/parity/crop-energy-stretch-float-isolation.md).

| Everyday word | What we mean |
|---------------|----------------|
| **Oracle** | Saved MATLAB answers for that photo |
| **Crop** | Small real cut-out of the big photo |
| **Tile / chunk** | Overlapping piece processed with extra border, then the center is kept |
| **Close enough (`allclose`)** | Ship bar — not identical last digits |
| **Stretch / 100%** | Identical last digits under `--strict-floats` |

---

## 🚀 Quick Start

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[app,workspace]"

# Run pipeline
slavv run -i volume.tif -o output --export csv json

# Monitor jobs
slavv jobs list
slavv jobs history

# Quality checks
python -m pytest -m "unit or integration"
python -m ruff format slavv_python tests
python -m ruff check slavv_python tests --fix
python -m mypy
```

---

## 📚 Where to Find Things

| I need to... | Go to... |
|--------------|----------|
| 🆕 Brand-new to the repo | [NEW_ENGINEER_START_HERE.md](reference/core/NEW_ENGINEER_START_HERE.md) ⭐ |
| 🤖 Start AI work | [AGENTS.md](../AGENTS.md) ⭐ → [TODO.md](TODO.md) |
| 🔬 Check parity status | [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md) ⭐ |
| 🧭 Who owns which doc | [docs/README.md authority map](README.md#documentation-authority-map-one-concept--one-home) (+ deprecated table) |
| 🧪 Run parity tests | [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md) |
| 📊 Monitor long jobs | [PARITY_JOB_MONITORING.md](reference/workflow/PARITY_JOB_MONITORING.md) |
| 📖 Understand term | [AGENTS.md § Glossary](../AGENTS.md#domain-glossary) (Paper Path vs publication; Edge Set / Candidate Set) |
| 📄 Which paper? | [papers/README.md](reference/papers/README.md) |
| 🛠️ Find task | [TODO.md](TODO.md) |
| 🏗️ Understand arch | [TECHNICAL_ARCHITECTURE.md](reference/core/TECHNICAL_ARCHITECTURE.md) |
| 📝 Coding style | [PYTHON_NAMING_GUIDE.md](reference/workflow/PYTHON_NAMING_GUIDE.md) |
| 🧩 Place tests | [tests/README.md](../tests/README.md) |

---

## 🔬 Parity Workflow (Quick)

The full cold-start protocol and command list live in one place — do not
duplicate them here.

**Start here:** [docs/README.md § Parity Closure Fast Path](README.md#-parity-closure-fast-path)
→ [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) (live status)
→ [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md) (commands)

---

## 🛠️ Common Commands

### Pipeline
```powershell
slavv run -i volume.tif -o output --export csv json
slavv run -i volume.tif -o output --profile matlab_compat
slavv analyze -i output/network.json
slavv plot -i output/network.json -o plots.html
slavv-app  # Launch Streamlit
```

### Monitoring
```powershell
slavv jobs list                              # Active jobs
slavv jobs history --run-dir workspace/runs  # Job history
slavv jobs kill <job-id>                     # Kill job
slavv jobs daemon status                     # Daemon status
slavv monitor --run-dir <run_root>          # Watch run
```

### Quality
```powershell
python -m ruff format slavv_python tests     # Format
python -m ruff check slavv_python tests --fix  # Lint
python -m mypy                               # Type check
python -m pytest tests/                      # All tests
python -m pytest -m "unit or integration"    # Fast tests
```

### Git Workflow
```powershell
git add .
git commit -m "type: description"
git push origin main
```

---

## 📖 Domain Terms (Essential)

| Term | Definition |
|------|------------|
| **Pipeline** | Energy → Vertices → Edges → Network stages |
| **Oracle** | Saved MATLAB answers for that photo |
| **Parity Run** | Python run compared against those saved answers |
| **Certification** | Close enough to ship: Energy/Vertices exact discrete + `np.allclose` floats (ADR 0011). Edges: who-owns-which-voxel map (ADR 0012). Network: matching strand/junction bags (ADR 0012). Not identical last digits. |
| **Canonical Volume** | Full `180709_E` volume for Phase 1 cert |
| **Crop Harness** | `180709_E_crop_M` subvolume for faster testing |
| **Vertex** | Point of interest with 3D position and radius |
| **Edge Discovery** | Finding connectivity between vertices |
| **Checkpoint** | Intermediate state for resumable runs |

**Full glossary:** [AGENTS.md § Domain Glossary](../AGENTS.md#domain-glossary)

---

## 🎯 Decision Tree (Simplified)

```
What are you working on?

├─ 🔬 MATLAB Parity
│  └─ Read: EXACT_PROOF_FINDINGS.md → PARITY_PRE_GATE.md
│
├─ 🐛 Bug Fix
│  └─ Read impacted module → Check PYTHON_NAMING_GUIDE.md
│
├─ ✨ New Feature
│  └─ Check TODO.md → Follow PYTHON_NAMING_GUIDE.md
│
├─ 🤖 AI Agent Setup
│  └─ Read: AGENTS.md → Follow decision tree
│
└─ 📚 Exploring Codebase
   └─ Start: TECHNICAL_ARCHITECTURE.md → Repository Map in AGENTS.md
```

**Full decision tree:** [AGENTS.md § Work Decision Tree](../AGENTS.md#-work-decision-tree)

---

## ⚠️ Common Pitfalls

### Documentation

❌ Don't duplicate status in TODO.md and EXACT_PROOF_FINDINGS.md  
✅ Tasks → TODO.md, Status → EXACT_PROOF_FINDINGS.md

❌ Don't create files >1000 lines  
✅ Break into focused modules

❌ Don't put brainstorm + spec in plans/  
✅ brainstorms/ only before spec exists

### Code

❌ Don't use `print()` in library code  
✅ Use `logging`

❌ Don't approximate in parity code  
✅ Exact MATLAB reproduction required

❌ Don't mix test types in same file  
✅ Follow tests/README.md placement guide

### Parity

❌ Don't start writers on active run directories  
✅ Check `slavv jobs list` first

❌ Don't skip preflight on long runs  
✅ Run `preflight-exact` before 4+ hour jobs

❌ Don't call Phase 1 “100%” or reopen it for last-digit Energy diffs  
✅ Ship bar is close enough (`allclose`); identical last digits is stretch, still open on crop Energy

---

**Last Updated**: 2026-08-17  
**Related**: [README.md](README.md), [AGENTS.md](../AGENTS.md)
