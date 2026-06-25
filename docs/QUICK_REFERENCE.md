# Quick Reference Card

[Up: Documentation Index](README.md)

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
| 🤖 Start AI work | [AGENTS.md](../AGENTS.md) ⭐ → [TODO.md](TODO.md) |
| 🔬 Check parity status | [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md) ⭐ |
| 🧪 Run parity tests | [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md) |
| 📊 Monitor long jobs | [PARITY_JOB_MONITORING.md](reference/workflow/PARITY_JOB_MONITORING.md) |
| 📖 Understand term | [AGENTS.md § Glossary](../AGENTS.md#domain-glossary) |
| 🛠️ Find task | [TODO.md](TODO.md) |
| 🏗️ Understand arch | [TECHNICAL_ARCHITECTURE.md](reference/core/TECHNICAL_ARCHITECTURE.md) |
| 📝 Coding style | [PYTHON_NAMING_GUIDE.md](reference/workflow/PYTHON_NAMING_GUIDE.md) |
| 🧩 Place tests | [tests/README.md](../tests/README.md) |

---

## 🔬 Parity Workflow (Quick)

The full cold-start protocol and command list live in one place — do not
duplicate them here.

**Start here:** [docs/README.md § Parity Closure Fast Path](README.md#-parity-closure-fast-path)
→ [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md) (live status)
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
| **Oracle** | Preserved MATLAB truth vectors for comparison |
| **Parity Run** | Python run compared against MATLAB oracle |
| **Certification** | Zero missing/extra on discrete/topological fields; `np.allclose` on continuous floats (ADR 0011) |
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

❌ Don't use "close enough" for parity  
✅ Zero missing/extra on discrete/topological fields; `np.allclose` on continuous floats (ADR 0011)

---

**Last Updated**: 2026-06-24  
**Related**: [README.md](README.md), [AGENTS.md](../AGENTS.md), [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)
