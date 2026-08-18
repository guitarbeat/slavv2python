# SLAVV Python

Python port of SLAVV for 3D vascular network extraction from microscopy volumes.
MATLAB source lives in `external/Vectorization-Public/source/`.

**Status:** Phase 1 is **CLOSED** — close enough to ship on the full volume, not identical last digits. Live numbers: [ONE TRUTH](docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk). Tasks: [TODO.md](docs/TODO.md). Commands: [HANDOFF](.claude/HANDOFF.md).

The public workflow is paper-first: `slavv run` with the default `paper` profile, then export `network.json`. Exact MATLAB proof is a separate developer track (`slavv parity`).

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[app,workspace]"
slavv info
slavv run -i volume.tif -o slavv_output --export csv json
```

```python
from slavv_python import SlavvPipeline, load_tiff_volume

image = load_tiff_volume("volume.tif")
pipeline = SlavvPipeline()
results = pipeline.run(image, {"pipeline_profile": "paper"})
```

More commands: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md). First extraction: [docs/TUTORIAL.md](docs/TUTORIAL.md). Method overview: [SLAVV_METHOD_EXPLAINED.md](docs/reference/core/SLAVV_METHOD_EXPLAINED.md).

## What's in this repo

| Path | What it is |
|------|------------|
| `slavv_python/` | Installable package |
| `tests/` | CI test suite |
| `docs/` | Maintained docs (start at [docs/README.md](docs/README.md)) |
| `figures/` | Publication figures |
| `scripts/` | Developer probes (not the public CLI) |
| `workspace/` | Experiment Root (binaries local/USB; scratch gitignored). See [Experiment Root](AGENTS.md#experiment-root). |
| `external/` | Vendored MATLAB source |

Folder rules: [FOLDER_PURPOSE_GUIDE.md](docs/reference/core/FOLDER_PURPOSE_GUIDE.md). Agent instructions: [AGENTS.md](AGENTS.md). Product strategy: [STRATEGY.md](STRATEGY.md).

## Where to go next

| I need… | Go here |
|---------|---------|
| Full documentation map | [docs/README.md](docs/README.md) |
| Live parity status | [ONE TRUTH](docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) |
| Stretch leftover (plain English) | [crop-energy-stretch-float-isolation.md](docs/solutions/parity/crop-energy-stretch-float-isolation.md) |
| Contribute / quality gate | [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) |
| Test placement | [tests/README.md](tests/README.md) |
