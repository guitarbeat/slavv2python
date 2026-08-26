# Developer probe scripts

## In short

Local operator tools, **not** the public `slavv` CLI. Product workflows stay on
`slavv parity <subcommand>`. One-off scratch stays in `workspace/scratch/`.

```text
scripts/
├── edges/        # Watershed / Edge Selection probes
├── stretch/      # True zero-tolerance Energy isolation
├── ladder/       # Synthetic complexity ladder
├── monitor/      # Parity run health / throughput
├── docs/         # Docs integrity verifier
├── curation/     # Interactive MATLAB comparison launchers
└── profiling/    # Phase 2 baseline recorder
```

## edges/

Crop / full-volume watershed and Edge Selection probes. Run from the repo root.

| Script | Role |
|--------|------|
| [`frontier_diff.py`](edges/frontier_diff.py) | Diff Python watershed trace vs MATLAB golden frontier |
| [`candidate_gap.py`](edges/candidate_gap.py) | Oracle Edge Set coverage by raw Candidate Set emission |
| [`strel_state.py`](edges/strel_state.py) | Compare MATLAB vs Python watershed strel-state JSONL |
| [`emission_order.py`](edges/emission_order.py) | Per-vertex candidate emission-order probe |
| [`selection_funnel.py`](edges/selection_funnel.py) | Replay Edge Selection funnel vs oracle pair set |
| [`clean_edge_pairs_matlab.py`](edges/clean_edge_pairs_matlab.py) | Python vs MATLAB `clean_edge_pairs` row order |
| [`persist_crop_selection.py`](edges/persist_crop_selection.py) | Re-run crop Edge Selection on existing `candidates.pkl` |
| [`align_raw_candidates.py`](edges/align_raw_candidates.py) | Align MATLAB raw candidates onto Python vertex indexing |

```powershell
.\.venv\Scripts\python.exe scripts\edges\frontier_diff.py `
  --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
  --oracle-root workspace/oracles/180709_E_crop_M_v2 `
  --regenerate-python
```

## stretch/

Energy last-digit isolation. Python operators dropped the redundant `stretch_`
prefix; **MATLAB helpers keep function-matched names** (`stretch_energy_chunk_v202.m`
and siblings) because MATLAB requires filename = function name and the engine
host addpaths this folder.

| Script | Role |
|--------|------|
| [`engine_worker.py`](stretch/engine_worker.py) | Python 3.7 MATLAB Engine worker (called by `matlab_engine_host`) |
| [`one_production_chunk.py`](stretch/one_production_chunk.py) | Re-run one v2 production crop Energy chunk |
| [`helper_body_isolation.py`](stretch/helper_body_isolation.py) | Helper body vs original `get_energy_V202` chunk math |
| [`lattice_params_isolation.py`](stretch/lattice_params_isolation.py) | v2 lattice/params vs original MATLAB |
| [`synthetic_original_compare.py`](stretch/synthetic_original_compare.py) | Tiny synthetic helper vs original MATLAB |
| [`whole_crop_get_energy_v202.py`](stretch/whole_crop_get_energy_v202.py) | Scratch-only whole-crop MATLAB `get_energy_V202` |

## ladder/

| Script | Role |
|--------|------|
| [`run.py`](ladder/run.py) | Dual-run synthetic complexity ladder until first break |
| [`vectorize_ladder_rung.m`](ladder/vectorize_ladder_rung.m) | MATLAB driver (function name stays `vectorize_ladder_rung`) |

```powershell
.\.venv\Scripts\python.exe scripts\ladder\run.py
```

## monitor/

| Script | Role |
|--------|------|
| [`check_run.py`](monitor/check_run.py) | One-shot RUNNING/STALLED/COMPLETED/FAILED from the run dir |
| [`throughput.py`](monitor/throughput.py) | Joblib `Done N tasks` chunk rate + ETA |

```powershell
python scripts\monitor\check_run.py --run-dir workspace\runs\oracle_180709_E\canonical_full_v3
python scripts\monitor\throughput.py --run-dir <run> --log <run-log> --total-chunks <N>
```

## docs/

| Script | Role |
|--------|------|
| [`verify_integrity.py`](docs/verify_integrity.py) | Authority, banners, wiring, and relative-link checks |

```powershell
python scripts\docs\verify_integrity.py
```

## curation/

| Script | Role |
|--------|------|
| [`launch_matlab_curator_sample.m`](curation/launch_matlab_curator_sample.m) | Open the original MATLAB Vertex Curator on the preserved `y_junction_32` sample; saved edits stay in `workspace/scratch/` |

From MATLAB R2019a with the repository as the working directory:

```matlab
run('scripts/curation/launch_matlab_curator_sample.m')
```

## profiling/

| Script | Role |
|--------|------|
| [`phase2_baseline.py`](profiling/phase2_baseline.py) | Read-only Phase 2 profiling baseline from the frozen claim dest |
