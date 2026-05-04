
import sys
import shutil
from pathlib import Path
import json

# Add the repo root to sys.path
sys.path.append(str(Path.cwd()))

from source.analysis.telemetry_normalizer import TelemetryNormalizer

# Setup a fake run dir
run_dir = Path("scratch/fake_run")
if run_dir.exists():
    shutil.rmtree(run_dir)
metadata_dir = run_dir / "99_Metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)

snapshot = {"status": "completed", "overall_progress": 1.0}
with open(metadata_dir / "run_snapshot.json", "w") as f:
    json.dump(snapshot, f)

normalizer = TelemetryNormalizer(output_dir="scratch/analysis_out")
results = normalizer.process_run(run_dir)

if "run_snapshot" in results:
    print("SUCCESS: Found run_snapshot.json")
else:
    print("FAILURE: Did not find run_snapshot.json in 99_Metadata via process_run")
    # Check if it would find it if it was at the root
    with open(run_dir / "run_snapshot.json", "w") as f:
        json.dump(snapshot, f)
    results_root = normalizer.process_run(run_dir)
    if "run_snapshot" in results_root:
        print("CONFIRMED: process_run expects files at the root of run_dir")
