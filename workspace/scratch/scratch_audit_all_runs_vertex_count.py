import joblib
from pathlib import Path

runs_dir = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\runs")

for run_path in sorted(runs_dir.iterdir()):
    if not run_path.is_dir():
        continue
    
    checkpoint_path = run_path / "02_Output" / "python_results" / "checkpoints" / "checkpoint_vertices.pkl"
    if checkpoint_path.exists():
        try:
            data = joblib.load(checkpoint_path)
            pos = data.get("positions", [])
            print(f"RUN: {run_path.name: <40} -> VERTICES: {len(pos)}")
        except Exception:
            pass
