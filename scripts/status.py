import json
import time
import sys
from pathlib import Path

def print_status(snapshot_path):
    if not snapshot_path.exists():
        print(f"Snapshot not found at {snapshot_path}")
        return

    with open(snapshot_path) as f:
        data = json.load(f)

    status = data.get("status", "unknown")
    overall = data.get("overall_progress", 0.0) * 100
    stage = data.get("current_stage", "idle")
    detail = data.get("current_detail", "")
    elapsed = data.get("elapsed_seconds", 0.0)
    eta = data.get("eta_seconds", 0.0)

    # Clear screen
    print("\033[H\033[J", end="")
    print("=" * 60)
    print(f" SLAVV RUN STATUS: {status.upper()}")
    print("=" * 60)
    print(f" Run ID:   {data.get('run_id')}")
    print(f" Stage:    {stage}")
    print(f" Detail:   {detail}")
    print("-" * 60)
    print(f" Progress: [{('#' * int(overall // 2)).ljust(50)}] {overall:.1f}%")
    print("-" * 60)
    print(f" Elapsed:  {elapsed/60:.1f} min")
    print(f" ETA:      {eta/60:.1f} min")
    print("=" * 60)

    for stage_name, s_data in data.get("stages", {}).items():
        s_status = s_data.get("status", "pending")
        s_prog = s_data.get("progress", 0.0) * 100
        print(f" {stage_name.ljust(12)}: {s_status.ljust(10)} {s_prog:5.1f}%")

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("workspace/runs/final_cert_run_v2/99_Metadata/run_snapshot.json")
    while True:
        try:
            print_status(path)
            time.sleep(5)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
