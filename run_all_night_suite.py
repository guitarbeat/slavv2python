import subprocess
import os
import json
from pathlib import Path

# Configuration
ORACLE_ROOT = "workspace/oracles/180709_E_batch_190910-103039"
SOURCE_RUN = "workspace/runs/oracle_180709_E/validation_strel_fix"
DEST_RUN_BASE = "workspace/runs/oracle_180709_E/all_night_suite"
LOG_FILE = "all_night_parity.log"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def run_cmd(cmd, desc):
    log(f"--- {desc} ---")
    log(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode != 0:
            log(f"ERROR (code {result.returncode}):\n{result.stderr}")
        else:
            log(f"SUCCESS:\n{result.stdout[:2000]}...") # Limit output in log
        return result
    except Exception as e:
        log(f"EXCEPTION: {str(e)}")
        return None

def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    log("Starting All-Night Comprehensive Parity Suite")
    
    # 1. Fresh baseline capture (v31)
    v31_dest = f"{DEST_RUN_BASE}_v31"
    run_cmd(
        f"python scripts/cli/parity_experiment.py capture-candidates --source-run-root {SOURCE_RUN} --oracle-root {ORACLE_ROOT} --dest-run-root {v31_dest}",
        "Capturing candidates for v31 baseline (aligned params)"
    )
    
    # 2. Detailed gap diagnosis
    run_cmd(
        f"python scripts/cli/parity_experiment.py diagnose-gaps --run-root {v31_dest}",
        "Generating detailed gap diagnosis for v31"
    )
    
    # 3. Exhaustive Tracing of top discrepant vertices
    # Based on v29 findings:
    missing_vertices = [92, 329, 230, 31, 229, 23, 29, 48, 56, 65]
    extra_vertices = [1005, 25, 593, 1367, 1373, 1371, 22, 499, 10, 222]
    
    trace_dir = Path(v31_dest) / "03_Analysis" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    for v_idx in missing_vertices:
        run_cmd(
            f"python scripts/cli/parity_experiment.py trace-vertex --source-run-root {v31_dest} --vertex-idx {v_idx} --output-trace {trace_dir}/missing_v{v_idx}.jsonl",
            f"Tracing missing vertex {v_idx}"
        )
        
    for v_idx in extra_vertices:
        run_cmd(
            f"python scripts/cli/parity_experiment.py trace-vertex --source-run-root {v31_dest} --vertex-idx {v_idx} --output-trace {trace_dir}/extra_v{v_idx}.jsonl",
            f"Tracing extra vertex {v_idx}"
        )
        
    # 4. Native Pipeline Stability Check on other TIFFs
    other_tiffs = [
        "external/neurovasc-db/data/raw/scans/180709_EL.tif",
        "external/neurovasc-db/data/raw/scans/200804.tif",
        "external/neurovasc-db/data/raw/scans/200806.tif"
    ]
    
    for tiff in other_tiffs:
        name = Path(tiff).stem
        dest = f"workspace/runs/stability_check_{name}"
        log(f"Running stability check for {tiff}...")
        # Use matlab_compat profile to exercise the watershed code
        run_cmd(
            f"python -m slavv_python.interface.cli.entrypoint run -i {tiff} -o {dest} --profile matlab_compat --edge-method watershed --number-of-edges-per-vertex 4",
            f"Stability run for {name}"
        )
        
    log("Suite Complete. Results ready for investigation.")

if __name__ == "__main__":
    main()
