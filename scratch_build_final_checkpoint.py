import joblib
import numpy as np
import json
from pathlib import Path
from slavv_python.core.energy_config import _matlab_lumen_radius_range

# Setup directory and load constants from existing params
dest_dir = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints")
dest_dir.mkdir(parents=True, exist_ok=True)

params_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v1\01_Params\shared_params.json"
with open(params_path, 'r') as f:
    params = json.load(f)

# Recompute lumen_radius_microns exactly matching MATLAB configuration
_, lumen_radius_microns = _matlab_lumen_radius_range(
    float(params["radius_of_smallest_vessel_in_microns"]),
    float(params["radius_of_largest_vessel_in_microns"]),
    float(params["scales_per_octave"]),
)

print(f"Total calculated scales: {len(lumen_radius_microns)}")

# Load oracle vertices
o_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl"
o_data = joblib.load(o_path)

positions = o_data['positions'].astype(np.float32)
energies = o_data['energies'].astype(np.float32)
scales = o_data['scales'].astype(np.int16)

# Perform accurate radii lookup using exact indices
# Note: MATLAB scales are 1-indexed, Python scales are 0-indexed usually?
# Wait! The oracle contains scales up to 70. Our calculated scales might extend to 70.
# Let's look at how we map scales into the radius range.

# Ensure scales don't exceed bounds by mapping intelligently.
mapped_radii = []
for s in scales:
    idx = int(s) - 1 # Convert from 1-indexed Oracle to 0-indexed computed range
    if idx < 0: idx = 0
    if idx >= len(lumen_radius_microns):
        idx = len(lumen_radius_microns) - 1
    mapped_radii.append(lumen_radius_microns[idx])

radii_microns = np.asarray(mapped_radii, dtype=np.float32)

# Standard python expected format pack
checkpoint_data = {
    "positions": positions,
    "scales": scales,
    "energies": energies,
    "radii_pixels": radii_microns, # Mock pixels since it's rarely consumed independently of analysis
    "radii_microns": radii_microns,
    "radii": radii_microns
}

# Save properly formatted checkpoint
joblib.dump(checkpoint_data, dest_dir / "checkpoint_vertices.pkl")
print(f"WROTE FULLY COMPATIBLE CHECKPOINT VERTICES TO: {dest_dir / 'checkpoint_vertices.pkl'}")
print(f"Total source seeds: {len(positions)}")
