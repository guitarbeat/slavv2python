import json
from pathlib import Path

params_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v1\99_Metadata\validated_params.json"
with open(params_path, 'r') as f:
    p = json.load(f)
    
print(f"Lumen radii exist: {'lumen_radius_microns' in p}")
if 'lumen_radius_microns' in p:
    print(p['lumen_radius_microns'])
