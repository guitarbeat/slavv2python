import sys
from pathlib import Path
import argparse

# Mock context for proofs
from slavv_python.analysis.parity.models import ExactProofSourceSurface
from slavv_python.analysis.parity.execution import load_oracle_surface
from slavv_python.analysis.parity.proofs import _load_exact_vertices_payload

oracle_root = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039")
run_root = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v1")

oracle_surface = load_oracle_surface(oracle_root)
source_surface = ExactProofSourceSurface(
    run_root=run_root,
    checkpoints_dir=run_root / "02_Output" / "python_results" / "checkpoints",
    validated_params_path=run_root / "99_Metadata" / "validated_params.json",
    oracle_surface=oracle_surface,
    matlab_batch_dir=oracle_surface.matlab_batch_dir,
    matlab_vector_paths=oracle_surface.matlab_vector_paths,
)

payload = _load_exact_vertices_payload(source_surface)
print(f"VERTICES LOADED FROM MATLAB DIRECTORY: {len(payload['positions'])}")
print(f"MAX SCALE: {payload['scales'].max()}")
print(f"FIRST VERTEX POSITION: {payload['positions'][0]}")
