import sys
from pathlib import Path
import argparse

# Mock context for proofs
from slavv_python.analysis.parity.models import ExactProofSourceSurface
from slavv_python.analysis.parity.execution import load_oracle_surface

oracle_root = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039")
oracle_surface = load_oracle_surface(oracle_root)

print(f"ORACLE MATLAB BATCH DIR: {oracle_surface.matlab_batch_dir}")
print("\nALL .MAT FILES IN THAT DIR:")
p = Path(oracle_surface.matlab_batch_dir)
for f in p.rglob("*.mat"):
    print(f)
