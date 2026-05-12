import sys
from pathlib import Path

oracle_root = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039")
print("\nALL FILES WITH 'VERT' OR 'VERTICES' IN THE ENTIRE ORACLE ROOT:")
for f in oracle_root.rglob("*"):
    if 'vert' in f.name.lower():
        print(f)
