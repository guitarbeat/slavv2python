from pathlib import Path

oracle_root = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039")
vectors_dir = oracle_root / "01_Input" / "matlab_results" / "batch_190910-103039_canonical" / "vectors"

print("\nALL FILES IN VECTORS DIRECTORY:")
for f in sorted(vectors_dir.glob("*")):
    print(f.name)
