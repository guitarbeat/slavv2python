import joblib
import numpy as np

oracle_pickle = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl"
data = joblib.load(oracle_pickle)

print(f"NORMALIZED ORACLE PICKLE CONTAINS:")
print(f" - Positions array length: {len(data['positions'])}")
print(f" - Scales array length: {len(data['scales'])}")
print(f" - Energies array length: {len(data['energies'])}")
