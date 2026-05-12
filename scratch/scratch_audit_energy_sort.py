import joblib
import numpy as np

v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = np.rint(v_data['positions']).astype(int)
energy = e_data['energy']
# Energy was transposed in reality but scalar sampling doesn't strictly depend on that if we sample at raw indices first
# Let's sample at exact stored raw coordinates
v_energies = []
for i in range(len(v_pos)):
    z, y, x = v_pos[i]
    val = energy[z, y, x]
    v_energies.append(val)

v_energies = np.array(v_energies)

# Check if sorted ascending or descending
is_asc = np.all(np.diff(v_energies) >= 0)
is_desc = np.all(np.diff(v_energies) <= 0)

print(f"Number of vertices: {len(v_energies)}")
print(f"First 5 energies: {v_energies[:5]}")
print(f"Last 5 energies: {v_energies[-5:]}")
print(f"Is energy sorted ASCENDING? {is_asc}")
print(f"Is energy sorted DESCENDING? {is_desc}")

# Now check Oracle Preserved Vertices to see if THEY are sorted!
oracle_edges = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl")
# Wait, edges file doesn't list raw vertices. I should load oracle vertices
oracle_v = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl")
o_e = oracle_v['energies']
is_o_asc = np.all(np.diff(o_e) >= 0)
is_o_desc = np.all(np.diff(o_e) <= 0)
print(f"\n--- ORACLE DATA ---")
print(f"Oracle vertex count: {len(o_e)}")
print(f"Oracle first 5: {o_e[:5]}")
print(f"Oracle last 5: {o_e[-5:]}")
print(f"Is Oracle energy sorted ASCENDING? {is_o_asc}")
print(f"Is Oracle energy sorted DESCENDING? {is_o_desc}")
