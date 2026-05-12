import joblib
import numpy as np

v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
# Note: These are the actual energies we EXPECT to find at the vertex locations
v_energies = v_data.get('energies', None)
energy = e_data['energy']

print(f"Shape of energy volume: {energy.shape}")
# Pick first 5 vertices
v_rounded = np.rint(v_pos[:5]).astype(int)
print("\nFIRST 5 VERTICES:")
for i in range(5):
    print(f"Vertex {i}: {v_rounded[i]} (Expected Energy stored in data: {v_energies[i] if v_energies is not None else 'N/A'})")

print("\n--- TESTING RAW UN-TRANSPOSED VOLUME ---")
for i in range(5):
    z, y, x = v_rounded[i]
    # Wait, standard indexing [z, y, x]
    val = energy[z, y, x]
    print(f"Raw value at index [z,y,x]: {val}")

print("\n--- TESTING TRANSPOSED (0, 2, 1) VOLUME ---")
aligned_energy = np.transpose(energy, (0, 2, 1))
for i in range(5):
    z, y, x = v_rounded[i]
    # Note: We keep the SAME INDEX tuple from the vertex data file!
    val = aligned_energy[z, y, x]
    print(f"Transposed value at same indices: {val}")

# Now let's load the ORACLE vertices explicitly to see their stored coordinate layout!
oracle_v = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl")
o_pos = oracle_v['positions']
o_energies = oracle_v['energies']
print("\n--- ORACLE VERIFIED VERTICES (FIRST 5) ---")
for i in range(5):
    print(f"Oracle V {i}: {o_pos[i]} (Oracle Energy: {o_energies[i]})")
