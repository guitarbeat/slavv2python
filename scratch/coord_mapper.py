import joblib
import numpy as np
import scipy.io

def find_mapping():
    ckpt_path = r"workspace\runs\measure2_experiment_evaluation\02_Output\python_results\checkpoints\checkpoint_vertices.pkl"
    mat_path = r"workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical\vectors\curated_vertices_190910-172151_tie2gfp16 9juyly2018 870nm region a-082-1.mat"
    
    ckpt = joblib.load(ckpt_path)
    mat = scipy.io.loadmat(mat_path)
    
    # Print first few checkpoint positions
    cpos = ckpt['positions'][:5]
    print("Checkpoint Positions (first 5):")
    print(cpos)
    
    # Print first few raw positions
    rpos = mat['vertex_space_subscripts'][:5]
    print("Oracle Raw Positions (first 5):")
    print(rpos)
    
    # Now let's find the MATCH for Oracle's FIRST ROW: [309, 213, 11]
    # We'll search the WHOLE checkpoint for ANY vertex near those numbers.
    all_cpos = ckpt['positions']
    print("\nSearching checkpoint for anything containing coordinate values like 309, 213, or 11...")
    matches = []
    for i in range(len(all_cpos)):
        p = all_cpos[i]
        # Test permutations of [308, 212, 10]
        target = set([308, 212, 10])
        # Accept +/- 2 range
        cnt = 0
        for val in p:
            for t in target:
                if abs(val - t) <= 1:
                    cnt += 1
        if cnt >= 3:
            matches.append((i, p))
            
    print(f"Found {len(matches)} potential matches for the first oracle vertex!")
    if matches:
        print("Sample match from checkpoint:", matches[0])

if __name__ == "__main__":
    find_mapping()
