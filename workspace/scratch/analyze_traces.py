import joblib
import numpy as np

import scipy.io
import glob
path = glob.glob(r'workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical\vectors\edges_*.mat')[0]
d = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
raw_traces = d['edge_space_subscripts']
# Ensure we handle potential variable-sized numpy arrays from MATLAB correctly
traces_sets = []
for t in raw_traces:
    if hasattr(t, '__len__') and len(t) > 0:
        # Handle case where trace is Nx3 array
        if hasattr(t, 'shape') and len(t.shape) == 2:
            traces_sets.append(set(map(tuple, t)))
        else:
            pass # Skip invalid scalar values if any
print(f"Analyzing {len(traces_sets)} MATLAB traces...")
num_overlaps = 0
max_overlap = 0

for i in range(len(traces_sets)):
    for j in range(i+1, len(traces_sets)):
        intersect = traces_sets[i].intersection(traces_sets[j])
        if len(intersect) > 1:
            num_overlaps += 1
            max_overlap = max(max_overlap, len(intersect))

print(f"Pairs with >1 point overlap: {num_overlaps}")
print(f"Max point overlap: {max_overlap}")

# Also check for self-cycles or duplicates within a single trace
self_repeats = 0
for i, t in enumerate(raw_traces):
    if len(t) != len(set(map(tuple, t))):
        self_repeats += 1
        print(f"Trace {i} has repeated points! Length: {len(t)}, Unique: {len(set(map(tuple, t)))}")

print(f"Traces with self-repeats: {self_repeats}")
