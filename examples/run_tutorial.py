#!/usr/bin/env python
"""
Minimal programmatic usage example for SLAVV.

Run from the repository root:
    python examples/run_tutorial.py

Or after pip install -e .:
    python -m examples.run_tutorial
"""
import numpy as np

from slavv import SLAVVProcessor
from slavv.utils import validate_parameters

# Use a small synthetic volume for the demo (simple 3D array)
image_data = np.zeros((32, 32, 32), dtype=np.float32)
image_data[8:24, 8:24, :] = 1.0  # Simple vessel-like structure

# Default parameters (validated)
params = validate_parameters({})

# Initialize and run
processor = SLAVVProcessor()
results = processor.process_image(image_data, params)

print("[Tutorial]: Vertices:", len(results["vertices"]["positions"]))
print("[Tutorial]: Edges:", len(results["edges"]["traces"]))
print("[Tutorial]: Done.")
