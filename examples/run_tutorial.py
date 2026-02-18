#!/usr/bin/env python
"""
Minimal programmatic usage example for SLAVV.

Run from the repository root:
    python examples/run_tutorial.py

Or after pip install -e .:
    python -m examples.run_tutorial
"""

from slavv import SLAVVProcessor
from slavv.utils import validate_parameters, generate_synthetic_vessel_volume

# Use a synthetic volume for the demo
image_data = generate_synthetic_vessel_volume(shape=(32, 32, 32))

# Default parameters (validated)
params = validate_parameters({})

# Initialize and run
processor = SLAVVProcessor()
results = processor.process_image(image_data, params)

print("[Tutorial]: Vertices:", len(results["vertices"]["positions"]))
print("[Tutorial]: Edges:", len(results["edges"]["traces"]))
print("[Tutorial]: Done.")
