
import sys
import os
from pathlib import Path

# Add source to sys.path
source_path = Path("source").resolve()
sys.path.insert(0, str(source_path))

print(f"Added {source_path} to sys.path")

try:
    import slavv.evaluation.matlab_parser
    print("Successfully imported slavv.evaluation.matlab_parser")
except Exception as e:
    print(f"Failed to import slavv.evaluation.matlab_parser: {e}")

try:
    import slavv.evaluation.metrics
    print("Successfully imported slavv.evaluation.metrics")
except Exception as e:
    print(f"Failed to import slavv.evaluation.metrics: {e}")

try:
    import slavv.evaluation.setup_checks
    print("Successfully imported slavv.evaluation.setup_checks")
except Exception as e:
    print(f"Failed to import slavv.evaluation.setup_checks: {e}")
