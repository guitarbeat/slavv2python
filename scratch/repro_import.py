
import sys
from pathlib import Path

# Add the repo root to sys.path
sys.path.append(str(Path.cwd()))

try:
    from source.runtime.run_tracking import STATUS_COMPLETED
    print("SUCCESS: Import successful")
except ImportError as e:
    print(f"FAILURE: Import failed: {e}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
