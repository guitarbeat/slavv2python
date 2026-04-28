"""Wrapper to run parity experiment with correct imports."""
import sys
from pathlib import Path

# Add source directory to path so 'source' module is importable
REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
sys.path.insert(0, str(REPO_ROOT))

# Now run the parity experiment script
exec(open("dev/scripts/cli/parity_experiment.py").read())
