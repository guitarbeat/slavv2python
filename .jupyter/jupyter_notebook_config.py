"""
Jupyter config to reduce temp files in the project directory.

Use with: JUPYTER_CONFIG_DIR=.jupyter jupyter notebook
Or add to your shell profile when working in this repo.
"""
import os
import tempfile

# Store checkpoints in system temp instead of .ipynb_checkpoints in each notebook dir
c.FileCheckpoints.checkpoint_dir = os.path.join(tempfile.gettempdir(), "slavv_jupyter_checkpoints")
