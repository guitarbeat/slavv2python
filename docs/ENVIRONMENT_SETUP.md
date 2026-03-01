# Virtual Environment Setup Guide

This guide helps you set up a Python virtual environment for SLAVV development and analysis.

## Quick Start

### Option 1: `venv` (recommended)

```powershell
cd /path/to/slavv2python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[notebooks]"
python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
```

### Option 2: Conda

```powershell
cd /path/to/slavv2python
conda create -n slavv-env python=3.10 -y
conda activate slavv-env
pip install -e .
pip install -e ".[notebooks]"
python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
```

## Using the Environment

1. Start Jupyter:

```powershell
jupyter lab
# or
jupyter notebook
```

2. In Jupyter, select **Kernel -> Change Kernel -> Python (SLAVV)**.

3. Verify interpreter:

```python
import sys
print(sys.executable)
```

## Verify Setup

```python
import sys
print(f"Python: {sys.executable}")

try:
    import numpy
    import scipy
    import matplotlib
    import tifffile
    from slavv.pipeline import SLAVVProcessor
    print("All packages imported successfully!")
except ImportError as e:
    print(f"Import failed: {e}")
```

## Troubleshooting

### `No module named 'source'`

Install in editable mode from project root:

```powershell
pip install -e .
```

### Missing dependency after install

Reinstall dependencies:

```powershell
pip install --force-reinstall -e .
```

### Wrong Jupyter kernel

```powershell
jupyter kernelspec list
python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
```

## Next Steps

1. Run `workspace/notebooks/00_Setup_and_Validation.ipynb`.
2. Select kernel `Python (SLAVV)`.
3. Confirm validation cells complete.

## Install Extras

```powershell
pip install -e .
pip install -e ".[notebooks]"
pip install -e ".[app]"
pip install -e ".[ml]"
pip install -e ".[dev]"
pip install -e ".[all]"
pip install -e ".[notebooks,dev]"
```
