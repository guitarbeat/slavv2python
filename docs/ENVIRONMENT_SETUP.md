# Virtual Environment Setup Guide

This guide helps you set up a proper Python virtual environment for SLAVV development and analysis.

## Why Do I Need a Virtual Environment?

A virtual environment ensures:
- **Isolated dependencies**: Your SLAVV packages won't conflict with system packages
- **Reproducible setup**: Same environment across different machines
- **Clean testing**: Easy to reset if something goes wrong

## Quick Start (Recommended)

### Option 1: Using `venv` (Built-in Python)

```powershell
# Navigate to project root
cd "C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python"

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install the package in editable mode with all dependencies
pip install -e .

# Install Jupyter support
pip install jupyter ipykernel

# Register this environment as a Jupyter kernel
python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
```

### Option 2: Using Conda/Anaconda

```powershell
# Navigate to project root
cd "C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python"

# Create conda environment
conda create -n slavv-env python=3.10 -y

# Activate it
conda activate slavv-env

# Install the package in editable mode with all dependencies
pip install -e .

# Install Jupyter support
conda install jupyter ipykernel -y

# Register this environment as a Jupyter kernel
python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
```

## Using Your New Environment

### In Jupyter Notebooks

1. **Open Jupyter Lab or Notebook**:
   ```powershell
   jupyter lab
   # or
   jupyter notebook
   ```

2. **Select the correct kernel**:
   - In the notebook interface, click **Kernel** → **Change Kernel** → **Python (SLAVV)**
   - Or use the kernel selector in the top-right corner

3. **Verify the environment**:
   ```python
   import sys
   print(sys.executable)  # Should point to your .venv or conda env
   ```

### In Terminal/Command Line

Always activate the environment first:

```powershell
# For venv
.\.venv\Scripts\Activate.ps1

# For conda
conda activate slavv-env
```

You'll see `(slavv-env)` or `(.venv)` in your prompt when activated.

## Verifying Your Setup

Run this in a notebook cell or Python REPL:

```python
import sys
print(f"Python: {sys.executable}")

# Try importing all key packages
try:
    import numpy
    import scipy
    import matplotlib
    import tifffile
    from source.slavv.pipeline import SLAVVProcessor
    print("✅ All packages imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

## Troubleshooting

### Issue: "No module named 'source'"

**Solution**: Install the package in editable mode:
```powershell
pip install -e .
```

This tells Python to find `source.slavv.*` modules from your current directory.

### Issue: "No module named 'matplotlib'" (or other dependency)

**Solution**: Reinstall dependencies:
```powershell
pip install -e .
# Or force reinstall
pip install --force-reinstall -e .
```

### Issue: Jupyter notebook uses wrong kernel

**Solution**: 
1. List available kernels:
   ```powershell
   jupyter kernelspec list
   ```

2. Remove old kernels if needed:
   ```powershell
   jupyter kernelspec remove old-kernel-name
   ```

3. Re-register your environment:
   ```powershell
   # Activate your environment first!
   python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
   ```

### Issue: `pip install -e .` fails

**Possible causes**:
- **Old pip**: Update it with `python -m pip install --upgrade pip setuptools wheel`
- **Permission errors**: Don't use `sudo` or admin mode; use a virtual environment instead
- **Path issues**: Make sure you're in the project root (where `pyproject.toml` is)

## Best Practices

1. **Always activate** your environment before running scripts or notebooks
2. **Use the same environment** for development and testing
3. **Update dependencies** when `pyproject.toml` changes:
   ```powershell
   pip install --upgrade -e .
   ```
4. **Restart Jupyter kernel** after installing new packages

## Next Steps

After setting up your environment:
1. Run `scripts/0_Setup_and_Validation.ipynb` to verify everything works
2. Make sure to select the **Python (SLAVV)** kernel in the notebook
3. All validation checks should pass ✅
