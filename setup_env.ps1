# Quick Setup Script for SLAVV Development Environment
# Run this from PowerShell in the project root directory

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "SLAVV Environment Setup" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "❌ Error: pyproject.toml not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Project root detected" -ForegroundColor Green
Write-Host ""

# Ask user which method they prefer
Write-Host "Choose your virtual environment method:" -ForegroundColor Yellow
Write-Host "  1) venv (built-in Python, recommended)"
Write-Host "  2) conda (if you have Anaconda/Miniconda)"
Write-Host ""
$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host "`nCreating venv environment..." -ForegroundColor Cyan
    python -m venv .venv
    
    Write-Host "Activating environment..." -ForegroundColor Cyan
    & .\.venv\Scripts\Activate.ps1
    
    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade pip setuptools wheel
    
    Write-Host "Installing SLAVV package and dependencies..." -ForegroundColor Cyan
    pip install -e .
    
    Write-Host "Installing Jupyter support..." -ForegroundColor Cyan
    pip install jupyter ipykernel
    
    Write-Host "Registering Jupyter kernel..." -ForegroundColor Cyan
    python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
    
    Write-Host "`n✅ Setup complete!" -ForegroundColor Green
    Write-Host "`nTo activate this environment in the future, run:" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    
} elseif ($choice -eq "2") {
    Write-Host "`nCreating conda environment..." -ForegroundColor Cyan
    conda create -n slavv-env python=3.10 -y
    
    Write-Host "Activating environment..." -ForegroundColor Cyan
    conda activate slavv-env
    
    Write-Host "Installing SLAVV package and dependencies..." -ForegroundColor Cyan
    pip install -e .
    
    Write-Host "Installing Jupyter support..." -ForegroundColor Cyan
    conda install jupyter ipykernel -y
    
    Write-Host "Registering Jupyter kernel..." -ForegroundColor Cyan
    python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
    
    Write-Host "`n✅ Setup complete!" -ForegroundColor Green
    Write-Host "`nTo activate this environment in the future, run:" -ForegroundColor Yellow
    Write-Host "  conda activate slavv-env" -ForegroundColor White
    
} else {
    Write-Host "❌ Invalid choice. Exiting." -ForegroundColor Red
    exit 1
}

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "1. Open Jupyter: jupyter lab (or jupyter notebook)"
Write-Host "2. Open: scripts/0_Setup_and_Validation.ipynb"
Write-Host "3. Select kernel: 'Python (SLAVV)'"
Write-Host "4. Run all cells to validate your setup"
Write-Host ""
Write-Host "For more help, see: docs/ENVIRONMENT_SETUP.md" -ForegroundColor Yellow
