param (
    [string]$Target = ""
)

function Install-Dependencies {
    Write-Host "Installing dependencies..."
    pip install -e ".[dev]"
    pre-commit install
}

function Format-Code {
    Write-Host "Formatting code..."
    python -m ruff format source tests
}

function Lint-Code {
    Write-Host "Linting code..."
    python -m ruff check source tests --fix
}

function Typecheck-Code {
    Write-Host "Typechecking code..."
    python -m mypy source
}

function Run-Tests {
    Write-Host "Running tests..."
    python -m pytest tests/
}

function Clean-Cache {
    Write-Host "Cleaning cache directories..."
    Get-ChildItem -Path . -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Recurse -Filter ".pytest_cache" -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Recurse -Filter ".ruff_cache" -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Recurse -Filter ".mypy_cache" -Directory | Remove-Item -Recurse -Force
}

switch ($Target) {
    "install" { Install-Dependencies }
    "format" { Format-Code }
    "lint" { Lint-Code }
    "typecheck" { Typecheck-Code }
    "test" { Run-Tests }
    "clean" { Clean-Cache }
    "all" {
        Format-Code
        Lint-Code
        Typecheck-Code
        Run-Tests
    }
    default {
        Write-Host "Usage: .\make.ps1 [install|format|lint|typecheck|test|clean|all]"
    }
}
