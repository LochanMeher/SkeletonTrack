# PowerShell script to activate virtual environment and run the pose detection app
# Usage: .\run.ps1

# Navigate to project directory
Set-Location $PSScriptRoot

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Check Python version
Write-Host "`nPython version:" -ForegroundColor Cyan
python --version

# Run the main script
Write-Host "`nStarting pose detection application...`n" -ForegroundColor Green
python main.py
