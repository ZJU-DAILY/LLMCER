# One-shot: create venv, install deps, run the no-API validation scripts.
# Usage (from project root, PowerShell):
#   .\setup_and_test.ps1

$ErrorActionPreference = "Stop"

# 1. Create the virtual environment if it doesn't exist.
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    python -m venv .venv
}

# 2. Activate it.
Write-Host "Activating .venv ..."
& .\.venv\Scripts\Activate.ps1

# 3. Install dependencies.
Write-Host "Upgrading pip and installing requirements ..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Run the validation scripts (no API key needed) and capture output.
New-Item -ItemType Directory -Force -Path "issue_experiments\results" | Out-Null

Write-Host "`n===== Dataset diagnostic =====" -ForegroundColor Cyan
python issue_experiments\check_datasets.py | Tee-Object -FilePath "issue_experiments\results\check_datasets.txt"

Write-Host "`n===== Per-issue tests =====" -ForegroundColor Cyan
python issue_experiments\test_issues.py | Tee-Object -FilePath "issue_experiments\results\test_issues.txt"

Write-Host "`n===== End-to-end oracle test =====" -ForegroundColor Cyan
python issue_experiments\test_end_to_end.py | Tee-Object -FilePath "issue_experiments\results\test_end_to_end.txt"

Write-Host "`nDone. Results saved under issue_experiments\results\." -ForegroundColor Green
