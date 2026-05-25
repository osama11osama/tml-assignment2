# Worker 0 of 4 — suspects 0, 4, 8, …, 356 (90 suspects)
#   .\scripts\cluster\worker0_extract.ps1

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_venv_python.ps1"
Set-Location $ProjectRoot

Write-Host "Using Python: $VenvPython"

if (-not (Test-Path "results\cache_master\config.json")) {
    Write-Host "Running target precompute first..."
    Invoke-VenvPython scripts/master_precompute_target.py --device cuda --import-legacy-40k --batch-size 128
}

Invoke-VenvPython scripts/master_extract.py --device cuda --batch-size 128 --worker-index 0 --num-workers 4 --worker-name local-worker0

Invoke-VenvPython scripts/master_status.py --worker-index 0 --num-workers 4

Write-Host "Worker 0 done."
