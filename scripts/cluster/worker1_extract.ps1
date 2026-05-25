# Worker 1 of 4
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_venv_python.ps1"
Set-Location $ProjectRoot
if (-not (Test-Path "results\cache_master\config.json")) {
    Invoke-VenvPython scripts/master_precompute_target.py --device cuda --import-legacy-40k --batch-size 128
}
Invoke-VenvPython scripts/master_extract.py --device cuda --batch-size 128 --worker-index 1 --num-workers 4 --worker-name local-worker1
Invoke-VenvPython scripts/master_status.py --worker-index 1 --num-workers 4
