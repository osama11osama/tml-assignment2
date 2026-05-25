# PC worker 0 of 4 (suspects 0,4,8,...,356)
# Usage:
#   cd Assignment2
#   .\.venv\Scripts\Activate.ps1
#   .\scripts\cluster\run_local_worker0.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

if (-not (Test-Path "results\cache_master\config.json")) {
    Write-Host "Run target precompute first:"
    Write-Host "  python scripts/master_precompute_target.py --device cuda --import-legacy-40k"
    exit 1
}

python scripts/master_extract.py `
    --device cuda `
    --worker-index 0 `
    --num-workers 4 `
    --worker-name local-5060 `
    --batch-size 128

python scripts/master_status.py --worker-index 0 --num-workers 4
